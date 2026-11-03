#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盤中即時報價：開盤時間每隔幾秒抓證交所 MIS 的即時行情（上市、上櫃都有），存在 db/local.db：
  quotes        每檔當天最新一筆（開、高、低、現價、昨收、累計量、時間）
  quote_ticks   當天逐筆現價，畫盤中走勢用；只留最近 7 天
  watchlist     自選清單，前端「個股」頁可以加減

追蹤哪些股票（watch_codes）：所有模擬帳戶的持股與待成交委託、今天的候選股（5 日模型前 12 名，
和 LLM 選股同一個候選池）、自選清單，再加 0050（閒置資金）和加權指數 t00。每分鐘重讀一次，
盤中新下的單或新加的自選也會跟著追。

用途
----
  1. 前端「個股」「模擬交易」頁顯示盤中現價
  2. 模擬帳戶盤中成交：paper_trade.settle 把今天的盤中 K 棒（live_bars）接在日線後面，
     昨晚下的單用今天開盤價成交，停損／停利碰到就成交，規則和日線版相同（forecast.simulate_exit）。
     已成交的單之後結算不會再改，晚上日線進來也不會被改掉。
  3. 地端問答（local_ask.py）視角是今天時，帶入盤中價

價格口徑：DB 的 prices／etf_prices 是還原價，盤中報價是原始價。live_bars 乘上「DB 最後一天收盤 ÷
證交所昨收」接在日線後面才連續（0050 的含息還原價差很多，個股多半剛好是 1）。
DB 還沒有上一個交易日的日線時不給盤中 K 棒：中間缺一天的話，委託會成交在錯的日子。

證交所沒有公開說明頻率限制，一般建議每 5 秒不超過 3 次請求；這裡一輪只發一兩次請求、每 10 秒一輪。
08:30～09:00 的試撮價格不是真的成交，所以 09:00 才開始抓。

用法
----
  python intraday.py              開盤時間內每 10 秒抓一次，收盤後結束（排程 08:55 啟動，會等到 09:00）
  python intraday.py --once       只抓一次（測試用，不管現在幾點）
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / "db" / "local.db"
MIS = "https://mis.twse.com.tw/stock"
INDEX, ETF = "t00", "0050"
OPEN, CLOSE, END = "09:00:00", "13:30:00", "13:36:00"   # 13:30 收盤集合競價的結果要等一下才出來
BATCH = 50                                              # 一次請求幾檔（太長的網址會被擋）
KEEP_DAYS = 7


def connect(db: str | Path = DB) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=30, check_same_thread=False)
    ensure(con)
    return con


def ensure(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS quotes (
        code TEXT PRIMARY KEY, name TEXT, ex TEXT, date TEXT, time TEXT,
        open REAL, high REAL, low REAL, last REAL, prev_close REAL, volume REAL, updated_at TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS quote_ticks (code TEXT, ts TEXT, price REAL, volume REAL)")
    con.execute("CREATE INDEX IF NOT EXISTS quote_ticks_code_ts ON quote_ticks (code, ts)")
    con.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        code TEXT PRIMARY KEY, added_at TEXT DEFAULT (datetime('now', 'localtime')))""")
    con.commit()


# ---------------------------------------------------------------- 自選清單

def watchlist(con) -> list[str]:
    ensure(con)
    return [c for (c,) in con.execute("SELECT code FROM watchlist ORDER BY added_at")]


def add_watch(con, code: str) -> None:
    ensure(con)
    con.execute("INSERT OR IGNORE INTO watchlist (code) VALUES (?)", [code])
    con.commit()


def remove_watch(con, code: str) -> None:
    con.execute("DELETE FROM watchlist WHERE code = ?", [code])
    con.commit()


def candidates(con) -> list[str]:
    """今天的候選股：最新一天 5 日模型前 12 名（llm_judge.candidates，不查處置股，免得多打外部網站）。"""
    try:
        import llm_judge as J
        return J.candidates(con, datetime.now().strftime("%Y-%m-%d"), 12, exclude_restricted=False)["code"] \
            .astype(str).tolist()
    except Exception:
        return []


def watch_codes(con, cands: list[str] | None = None) -> list[str]:
    held = [c for (c,) in con.execute(
        "SELECT DISTINCT code FROM paper_orders WHERE status IN ('pending', 'filled')")] \
        if con.execute("SELECT 1 FROM sqlite_master WHERE name = 'paper_orders'").fetchone() else []
    codes = [INDEX, ETF] + held + (cands or []) + watchlist(con)
    return list(dict.fromkeys(str(c) for c in codes))


# ---------------------------------------------------------------- 抓報價

def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


class Feed:
    """證交所 MIS。先開首頁拿 cookie，之後同一個 session 連續查。上市或上櫃不知道時兩個都問，記住有回應的那個。"""

    def __init__(self):
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0"
        self.ex: dict[str, str] = {INDEX: "tse", ETF: "tse"}
        self.ready = False

    def fetch(self, codes: list[str]) -> list[dict]:
        if not self.ready:
            self.s.get(f"{MIS}/index.jsp", timeout=10)
            self.ready = True
        keys = []
        for c in codes:
            keys += [f"{self.ex[c]}_{c}.tw"] if c in self.ex else [f"tse_{c}.tw", f"otc_{c}.tw"]
        out = []
        for i in range(0, len(keys), BATCH):
            r = self.s.get(f"{MIS}/api/getStockInfo.jsp", timeout=10, params={
                "ex_ch": "|".join(keys[i:i + BATCH]), "json": "1", "delay": "0", "_": int(time.time() * 1000)})
            r.encoding = "utf-8"
            for m in r.json().get("msgArray", []):
                if m.get("c") and m.get("d"):
                    self.ex[m["c"]] = m["ex"]
                    out.append(m)
            if i + BATCH < len(keys):
                time.sleep(1)
        return out


def _bid(m: dict):
    b = (m.get("b") or "").split("_")[0]
    return _f(b)


def save(con, msgs: list[dict]) -> int:
    """寫進 quotes 與 quote_ticks。這一刻沒有成交（z 是 -）時沿用上一筆現價，沒有就用最佳買價。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old = dict(con.execute("SELECT code, last FROM quotes WHERE date = ?", [datetime.now().strftime("%Y-%m-%d")]))
    rows, ticks = [], []
    for m in msgs:
        d = f"{m['d'][:4]}-{m['d'][4:6]}-{m['d'][6:]}"
        last = _f(m.get("z")) or old.get(m["c"]) or _bid(m)
        vol = _f(m.get("v"))
        rows.append((m["c"], m.get("n"), m.get("ex"), d, m.get("t"), _f(m.get("o")), _f(m.get("h")), _f(m.get("l")),
                     last, _f(m.get("y")), vol, now))
        if last:
            ticks.append((m["c"], f"{d} {m.get('t')}", last, vol))
    con.executemany("INSERT OR REPLACE INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.executemany("INSERT INTO quote_ticks VALUES (?, ?, ?, ?)", ticks)
    con.commit()
    return len(rows)


# ---------------------------------------------------------------- 給其他程式用

def quote(con, code: str) -> dict | None:
    """這檔今天的盤中報價（原始價）；今天沒有就回傳 None。"""
    try:
        r = pd.read_sql("SELECT * FROM quotes WHERE code = ? AND date = ?", con,
                        params=[code, datetime.now().strftime("%Y-%m-%d")])
    except Exception:
        return None
    return None if r.empty else r.iloc[0].to_dict()


def ticks(con, code: str) -> pd.DataFrame:
    """這檔今天的逐筆現價（index 是時間）。"""
    try:
        t = pd.read_sql("SELECT ts, price, volume FROM quote_ticks WHERE code = ? AND ts >= ? ORDER BY ts", con,
                        params=[code, datetime.now().strftime("%Y-%m-%d")], parse_dates=["ts"])
    except Exception:
        return pd.DataFrame(columns=["price", "volume"])
    return t.drop_duplicates("ts", keep="last").set_index("ts")


def is_open(now: datetime | None = None) -> bool:
    """平日 09:00～13:36（國定假日沒有扣；那天沒有報價，頁面會顯示沒有盤中資料）。"""
    now = now or datetime.now()
    return now.weekday() < 5 and OPEN <= now.strftime("%H:%M:%S") <= END


def live_bars(con) -> dict:
    """今天的盤中 K 棒，換算成 DB 的還原價口徑：{code: {date, open, high, low, close}}。
    DB 已經有今天的日線，或還沒有上一個交易日的日線，就回傳空的（見檔頭）。"""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        q = pd.read_sql("SELECT code, open, high, low, last, prev_close FROM quotes WHERE date = ? AND code != ? "
                        "AND open IS NOT NULL AND prev_close IS NOT NULL", con, params=[today, INDEX])
    except Exception:
        return {}
    last = con.execute("SELECT MAX(date) FROM market WHERE symbol = 'TWII'").fetchone()[0]
    if q.empty or not last or last >= today or np.busday_count(
            (pd.Timestamp(last) + timedelta(days=1)).date(), pd.Timestamp(today).date()) > 0:
        return {}
    codes = q["code"].tolist()
    ph = ",".join("?" * len(codes))
    base = dict(con.execute(f"SELECT code, close FROM prices WHERE date = ? AND code IN ({ph}) UNION ALL "
                            f"SELECT code, close FROM etf_prices WHERE date = ? AND code IN ({ph})",
                            [last, *codes, last, *codes]))
    out = {}
    for r in q.itertuples():
        if r.code not in base:
            continue
        k = base[r.code] / r.prev_close
        out[r.code] = dict(date=today, open=r.open * k, high=(r.high or r.open) * k, low=(r.low or r.open) * k,
                           close=(r.last or r.open) * k)
    return out


# ---------------------------------------------------------------- 主程式

def _settle(con) -> int:
    """有待成交的單才結算（盤中停損／停利、昨晚下的單開盤成交）。"""
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name = 'paper_orders'").fetchone() or \
            not con.execute("SELECT 1 FROM paper_orders WHERE status = 'pending' LIMIT 1").fetchone():
        return 0
    import paper_trade as P
    return P.settle(con, None)


def run(every: float = 10, once: bool = False) -> int:
    con = connect()
    feed = Feed()
    today = datetime.now().strftime("%Y-%m-%d")
    con.execute("DELETE FROM quote_ticks WHERE ts < ?",
                [(datetime.now() - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")])
    con.commit()
    if not once:
        wait = (datetime.strptime(f"{today} {OPEN}", "%Y-%m-%d %H:%M:%S") - datetime.now()).total_seconds()
        if wait > 0:
            print(f"等到 {OPEN} 開盤（{wait:.0f} 秒）", flush=True)
            time.sleep(wait + 3)
    cands = candidates(con)
    codes, t_codes, n, holiday_tries = [], 0.0, 0, 0
    while True:
        if time.time() - t_codes > 60:
            codes, t_codes = watch_codes(con, cands), time.time()
        try:
            msgs = feed.fetch(codes)
        except Exception as e:                       # 網路斷一下就下一輪再試
            print(f"{datetime.now():%H:%M:%S} 抓取失敗：{type(e).__name__}: {e}", flush=True)
            feed.ready = False
            msgs = []
        idx = next((m for m in msgs if m["c"] == INDEX), None)
        if msgs and not once and idx and idx["d"] != today.replace("-", ""):
            # 證交所還是昨天的資料：國定假日，或剛開盤還沒更新
            holiday_tries += 1
            if holiday_tries >= 30:
                print(f"證交所的資料日期是 {idx['d']}，今天沒有開盤，結束。", flush=True)
                return 0
        elif msgs:
            save(con, msgs)
            n += 1
            try:
                filled = _settle(con)
            except Exception as e:
                filled = 0
                print(f"{datetime.now():%H:%M:%S} 結算失敗：{type(e).__name__}: {e}", flush=True)
            if filled or n % 30 == 1:
                print(f"{datetime.now():%H:%M:%S} 第 {n} 輪：{len(msgs)}／{len(codes)} 檔"
                      + (f"，加權 {idx['z']}" if idx else "") + (f"，模擬帳戶新成交 {filled} 筆" if filled else ""),
                      flush=True)
        if once or datetime.now().strftime("%H:%M:%S") > END:
            break
        time.sleep(every)
    print(f"收盤，共抓 {n} 輪。", flush=True)
    return 0


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="盤中即時報價")
    ap.add_argument("--every", type=float, default=10, help="幾秒抓一次（預設 10）")
    ap.add_argument("--once", action="store_true", help="只抓一次")
    a = ap.parse_args(argv)
    return run(a.every, a.once)


if __name__ == "__main__":
    sys.exit(main())
