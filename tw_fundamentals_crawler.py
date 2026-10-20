#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面資料爬蟲：證交所「個股日本益比、殖利率及股價淨值比」（BWIBBU_d）。

為什麼需要這一份
----------------
專案目前只有股價與三大法人買賣超，完全沒有基本面。要把「跌深反彈」策略加上
基本面判斷（這一跌是錯殺，還是公司真的變差了），至少需要估值指標。

這個端點的好處是一次回傳全市場約 1,000 檔、且是逐日的，所以 15 年也只要
約 3,600 次請求；若改用「一檔一個月」的端點，383 檔 × 180 個月要近 7 萬次。

欄位
----
    本益比       盈餘的倍數；跌深後本益比同步下降，代表價格跌但盈餘還在
    股價淨值比   相對帳面價值；用來判斷是否跌破淨值
    殖利率       股利 / 股價；跌深後殖利率上升，是常見的支撐依據
    財報年/季    上述數字根據哪一期財報，用來確認資料的新鮮度

注意：這些是「以最近一期財報計算」的靜態指標，不是預估值。
公司獲利真正轉差時，本益比會先因股價下跌而變低、等財報公布後才跳高，
所以低本益比本身不等於便宜——這一點在使用時必須考慮。

輸出：data/fundamentals/bwibbu_{年}.csv
斷點續跑：data/fundamentals/checkpoint.json 記錄已完成的日期。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tw_stock_crawler as C  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "fundamentals"
CKPT = OUT / "checkpoint.json"
URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"

COLS = ["code", "name", "close", "yield_pct", "div_year", "pe", "pb", "fin_period"]


def trading_days(start: str, end: str) -> list:
    """用加權指數的交易日當行事曆，避免對非交易日發出無效請求。"""
    f = ROOT / "data" / "market" / "TWII.csv"
    d = pd.read_csv(f, parse_dates=["date"])["date"]
    d = d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    return [x.strftime("%Y%m%d") for x in sorted(d)]


def load_ckpt() -> set:
    if CKPT.exists():
        return set(json.loads(CKPT.read_text(encoding="utf-8")))
    return set()


def save_ckpt(done: set) -> None:
    CKPT.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")


# 證交所的欄位會隨年代改變：2017 年中以前只有 5 欄
# ['證券代號','證券名稱','本益比','殖利率(%)','股價淨值比']，之後變成 8 欄
# ['證券代號','證券名稱','收盤價','殖利率(%)','股利年度','本益比','股價淨值比','財報年/季']。
# 所以絕對不能按位置硬套欄名——那會讓舊格式的本益比被存成「收盤價」、
# 股價淨值比被存成「股利年度」。一律依回傳的 fields 名稱對應。
FIELD_MAP = {
    "證券代號": "code", "證券名稱": "name", "收盤價": "close",
    "殖利率(%)": "yield_pct", "股利年度": "div_year",
    "本益比": "pe", "股價淨值比": "pb", "財報年/季": "fin_period",
}


def fetch_day(date_str: str) -> pd.DataFrame:
    d = C.http_get(URL, params={"date": date_str, "selectType": "ALL",
                                "response": "json"})
    if d.get("stat") != "OK":
        return pd.DataFrame()
    rows = d.get("data") or []
    fields = d.get("fields") or []
    if not rows or not fields:
        return pd.DataFrame()
    names = [FIELD_MAP.get(str(f).strip(), str(f).strip()) for f in fields]
    df = pd.DataFrame(rows, columns=names[:len(rows[0])])
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df["date"] = pd.to_datetime(date_str)
    for c in ("close", "yield_pct", "pe", "pb"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "")
                                  .replace({"-": None, "": None}), errors="coerce")
    return df


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="證交所基本面（本益比／股價淨值比／殖利率）")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--delay", type=float, default=1.2, help="每次請求間隔秒數")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    days = trading_days(args.start, args.end)
    done = load_ckpt()
    todo = [d for d in days if d not in done]
    print(f"交易日 {len(days)} 天，已完成 {len(done)}，待抓 {len(todo)}")
    print(f"預估約 {len(todo) * args.delay / 60:.0f} 分鐘\n")

    buf, n_ok, n_empty = {}, 0, 0
    for i, ds in enumerate(todo, 1):
        try:
            df = fetch_day(ds)
        except Exception as exc:  # noqa: BLE001
            print(f"  {ds} 失敗：{type(exc).__name__}")
            time.sleep(args.delay * 2)
            continue
        if df.empty:
            n_empty += 1
        else:
            buf.setdefault(ds[:4], []).append(df)
            n_ok += 1
        done.add(ds)

        # 每 100 天寫一次檔並存檔斷點，中途關掉不會全部重來
        if i % 100 == 0 or i == len(todo):
            for year, frames in buf.items():
                path = OUT / f"bwibbu_{year}.csv"
                new = pd.concat(frames, ignore_index=True)
                if path.exists():
                    old = pd.read_csv(path, parse_dates=["date"])
                    new = pd.concat([old, new], ignore_index=True)
                new = new.drop_duplicates(subset=["date", "code"], keep="last")
                new.sort_values(["date", "code"]).to_csv(
                    path, index=False, encoding="utf-8-sig")
            buf.clear()
            save_ckpt(done)
            print(f"  進度 {i}/{len(todo)}　成功 {n_ok}　空白 {n_empty}", flush=True)
        time.sleep(args.delay)

    print("\n完成。各年度檔案：")
    for f in sorted(OUT.glob("bwibbu_*.csv")):
        d = pd.read_csv(f)
        print(f"  {f.name}　{len(d):,} 列　{d['code'].nunique()} 檔　"
              f"{d['date'].min()} ~ {d['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
