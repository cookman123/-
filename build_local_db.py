#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把散在 data/ 與 results/ 的 CSV 整理成一個本機 SQLite 資料庫（流程圖裡的 DB）。

用途
----
每日流程（爬蟲 → 打分 → 操作 → 回傳）與地端模型（Ollama RAG）都讀寫同一個檔：
    db/local.db
表名與欄位一律用英文，地端模型寫 SQL 或看 schema 時比較不會出錯；
說明寫在 meta 表與 地端部署說明.md。

設計原則
--------
- 股價一律經過 davis_tw_train.load_price：去掉停牌假列、套用 data/price_fixes.csv，
  和研究用的價格完全相同，不會出現「DB 裡的數字和回測對不上」。
- 每張表都附「什麼時候才知道」的欄位（news.ts、revenue.avail_date、events.exit 等），
  RAG 檢索時用它過濾，避免撈到決策當下還不知道的資料。
- 每張表獨立重建（DROP 後重寫），可以只更新其中幾張：--tables prices news

表
--
  stocks             代號、名稱、產業、市值、下市日
  prices             日線（含下市股，is_delisted 標記）
  market             大盤與海外指數日線
  chips              三大法人買賣超（股）
  valuation          本益比、股價淨值比、殖利率（證交所 BWIBBU）
  revenue            月營收（avail_date＝公告日）
  quarterly          季財報（avail_date＝公告日）
  news, news_stock   新聞（跨股票去重）與新聞—股票對應
  events             跌深事件庫（RAG 檢索庫；exit 之後答案才揭曉）
  rag_decisions      候選 RAG 規則的逐事件分數與排除結果
  strategy_daily     高波動＋趨勢過濾策略的逐日報酬
  significance       週期性邏輯檢驗的逐期結果
  picks              最新一次換倉清單
  ops_log            每日操作回寫（流程圖最後一步；建立空表）
  meta               建置時間、來源、各表筆數與說明
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import davis_tw_train as T  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES = ROOT / "results"
DB = ROOT / "db" / "local.db"

DESCRIPTIONS = {
    "stocks": "股票基本資料：code 代號、name 名稱、category 產業、market_cap 市值、delist_date 下市日（未下市為空）",
    "prices": "日線 OHLCV（已還原、已套 price_fixes 修正；is_delisted=1 為下市股）",
    "market": "大盤與海外指數日線；symbol 如 TWII、SOX、GSPC",
    "chips": "三大法人買賣超（股）；收盤後公布，當日收盤後才知道",
    "valuation": "證交所本益比 pe、股價淨值比 pb、殖利率 yield_pct",
    "revenue": "月營收；yoy_pct 年增率；avail_date 公告日，之前不能用",
    "quarterly": "季財報；avail_date 公告日，之前不能用",
    "news": "新聞；ts 發布時間（台北時間），收盤後發布者要算到下一個交易日",
    "news_stock": "新聞與股票的對應（同一則新聞可對到多檔）",
    "events": "跌深事件（5 日內跌 8%）；signal 訊號日、entry 進場、exit 出場、net 淨報酬；exit 之後答案才揭曉",
    "rag_decisions": "候選 RAG 規則：score 相似事件加權淨報酬、threshold 滾動門檻、excluded=1 不做",
    "strategy_daily": "高波動＋120 日均線過濾策略逐日報酬；strategy 含過濾、strategy_nofilter 不含、market 大盤、in_market 是否持有",
    "significance": "週期性邏輯檢驗（每 42 交易日）；level 正常／注意／顯著",
    "picks": "最新一次換倉清單（規則輸出，不是投資建議）",
    "ops_log": "每日操作回寫：date、code、action（buy/sell/skip）、rule、score、note",
}


def _write(con: sqlite3.Connection, name: str, df: pd.DataFrame, index: list | None = None,
           pk: list | None = None) -> int:
    """整張表重建；pk 給了就建唯一索引，index 是一般索引。"""
    con.execute(f'DROP TABLE IF EXISTS "{name}"')
    df.to_sql(name, con, index=False, chunksize=50_000)
    if pk:
        con.execute(f'CREATE UNIQUE INDEX "ux_{name}" ON "{name}" ({", ".join(pk)})')
    for col in index or []:
        con.execute(f'CREATE INDEX "ix_{name}_{col}" ON "{name}" ({col})')
    con.commit()
    return len(df)


def _d(s: pd.Series) -> pd.Series:
    """日期統一存成 YYYY-MM-DD 字串，SQLite 才能直接比大小。"""
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _codes() -> list:
    return sorted(x.name for x in (DATA / "stocks").iterdir()
                  if x.is_dir() and any((x / "price").glob("*.csv")))


# ------------------------------------------------------------------ 各表

def build_stocks(con) -> int:
    m = pd.DataFrame(json.loads((DATA / "industry_stock_map.json").read_text(encoding="utf-8")))
    m["code"] = m["code"].astype(str)
    dl = pd.read_csv(DATA / "delisted" / "delisted.csv", dtype={"code": str}, encoding="utf-8-sig")
    dl = dl.rename(columns={"delist_date": "delist_date"})[["code", "name", "delist_date"]]
    df = pd.concat([m, dl[~dl["code"].isin(m["code"])]], ignore_index=True)
    df = df.merge(dl[["code", "delist_date"]], on="code", how="left", suffixes=("_x", ""))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_x")])
    df["delist_date"] = _d(df["delist_date"])
    return _write(con, "stocks", df[["code", "name", "category", "market_cap", "delist_date"]],
                  pk=["code"])


def build_prices(con) -> int:
    parts = []
    for code in _codes():
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        p = px.reset_index()
        p["code"], p["is_delisted"] = code, 0
        parts.append(p)
    for f in sorted((DATA / "delisted" / "price_adj").glob("*.csv")):
        if f.stem.startswith("_"):            # _events.csv 是除權息事件紀錄，不是股價
            continue
        p = pd.read_csv(f, encoding="utf-8-sig")
        p["code"], p["is_delisted"] = f.stem, 1
        parts.append(p[["date", "open", "high", "low", "close", "volume", "code", "is_delisted"]])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = _d(df["date"])
    df = df.dropna(subset=["date", "close"]).drop_duplicates(["code", "date"], keep="first")
    return _write(con, "prices", df[["code", "date", "open", "high", "low", "close",
                                     "volume", "is_delisted"]],
                  pk=["code", "date"], index=["date"])


def build_market(con) -> int:
    parts = []
    for f in sorted((DATA / "market").glob("*.csv")):
        if f.stem == "turnover_rank" or f.stem.endswith("_recent"):
            continue
        p = pd.read_csv(f, encoding="utf-8-sig")
        if not {"date", "close"} <= set(p.columns):
            continue
        p["symbol"] = f.stem
        parts.append(p[["symbol", "date"] + [c for c in ("open", "high", "low", "close", "volume")
                                             if c in p.columns]])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = _d(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(["symbol", "date"], keep="last")
    return _write(con, "market", df, pk=["symbol", "date"])


def build_chips(con) -> int:
    parts = []
    for f in sorted((DATA / "chips").glob("*.csv")):
        p = pd.read_csv(f, encoding="utf-8-sig")
        p["code"] = f.stem
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = _d(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(["code", "date"], keep="last")
    cols = ["code", "date"] + [c for c in ("foreign_investor", "investment_trust", "dealer",
                                           "dealer_self", "dealer_hedge", "inst_total")
                               if c in df.columns]
    return _write(con, "chips", df[cols], pk=["code", "date"])


def build_valuation(con) -> int:
    df = pd.concat([pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
                    for f in sorted((DATA / "fundamentals").glob("bwibbu_*.csv"))],
                   ignore_index=True)
    df["code"] = df["code"].str.strip()
    df["date"] = _d(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(["code", "date"], keep="last")
    return _write(con, "valuation", df[["code", "date", "close", "pe", "pb", "yield_pct"]],
                  pk=["code", "date"])


def build_financials(con) -> int:
    rv = pd.read_csv(DATA / "financials" / "revenue.csv", dtype={"code": str}, encoding="utf-8-sig")
    rv["avail_date"] = _d(rv["avail_date"])
    n = _write(con, "revenue", rv.drop_duplicates(["code", "ym"], keep="last"),
               pk=["code", "ym"], index=["avail_date"])
    q = pd.read_csv(DATA / "financials" / "quarterly.csv", dtype={"code": str}, encoding="utf-8-sig")
    q["avail_date"] = _d(q["avail_date"])
    n += _write(con, "quarterly", q.drop_duplicates(["code", "year", "q"], keep="last"),
                pk=["code", "year", "q"], index=["avail_date"])
    return n


def _news_id(row) -> str:
    """鉅亨有 news_id；Yahoo 沒有，用網址（沒有網址就用時間＋標題）雜湊。"""
    if isinstance(row.get("news_id"), (int, float, str)) and str(row.get("news_id")) not in ("", "nan"):
        return f"cnyes:{int(float(row['news_id']))}"
    key = row.get("url") if isinstance(row.get("url"), str) and row.get("url") else \
        f"{row.get('date')}|{row.get('title')}"
    return f"{row.get('source_tag', 'x')}:{hashlib.md5(key.encode('utf-8')).hexdigest()[:16]}"


def build_news(con) -> int:
    seen: set = set()
    news, links = [], []
    codes = _codes()
    for i, code in enumerate(codes, 1):
        for pattern, tag in ((f"cnyes_news_{code}.csv", "cnyes"), (f"yahoo_news_{code}.csv", "yahoo")):
            f = DATA / "stocks" / code / "news" / pattern
            if not f.exists():
                continue
            d = pd.read_csv(f, encoding="utf-8-sig")
            if "date" not in d.columns:
                continue
            d["source_tag"] = tag
            d["nid"] = [_news_id(r) for r in d.to_dict("records")]
            links.append(pd.DataFrame({"news_id": d["nid"], "code": code}))
            new = d[~d["nid"].isin(seen)].drop_duplicates("nid")
            seen.update(new["nid"])
            ts = pd.to_datetime(new["date"], errors="coerce", utc=True)
            news.append(pd.DataFrame({
                "news_id": new["nid"],
                "ts": ts.dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": new.get("title", pd.Series("", index=new.index)).fillna("").astype(str),
                "summary": new.get("summary", pd.Series("", index=new.index)).fillna("").astype(str),
                "text": new.get("text", pd.Series("", index=new.index)).fillna("").astype(str),
                "source": new.get("source", pd.Series(tag, index=new.index)).fillna(tag).astype(str),
                "url": new.get("url", pd.Series("", index=new.index)).fillna("").astype(str),
            }))
        if i % 100 == 0:
            print(f"    新聞 {i}/{len(codes)} 檔，累計 {len(seen):,} 則", flush=True)
    nw = pd.concat(news, ignore_index=True).dropna(subset=["ts"])
    lk = pd.concat(links, ignore_index=True).drop_duplicates()
    lk = lk[lk["news_id"].isin(set(nw["news_id"]))]
    n = _write(con, "news", nw, pk=["news_id"], index=["ts"])
    _write(con, "news_stock", lk, pk=["news_id", "code"], index=["code"])
    return n


def build_research(con) -> int:
    """研究產出：事件庫、RAG 決策、策略逐日報酬、檢驗結果、換倉清單。"""
    n = 0
    ev_f = RES / "event_model" / "events.csv"
    if ev_f.exists():
        ev = pd.read_csv(ev_f, dtype={"code": str}, encoding="utf-8-sig")
        for c in ("news_day", "drop_day", "signal", "entry", "exit"):
            if c in ev.columns:
                ev[c] = _d(ev[c])
        ev.insert(0, "event_id", ev["code"] + "_" + ev["entry"].astype(str))
        ev = ev.drop_duplicates("event_id")
        n += _write(con, "events", ev, pk=["event_id"], index=["signal", "exit", "code"])
    rd_f = RES / "rag_rule" / "decisions.csv"
    if rd_f.exists():
        rd = pd.read_csv(rd_f, dtype={"code": str}, encoding="utf-8-sig").rename(
            columns={"門檻": "threshold", "排除": "excluded"})
        for c in ("entry", "exit"):
            rd[c] = _d(rd[c])
        rd["excluded"] = rd["excluded"].astype(bool).astype(int)
        n += _write(con, "rag_decisions", rd, index=["entry"])
    sd_f = RES / "spec" / "daily.csv"
    if sd_f.exists():
        sd = pd.read_csv(sd_f, encoding="utf-8-sig").rename(columns={
            "策略": "strategy", "無過濾": "strategy_nofilter", "大盤": "market", "在場": "in_market"})
        sd["date"] = _d(sd["date"])
        n += _write(con, "strategy_daily", sd, pk=["date"])
    sg_f = RES / "significance" / "history.csv"
    if sg_f.exists():
        sg = pd.read_csv(sg_f, encoding="utf-8-sig").rename(columns={
            "日期": "check_date", "本期超額": "excess", "百分位": "pctile",
            "第5百分位": "p05", "第10百分位": "p10", "前期超額": "prev_excess",
            "前期百分位": "prev_pctile", "近一年日超額": "recent_daily_excess",
            "參考期日超額": "ref_daily_excess", "漂移t": "drift_t", "漂移p": "drift_p",
            "A單期極端": "rule_a", "B連續偏弱": "rule_b", "C長期漂移": "rule_c", "結論": "level"})
        sg["check_date"] = _d(sg["check_date"])
        for c in ("rule_a", "rule_b", "rule_c"):
            sg[c] = sg[c].astype(bool).astype(int)
        n += _write(con, "significance", sg, pk=["check_date"])
    pk_f = RES / "spec" / "current_picks.csv"
    if pk_f.exists():
        pk = pd.read_csv(pk_f, dtype={"代號": str}, encoding="utf-8-sig").rename(columns={
            "代號": "code", "名稱": "name", "產業": "category", "排名": "rank",
            "年化波動": "ann_vol", "動作": "action"})
        pk.insert(0, "asof", time.strftime("%Y-%m-%d"))
        n += _write(con, "picks", pk)
    return n


def build_ops_log(con) -> int:
    """流程圖最後一步的回寫表；已存在就保留（裡面是實際操作紀錄，不能重建）。"""
    con.execute("""CREATE TABLE IF NOT EXISTS ops_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL, code TEXT NOT NULL, action TEXT NOT NULL,
        rule TEXT, score REAL, note TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')))""")
    con.commit()
    return con.execute("SELECT COUNT(*) FROM ops_log").fetchone()[0]


BUILDERS = {
    "stocks": build_stocks, "prices": build_prices, "market": build_market,
    "chips": build_chips, "valuation": build_valuation, "financials": build_financials,
    "news": build_news, "research": build_research, "ops_log": build_ops_log,
}


def write_meta(con) -> None:
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('meta','sqlite_sequence') "
        "AND name NOT LIKE 'kb_%' AND name NOT LIKE 'sqlite_%'")]
    rows = [("built_at", time.strftime("%Y-%m-%d %H:%M:%S"))]
    for t in tables:
        cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        rows.append((f"rows.{t}", str(cnt)))
        if t in DESCRIPTIONS:
            rows.append((f"desc.{t}", DESCRIPTIONS[t]))
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", rows)
    con.commit()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="整理成本機 SQLite（db/local.db）")
    ap.add_argument("--tables", nargs="*", default=list(BUILDERS),
                    help=f"只重建這幾組：{' '.join(BUILDERS)}")
    ap.add_argument("--db", default=str(DB))
    a = ap.parse_args(argv)

    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.db)
    con.execute("PRAGMA journal_mode=WAL")
    for name in a.tables:
        t0 = time.time()
        print(f"  {name:<12}…", end="", flush=True)
        n = BUILDERS[name](con)
        print(f" {n:>12,} 列　{time.time() - t0:6.1f} 秒", flush=True)
    write_meta(con)
    con.execute("PRAGMA optimize")
    con.close()
    print(f"\n  完成：{a.db}（{Path(a.db).stat().st_size / 1e9:.2f} GB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
