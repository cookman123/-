#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取指數型標的（ETF 與指數）的歷史報價，供 news_reversal.py 的大盤策略使用。

為什麼要多抓幾檔
----------------
只用加權指數（TWII）會有兩個限制：一是它是價格指數、不可直接交易，
二是單一標的的事件數太少（15 年只有 17~20 次符合條件的下跌），
掃參數後根本無法通過多重檢定。多找幾個「指數型、不會下市」的標的，
就能在不引入生存者偏誤的前提下把樣本數撐起來。

選取原則：全部是追蹤指數或一籃子的標的，不含個股，因此沿用
「指數不會下市」的性質。資料期間較短的（例如 2020 年才上市的高股息 ETF）
也納入——樣本短不是問題，把它們合起來看才是重點。

輸出：data/market/{代號}.csv，欄位與 TWII.csv 一致（date/open/high/low/close/volume）。
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tw_stock_crawler as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "market"

# (Yahoo 代號, 檔名, 中文名稱, 類型)
TARGETS = [
    ("0050.TW",   "0050",   "元大台灣50",        "大盤型"),
    ("006208.TW", "006208", "富邦台50",          "大盤型"),
    ("0051.TW",   "0051",   "元大中型100",       "中型股"),
    ("0056.TW",   "0056",   "元大高股息",        "高股息"),
    ("00878.TW",  "00878",  "國泰永續高股息",    "高股息"),
    ("00713.TW",  "00713",  "元大台灣高息低波",  "高股息"),
    ("00850.TW",  "00850",  "元大臺灣ESG永續",   "大盤型"),
    ("00692.TW",  "00692",  "富邦公司治理",      "大盤型"),
    ("0052.TW",   "0052",   "富邦科技",          "產業型"),
    ("0053.TW",   "0053",   "元大電子",          "產業型"),
    ("0055.TW",   "0055",   "元大MSCI金融",      "產業型"),
    ("00881.TW",  "00881",  "國泰台灣5G+",       "產業型"),
    ("00631L.TW", "00631L", "元大台灣50正2",     "槓桿型"),
    ("00632R.TW", "00632R", "元大台灣50反1",     "反向型"),
    ("^TWOII",    "TWOII",  "櫃買指數",          "指數"),

    # 海外指數：對台股是「領先」而不只是「相關」——費半、那斯達克都在台北時間
    # 凌晨收盤，台股開盤前就已知。這是不需要拚速度的資訊優勢。
    ("^SOX",      "SOX",    "費城半導體指數",    "海外領先"),
    ("^IXIC",     "IXIC",   "那斯達克指數",      "海外領先"),
    ("^GSPC",     "GSPC",   "標普500",          "海外領先"),
    ("SOXX",      "SOXX",   "iShares半導體ETF",  "海外領先"),
    ("SMH",       "SMH",    "VanEck半導體ETF",   "海外領先"),
    ("TSM",       "TSM",    "台積電ADR",         "海外領先"),
    ("^KS11",     "KS11",   "韓國KOSPI",        "海外領先"),
    ("^N225",     "N225",   "日經225",          "海外領先"),
    # 台灣掛牌、追蹤費半的 ETF（以台幣交易，可直接下單）
    ("00830.TW",  "00830",  "國泰費城半導體",    "產業型"),
    ("00646.TW",  "00646",  "元大S&P500",       "海外連結"),
    # 油價：Yahoo 的近月連續期貨。換月時會有價差跳動，2020-04-20 的 WTI 還出現負價
    # （−37.63），算報酬前要注意。Brent 比較少受美國庫存交割問題影響，兩個都抓
    ("CL=F",      "WTI",    "西德州原油期貨",    "原物料"),
    ("BZ=F",      "BRENT",  "布蘭特原油期貨",    "原物料"),
]

# 台指期：Yahoo 不提供台灣期交所的連續月合約，需自行向期交所取得。
# 在本研究中以 0050／TWII 搭配 --instrument futures 的成本結構近似
# （期交稅 0.002%、手續費約 0.003%，來回約 0.01%）。

START = date(2010, 1, 1)
END = date(2024, 12, 31)


def fetch(symbol: str) -> pd.DataFrame:
    frames = []
    for year, y0, y1 in C.year_chunks(START, END):
        try:
            data = C.http_get(
                C.YAHOO_CHART_URL.format(symbol=symbol),
                params={
                    "period1": int(datetime(y0.year, y0.month, y0.day).timestamp()),
                    "period2": int(datetime(y1.year, y1.month, y1.day).timestamp()) + 86400,
                    "interval": "1d",
                    "events": "history",
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    {year} 失敗：{type(exc).__name__}")
            time.sleep(1.0)
            continue
        res = (data.get("chart") or {}).get("result")
        if not res:
            time.sleep(0.6)
            continue
        r0 = res[0]
        ts = r0.get("timestamp") or []
        q = (r0.get("indicators") or {}).get("quote")
        if not ts or not q:
            time.sleep(0.6)
            continue
        q = q[0]
        frames.append(pd.DataFrame({
            "date": [datetime.utcfromtimestamp(t).date() for t in ts],
            "open": q.get("open"), "high": q.get("high"),
            "low": q.get("low"), "close": q.get("close"),
            "volume": q.get("volume"),
        }))
        time.sleep(0.6)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = (df.dropna(subset=["close"]).drop_duplicates(subset=["date"], keep="last")
            .sort_values("date").reset_index(drop=True))
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"抓取 {len(TARGETS)} 個指數型標的　{START} ~ {END}\n")
    ok = []
    for symbol, name, label, kind in TARGETS:
        path = OUT / f"{name}.csv"
        if path.exists():
            d = pd.read_csv(path)
            print(f"  [已存在] {name:<8}{label:<16}{len(d):>5} 筆，略過")
            ok.append((name, label, kind, len(d)))
            continue
        print(f"  [抓取中] {symbol:<11}{label}")
        df = fetch(symbol)
        if df.empty:
            print(f"    無資料（可能 Yahoo 不提供此代號）")
            continue
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"    {len(df):>5} 筆　{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        ok.append((name, label, kind, len(df)))

    print(f"\n可用標的 {len(ok)} 個：")
    for name, label, kind, n in ok:
        print(f"  {name:<8}{label:<16}{kind:<8}{n:>6} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
