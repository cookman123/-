#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台指期（TX）每日行情爬蟲——期交所官方下載端點。

為什麼需要
----------
現貨指數測到「費半隔夜 → 台股盤中」有 t=8.23 的訊號，但現貨指數不能直接交易。
實際要下單的是台指期，而台指期有兩個關鍵差異：

  1. 8:45 開盤，比現貨早 15 分鐘
  2. 自 2017-05-15 起有盤後交易時段（15:00~次日 05:00），**涵蓋美股盤中**

第 2 點是致命的：若費半的資訊在夜盤就被套利掉，日盤 8:45 開盤價就已反映完畢，
那個 t=8.23 的訊號就吃不到。期交所的資料有「交易時段」欄位，可以把
夜盤變化與日盤盤中分開算，直接回答這個問題。

抓取內容
--------
每日每契約的開高低收、成交量、交易時段（一般／盤後）。
只保留月契約（到期月份為 YYYYMM），排除週契約（YYYYMMWn），並取最近月份。

輸出：data/taifex/tx_{年}.csv
斷點：data/taifex/checkpoint.json（記錄已完成的月份）
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "taifex"
CKPT = OUT / "checkpoint.json"
URL = "https://www.taifex.com.tw/cht/3/futDataDown"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (graduation research; polite crawler)",
    "Referer": "https://www.taifex.com.tw/cht/3/futDataDown",
}
COLS = ["date", "contract", "expiry", "open", "high", "low", "close",
        "chg", "chg_pct", "volume", "settle", "oi", "bid", "ask",
        "hist_high", "hist_low", "halt", "session", "spread_vol"]


def load_ckpt() -> set:
    return set(json.loads(CKPT.read_text(encoding="utf-8"))) if CKPT.exists() else set()


def save_ckpt(d: set) -> None:
    CKPT.write_text(json.dumps(sorted(d), ensure_ascii=False), encoding="utf-8")


def fetch_month(y: int, mth: int) -> pd.DataFrame:
    last = (date(y + (mth == 12), mth % 12 + 1, 1) - pd.Timedelta(days=1)).date() \
        if False else (pd.Timestamp(y, mth, 1) + pd.offsets.MonthEnd(1)).date()
    r = requests.post(URL, headers=HEADERS, timeout=60, data={
        "down_type": "1", "commodity_id": "TX",
        "queryStartDate": f"{y}/{mth:02d}/01",
        "queryEndDate": f"{last.year}/{last.month:02d}/{last.day:02d}",
    })
    r.raise_for_status()
    txt = r.content.decode("big5", errors="replace")
    df = pd.read_csv(io.StringIO(txt), skiprows=1, header=None, on_bad_lines="skip")
    if df.empty:
        return pd.DataFrame()
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS[:df.shape[1]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiry"] = df["expiry"].astype(str).str.strip()
    # 只留月契約（YYYYMM），排除週契約（YYYYMMWn）與價差契約
    df = df[df["expiry"].str.fullmatch(r"\d{6}")]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["date", "open", "close"])


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="台指期每日行情")
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    done = load_ckpt()
    todo = [(y, m) for y in range(args.start_year, args.end_year + 1)
            for m in range(1, 13) if f"{y}-{m:02d}" not in done]
    print(f"待抓 {len(todo)} 個月，預估約 {len(todo) * (args.delay + 1) / 60:.0f} 分鐘\n")

    buf: dict = {}
    for i, (y, m) in enumerate(todo, 1):
        try:
            d = fetch_month(y, m)
        except Exception as exc:  # noqa: BLE001
            print(f"  {y}-{m:02d} 失敗：{type(exc).__name__}")
            time.sleep(args.delay * 2)
            continue
        if not d.empty:
            buf.setdefault(y, []).append(d)
        done.add(f"{y}-{m:02d}")
        if i % 12 == 0 or i == len(todo):
            for yy, parts in buf.items():
                path = OUT / f"tx_{yy}.csv"
                new = pd.concat(parts, ignore_index=True)
                if path.exists():
                    new = pd.concat([pd.read_csv(path, parse_dates=["date"]), new],
                                    ignore_index=True)
                new = new.drop_duplicates(subset=["date", "expiry", "session"],
                                          keep="last")
                new.sort_values(["date", "expiry"]).to_csv(
                    path, index=False, encoding="utf-8-sig")
            buf.clear()
            save_ckpt(done)
            print(f"  {i}/{len(todo)}　{y}-{m:02d} 完成", flush=True)
        time.sleep(args.delay)

    print("\n完成：")
    tot = 0
    for f in sorted(OUT.glob("tx_*.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        tot += len(d)
        ses = d["session"].astype(str).value_counts().to_dict()
        print(f"  {f.name}　{len(d):>6,} 列　交易日 {d['date'].nunique():>4}　{ses}")
    print(f"合計 {tot:,} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
