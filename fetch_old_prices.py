#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
補抓 2000–2010 的台股歷史股價，把樣本從 14 年拉長到 25 年。

為什麼要補
----------
動能因子在 2011–2024 的最佳組超額 +8.29%、未校正 t = 1.52、
Reality Check p = 0.227 —— 方向對但樣本撐不起顯著性。
動能在文獻上是跨市場重複驗證過的異象，最可能的限制就是樣本長度。
t 值大致隨 √樣本成長，14 年 → 25 年約可把 1.52 推到 2.0 附近。

拉長樣本不改變任何策略設定，只增加證據，是最乾淨的一步：
它無法把假訊號變成真的，只能讓真訊號浮出來、讓假訊號更清楚地消失。

輸出沿用既有結構 data/stocks/{代號}/price/yahoo_{代號}_{年}.csv，
load_price 會自動把新舊年度併起來並依日期去重。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
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
CKPT = ROOT / "data" / "old_price_checkpoint.json"


def load_ckpt() -> set:
    return set(json.loads(CKPT.read_text(encoding="utf-8"))) if CKPT.exists() else set()


def save_ckpt(d: set) -> None:
    CKPT.write_text(json.dumps(sorted(d), ensure_ascii=False), encoding="utf-8")


def fetch(code: str, year: int) -> pd.DataFrame:
    d = C.http_get(C.YAHOO_CHART_URL.format(symbol=f"{code}.TW"), params={
        "period1": int(datetime(year, 1, 1).timestamp()),
        "period2": int(datetime(year, 12, 31).timestamp()) + 86400,
        "interval": "1d", "events": "history",
    })
    res = (d.get("chart") or {}).get("result")
    if not res:
        return pd.DataFrame()
    r0 = res[0]
    ts = r0.get("timestamp") or []
    q = (r0.get("indicators") or {}).get("quote")
    if not ts or not q:
        return pd.DataFrame()
    q = q[0]
    return pd.DataFrame({
        "date": [datetime.utcfromtimestamp(t).date() for t in ts],
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"), "volume": q.get("volume"),
    }).dropna(subset=["close"])


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="補抓 2000-2010 股價")
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--to-year", type=int, default=2010)
    ap.add_argument("--delay", type=float, default=0.45)
    a = ap.parse_args(argv)

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and (x / "price").exists())
    years = list(range(a.from_year, a.to_year + 1))
    done = load_ckpt()
    todo = [(c, y) for c in codes for y in years if f"{c}|{y}" not in done]
    print(f"股票 {len(codes)} 檔 × 年份 {len(years)} 個 = {len(codes)*len(years)} 組")
    print(f"待抓 {len(todo)}，預估約 {len(todo)*(a.delay+0.35)/60:.0f} 分鐘\n")

    ok = empty = fail = 0
    # 連續多年無資料就跳過該檔剩下的年份（多數公司在 2000 年尚未上市）
    skip_code, miss = None, 0
    for i, (code, year) in enumerate(todo, 1):
        if code == skip_code:
            done.add(f"{code}|{year}")
            continue
        try:
            df = fetch(code, year)
        except Exception:
            fail += 1
            done.add(f"{code}|{year}")
            time.sleep(a.delay * 2)
            continue
        if df.empty:
            empty += 1
            miss += 1
            if miss >= 4:          # 連 4 年沒資料，視為當時尚未上市
                skip_code, miss = code, 0
        else:
            miss = 0
            out = ROOT / "data" / "stocks" / code / "price" / f"yahoo_{code}_{year}.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            ok += 1
        done.add(f"{code}|{year}")
        if i % 200 == 0:
            save_ckpt(done)
            print(f"  {i}/{len(todo)}　成功 {ok}　無資料 {empty}　失敗 {fail}", flush=True)
        if code != skip_code:
            time.sleep(a.delay)
    save_ckpt(done)

    print(f"\n完成：成功 {ok}、無資料 {empty}、失敗 {fail}")
    # 抽樣確認
    import davis_tw_train as T
    for code in ("2330", "2317", "1301"):
        px = T.load_price(code)
        if px is not None:
            print(f"  {code}：{len(px):,} 個交易日　{px.index.min():%Y-%m-%d} ~ {px.index.max():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
