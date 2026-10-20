#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大盤（發行量加權股價指數 ^TWII）——所有績效的最終對照組。

為什麼要有這個：買進持有「個股」只能說明模型有沒有打敗那一檔，
但投資人真正的替代方案是買大盤（0050 之類的指數型商品）。
提案 §4 的回測要回答「這個模型值不值得做」，就必須跟大盤比：

    模型報酬 − 大盤報酬 = 超額報酬（alpha 的粗略版本）

資料來源用 Yahoo 的同一個報價端點，與個股股價一致，不必另外處理對齊問題。
第一次呼叫會下載並快取到 data/market/TWII.csv，之後直接讀檔。

單獨執行可更新快取：
    python tw_market_index.py              # 補到最新
    python tw_market_index.py --refresh    # 整段重抓
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "market" / "TWII.csv"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")


def fetch_twii(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """向 Yahoo 取加權指數日線。"""
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int((datetime.fromisoformat(end) if end else datetime.now())
             .replace(tzinfo=timezone.utc).timestamp())
    r = requests.get(YAHOO_CHART, headers={"User-Agent": UA},
                     params={"period1": p1, "period2": p2,
                             "interval": "1d", "events": "history"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(t, timezone.utc).date() for t in res["timestamp"]],
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"), "volume": q.get("volume"),
    })
    return df.dropna(subset=["close"])


def load_twii(refresh: bool = False) -> pd.Series:
    """
    回傳加權指數收盤價（以日期為索引）。沒有快取就自動下載一次。
    抓不到時回傳空序列，讓呼叫端把大盤欄位留白，而不是整個流程失敗。
    """
    if CACHE.exists() and not refresh:
        df = pd.read_csv(CACHE, encoding="utf-8-sig")
    else:
        try:
            df = fetch_twii()
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE, index=False, encoding="utf-8-sig")
        except Exception as e:                       # noqa: BLE001
            print(f"[大盤] 取得 ^TWII 失敗（{e}），本次略過大盤比較", file=sys.stderr)
            return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).set_index("date")["close"].sort_index()


def benchmark_stats(index_close: pd.Series, test_index: pd.DatetimeIndex,
                    trading_days: int = 252) -> dict:
    """
    測試期間的大盤表現：總報酬與夏普。

    只取測試期的第一天與最後一天，跟個股的買進持有算法一致，
    這樣「模型 vs 買進持有個股 vs 買進持有大盤」三者可以並排比較。
    """
    if index_close is None or index_close.empty or len(test_index) < 2:
        return {"bench_return": float("nan"), "bench_sharpe": float("nan")}
    seg = index_close.reindex(test_index).ffill().dropna()
    if len(seg) < 2:
        return {"bench_return": float("nan"), "bench_sharpe": float("nan")}
    daily = seg.pct_change().dropna()
    sharpe = (float(daily.mean() / daily.std(ddof=1) * (trading_days ** 0.5))
              if daily.std(ddof=1) > 0 else float("nan"))
    return {"bench_return": float(seg.iloc[-1] / seg.iloc[0] - 1), "bench_sharpe": sharpe}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="更新大盤（加權指數）快取")
    ap.add_argument("--refresh", action="store_true", help="整段重抓，不沿用既有快取")
    args = ap.parse_args(argv)
    s = load_twii(refresh=args.refresh)
    if s.empty:
        return 1
    print(f"加權指數快取：{CACHE}")
    print(f"  {len(s)} 個交易日　{s.index.min():%Y-%m-%d} ~ {s.index.max():%Y-%m-%d}")
    print(f"  首日 {s.iloc[0]:,.2f}　末日 {s.iloc[-1]:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
