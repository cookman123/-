#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
時點正確（point-in-time）的選股範圍。

要解決什麼問題
--------------
先前用「當期市值前 N 大」當標的清單，那是今天才知道的資訊。
2024 年的市值排名拿去決定 2011 年要買哪些股票，等於偷看答案：
清單裡必然包含一些當年還很小、後來才長大的公司，而那正是最會漲的一批。
研究期間越長，這個偏誤越嚴重。

改法
----
改用「近 60 個交易日的平均成交金額」逐日重新排名。成交金額在當天就知道，
不含任何未來資訊，而且同時代表流動性——本策略要能實際成交，
流動性本來就是必要條件，用它當篩選標準比市值更貼近實務。

輸出一張 date × code 的布林表：某檔在某日是否屬於當時的前 N 名。
交易只在「訊號日當天屬於前 N 名」時才採用。
"""
from __future__ import annotations

import sys
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
CACHE = ROOT / "data" / "market" / "turnover_rank.csv"


def build_turnover_panel(codes: list, start: str, end: str,
                         window: int = 60) -> pd.DataFrame:
    """每檔每日的近 window 日平均成交金額（收盤價 × 成交量）。"""
    cols = {}
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        if len(px) < window:
            continue
        # shift(1)：訊號日當天用的是「前一日為止」的平均成交金額，更保守
        cols[code] = (px["close"] * px["volume"]).rolling(window).mean().shift(1)
        if i % 100 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔")
    return pd.DataFrame(cols).sort_index()


def top_n_mask(turnover: pd.DataFrame, n: int) -> pd.DataFrame:
    """每一天取成交金額前 n 名；回傳同形狀的布林表。"""
    rank = turnover.rank(axis=1, ascending=False, method="first")
    return rank <= n


def get_mask(start: str, end: str, n: int, refresh: bool = False) -> pd.DataFrame:
    """取得（必要時重建）時點正確的前 n 名布林表。"""
    if CACHE.exists() and not refresh:
        t = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    else:
        codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                       if x.is_dir() and any((x / "price").glob("*.csv")))
        print(f"建立成交金額面板（{len(codes)} 檔，{start} ~ {end}）…")
        t = build_turnover_panel(codes, start, end)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(CACHE, encoding="utf-8-sig")
        print(f"已快取至 {CACHE}")
    t = t[(t.index >= pd.Timestamp(start)) & (t.index <= pd.Timestamp(end))]
    return top_n_mask(t, n)


def filter_trades(trades: pd.DataFrame, mask: pd.DataFrame,
                  date_col: str = "drop_day") -> pd.DataFrame:
    """只保留「訊號日當天屬於前 N 名」的交易。"""
    if trades.empty:
        return trades
    d = trades.copy()
    dates = pd.to_datetime(d[date_col])
    keep = []
    for dt, code in zip(dates, d["code"].astype(str)):
        try:
            keep.append(bool(mask.at[dt, code]))
        except (KeyError, IndexError):
            keep.append(False)
    return d[pd.Series(keep, index=d.index)]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="時點正確的選股範圍")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    m = get_mask(a.start, a.end, a.n, a.refresh)
    print(f"\n前 {a.n} 名布林表：{m.shape[0]:,} 天 × {m.shape[1]} 檔")
    print(f"每日平均入選 {m.sum(axis=1).mean():.1f} 檔")

    # 看看清單隨時間換了多少——換得越多，代表用固定清單的偏誤越大
    yearly = m.groupby(m.index.year).apply(
        lambda g: set(g.columns[g.any()]), include_groups=False)
    print(f"\n{'年':<6}{'該年曾入選檔數':>14}{'與最後一年的重疊':>16}")
    last = yearly.iloc[-1]
    for y, s in yearly.items():
        print(f"{y:<6}{len(s):>14}{len(s & last) / max(len(last), 1):>15.0%}")
    print("\n重疊率越低，代表「用當期市值挑過去的股票」偏誤越嚴重。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
