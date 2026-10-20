#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把可交易性限制套到隔日模型的預測上。

為什麼需要這支
--------------
`next_day_stocks.py` 的原始輸出是「多空 10 檔、扣成本後年化 +161.65%、
夏普 3.82」。那個回測有三個不成立的隱含假設：

    一、能用當日收盤價成交 —— 但訊號也是用同一個收盤價算的
    二、漲停鎖死的股票買得到、跌停鎖死的賣得掉 —— 實際上不行
    三、極端分位的小型股掛單不影響價格 —— 日換手 0.8 做不到

`nextday_diagnose.py` 已證明天真訊號在收盤與開盤兩種成交方式下差了
100 個百分點，代表收盤價本身帶著微結構雜訊。本程式對**模型預測**
做同一組檢查，看那 +161% 剩下多少。

每一欄的意思
------------
    收盤進出    t 日收盤進、t+1 收盤出（原始作法，最寬鬆）
    開盤進出    t+1 開盤進、t+2 開盤出（訊號與成交價分離，較誠實）
    剔除漲跌停  把 t 日觸及 ±10% 且收在極端的個股從可選清單剔除
    延遲一日    t+1 收盤才進場，測訊號是不是只活在成交當下
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import davis_tw_train as T  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from nextday_diagnose import panels, perf, run_rank  # noqa: E402

OUT = ROOT / "results" / "nextday"
FEE, TAX = 0.001425 * 0.3, 0.003


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="隔日模型的可交易性檢查")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", default="2014-12-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args(argv)

    pf = OUT / "pred_stocks.csv"
    if not pf.exists():
        print(f"找不到 {pf}，請先跑 next_day_stocks.py"); return 1
    p = pd.read_csv(pf, encoding="utf-8-sig", parse_dates=["date"],
                    dtype={"code": str})
    pred = p.pivot(index="date", columns="code", values="pred").sort_index()
    print(f"預測 {p.shape[0]:,} 列　{pred.index[0]:%Y-%m} ~ {pred.index[-1]:%Y-%m}"
          f"　{pred.shape[1]} 檔")

    codes = sorted(pred.columns)
    print(f"載入 {len(codes)} 檔價格面板…", flush=True)
    P = panels(codes, a.start, a.end)
    C, O, H, L = P["close"], P["open"], P["high"], P["low"]
    pred = pred.reindex(index=C.index, columns=C.columns)

    m250 = U.get_mask(a.start, a.end, 250).reindex(
        index=C.index, columns=C.columns).fillna(False)
    m100 = U.get_mask(a.start, a.end, 100).reindex(
        index=C.index, columns=C.columns).fillna(False)
    prev = C.shift(1)
    lim = ((C >= prev * 1.095) & (C >= H * 0.999)) | \
          ((C <= prev * 0.905) & (C <= L * 1.001))
    trad = m250 & ~lim

    fwd_cc = C.pct_change().shift(-1)              # t 收盤 → t+1 收盤
    fwd_oo = O.shift(-2) / O.shift(-1) - 1         # t+1 開盤 → t+2 開盤
    fwd_lag = C.pct_change().shift(-2)             # t+1 收盤 → t+2 收盤
    ew = fwd_cc[m250].mean(axis=1)
    cost = 2 * FEE + TAX

    cases = [
        ("收盤進出／前250（原始作法）", fwd_cc, m250),
        ("收盤進出／剔除漲跌停", fwd_cc, trad),
        ("開盤進出／前250", fwd_oo, m250),
        ("開盤進出／剔除漲跌停", fwd_oo, trad),
        ("開盤進出／前100", fwd_oo, m100),
        ("延遲一日收盤進出／前250", fwd_lag, m250),
    ]
    print(f"\n{'=' * 104}")
    print(f"模型預測在各種可交易性假設下的表現（{a.n} 檔，母體等權對照 "
          f"{perf(ew)['年化']:+.2%}）")
    print("=" * 104)
    print(f"  {'情境':<30}{'方式':<6}{'零成本':>12}{'扣成本':>12}{'夏普':>8}{'超額t':>8}")
    rows = []
    for name, fwd, msk in cases:
        for side in ("多空", "多單"):
            r0 = run_rank(pred, fwd, msk, a.n, side, 0.0)
            r1 = run_rank(pred, fwd, msk, a.n, side, cost)
            s0, s1 = perf(r0), perf(r1, ew)
            if not s0 or not s1:
                continue
            rows.append({"情境": name, "方式": side,
                         "年化_零成本": s0["年化"], "年化_扣成本": s1["年化"],
                         "夏普": s1["夏普"], "超額t": s1["超額t"]})
            print(f"  {name:<30}{side:<6}{s0['年化']:>+12.2%}"
                  f"{s1['年化']:>+12.2%}{s1['夏普']:>8.2f}{s1['超額t']:>8.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tradable.csv", index=False, encoding="utf-8-sig")

    # 極端分位裡有多少是漲跌停股 —— 直接量「押在買不到的股票上」的比例
    share = []
    for d in pred.index:
        s = pred.loc[d].dropna()
        s = s[[c for c in s.index if bool(m250.at[d, c])]] if d in m250.index else s
        if len(s) < a.n * 2:
            continue
        s = s.sort_values(ascending=False)
        pick = list(s.index[:a.n]) + list(s.index[-a.n:])
        share.append(float(lim.loc[d, pick].mean()) if d in lim.index else np.nan)
    print(f"\n  模型選中的極端 {2*a.n} 檔裡，當日觸及漲跌停的比例："
          f"平均 {np.nanmean(share):.1%}")
    print(f"  （母體整體的漲跌停比例 {float(lim[m250].stack().mean()):.1%}）")

    print(f"\n判讀")
    print(f"  原始作法的數字若在『開盤進出』或『剔除漲跌停』後大幅縮水，")
    print(f"  代表它賺的是成交不到的價格。扣成本後仍為正才算數。")
    print(f"\n輸出：{OUT / 'tradable.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
