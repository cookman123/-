#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沖掉 beta 之後還有沒有 alpha：對波動度溢酬做 beta 中性化與風險對等比較。

為什麼要做
----------
前一輪的結論是「那 +24.45% 是 beta，不是 alpha」（beta 1.45、t=12.43；
alpha t 僅 1.49）。但「它是 beta」不是終點，真正該問的是兩件事：

一、**把 beta 沖掉之後還剩什麼？** 用台指期空單對沖市場曝險，來回成本
    約 0.01%，所以這個對沖在台股實際做得到，不是紙上操作。

二、**在相同風險下，它贏不贏槓桿化的大盤？** 策略季波動是大盤的 1.88 倍。
    把大盤放大到同樣波動度再比，才是公平的對照。若贏不過，那報酬就只是
    槓桿的結果，用台指期買就好，不需要選股。

不偷看未來
----------
    beta 與波動比**只用該季之前的資料估計**（滾動窗，不含當季）。
    用全期 beta 去對沖會把未來資訊帶進來，對沖後的 alpha 會虛高 ——
    這正是本研究第 3.1 節那個錯誤的同一類問題。
    對沖部位在季初建立，當季不調整。
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
import feature_engineering as FE  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from quarterly_factors import forward_returns, quarter_ends  # noqa: E402
from quarterly_verify import book  # noqa: E402

OUT = ROOT / "results" / "quarterly"
FUT_COST = 0.0001        # 台指期來回約 0.01%（期交稅 0.002%×2 ＋ 手續費）
MIN_WIN = 12             # 估 beta 至少要 12 季，不足就不對沖該季


def ann(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (4 / len(r)) - 1 if len(r) >= 8 else np.nan


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(4)) if len(r) > 4 and r.std() else np.nan


def rolling_beta(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
    """
    滾動 beta，只用「該季之前」的資料。shift(1) 是關鍵：
    第 t 季要用的 beta 必須由 t-1 季為止的資料估出，否則就是用未來估對沖比例。
    """
    cov = y.rolling(win, min_periods=MIN_WIN).cov(x)
    var = x.rolling(win, min_periods=MIN_WIN).var()
    return (cov / var.replace(0, np.nan)).shift(1)


def rolling_volratio(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
    a = y.rolling(win, min_periods=MIN_WIN).std()
    b = x.rolling(win, min_periods=MIN_WIN).std()
    return (a / b.replace(0, np.nan)).shift(1)


def report(name: str, r: pd.Series, base: pd.Series | None = None) -> dict:
    """
    base 給定時，整列都要描述「差額」—— 年化、夏普、t 全部算在 r−base 上。
    先前版本的 t 算在差額、年化卻印 r 本身，同一列兩個數字講不同的東西，
    讀起來會以為超額有 +24% 而其實那是策略自己的年化。
    """
    x = r.dropna() if base is None else (r - base).dropna()
    t, p = nw_tstat(x)
    d = {"組合": name, "年化": ann(x), "夏普": sharpe(x),
         "季均": float(x.mean()), "t": t, "p": p, "季數": int(len(x))}
    lab = "平均報酬 t" if base is None else "超額 t"
    print(f"  {name:<28}{ann(x):>+10.2%}{sharpe(x):>8.2f}"
          f"{t:>9.2f}{p:>9.4f}　{lab}")
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="beta 中性化後的 alpha")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sig", default="cs_ts_vol60")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--win", type=int, default=20, help="估 beta 的滾動季數")
    a = ap.parse_args(argv)

    store = ROOT / "data" / "features" / f"panel{a.tag}.pkl"
    if not store.exists():
        print(f"找不到特徵庫 {store}"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(store)
    if a.sig not in df.columns:
        print(f"特徵庫裡沒有 {a.sig}"); return 1

    P = FE.load_panels("2010-01-01", "2026-12-31", quiet=True)
    O = P["open"]
    qe = [d for d in quarter_ends(O.index)
          if pd.Timestamp(d) in df.index.get_level_values("date")]
    fwd = forward_returns(O, qe)
    qd = [d for d in fwd.index if d in df.index.get_level_values("date")]
    sub = df.loc[df.index.get_level_values("date").isin(qd)]

    # 大盤季報酬：同樣隔日開盤進出，與策略一致
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv",
                     parse_dates=["date"]).set_index("date")["close"].sort_index()
    tw = tw[~tw.index.duplicated(keep="last")]
    twv = pd.Series({d: tw.asof(pd.Timestamp(d)) for d in O.index}, index=O.index)
    pos = {d: i for i, d in enumerate(O.index)}
    mkt = {}
    for x, y in zip(qe[:-1], qe[1:]):
        ix, iy = pos[pd.Timestamp(x)], pos[pd.Timestamp(y)]
        if ix + 1 < len(O.index) and iy + 1 < len(O.index):
            mkt[pd.Timestamp(x)] = twv.iloc[iy + 1] / twv.iloc[ix + 1] - 1
    mkt = pd.Series(mkt).sort_index()

    strat = book(sub, fwd, qd, a.sig, 10, "多單", n_fixed=a.n)["r"]
    m = mkt.reindex(strat.index)
    ew = pd.Series({d: float(fwd.loc[d].reindex(sub.loc[d].index).dropna().mean())
                    for d in qd}).sort_index().reindex(strat.index)

    print(f"特徵庫 {store.name}　訊號 {a.sig}　{a.n} 檔"
          f"　{strat.index[0]:%Y-%m} ~ {strat.index[-1]:%Y-%m}（{len(strat)} 季）")
    print(f"策略季波動 {strat.std():.2%}　大盤 {m.std():.2%}"
          f"　比值 {strat.std() / m.std():.2f}\n")

    b = rolling_beta(strat, m, a.win)
    v = rolling_volratio(strat, m, a.win)
    print(f"滾動 beta（{a.win} 季窗，已 shift 一季避免前視）："
          f"中位數 {b.median():.2f}　範圍 {b.min():.2f} ~ {b.max():.2f}")
    print(f"可用於對沖的季數 {int(b.notna().sum())}／{len(b)}"
          f"（前 {MIN_WIN} 季不足無法估）\n")

    rows = []
    print(f"{'=' * 96}")
    print(f"  {'組合':<28}{'年化':>10}{'夏普':>8}{'t':>9}{'p':>9}")
    print(f"{'=' * 96}")
    rows.append(report("策略（未對沖）", strat))
    rows.append(report("大盤買進持有", m))
    rows.append(report("策略 − 大盤", strat, m))
    rows.append(report("策略 − 母體等權", strat, ew))

    # 一、beta 中性化：季初用事前 beta 建台指期空單
    hedge = (strat - b * m - b.abs() * FUT_COST).dropna()
    print()
    rows.append(report("beta 中性（事前 beta）", hedge))

    # 二、風險對等：把大盤放大到和策略同波動
    lev = (v * m - (v - 1).abs() * FUT_COST).dropna()
    idx = strat.index.intersection(lev.index)
    rows.append(report("槓桿大盤（同波動）", lev.reindex(idx)))
    rows.append(report("策略 − 槓桿大盤", strat.reindex(idx), lev.reindex(idx)))

    print(f"\n{'=' * 96}\n判讀\n{'=' * 96}")
    hr = [r for r in rows if r["組合"].startswith("beta 中性")][0]
    vr = [r for r in rows if r["組合"] == "策略 − 槓桿大盤"][0]
    print(f"  一、沖掉 beta 後年化 {hr['年化']:+.2%}、t = {hr['t']:.2f}"
          f"（p = {hr['p']:.4f}）"
          f" → {'仍有 alpha' if hr['p'] < 0.05 else 'alpha 不顯著'}")
    print(f"  二、相同風險下對槓桿大盤的超額年化 {vr['年化']:+.2%}"
          f"、t = {vr['t']:.2f}（p = {vr['p']:.4f}）"
          f" → {'贏得過' if vr['p'] < 0.05 and vr['t'] > 0 else '贏不過'}")
    print(f"\n  兩項都不顯著，就表示這個效應能用台指期複製，不需要選股；")
    print(f"  任一項顯著，才代表選股本身有無法被槓桿取代的貢獻。")

    pd.DataFrame(rows).to_csv(OUT / f"alpha{a.tag}.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\n輸出：{OUT / f'alpha{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
