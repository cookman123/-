#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
營收年增率因子的五道驗證（前四道沿用波動度溢酬那一套，第八、九節）。

  1. 是不是價格動能換個名字：每季橫斷面迴歸（Fama-MacBeth），同時放入
     營收年增率與 60 日報酬、120 日均線乖離、120 日區間位置、60 日波動度，
     看營收的係數還在不在
  2. 純做多：前十分位、固定 30 檔，對母體等權的超額（扣成本）
  3. 是不是 beta：對大盤的季頻 CAPM（Newey-West），以及多空組合本身的 beta
  4. 子期間與逐年
  5. 是不是產業或規模：每季先轉百分位、扣掉產業平均，再對 log 成交金額迴歸取殘差，
     看中性化後的 IC 與十分位多空還在不在（做法取自 jameschen108/tw-stock-quant 的
     backtest.neutralize；原版用市值，本專案沒有時點正確的市值，改用 60 日均成交金額）

全部用季末訊號、隔日開盤進出，與 quarterly_factors.py 相同。
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
import fundamental_factors as FF  # noqa: E402
import quarterly_factors as QF  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402

CONTROLS = ["ts_ret60", "ts_ma120_dev", "ts_pos120", "ts_vol60"]


def ols_nw(y: pd.Series, x: pd.Series, lags: int = 4) -> tuple:
    """y = a + b x，回傳 (a, t_a, b, t_b)，Newey-West 標準誤。"""
    d = pd.concat([y, x], axis=1).dropna()
    Y, X = d.iloc[:, 0].to_numpy(), np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy()])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ beta
    n = len(Y)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ S @ XtX
    se = np.sqrt(np.diag(V))
    return beta[0], beta[0] / se[0], beta[1], beta[1] / se[1]


def ann(r: pd.Series) -> float:
    return (1 + r).prod() ** (4 / len(r)) - 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="營收年增率因子的驗證")
    ap.add_argument("--end", default="2026-12-31", help="例如 2024-12-31 可排除 2025–2026")
    ap.add_argument("--tag", default="", help="_surv：含下市股（搭配 tool/run_surv.py）")
    a = ap.parse_args(argv)
    print(f"期間 2010 ~ {a.end}　特徵庫 panel{a.tag}.pkl", flush=True)
    P = FE.load_panels("2010-01-01", a.end, quiet=True)
    O = P["open"]
    codes = list(O.columns)
    qe = QF.quarter_ends(O.index)
    fwd = QF.forward_returns(O, qe)
    qd = list(fwd.index)
    mask = U.get_mask("2010-01-01", a.end, 250).reindex(index=qd, columns=codes).fillna(False)

    rv = FF.revenue_table()
    sig = {"fin_rev_yoy": FF.asof(rv, {"fin_rev_yoy": "yoy"}, qd, codes)["fin_rev_yoy"],
           "fin_rev_yoy3": FF.asof(rv, {"x": "yoy3"}, qd, codes)["x"]}
    df = pd.read_pickle(QF.STORE.with_name(f"panel{a.tag}.pkl"))
    sub = df.loc[df.index.get_level_values("date").isin(qd), CONTROLS]
    ctrl = {c: sub[c].unstack().reindex(index=qd, columns=codes) for c in CONTROLS}
    del df, sub
    ind = pd.Series(FE.industry_map()).reindex(codes).fillna("NA")
    turn = pd.read_csv(U.CACHE, index_col=0, parse_dates=True).reindex(index=qd, columns=codes)

    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]).set_index("date")["open"]
    tw = tw[~tw.index.duplicated()]
    pos = {d: i for i, d in enumerate(tw.index)}
    mkt = {}
    for a, b in zip(qe[:-1], qe[1:]):
        ia, ib = tw.index.searchsorted(a, "right"), tw.index.searchsorted(b, "right")
        if ib < len(tw):
            mkt[pd.Timestamp(a)] = tw.iloc[ib] / tw.iloc[ia] - 1
    mkt = pd.Series(mkt)

    for name, S in sig.items():
        S = S.astype(float).where(mask)
        print(f"\n{'=' * 96}\n{name}\n{'=' * 96}")

        # 1. Fama-MacBeth（全部用橫斷面百分位，係數可直接比較）
        coefs = []
        for d in qd:
            X = pd.DataFrame({"rev": S.loc[d], **{c: ctrl[c].loc[d] for c in CONTROLS}})
            X["y"] = fwd.loc[d]
            X = X[mask.loc[d]].dropna()
            if len(X) < 60:
                continue
            R = X.rank(pct=True)
            A = np.column_stack([np.ones(len(R)), R.drop(columns="y").to_numpy()])
            b = np.linalg.lstsq(A, R["y"].to_numpy(), rcond=None)[0]
            coefs.append(pd.Series(b[1:], index=["rev"] + CONTROLS, name=d))
        C = pd.DataFrame(coefs)
        t = C.mean() / (C.std(ddof=1) / np.sqrt(len(C)))
        print(f"  1. Fama-MacBeth（{len(C)} 季，百分位迴歸）：")
        for k in C.columns:
            print(f"       {k:<14} 係數 {C[k].mean():+.4f}　t {t[k]:+.2f}")

        # 2. 純做多與多空
        rows = {}
        for d in qd:
            s = S.loc[d].dropna()
            r = fwd.loc[d].reindex(s.index)
            ok = r.notna()
            s, r = s[ok], r[ok]
            if len(s) < 60:
                continue
            uni = fwd.loc[d][mask.loc[d]].mean()
            o = s.sort_values(ascending=False)
            k = len(o) // 10
            rows[d] = {"top10": r[o.index[:k]].mean(), "bot10": r[o.index[-k:]].mean(),
                       "top30": r[o.index[:30]].mean(), "uni": uni}
        R = pd.DataFrame(rows).T
        cost = QF.COST
        top10, top30 = R["top10"] - cost, R["top30"] - cost
        ls = R["top10"] - R["bot10"] - 2 * cost
        print(f"  2. 組合（{len(R)} 季，{R.index[0]:%Y-%m} ~ {R.index[-1]:%Y-%m}，扣成本）：")
        print(f"       母體等權 年化 {ann(R['uni']):+.2%}　大盤 {ann(mkt.reindex(R.index)):+.2%}")
        for lab, x in (("前十分位", top10), ("固定 30 檔", top30), ("十分位多空", ls)):
            ex = x - R["uni"] if lab != "十分位多空" else x
            tt, _ = nw_tstat(ex)
            print(f"       {lab:<8} 年化 {ann(x):+.2%}　"
                  f"{'對母體超額' if lab != '十分位多空' else '本身'} 季均 {ex.mean():+.2%}（t {tt:.2f}）")

        # 3. CAPM
        m = mkt.reindex(R.index)
        print("  3. 對大盤的 CAPM（季頻，NW）：")
        for lab, x in (("固定 30 檔", top30), ("母體等權", R["uni"]), ("十分位多空", ls)):
            a, ta, b, tb = ols_nw(x, m)
            print(f"       {lab:<8} alpha 年化 {a * 4:+.2%}（t {ta:.2f}）　beta {b:.2f}（t {tb:.2f}）")
        a, ta, b, tb = ols_nw(top30 - R["uni"], R["uni"] - m)
        print(f"       固定 30 檔 − 母體 對（母體 − 大盤）：alpha 年化 {a * 4:+.2%}（t {ta:.2f}）　斜率 {b:.2f}")

        # 4. 子期間與逐年
        h = len(R) // 2
        ex30 = top30 - R["uni"]
        print(f"  4. 子期間：前半 {R.index[0]:%Y}–{R.index[h - 1]:%Y} 對母體季均 {ex30.iloc[:h].mean():+.2%}"
              f"（t {nw_tstat(ex30.iloc[:h])[0]:.2f}）　後半 {ex30.iloc[h:].mean():+.2%}"
              f"（t {nw_tstat(ex30.iloc[h:])[0]:.2f}）")
        yr = ex30.groupby(ex30.index.year).apply(lambda x: (1 + x).prod() - 1)
        print(f"       逐年（固定 30 檔 − 母體）勝 {int((yr > 0).sum())}/{len(yr)} 年："
              + "　".join(f"{y} {v:+.0%}" for y, v in yr.items()))

        # 5. 產業＋成交金額中性
        res = {"原始": ([], {}), "中性化": ([], {})}
        for d in qd:
            X = pd.DataFrame({"s": S.loc[d], "ind": ind, "lt": np.log(turn.loc[d]), "y": fwd.loc[d]})
            X = X[mask.loc[d]].replace([np.inf, -np.inf], np.nan).dropna(subset=["s", "lt", "y"])
            if len(X) < 60:
                continue
            y = X["s"].rank(pct=True)
            y = y - y.groupby(X["ind"]).transform("mean")
            lt = X["lt"] - X["lt"].groupby(X["ind"]).transform("mean")
            neu = y - float((y * lt).sum() / (lt ** 2).sum()) * lt
            for lab, sc in (("原始", X["s"]), ("中性化", neu)):
                res[lab][0].append(float(sc.corr(X["y"], method="spearman")))
                res[lab][1][d] = QF.decile_spread(sc, X["y"])[0] - 2 * cost
        print("  5. 產業＋成交金額中性（每季百分位扣產業平均，再對 log 成交金額取殘差）：")
        for lab, (ic, sp) in res.items():
            ic, sp = np.array(ic), pd.Series(sp).sort_index()
            t_ic = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
            print(f"       {lab:<6} IC {ic.mean():+.4f}（t {t_ic:.2f}）　十分位多空扣成本 年化 {ann(sp):+.2%}"
                  f"（季均 t {nw_tstat(sp)[0]:.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
