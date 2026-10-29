#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
財報因子的季頻檢定：月營收、獲利、財務結構有沒有季頻的選股力。

資料來自 financials_crawler.py（MOPS 彙總表）與 tw_fundamentals_crawler.py（證交所本益比等）。
檢定方法完全沿用 quarterly_factors.py，結果才能和 105 個價量特徵直接比：
    季末取訊號、隔日開盤進場、下一季隔日開盤出場；母體為時點正確的成交金額前 250 名；
    逐季 Spearman IC、十分位多空扣成本、BH FDR、White Reality Check／SPA。

時點正確
--------
季末那天只用 avail_date 早於該日的資料（月營收：次月 10 日；季報：法定期限中最晚者）。
所以 3 月底只看得到前一年第三季的季報與 2 月的月營收，這是刻意的保守。

刻意不做的
----------
不用財報數字除以股價（例如自己算 E/P）：Yahoo 的價格會因日後配股往回調整，
過去某天的「價格」並不是當天真正的價格，相除會引入未來資訊。
估值類只用證交所當天公布的本益比、股價淨值比、殖利率（2015 年起）。

特徵（越高越該買，方向由檢定決定，不預設）
----
  fin_rev_yoy       最新一個月營收年增率
  fin_rev_yoy3      最近 3 個月合計營收年增率
  fin_rev_accel     最新月年增率 − 前 3 個月平均年增率
  fin_ni_yoy        單季淨利年增率（分母取絕對值）
  fin_sue           EPS 年差的標準化（除以過去 8 季年差的標準差）
  fin_roe           近四季淨利 / 權益
  fin_gm            單季毛利率
  fin_gm_chg        毛利率年差
  fin_opm           單季營業利益率
  fin_debt          負債比
  fin_ep, fin_bp, fin_dy   1/本益比、1/股價淨值比、殖利率（證交所，2015 起）

--extra 另外加入三個取自 jameschen108/tw-stock-quant（factors.py）的定義。
預設不加，原本 118 個特徵的多重比較結果才能原樣重現：
  fin_rev_vs_high   單月營收 / 過去所有月份最高 − 1（至少 24 個月歷史）
  fin_rev_yoy12     近 12 個月合計營收年增率
  fin_asset_growth  總資產年增率（高資產成長通常是負向，方向一樣交給檢定）
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
import quarterly_factors as QF  # noqa: E402
import universe_pit as U  # noqa: E402
from reality_check import reality_check  # noqa: E402

FIN = ROOT / "data" / "financials"
OUT = ROOT / "results" / "fundamental"


def _growth(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a − b) / |b|，分母為 0 給 NaN。虧轉盈、盈轉虧也能算出方向。"""
    return (a - b) / b.abs().replace(0, np.nan)


def revenue_table() -> pd.DataFrame:
    r = pd.read_csv(FIN / "revenue.csv", dtype={"code": str}, parse_dates=["avail_date"])
    r["p"] = pd.PeriodIndex(r["ym"], freq="M")
    r = r.sort_values(["code", "p"])
    g = r.groupby("code")
    # 用期數而不是列數對齊，中間缺月就不硬接
    prev12 = r.set_index(["code", "p"])["revenue"]
    key12 = list(zip(r["code"], r["p"] - 12))
    r["rev_ly"] = prev12.reindex(key12).to_numpy()
    r["yoy"] = _growth(r["revenue"], r["rev_ly"])
    s3 = g["revenue"].transform(lambda x: x.rolling(3).sum())
    s3_ly = g["rev_ly"].transform(lambda x: x.rolling(3).sum())
    r["yoy3"] = _growth(s3, s3_ly)
    r["accel"] = r["yoy"] - g["yoy"].transform(lambda x: x.shift(1).rolling(3).mean())
    s12 = g["revenue"].transform(lambda x: x.rolling(12).sum())
    s12_ly = g["rev_ly"].transform(lambda x: x.rolling(12).sum())
    r["yoy12"] = _growth(s12, s12_ly)
    prior_max = g["revenue"].transform(lambda x: x.shift(1).cummax())
    r["vs_high"] = (r["revenue"] / prior_max.where(prior_max > 0) - 1).where(g.cumcount() >= 24)
    return r


def quarterly_table() -> pd.DataFrame:
    q = pd.read_csv(FIN / "quarterly.csv", dtype={"code": str}, parse_dates=["avail_date"])
    q["p"] = pd.PeriodIndex.from_fields(year=q["year"], quarter=q["q"], freq="Q")
    q = q.sort_values(["code", "p"]).reset_index(drop=True)
    idx = q.set_index(["code", "p"])
    ly = idx.reindex(list(zip(q["code"], q["p"] - 4)))
    q["ni_yoy"] = _growth(q["net_income"], pd.Series(ly["net_income"].to_numpy(), index=q.index))
    d_eps = q["eps"] - ly["eps"].to_numpy()
    q["d_eps"] = d_eps
    q["sue"] = d_eps / q.groupby("code")["d_eps"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=4).std())
    # 近四季淨利：要求四季連續
    ttm = q.groupby("code")["net_income"].transform(lambda x: x.rolling(4).sum())
    span = q.groupby("code")["p"].transform(lambda x: x.astype("int64").diff(3))
    q["roe"] = (ttm / q["equity"]).where(span == 3)
    q["gm"] = q["gross_profit"] / q["revenue"].where(q["revenue"] > 0)
    q["gm_chg"] = q["gm"] - (ly["gross_profit"] / ly["revenue"].where(ly["revenue"] > 0)).to_numpy()
    q["opm"] = q["op_income"] / q["revenue"].where(q["revenue"] > 0)
    q["debt"] = q["total_liab"] / q["total_assets"]
    q["asset_growth"] = q["total_assets"] / ly["total_assets"].where(ly["total_assets"] > 0).to_numpy() - 1
    return q


def asof(tab: pd.DataFrame, cols: dict, dates: list, codes: list) -> dict:
    """每個季末日、每檔股票，取 avail_date 早於該日的最新一筆。"""
    out = {k: pd.DataFrame(index=dates, columns=codes, dtype=float) for k in cols}
    t = tab[tab["code"].isin(codes)].sort_values("avail_date")
    for d in dates:
        last = t[t["avail_date"] < d].groupby("code").tail(1).set_index("code")
        for k, c in cols.items():
            out[k].loc[d] = last[c].reindex(codes)
    return out


def bwibbu(dates: list, codes: list) -> dict:
    files = sorted((ROOT / "data" / "fundamentals").glob("bwibbu_*.csv"))
    d = pd.concat([pd.read_csv(f, dtype={"code": str}, parse_dates=["date"]) for f in files])
    d["code"] = d["code"].str.strip()
    out = {}
    for k, col, inv in (("fin_ep", "pe", True), ("fin_bp", "pb", True), ("fin_dy", "yield_pct", False)):
        w = d.pivot_table(index="date", columns="code", values=col, aggfunc="last")
        w = w.where(w > 0)
        w = (1 / w) if inv else w
        # 季末當天的值（證交所收盤後公布，與用季末收盤特徵的其他因子同一時點）
        out[k] = w.reindex(columns=codes).reindex(dates)
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="財報因子的季頻檢定")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--uni", type=int, default=250)
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--with-price", action="store_true",
                    help="把 105 個價量特徵一起放進多重比較（較誠實，也較難通過）")
    ap.add_argument("--extra", action="store_true",
                    help="加入取自 tw-stock-quant 的 fin_rev_vs_high、fin_rev_yoy12、fin_asset_growth")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    print("載入開盤價面板…", flush=True)
    P = FE.load_panels(a.start, a.end, quiet=True)
    O = P["open"]
    codes = list(O.columns)
    qe = QF.quarter_ends(O.index)
    fwd = QF.forward_returns(O, qe)
    mask = U.get_mask(a.start, a.end, a.uni).reindex(index=fwd.index, columns=codes).fillna(False)
    qd = list(fwd.index)
    print(f"  季數 {len(qd)}　{qd[0]:%Y-%m} ~ {qd[-1]:%Y-%m}　母體每季約 {mask.sum(axis=1).mean():.0f} 檔")

    print("建立時點正確的財報面板…", flush=True)
    rv = revenue_table()
    qt = quarterly_table()
    F = {}
    F.update(asof(rv, {"fin_rev_yoy": "yoy", "fin_rev_yoy3": "yoy3", "fin_rev_accel": "accel"}, qd, codes))
    F.update(asof(qt, {"fin_ni_yoy": "ni_yoy", "fin_sue": "sue", "fin_roe": "roe", "fin_gm": "gm",
                       "fin_gm_chg": "gm_chg", "fin_opm": "opm", "fin_debt": "debt"}, qd, codes))
    F.update(bwibbu(qd, codes))
    if a.extra:
        F.update(asof(rv, {"fin_rev_vs_high": "vs_high", "fin_rev_yoy12": "yoy12"}, qd, codes))
        F.update(asof(qt, {"fin_asset_growth": "asset_growth"}, qd, codes))
    for k in F:
        F[k] = F[k].astype(float).where(mask)
    cov = pd.DataFrame({k: v.notna().sum(axis=1) for k, v in F.items()})
    first = {k: (cov[k] >= 30).idxmax() for k in F}
    print("  各因子可用起點：" + "、".join(f"{k} {v:%Y-%m}" for k, v in first.items()))

    if a.with_price:
        df = pd.read_pickle(QF.STORE)
        feats = [c for c in df.columns if not c.startswith("y_")]
        sub = df.loc[df.index.get_level_values("date").isin(qd), feats]
        for c in feats:
            F[c] = sub[c].unstack().reindex(index=qd, columns=codes)
        del df, sub
        print(f"  加入 {len(feats)} 個價量特徵一起檢定")

    from scipy import stats as st
    rows, spreads, longs = [], {}, {}
    for c, W in F.items():
        ics, sps, hs = [], [], []
        for d in qd:
            s = W.loc[d].dropna()
            r = fwd.loc[d].reindex(s.index)
            ok = r.notna()
            if ok.sum() < 30:
                continue
            ics.append(float(s[ok].corr(r[ok], method="spearman")))
            sp, hi, _ = QF.decile_spread(s[ok], r[ok])
            uni_r = fwd.loc[d][mask.loc[d]].mean()
            sps.append((d, sp)); hs.append((d, hi - uni_r))
        if len(ics) < 16:
            continue
        ic = np.array(ics); ic = ic[np.isfinite(ic)]
        t_ic = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
        p_ic = float(2 * (1 - st.t.cdf(abs(t_ic), len(ic) - 1)))
        sp = pd.Series(dict(sps)).dropna().sort_index() - 2 * QF.COST
        hl = pd.Series(dict(hs)).dropna().sort_index()
        # 方向不預設：IC 為負就把多空反過來
        if ic.mean() < 0:
            sp = -sp - 4 * QF.COST
        ann = (1 + sp).prod() ** (4 / len(sp)) - 1
        rows.append({"特徵": c, "季數": len(ic), "起": f"{sp.index[0]:%Y-%m}",
                     "IC平均": ic.mean(), "IC_t": t_ic, "IC_p": p_ic,
                     "多空年化_扣成本": ann, "勝率": float((sp > 0).mean()),
                     "前十分位對母體_季均": float(hl.mean())})
        spreads[c] = sp
    res = pd.DataFrame(rows).sort_values("IC_t", key=abs, ascending=False)
    res["FDR通過"] = QF.bh_fdr(res["IC_p"].to_numpy(), a.fdr)
    tag = ("_with_price" if a.with_price else "") + ("_extra" if a.extra else "")
    res.to_csv(OUT / f"factors{tag}.csv", index=False, encoding="utf-8-sig")

    fin = res[res["特徵"].str.startswith("fin_")]
    print(f"\n{'=' * 108}\n財報因子（季頻，前 {a.uni} 名，方向依 IC 符號）\n{'=' * 108}")
    print(f"  {'特徵':<16}{'起':>9}{'季數':>5}{'IC平均':>9}{'IC_t':>8}{'IC_p':>9}"
          f"{'多空年化':>10}{'勝率':>7}{'前十分位-母體':>13}{'FDR':>5}")
    for _, r in fin.iterrows():
        print(f"  {r['特徵']:<16}{r['起']:>9}{int(r['季數']):>5}{r['IC平均']:>+9.4f}{r['IC_t']:>8.2f}"
              f"{r['IC_p']:>9.4f}{r['多空年化_扣成本']:>+10.2%}{r['勝率']:>7.0%}"
              f"{r['前十分位對母體_季均']:>+13.2%}{'✓' if r['FDR通過'] else '':>5}")
    print(f"\n  共 {len(res)} 個特徵一起做 FDR（{'含' if a.with_price else '不含'}價量特徵），"
          f"財報因子通過 {int(fin['FDR通過'].sum())} 個")

    print(f"\n{'=' * 108}\n多重比較：十分位多空（扣成本）一起送 Reality Check ／ SPA\n{'=' * 108}")
    for name, cols in (("只有財報因子", list(fin["特徵"])),
                       ("財報＋其他", list(spreads)) if a.with_price else (None, None)):
        if not name:
            continue
        X = pd.DataFrame({c: spreads[c] for c in cols}).dropna(how="all")
        X = X.loc[:, X.notna().sum() >= 16].fillna(0.0)
        best = X.mean().idxmax()
        rc = reality_check(X.to_numpy(), mean_block=4, n_boot=2000, seed=42)
        print(f"  {name}：{X.shape[1]} 組 × {X.shape[0]} 季　最佳 {best}"
              f"（季均 {X[best].mean():+.2%}）")
        for k, v in rc.items():
            print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")
    print(f"\n輸出：{OUT / f'factors{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
