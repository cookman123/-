#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季頻因子篩選：105 個特徵裡有沒有任何一個賺得到季頻的錢。

為什麼換到季頻
--------------
隔日那條線已經用三種方式否證：毛價差 0.426%/日打不過兩腿合計 0.62%/日的成本，
而且 RC／SPA 的 p 值都是 1.0000。擋住它的從頭到尾是成本與流動性，不是預測力。

季頻的成本結構完全不同：一年換四次，每次來回 0.39%，總共約 1.6%/年。
日頻要付 0.62% × 250 天。同一個訊號在季頻上有兩個數量級的成本餘裕，
所以「日頻無效」完全不蘊含「季頻無效」—— 這是值得獨立測的問題。
本專題唯一存活的效應（橫斷面波動度溢酬）本來就是季頻的。

怎麼測
------
    每季最後一個交易日取特徵，隔日開盤進場，下一季隔日開盤出場
      （一日實施落差，季頻下成本可忽略，但避免用收盤價同時當訊號與成交價）
    母體用時點正確的成交金額前 250 名
    逐特徵算每季橫斷面 Spearman IC，以季為叢集單位做 t 檢定
    另算十分位多空價差的年化績效，扣交易成本
    **多重比較一定要做**：105 個特徵隨機也會有幾個 p < 0.05。
      同時給 Benjamini-Hochberg FDR 與 White Reality Check。

判讀
----
單一特徵的 t 值不算數。要同時過 FDR 與 RC，而且逐季穩定，才值得接到策略上。
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
from reality_check import reality_check  # noqa: E402

STORE = ROOT / "data" / "features" / "panel.pkl"
OUT = ROOT / "results" / "quarterly"
FEE, TAX = 0.001425 * 0.3, 0.003
COST = 2 * FEE + TAX


def quarter_ends(idx: pd.DatetimeIndex) -> list:
    """每季最後一個交易日。"""
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.quarter]).last().values)


def forward_returns(O: pd.DataFrame, qe: list) -> pd.DataFrame:
    """
    季報酬：第 t 季末的隔一個交易日開盤進場，第 t+1 季末的隔一個交易日開盤出場。
    用開盤價而不是收盤價，訊號與成交價才不是同一個數字。

    出場日沒有開盤價（停牌、下市）時，用持有期間最後一個有效開盤價出場。
    原本直接留 NaN，之後選股時會被 notna() 濾掉 —— 等於換倉當天就知道誰會在這季
    停牌，約 2.5% 的名單因此被事先剔除（2026-10-15 修正）。進場日沒有報價的才剔除。
    """
    idx = O.index
    pos = {d: i for i, d in enumerate(idx)}
    rows = {}
    for a, b in zip(qe[:-1], qe[1:]):
        ia, ib = pos[pd.Timestamp(a)], pos[pd.Timestamp(b)]
        if ia + 1 >= len(idx) or ib + 1 >= len(idx):
            continue
        exit_ = O.iloc[ia + 1:ib + 2].ffill().iloc[-1]
        rows[pd.Timestamp(a)] = exit_ / O.iloc[ia + 1] - 1
    return pd.DataFrame(rows).T.sort_index()


def decile_spread(sig: pd.Series, ret: pd.Series, n_grp: int = 10) -> tuple:
    """最高與最低十分位的等權報酬差。"""
    s = sig.dropna()
    r = ret.reindex(s.index)
    ok = r.notna()
    s, r = s[ok], r[ok]
    if len(s) < n_grp * 3:
        return np.nan, np.nan, np.nan
    k = max(1, len(s) // n_grp)
    o = s.sort_values(ascending=False)
    hi = float(r.reindex(o.index[:k]).mean())
    lo = float(r.reindex(o.index[-k:]).mean())
    return hi - lo, hi, lo


def bh_fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg：回傳布林陣列，True 表示在 FDR=q 下顯著。"""
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        out[order[:kmax + 1]] = True
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="季頻因子篩選")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--tag", default="",
                    help="讀 panel<tag>.pkl 並把輸出也加同一後綴，例如 --tag _full")
    a = ap.parse_args(argv)

    store = STORE.with_name(f"panel{a.tag}.pkl")
    if not store.exists():
        print(f"找不到特徵庫 {store}"); return 1
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"載入特徵庫 {store.name}…", flush=True)
    df = pd.read_pickle(store)
    feats = [c for c in df.columns if not c.startswith("y_")]
    print(f"  {len(df):,} 列 × {len(feats)} 特徵", flush=True)

    print(f"載入價格面板（算季報酬用開盤價）…", flush=True)
    P = FE.load_panels(a.start, a.end, quiet=True)
    O = P["open"]
    qe = [d for d in quarter_ends(O.index)
          if pd.Timestamp(d) in df.index.get_level_values("date")]
    fwd = forward_returns(O, qe)
    print(f"  季數 {len(fwd)}　{fwd.index[0]:%Y-%m} ~ {fwd.index[-1]:%Y-%m}", flush=True)

    # 把季末那幾天的特徵抽出來，母體遮罩已經內含在特徵庫裡
    qd = [d for d in fwd.index if d in df.index.get_level_values("date")]
    sub = df.loc[df.index.get_level_values("date").isin(qd), feats]
    print(f"  季末樣本 {len(sub):,} 列（平均每季 {len(sub) / len(qd):.0f} 檔）\n",
          flush=True)

    rows, spread_series = [], {}
    for j, c in enumerate(feats, 1):
        ics, sps = [], []
        for d in qd:
            try:
                s = sub.loc[d, c]
            except KeyError:
                continue
            r = fwd.loc[d].reindex(s.index)
            s2 = s.dropna()
            if len(s2) < 30:
                continue
            r2 = r.reindex(s2.index)
            m = r2.notna()
            if m.sum() < 30:
                continue
            ics.append(float(s2[m].corr(r2[m], method="spearman")))
            sp, _, _ = decile_spread(s2[m], r2[m])
            sps.append((d, sp))
        if len(ics) < 20:
            continue
        ic = np.array(ics, dtype=float)
        ic = ic[np.isfinite(ic)]
        t_ic = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
        from scipy import stats as st
        p_ic = float(2 * (1 - st.t.cdf(abs(t_ic), len(ic) - 1)))
        sp = pd.Series(dict(sps)).dropna().sort_index()
        # 季頻每季重建組合，十分位多空兩腿都要付費
        sp_net = sp - 2 * COST
        ann = (1 + sp_net).prod() ** (4 / len(sp_net)) - 1 if len(sp_net) else np.nan
        rows.append({"特徵": c, "季數": len(ic), "IC平均": ic.mean(),
                     "IC_t": t_ic, "IC_p": p_ic,
                     "十分位價差_季均": float(sp.mean()),
                     "年化_扣成本": ann,
                     "勝率": float((sp_net > 0).mean())})
        spread_series[c] = sp_net
        if j % 20 == 0:
            print(f"  已測 {j}/{len(feats)}", flush=True)

    res = pd.DataFrame(rows).sort_values("IC_t", key=abs, ascending=False)
    res["FDR通過"] = bh_fdr(res["IC_p"].to_numpy(), a.fdr)
    res.to_csv(OUT / f"factors{a.tag}.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 104}\n季頻因子：IC t 值絕對值最大的 15 個\n{'=' * 104}")
    print(f"  {'特徵':<24}{'季數':>5}{'IC平均':>9}{'IC_t':>8}{'IC_p':>9}"
          f"{'季均價差':>10}{'年化扣成本':>11}{'勝率':>7}{'FDR':>5}")
    for _, r in res.head(15).iterrows():
        print(f"  {r['特徵']:<24}{int(r['季數']):>5}{r['IC平均']:>+9.4f}"
              f"{r['IC_t']:>8.2f}{r['IC_p']:>9.4f}"
              f"{r['十分位價差_季均']:>+10.2%}{r['年化_扣成本']:>+11.2%}"
              f"{r['勝率']:>7.0%}{'✓' if r['FDR通過'] else '':>5}")

    n_fdr = int(res["FDR通過"].sum())
    print(f"\n  {len(res)} 個特徵中，FDR={a.fdr:.0%} 下通過 {n_fdr} 個")

    # 多重比較：把所有特徵的十分位多空當成一族策略送 Reality Check
    print(f"\n{'=' * 104}\n多重比較：White Reality Check ／ SPA\n{'=' * 104}")
    X = pd.DataFrame(spread_series).dropna(how="all")
    X = X.loc[:, X.notna().sum() >= 20].fillna(0.0)
    print(f"  變體 {X.shape[1]} 組　季數 {X.shape[0]}")
    best = X.mean().idxmax()
    print(f"  最佳：{best}　季均 {X[best].mean():+.2%}"
          f"　年化 {(1 + X[best]).prod() ** (4 / len(X)) - 1:+.2%}")
    rc = reality_check(X.to_numpy(), mean_block=4, n_boot=2000, seed=42)
    for k, v in rc.items():
        mk = ""
        if isinstance(v, float) and ("_p" in k.lower() or k.lower().endswith("p")):
            mk = "　→ 通過" if v < 0.05 else "　→ 未通過"
        print(f"  {k} = {v:.4f}{mk}" if isinstance(v, float) else f"  {k} = {v}")

    print(f"\n判讀")
    print(f"  單一特徵的 t 值不算數：{len(res)} 個特徵隨機也會冒出幾個 p<0.05。")
    print(f"  要同時過 FDR 與 Reality Check，而且逐季穩定，才值得接到策略上。")
    print(f"\n輸出：{OUT / f'factors{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
