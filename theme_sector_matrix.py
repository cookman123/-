#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每個產業類股 × 每個新聞主題的完整敏感度矩陣。

theme_sector_test.py 只測事先指定的 55 組配對（例如戰爭→航運）。這裡把 18 個主題
（themes.py）對所有產業類股全部算一遍，回答「每個類股對每個主題的比重是多少」。

比重怎麼定義
------------
    主題熱度：該主題當日新聞則數，相對自身近 60 日的 z 分數（theme_sector_test.heat）。
              盤後（13:30 後）的新聞歸到下一個交易日，只用當天（含）以前的新聞。
    類股超額報酬：類股指數未來 20 個交易日報酬 − 加權指數同期報酬。
              扣掉大盤，否則每個主題都會跟所有類股一起相關。
    比重（β）：類股超額報酬 = a + β × 主題熱度 + γ × 類股過去 20 日報酬 + e
              β 的單位是「主題熱度高一個標準差時，類股未來 20 日相對大盤多（少）幾 %」。
              控制自身動能，因為新聞常是在報導已經漲跌的類股（反向因果）。
    t 值：Newey-West（落後 20 日），因為 20 日報酬每天重疊。
    多重比較：全部格子一起做 Benjamini-Hochberg（FDR 10%）。格子有幾百個，
              5% 水準下純機率就會有一堆「顯著」，只有過 FDR 的才值得看。

限制
----
    產業分類用現行清單（不是時點正確）；主題詞表是人工編的（可能帶有事後知識）；
    類股指數由現有股票池等權合成。
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
import theme_sector_test as TS  # noqa: E402
from quarterly_factors import bh_fdr  # noqa: E402
from themes import THEMES, THEME_SECTOR  # noqa: E402

OUT = ROOT / "results" / "theme"
H = 20


def ols_nw(y: np.ndarray, X: np.ndarray, lags: int) -> tuple:
    """OLS 係數與 Newey-West 標準誤（Bartlett 權重）。"""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="主題 × 類股完整敏感度矩陣")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args(argv)

    imap = T.load_industry_map()
    sectors = {}
    for name, codes in imap.items():
        idx, used = T.build_sector_index(codes, a.start, a.end)
        if idx is not None and not idx.empty and len(used) >= 3:
            sectors[name] = idx["close"]
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]) \
        .set_index("date")["close"].sort_index()
    tw = tw[~tw.index.duplicated(keep="last")]

    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in sectors.values()])))
    th = TS.build_theme_series(cal)
    print(f"類股 {len(sectors)} 個　主題 {len([t for t in THEMES if t in th.columns])} 個　"
          f"{cal[0]:%Y-%m} ~ {cal[-1]:%Y-%m}")
    mfwd = tw.reindex(cal).ffill()
    mfwd = mfwd.shift(-H) / mfwd - 1

    expected = {(t, s): d for t, ps in THEME_SECTOR.items() for s, d, _ in ps}
    rows = []
    for theme in THEMES:
        if theme not in th.columns:
            continue
        h = TS.heat(th[theme])
        for sec, px in sectors.items():
            fwd = (px.shift(-H) / px - 1) - mfwd.reindex(px.index)
            own = px.pct_change(20)
            d = pd.DataFrame({"h": h, "fwd": fwd, "own": own}).dropna()
            if len(d) < 250:
                continue
            X = np.column_stack([np.ones(len(d)), d["h"], d["own"]])
            b, se = ols_nw(d["fwd"].to_numpy(), X, lags=H)
            rows.append({"主題": theme, "類股": sec, "比重": b[1], "t": b[1] / se[1],
                         "天數": len(d), "事先預期": expected.get((theme, sec), 0)})
    res = pd.DataFrame(rows)
    from scipy import stats
    res["p"] = 2 * stats.norm.sf(res["t"].abs())
    res["FDR通過"] = bh_fdr(res["p"].to_numpy(), q=0.10)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "theme_sector_matrix.csv", index=False, encoding="utf-8-sig")

    n = len(res)
    print(f"\n共 {n} 格；|t| > 1.96 的 {int((res['t'].abs() > 1.96).sum())} 格"
          f"（純機率約 {n * 0.05:.0f} 格）；FDR 10% 通過 {int(res['FDR通過'].sum())} 格")
    pre = res[res["事先預期"] != 0]
    if len(pre):
        k = int((np.sign(pre["比重"]) == pre["事先預期"]).sum())
        print(f"事先預期的 {len(pre)} 組配對中，方向符合的 {k} 組"
              f"（二項檢定 p = {stats.binomtest(k, len(pre), 0.5).pvalue:.3f}）")
    show = res.reindex(res["t"].abs().sort_values(ascending=False).index).head(20).copy()
    show["比重"] = show["比重"].map(lambda x: f"{x:+.2%}")
    show["t"] = show["t"].map(lambda x: f"{x:+.2f}")
    print("\n|t| 最大的 20 格（比重＝主題熱度高 1 個標準差時，類股未來 20 日相對大盤的報酬）")
    print(show[["主題", "類股", "比重", "t", "FDR通過", "事先預期"]].to_string(index=False))

    # 熱圖
    M = res.pivot(index="類股", columns="主題", values="比重") * 100
    Tm = res.pivot(index="類股", columns="主題", values="t")
    Fm = res.pivot(index="類股", columns="主題", values="FDR通過")
    M = M.loc[M.abs().max(axis=1).sort_values(ascending=False).index]
    Tm, Fm = Tm.loc[M.index, M.columns], Fm.loc[M.index, M.columns]
    M.round(3).to_csv(OUT / "theme_sector_matrix_pivot.csv", encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Microsoft JhengHei", "axes.unicode_minus": False})
    v = np.nanmax(np.abs(M.to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(M) + 2))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(M.columns)), M.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(M.index)), M.index, fontsize=8.5)
    for i in range(len(M.index)):
        for j in range(len(M.columns)):
            t = Tm.iat[i, j]
            if pd.notna(t) and abs(t) > 1.96:
                ax.text(j, i, "●" if Fm.iat[i, j] else "·", ha="center", va="center",
                        fontsize=9 if Fm.iat[i, j] else 12, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("比重：主題熱度 +1σ 時，類股未來 20 日相對大盤（%）")
    ax.set_title("每個類股對每個新聞主題的比重（控制類股自身動能；· = |t|>1.96，● = 過 FDR 10%）",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "theme_sector_matrix.png", dpi=180)
    print(f"\n輸出：{OUT / 'theme_sector_matrix.csv'}、theme_sector_matrix_pivot.csv、theme_sector_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
