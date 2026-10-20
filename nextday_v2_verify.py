#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 結果的三道檢定：那個「誠實假設下 +30.21%、超額 t 2.01」站不站得住。

為什麼不能直接接受
------------------
t = 2.01 是**挑過之後**的那一組。整個探索過程試了兩種標籤 × 兩種可選清單 ×
兩種方向 × 兩種檔數，加上 v1 的變體與特徵集本身的選擇，單一 t 值沒有扣掉
多重比較。而且本專題已經有五次「t 值在更嚴格的檢定下消失」的紀錄。

三道檢定
--------
一、多重比較（White Reality Check ／ Hansen SPA）
    把整族變體一起送檢，問「最佳那組的績效能否在整族都無技能的虛無下出現」。

二、分年穩定性
    +30% 是十二年都穩，還是一兩年貢獻掉全部？逐年算超額並看勝率。

三、多空分腿與放空可行性
    多單是 −3.60%，表示價值全在空方。本節把兩腿拆開計算，並用
    Amihud 非流動性估計市場衝擊 —— 每天換手 80% 押在極端分位，
    掛單本身會推動價格，而前面所有數字都假設衝擊為零。

    台股放空的實務限制另外列出：融券額度、平盤以下不得放空（部分標的）、
    股東會與除權息強制回補、借券費。這些無法只用成本參數表達。
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
from portfolio_backtest import nw_tstat  # noqa: E402
from reality_check import reality_check  # noqa: E402

STORE = ROOT / "data" / "features" / "panel.pkl"
OUT = ROOT / "results" / "nextday"
FEE, TAX = 0.001425 * 0.3, 0.003
COST = 2 * FEE + TAX


def legs(pred: pd.Series, y: pd.Series, tradable: pd.Series, n: int,
         cost: float, amihud: pd.Series | None = None,
         impact_bp: float = 0.0) -> dict:
    """
    把多空兩腿分開算，並可加上以 Amihud 估計的市場衝擊。
    impact_bp 是「每單位 Amihud 的衝擊基點」，用來做敏感度而非精確估計。
    """
    rl, rs, turns, imp = [], [], [], []
    pl, ps = set(), set()
    for d, g in pred.groupby(level="date"):
        g = g.droplevel("date")
        if tradable is not None:
            ok = tradable.xs(d, level="date")
            g = g[ok.reindex(g.index).fillna(False).to_numpy()]
        if len(g) < n * 2:
            continue
        g = g.sort_values(ascending=False)
        L, S = list(g.index[:n]), list(g.index[-n:])
        yd = y.xs(d, level="date")
        a = float(yd.reindex(L).fillna(0.0).mean())
        b = float(yd.reindex(S).fillna(0.0).mean())
        tl = 1 - len(set(L) & pl) / n if pl else 1.0
        ts = 1 - len(set(S) & ps) / n if ps else 1.0
        k = 0.0
        if amihud is not None and impact_bp:
            ad = amihud.xs(d, level="date")
            k = float(ad.reindex(L + S).fillna(0.0).mean()) * impact_bp / 1e4
        rl.append((d, a - tl * cost - k))
        rs.append((d, -b - ts * cost - k))
        turns.append((tl + ts) / 2)
        imp.append(k)
        pl, ps = set(L), set(S)
    L = pd.Series(dict(rl)).sort_index()
    S = pd.Series(dict(rs)).sort_index()
    return {"多單": L, "空單": S, "多空": (L + S) / 2,
            "換手": float(np.mean(turns)), "衝擊": float(np.mean(imp))}


def ann(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 60 else np.nan


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="v2 結果的三道檢定")
    ap.add_argument("--label", default="y_oo")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)

    pf = OUT / f"pred_v2_{a.label}.csv"
    if not pf.exists():
        print(f"找不到 {pf}，請先跑 nextday_v2.py"); return 1
    df = pd.read_pickle(STORE)
    p = pd.read_csv(pf, encoding="utf-8-sig", parse_dates=["date"],
                    dtype={"code": str})
    pred = p.set_index(["date", "code"])["pred"]
    pred = pred[pred.index.isin(df.index)]
    y = df[a.label]
    tradable = df["trd_locked"] < 0.5
    amihud = df["liq_amihud20"]
    ew = y.reindex(pred.index).groupby(level="date").mean()
    print(f"預測 {len(pred):,} 列　標籤 {a.label}"
          f"　{pred.index.get_level_values('date').min():%Y-%m}"
          f" ~ {pred.index.get_level_values('date').max():%Y-%m}")

    # ---------- 一、多重比較 ----------
    print(f"\n{'=' * 96}\n一、多重比較：整族變體一起送檢\n{'=' * 96}")
    fam = {}
    for filt_name, filt in (("剔漲跌停", tradable), ("全母體", None)):
        for n in (10, 20, 30, 50):
            d = legs(pred, y, filt, n, COST)
            for side in ("多單", "空單", "多空"):
                fam[f"{filt_name}／{side}／{n}檔"] = d[side]
    idx = sorted(set().union(*[s.index for s in fam.values()]))
    base = ew.reindex(idx).fillna(0.0)
    excess = {k: (v.reindex(idx).fillna(0.0) - base) for k, v in fam.items()}
    X = pd.DataFrame(excess)
    print(f"  變體 {X.shape[1]} 組　共同交易日 {X.shape[0]:,}")
    best = X.mean().idxmax()
    print(f"  最佳變體：{best}　年化超額 {X[best].mean() * 252:+.2%}")
    t_best, p_best = nw_tstat(X[best])
    print(f"  該組單獨看的超額 t = {t_best:.2f}（p = {p_best:.4f}）"
          f" —— 這是挑過之後的 t，下面才是扣掉多重比較的結果")
    rc = reality_check(X.to_numpy(), mean_block=10, n_boot=2000, seed=42)
    for k, v in rc.items():
        mark = ""
        if isinstance(v, float) and k.lower().startswith(("rc", "spa", "p")):
            mark = "　→ 通過" if v < 0.05 else "　→ 未通過"
        print(f"  {k} = {v:.4f}{mark}" if isinstance(v, float)
              else f"  {k} = {v}{mark}")

    # ---------- 二、分年穩定性 ----------
    print(f"\n{'=' * 96}\n二、分年穩定性\n{'=' * 96}")
    d = legs(pred, y, tradable, a.n, COST)
    print(f"  {'年':<8}{'多空':>10}{'多單':>10}{'空單':>10}{'母體等權':>10}{'多空超額':>10}")
    yrs = []
    for yr, g in d["多空"].groupby(d["多空"].index.year):
        if len(g) < 60:
            continue
        l = d["多單"].reindex(g.index)
        s = d["空單"].reindex(g.index)
        b = ew.reindex(g.index).fillna(0.0)
        row = (yr, ann(g), ann(l), ann(s), ann(b), ann(g) - ann(b))
        yrs.append(row)
        print(f"  {yr:<8}{row[1]:>+10.1%}{row[2]:>+10.1%}{row[3]:>+10.1%}"
              f"{row[4]:>+10.1%}{row[5]:>+10.1%}")
    win = sum(1 for r in yrs if r[5] > 0)
    print(f"\n  勝過母體 {win}/{len(yrs)} 年")
    top2 = sorted(yrs, key=lambda r: -r[5])[:2]
    print(f"  貢獻最大的兩年：{top2[0][0]} ({top2[0][5]:+.1%})、"
          f"{top2[1][0]} ({top2[1][5]:+.1%})")
    rest = [r for r in yrs if r[0] not in (top2[0][0], top2[1][0])]
    if rest:
        print(f"  **剔除這兩年後的其餘 {len(rest)} 年平均超額 "
              f"{np.mean([r[5] for r in rest]):+.2%}**")

    # ---------- 三、分腿與市場衝擊 ----------
    print(f"\n{'=' * 96}\n三、多空分腿與市場衝擊敏感度\n{'=' * 96}")
    print(f"  日換手 {d['換手']:.2f}（約每年 {d['換手'] * 252:.0f} 次進出）")
    print(f"\n  {'情境':<22}{'多單':>11}{'空單':>11}{'多空':>11}")
    print(f"  {'零衝擊（前面的數字）':<22}{ann(d['多單']):>+11.2%}"
          f"{ann(d['空單']):>+11.2%}{ann(d['多空']):>+11.2%}")
    for bp in (1, 3, 5, 10):
        e = legs(pred, y, tradable, a.n, COST, amihud, bp)
        print(f"  {f'Amihud 衝擊 ×{bp}bp':<22}{ann(e['多單']):>+11.2%}"
              f"{ann(e['空單']):>+11.2%}{ann(e['多空']):>+11.2%}"
              f"　（平均扣 {e['衝擊'] * 252:.1%}/年）")

    print(f"\n  台股放空的實務限制（無法用成本參數表達）：")
    print(f"    融券額度有限，極端分位多為中小型股，常常借不到券")
    print(f"    部分標的平盤以下不得放空")
    print(f"    股東會與除權息前強制回補，被迫在不利時點平倉")
    print(f"    每天換手 {d['換手']:.0%} 意味著空單幾乎每日重建，上述限制會反覆觸發")

    pd.DataFrame(yrs, columns=["年", "多空", "多單", "空單", "母體等權", "超額"]) \
        .to_csv(OUT / "v2_yearly.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出：{OUT / 'v2_yearly.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
