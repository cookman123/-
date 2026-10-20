#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驗證季頻篩選的勝出者：`cs_ts_vol60` 的 +22.77% 是真的還是又一次僥倖。

為什麼要單獨驗
--------------
`quarterly_factors.py` 的 Reality Check 給 p = 0.027、SPA p = 0.016，兩者都過。
但那是 64 季、105 組變體的 bootstrap，樣本數偏薄；而且勝出者正好就是本專題
原有策略的訊號（60 日波動度橫斷面排名），所以要先確認一件事：

    這是「原有效應在更嚴格的檢定下存活」，還是「我換了規格之後數字自己變好」？

兩者結論天差地別。前者讓原本 t=1.30 的結果升級；後者只是又一次挑規格。

五道檢查
--------
一、逐年與逐季穩定性 —— 是不是一兩年貢獻全部
二、多腿分解 —— 原策略是純做多，若價值全在空腿（放空低波動股）則不可執行
三、規格敏感度 —— 分位數、母體大小、持有期改動後還剩多少
四、與原策略對照 —— 同期間、同成本下，新規格是否真的比原規格好
五、子期間 —— 前半與後半各自是否成立（效應會被套利掉）
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
from quarterly_factors import forward_returns, quarter_ends  # noqa: E402

STORE = ROOT / "data" / "features" / "panel.pkl"
OUT = ROOT / "results" / "quarterly"
FEE, TAX = 0.001425 * 0.3, 0.003
COST = 2 * FEE + TAX
SIG = "cs_ts_vol60"


def book(sub: pd.DataFrame, fwd: pd.DataFrame, qd: list, sig: str,
         n_grp: int, side: str, n_fixed: int | None = None) -> dict:
    """
    每季依 sig 排序建組合。side 為「多空」「多單」「空單」。
    n_fixed 給定時改成固定檔數（原策略是固定 30 檔，不是分位數）。
    成本依實際換手計：兩腿各自換手都要付費。
    """
    rows, pl, ps, turns = [], set(), set(), []
    for d in qd:
        try:
            s = sub.loc[d, sig].dropna()
        except KeyError:
            continue
        r = fwd.loc[d].reindex(s.index)
        m = r.notna()
        s, r = s[m], r[m]
        if len(s) < 60:
            continue
        o = s.sort_values(ascending=False)
        k = n_fixed if n_fixed else max(1, len(o) // n_grp)
        L = list(o.index[:k]) if side in ("多空", "多單") else []
        S = list(o.index[-k:]) if side in ("多空", "空單") else []
        a = float(r.reindex(L).mean()) if L else 0.0
        b = float(r.reindex(S).mean()) if S else 0.0
        tl = (1 - len(set(L) & pl) / len(L)) if (L and pl) else (1.0 if L else 0.0)
        ts = (1 - len(set(S) & ps) / len(S)) if (S and ps) else (1.0 if S else 0.0)
        if side == "多空":
            # 毛曝險 1：多空各半，報酬取半價差，兩腿換手各付一半成本
            ret = (a - b) / 2 - (tl + ts) / 2 * COST
            turn = (tl + ts) / 2
        elif side == "多單":
            ret, turn = a - tl * COST, tl
        else:
            ret, turn = -b - ts * COST, ts
        rows.append((d, ret))
        turns.append(turn)
        pl, ps = set(L), set(S)
    s = pd.Series(dict(rows)).sort_index()
    return {"r": s, "換手": float(np.mean(turns)) if turns else np.nan}


def ann(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (4 / len(r)) - 1 if len(r) >= 8 else np.nan


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(4)) if len(r) > 4 and r.std() else np.nan


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="季頻勝出者的五道驗證")
    ap.add_argument("--tag", default="",
                    help="讀 panel<tag>.pkl 並把輸出也加同一後綴，例如 --tag _full")
    ap.add_argument("--sig", default=SIG, help="要驗證的特徵名")
    a = ap.parse_args(argv)
    sig = a.sig

    store = STORE.with_name(f"panel{a.tag}.pkl")
    if not store.exists():
        print(f"找不到特徵庫 {store}"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"特徵庫 {store.name}　驗證訊號 {sig}")
    df = pd.read_pickle(store)
    if sig not in df.columns:
        print(f"特徵庫裡沒有 {sig}"); return 1
    P = FE.load_panels("2010-01-01", "2026-12-31", quiet=True)
    O = P["open"]
    qe = [d for d in quarter_ends(O.index)
          if pd.Timestamp(d) in df.index.get_level_values("date")]
    fwd = forward_returns(O, qe)
    qd = [d for d in fwd.index if d in df.index.get_level_values("date")]
    sub = df.loc[df.index.get_level_values("date").isin(qd)]
    ew = pd.Series({d: float(fwd.loc[d].reindex(
        sub.loc[d].index).dropna().mean()) for d in qd}).sort_index()
    print(f"季數 {len(qd)}　{qd[0]:%Y-%m} ~ {qd[-1]:%Y-%m}"
          f"　母體等權年化 {ann(ew):+.2%}\n")

    # ---------- 一、多腿分解 ----------
    print(f"{'=' * 96}\n一、多腿分解（十分位）\n{'=' * 96}")
    print(f"  {'腿':<8}{'年化':>10}{'夏普':>8}{'換手':>8}{'勝率':>8}")
    base = {}
    for side in ("多空", "多單", "空單"):
        d = book(sub, fwd, qd, sig, 10, side)
        base[side] = d["r"]
        print(f"  {side:<8}{ann(d['r']):>+10.2%}{sharpe(d['r']):>8.2f}"
              f"{d['換手']:>8.2f}{float((d['r'] > 0).mean()):>8.0%}")
    print(f"\n  原策略是純做多，所以「多單」那一列才是可比的數字。")

    # ---------- 二、逐年 ----------
    print(f"\n{'=' * 96}\n二、逐年（多單十分位 vs 母體等權）\n{'=' * 96}")
    L = base["多單"]
    yr = pd.DataFrame({"策略": L, "母體": ew.reindex(L.index)}).dropna()
    rows = []
    for y, g in yr.groupby(yr.index.year):
        if len(g) < 2:
            continue
        s = (1 + g["策略"]).prod() - 1
        b = (1 + g["母體"]).prod() - 1
        rows.append((y, s, b, s - b))
    print(f"  {'年':<7}{'策略':>10}{'母體':>10}{'差':>10}")
    for y, s, b, d in rows:
        print(f"  {y:<7}{s:>+10.1%}{b:>+10.1%}{d:>+10.1%}")
    win = sum(1 for r in rows if r[3] > 0)
    print(f"\n  勝過母體 {win}/{len(rows)} 年")
    top2 = sorted(rows, key=lambda r: -r[3])[:2]
    rest = [r for r in rows if r[0] not in (top2[0][0], top2[1][0])]
    print(f"  貢獻最大兩年：{top2[0][0]} ({top2[0][3]:+.1%})、"
          f"{top2[1][0]} ({top2[1][3]:+.1%})")
    print(f"  剔除後其餘 {len(rest)} 年平均超額 "
          f"{np.mean([r[3] for r in rest]):+.2%}")

    # ---------- 三、規格敏感度 ----------
    print(f"\n{'=' * 96}\n三、規格敏感度（多單）\n{'=' * 96}")
    print(f"  {'規格':<26}{'年化':>10}{'夏普':>8}{'超額':>10}")
    specs = [("十分位（約 25 檔）", dict(n_grp=10)),
             ("五分位（約 50 檔）", dict(n_grp=5)),
             ("二十分位（約 12 檔）", dict(n_grp=20)),
             ("固定 30 檔（原策略規格）", dict(n_grp=10, n_fixed=30)),
             ("固定 50 檔", dict(n_grp=10, n_fixed=50))]
    for name, kw in specs:
        d = book(sub, fwd, qd, sig, side="多單", **kw)
        print(f"  {name:<26}{ann(d['r']):>+10.2%}{sharpe(d['r']):>8.2f}"
              f"{ann(d['r']) - ann(ew):>+10.2%}")

    # ---------- 四、子期間 ----------
    print(f"\n{'=' * 96}\n四、子期間（效應會不會被套利掉）\n{'=' * 96}")
    d30 = book(sub, fwd, qd, sig, 10, "多單", n_fixed=30)["r"]
    half = len(d30) // 2
    for name, seg in (("前半", d30.iloc[:half]), ("後半", d30.iloc[half:])):
        b = ew.reindex(seg.index)
        print(f"  {name}（{seg.index[0]:%Y-%m} ~ {seg.index[-1]:%Y-%m}，"
              f"{len(seg)} 季）　策略 {ann(seg):+.2%}"
              f"　母體 {ann(b):+.2%}　超額 {ann(seg) - ann(b):+.2%}")

    # ---------- 五、與原策略對照 ----------
    print(f"\n{'=' * 96}\n五、對照原策略\n{'=' * 96}")
    print(f"  原策略（strategy_spec.py，含 120 日均線趨勢過濾）"
          f"年化 +18.24%、夏普 0.91、超額 t 1.30")
    print(f"  本節的固定 30 檔純做多（無趨勢過濾）年化 {ann(d30):+.2%}"
          f"、夏普 {sharpe(d30):.2f}")
    print(f"  差別在於：本節季頻換倉、開盤成交、無趨勢過濾；")
    print(f"  原策略日頻判斷趨勢訊號。兩者不是同一個東西，不能直接比優劣。")

    pd.DataFrame(rows, columns=["年", "策略", "母體", "超額"]).to_csv(
        OUT / f"verify_yearly{a.tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出：{OUT / f'verify_yearly{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
