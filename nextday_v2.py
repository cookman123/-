#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔日預測 v2：用新特徵庫重跑，並把誠實的成交假設設為預設。

和 v1 的三個差別
----------------
一、特徵從 43 個單檔技術指標擴到 105 個，補上原本完全沒有的橫斷面百分位、
    相對市場與產業、流動性與微結構。
二、預設標籤是 `y_oo`（t+1 開盤進、t+2 開盤出），不是 `y_cc`。
    v1 用收盤進出，而訊號也是用同一個收盤價算的 —— 那個假設撐出了
    年化 +161%，查下去發現是押在漲跌停（極端分位佔 9.4%，母體 1.1%）。
三、預設把當日鎖死（漲跌停）的個股從可選清單剔除。`trd_locked` 已經是
    特徵庫的欄位，直接拿來當過濾條件。

兩種假設都會跑，並列出來，所以「寬鬆假設下有多好」與「誠實假設下剩多少」
一次看得到。判讀規則不變：**扣成本後、誠實假設下仍為正才算數。**

成本
----
個股來回約 0.39%（手續費 0.1425%×3 折 ×2 ＋ 賣出證交稅 0.3%）。
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

STORE = ROOT / "data" / "features" / "panel.pkl"
OUT = ROOT / "results" / "nextday"
FEE, TAX = 0.001425 * 0.3, 0.003
COST = 2 * FEE + TAX


def walk_forward(df: pd.DataFrame, feats: list, label: str,
                 min_days: int, step_days: int) -> pd.Series:
    """
    依日期切分的 walk-forward；訓練集尾端切掉 2 個交易日。
    y_oo 的標籤跨到 t+2 開盤，所以要切 2 日而不是 1 日 —— 切 1 日
    會讓訓練集最後一列的答案由測試期第一天的開盤價決定。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    days = df.index.get_level_values("date").unique().sort_values()
    purge = 2
    out = []
    for k in range(min_days, len(days), step_days):
        tr_days = days[:max(0, k - purge)]
        te_days = days[k:min(k + step_days, len(days))]
        if len(te_days) == 0:
            break
        tr = df[df.index.get_level_values("date").isin(tr_days)]
        te = df[df.index.get_level_values("date").isin(te_days)]
        tr = tr.dropna(subset=[label])
        if len(tr) < 20000 or te.empty:
            continue
        # 本折內全為缺值或只有單一值的特徵要先剔除：HGB 的分箱會直接拋錯，
        # 一個退化欄位不該讓整趟執行失敗
        use = [c for c in feats if tr[c].nunique(dropna=True) >= 2]
        if len(use) < len(feats):
            print(f"      剔除退化特徵 {len(feats) - len(use)} 個："
                  f"{', '.join(c for c in feats if c not in use)[:70]}", flush=True)
        m = HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=200, early_stopping=False, random_state=42)
        m.fit(tr[use].to_numpy("float32"), tr[label].to_numpy("float32"))
        out.append(pd.Series(m.predict(te[use].to_numpy("float32")),
                             index=te.index))
        print(f"    {te_days[0]:%Y-%m} ~ {te_days[-1]:%Y-%m}"
              f"　訓練 {len(tr):,}　測試 {len(te):,}", flush=True)
    return pd.concat(out) if out else pd.Series(dtype=float)


def portfolio(pred: pd.Series, y: pd.Series, tradable: pd.Series,
              n: int, side: str, cost: float) -> tuple:
    """每天取預測最高／最低各 n 檔；tradable 為 False 的個股不納入可選清單。"""
    out, pl, ps, turns = [], set(), set(), []
    for d, g in pred.groupby(level="date"):
        g = g.droplevel("date")
        if tradable is not None:
            ok = tradable.xs(d, level="date")
            g = g[ok.reindex(g.index).fillna(False).to_numpy()]
        if len(g) < n * 2:
            continue
        g = g.sort_values(ascending=False)
        L = list(g.index[:n])
        S = list(g.index[-n:]) if side == "多空" else []
        yd = y.xs(d, level="date")
        a = float(yd.reindex(L).fillna(0.0).mean())
        b = float(yd.reindex(S).fillna(0.0).mean()) if S else 0.0
        tl = 1 - len(set(L) & pl) / n if pl else 1.0
        ts = (1 - len(set(S) & ps) / n) if (S and ps) else (1.0 if S else 0.0)
        # 多空的資本配置：毛曝險 1（多空各 0.5 單位），所以報酬取半價差，
        # 成本也取兩腿換手的一半。兩腿各自換手都要付費 —— 先前的版本
        # 把換手平均後只收一次卻用了完整毛價差，等於只收了該收的一半，
        # 那個錯誤把真實的 −21.8% 變成了 +30.2%。
        if S:
            r = (a - b) / 2 - (tl + ts) / 2 * cost
            turn = (tl + ts) / 2
        else:
            r = a - tl * cost
            turn = tl
        out.append((d, r))
        turns.append(turn)
        pl, ps = set(L), set(S)
    return pd.Series(dict(out)).sort_index(), float(np.mean(turns)) if turns else np.nan


def perf(r: pd.Series, base: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 60 or r.std() == 0:
        return {}
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    d = {"年化": (1 + r).prod() ** (1 / yrs) - 1,
         "夏普": r.mean() / r.std() * np.sqrt(252),
         "最大回撤": float((eq / eq.cummax() - 1).min())}
    if base is not None:
        idx = r.index.intersection(base.index)
        t, _ = nw_tstat(r.reindex(idx) - base.reindex(idx))
        d["超額t"] = t
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="隔日預測 v2（新特徵庫）")
    ap.add_argument("--min-days", type=int, default=1000)
    ap.add_argument("--step-days", type=int, default=252)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tag", default="",
                    help="讀 panel<tag>.pkl 並把輸出也加同一後綴，例如 --tag _full")
    a = ap.parse_args(argv)

    store = STORE.with_name(f"panel{a.tag}.pkl")
    if not store.exists():
        print(f"找不到特徵庫 {store}，請先跑 feature_engineering.py"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"載入特徵庫 {store.name}　{store.stat().st_size / 1e6:.0f} MB…", flush=True)
    df = pd.read_pickle(store)
    feats = [c for c in df.columns if not c.startswith("y_")]
    days = df.index.get_level_values("date").unique()
    print(f"  {len(df):,} 列 × {len(feats)} 特徵"
          f"　{len(days):,} 日　{days.min():%Y-%m} ~ {days.max():%Y-%m}")

    # 可交易性：trd_locked 是當日漲跌停鎖死旗標
    tradable = (df["trd_locked"] < 0.5) if "trd_locked" in df else None
    lock_rate = float(df["trd_locked"].mean()) if "trd_locked" in df else np.nan

    rows = []
    for label, lab_name, honest in (("y_oo", "開盤進出（誠實）", True),
                                    ("y_cc", "收盤進出（寬鬆）", False)):
        if label not in df.columns:
            continue
        print(f"\n{'=' * 104}\n【{lab_name}】walk-forward", flush=True)
        sub = df.dropna(subset=feats, how="all")
        pred = walk_forward(sub, feats, label, a.min_days, a.step_days)
        if pred.empty:
            print("  樣本不足"); continue
        y = df[label]
        act = y.reindex(pred.index)
        da = float((np.sign(pred) == np.sign(act))[act != 0].mean())
        ic = pd.concat([pred.rename("p"), act.rename("y")], axis=1) \
            .groupby(level="date").apply(
                lambda g: g["p"].corr(g["y"], method="spearman")
                if len(g) > 5 else np.nan, include_groups=False).dropna()
        icm = float(ic.mean())
        ict = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
        pred.rename("pred").reset_index().to_csv(
            OUT / f"pred_v2{a.tag}_{label}.csv", index=False, encoding="utf-8-sig")
        ew = act.groupby(level="date").mean()
        print(f"  方向準確率 {da:.2%}　每日 IC {icm:+.4f}（t={ict:.2f}）"
              f"　母體等權 {perf(ew)['年化']:+.2%}")
        print(f"  {'可選清單':<14}{'方式':<6}{'檔數':>5}{'換手':>7}"
              f"{'零成本':>12}{'扣成本':>12}{'夏普':>8}{'超額t':>8}")
        for filt_name, filt in (("剔除漲跌停", tradable), ("全母體", None)):
            for side in ("多空", "多單"):
                for n in (10, 30):
                    r0, _ = portfolio(pred, y, filt, n, side, 0.0)
                    r1, turn = portfolio(pred, y, filt, n, side, COST)
                    s0, s1 = perf(r0), perf(r1, ew)
                    if not s0 or not s1:
                        continue
                    rows.append({"標籤": lab_name, "可選清單": filt_name,
                                 "方式": side, "檔數": n, "DA": da, "IC": icm,
                                 "IC_t": ict, "換手": turn,
                                 "年化_零成本": s0["年化"],
                                 "年化_扣成本": s1["年化"],
                                 "夏普": s1["夏普"], "超額t": s1["超額t"]})
                    print(f"  {filt_name:<14}{side:<6}{n:>5}{turn:>7.2f}"
                          f"{s0['年化']:>+12.2%}{s1['年化']:>+12.2%}"
                          f"{s1['夏普']:>8.2f}{s1['超額t']:>8.2f}")

    res = pd.DataFrame(rows)
    if res.empty:
        print("\n沒有結果"); return 1
    res.to_csv(OUT / f"v2{a.tag}.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 104}\n判讀\n{'=' * 104}")
    print(f"  母體整體漲跌停鎖死比例 {lock_rate:.2%}")
    honest = res[(res["標籤"].str.startswith("開盤")) &
                 (res["可選清單"] == "剔除漲跌停")]
    if not honest.empty:
        best = honest.loc[honest["年化_扣成本"].idxmax()]
        print(f"  誠實假設下最佳：{best['方式']} {int(best['檔數'])} 檔"
              f"　扣成本年化 {best['年化_扣成本']:+.2%}"
              f"　超額 t {best['超額t']:.2f}")
        n_pos = int((honest["年化_扣成本"] > 0).sum())
        print(f"  誠實假設的 {len(honest)} 組變體中，扣成本後為正的有 {n_pos} 組")
    loose = res[res["標籤"].str.startswith("收盤")]
    if not loose.empty:
        print(f"  對照：寬鬆假設最佳扣成本年化 "
              f"{loose['年化_扣成本'].max():+.2%}"
              f"（這個數字不可用，僅供對比）")
    print(f"\n輸出：{OUT / f'v2{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
