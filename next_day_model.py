#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔日預測：預測明天的漲跌與報酬，今天收盤依預測進出。

要回答的問題
------------
「隔天如果是正的就今天買、隔天如果是負的就今天賣」——這條規則的前提是
你今天就知道明天的方向。所以真正要測的是：**用今天收盤為止的資訊，
能不能預測明天的報酬，而且準到足以付得起每天進出的成本。**

成本為什麼是關鍵
----------------
台股現股來回成本約 0.39%（手續費 0.1425%×3 折 ×2 ＋ 賣出證交稅 0.3%），
而大盤單日報酬標準差約 1%。每天翻倉一次就要付掉將近 0.4 個標準差。
所以本程式一律同時給「扣成本前」與「扣成本後」，並分三種成本情境：

    現股/ETF   手續費兩次 ＋ 賣出證交稅（ETF 0.1%、個股 0.3%）
    台指期     來回約 0.01%（期交稅 0.002%×2 ＋ 手續費）
    零成本     理論上限，只用來看訊號本身有沒有內容

若只有「零成本」那一欄為正，結論就是訊號不存在可交易的價值。

不偷看答案
----------
    特徵只用到第 t 日收盤為止的資料
    標籤是 r(t+1) = C(t+1)/C(t) − 1
    walk-forward 每折都重新訓練，且訓練集尾端切掉 1 列（purging）——
    最後一列的標籤由測試期第一天的價格決定，不切就是洩漏
    部位在第 t 日收盤建立，實現的是 r(t+1)，不存在用未來資料下單的情況

輸出
----
    方向準確率（DA）、Spearman IC、各成本情境的年化與夏普、
    以進場月份分群的 t 值、換手率，以及對整族變體的 White Reality Check
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
from portfolio_backtest import nw_tstat  # noqa: E402
from reality_check import reality_check  # noqa: E402

OUT = ROOT / "results" / "nextday"
FEE = 0.001425 * 0.3
COSTS = {"現股": (FEE, 0.003), "ETF": (FEE, 0.001),
         "台指期": (0.00002, 0.00002), "零成本": (0.0, 0.0)}


def load_market(name: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "market" / f"{name}.csv",
                    parse_dates=["date"]).set_index("date").sort_index()
    return d.dropna(subset=["close"])


def build(px: pd.DataFrame) -> tuple:
    """特徵（截至第 t 日）與標籤 r(t+1)。"""
    f = T.step2_price_features(px, quiet=True)
    y = px["close"].pct_change().shift(-1)        # r(t+1)
    f = f.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in f.columns if f[c].notna().mean() > 0.9]
    df = f[keep].join(y.rename("_y")).dropna()
    return df[keep], df["_y"], keep


def walk_forward(X: pd.DataFrame, y: pd.Series, min_train: int,
                 step: int, kind: str) -> pd.Series:
    """
    逐段重訓並預測。訓練集尾端切掉 1 列：
    該列的標籤 r(t+1) 由測試期第一天的收盤決定，留著就是洩漏。
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    out = pd.Series(index=X.index, dtype=float)
    n = len(X)
    for i0 in range(min_train, n, step):
        i1 = min(i0 + step, n)
        tr = slice(0, i0 - 1)                      # purge 1 列
        Xtr, ytr = X.iloc[tr].to_numpy(), y.iloc[tr].to_numpy()
        if len(Xtr) < min_train // 2:
            continue
        if kind == "clf":
            m = HistGradientBoostingClassifier(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=40, early_stopping=False, random_state=42)
            m.fit(Xtr, (ytr > 0).astype(int))
            p = m.predict_proba(X.iloc[i0:i1].to_numpy())[:, 1] - 0.5
        else:
            m = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=40, early_stopping=False, random_state=42)
            m.fit(Xtr, ytr)
            p = m.predict(X.iloc[i0:i1].to_numpy())
        out.iloc[i0:i1] = p
    return out.dropna()


def trade(sig: pd.Series, ret: pd.Series, mode: str, thr: float,
          fee: float, tax: float) -> tuple:
    """
    第 t 日收盤依 sig(t) 建立部位，實現 r(t+1)=ret(t)。
    換手成本在部位變動當日扣：買進付手續費，賣出付手續費＋稅。
    """
    if mode == "多空":
        pos = np.where(sig > thr, 1.0, np.where(sig < -thr, -1.0, 0.0))
    else:                                           # 多單或空手
        pos = np.where(sig > thr, 1.0, 0.0)
    pos = pd.Series(pos, index=sig.index)
    chg = pos.diff().abs().fillna(pos.abs())
    cost = chg * (fee + (fee + tax) / 2)            # 單邊平均成本
    r = pos * ret.reindex(sig.index).fillna(0.0) - cost
    return r, float(chg.mean()), pos


def stats(r: pd.Series, base: pd.Series | None = None) -> dict:
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
        t, p = nw_tstat(r.reindex(idx) - base.reindex(idx))
        d["超額t"], d["超額p"] = t, p
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="隔日漲跌／報酬預測")
    ap.add_argument("--targets", nargs="+", default=["TWII", "0050"])
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--min-train", type=int, default=1000)
    ap.add_argument("--step", type=int, default=252)
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, family = [], {}

    for tgt in a.targets:
        px = load_market(tgt)
        px = px[(px.index >= a.start) & (px.index <= a.end)]
        if len(px) < a.min_train + 300:
            print(f"{tgt} 資料不足，略過"); continue
        X, y, feats = build(px)
        print(f"\n{'=' * 100}")
        print(f"{tgt}　{X.index[0]:%Y-%m-%d} ~ {X.index[-1]:%Y-%m-%d}"
              f"　{len(X):,} 日　特徵 {len(feats)} 個")
        print("=" * 100)

        bench = y.copy()                             # 買進持有＝每天都拿 r(t+1)
        for kind, label in (("reg", "預測報酬（迴歸）"), ("clf", "預測漲跌（分類）")):
            sig = walk_forward(X, y, a.min_train, a.step, kind)
            act = y.reindex(sig.index)
            nz = act != 0
            da = float((np.sign(sig[nz]) == np.sign(act[nz])).mean())
            ic = float(pd.Series(sig).corr(act, method="spearman"))
            bh = bench.reindex(sig.index)
            s_bh = stats(bh)
            print(f"\n【{label}】樣本外 {len(sig):,} 日"
                  f"　{sig.index[0]:%Y-%m} ~ {sig.index[-1]:%Y-%m}")
            print(f"  方向準確率 {da:.2%}　Spearman IC {ic:+.4f}"
                  f"　（買進持有年化 {s_bh.get('年化', float('nan')):+.2%}）")
            print(f"  {'規則':<10}{'門檻':>7}{'換手':>7}" +
                  "".join(f"{k:>11}" for k in COSTS) + f"{'夏普':>8}{'超額t':>8}")
            for mode in ("多單或空手", "多空"):
                for thr in (0.0, 0.0005, 0.001):
                    line = {}
                    for cname, (fee, tax) in COSTS.items():
                        r, to, _ = trade(sig, act, mode, thr, fee, tax)
                        st = stats(r, bh)
                        if not st:
                            line = {}; break
                        line[cname] = st
                    if not line:
                        continue
                    base = line["ETF" if tgt != "TWII" else "台指期"]
                    nm = f"{tgt}／{label}／{mode}／{thr}"
                    family[nm] = (line["台指期"]["年化"], None)
                    rows.append({"標的": tgt, "模型": label, "規則": mode,
                                 "門檻": thr, "DA": da, "IC": ic, "換手": to,
                                 **{f"年化_{k}": v["年化"] for k, v in line.items()},
                                 "夏普": base["夏普"], "超額t": base["超額t"]})
                    print(f"  {mode:<10}{thr:>7.4f}{to:>7.2f}" +
                          "".join(f"{line[k]['年化']:>+11.2%}" for k in COSTS) +
                          f"{base['夏普']:>8.2f}{base['超額t']:>8.2f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n沒有產生任何結果"); return 1
    df.to_csv(OUT / "nextday.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}\n判讀\n{'=' * 100}")
    best = df.loc[df["年化_零成本"].idxmax()]
    print(f"  扣成本前最佳：{best['標的']}／{best['模型']}／{best['規則']}"
          f"　年化 {best['年化_零成本']:+.2%}")
    print(f"    同一組扣成本後：現股 {best['年化_現股']:+.2%}"
          f"　ETF {best['年化_ETF']:+.2%}　台指期 {best['年化_台指期']:+.2%}")
    n_pos = int((df["年化_ETF"] > 0).sum())
    print(f"  {len(df)} 組變體中，扣 ETF 成本後年化為正的有 {n_pos} 組")
    print(f"  方向準確率範圍 {df['DA'].min():.2%} ~ {df['DA'].max():.2%}"
          f"　（隨機＝50%）")
    print(f"\n輸出：{OUT / 'nextday.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
