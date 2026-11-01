#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似事件檢索（RAG 的行為模式）在跌深事件上有沒有用？

問題
----
流程圖裡「投資操作模型根據 RAG 的行為模式」的具體做法是：
    今天出現一個跌深事件 → 去歷史裡找「最像它」的 k 個過去事件
    → 看那些事件最後的淨報酬 → 當作今天這個事件的分數
以前的 DAViS 堆疊做過同樣的事（KNN），但樣本是每檔每天，87% 的日子沒事發生，
結果和其他模型無顯著差異、堆疊權重最小。這裡改成只在跌深事件之間檢索，
看訊號集中之後是否就有用。

與 event_model.py 完全相同的設定，才能公平比較
------------------------------------------------
  事件、時點正確選股範圍、特徵、測試窗、purging（訓練只收出場日早於測試窗 − 30 天）
  都直接沿用。檢索庫就是訓練集：只能撈到「答案已經揭曉」的過去事件。
  特徵標準化只用檢索庫的平均與標準差。

事先固定的比較組（不看結果再增減）
----------------------------------
  檢索 k=20、檢索 k=50   距離加權平均鄰居的淨報酬
  樹模型                 event_model.py 的梯度提升樹
  檢索＋樹               兩者分數各自轉百分位後平均（流程圖的「加權」）
四組一起看，任何一組的差異 t 過 2 都還要考慮多重比較。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_strategy as A  # noqa: E402
import event_model as E  # noqa: E402
import universe_pit as U  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "event_model" / "events.csv"
KS = (20, 50)
# 事後補做的敏感度檢查：主結果出來之後才加的，報告裡要和主測分開交代
KS_SENS = (10, 15, 20, 30, 40, 50, 100)
SPLIT = pd.Timestamp("2022-11-01")      # 測試期前後各半


def load_events(start: str, end: str, topn: int) -> tuple:
    """讀事件快取、套時點正確選股範圍；特徵挑選與 event_model.main 相同（不含基本面）。"""
    ev = pd.read_csv(CACHE, parse_dates=["entry", "exit", "signal"])
    ev = U.filter_trades(ev, U.get_mask(start, end, topn), "signal")
    not_feat = {"news_day", "drop_day", "entry", "exit", "signal", "campaign"}
    feats = [c for c in ev.columns
             if (c.startswith(("px_", "chip_", "news_"))
                 or c in ("drop", "drop_speed", "dist_hi60", "turnover_z"))
             and c not in not_feat]
    feats = [c for c in feats if ev[c].notna().mean() > 0.5]
    return ev.dropna(subset=feats + ["net"]), feats


def retrieve(tr: pd.DataFrame, te: pd.DataFrame, feats: list, k: int) -> np.ndarray:
    """在檢索庫（訓練集）裡找 k 個最近的事件，距離加權平均它們的淨報酬。"""
    from sklearn.neighbors import NearestNeighbors

    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    Xtr = ((tr[feats] - mu) / sd).to_numpy()
    Xte = ((te[feats] - mu) / sd).to_numpy()
    dist, idx = NearestNeighbors(n_neighbors=k).fit(Xtr).kneighbors(Xte)
    w = 1 / (dist + 1e-6)
    return (tr["net"].to_numpy()[idx] * w).sum(axis=1) / w.sum(axis=1)


def walk_forward(ev: pd.DataFrame, feats: list, ks: tuple = KS, test_days: int = 252,
                 min_train: int = 1500, embargo: int = 30, quiet: bool = False) -> pd.DataFrame:
    """切分與 event_model.walk_forward 相同；同一個 fold 同時算檢索與樹模型。"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    ev = ev.sort_values("entry").reset_index(drop=True)
    entry, exit_ = pd.to_datetime(ev["entry"]), pd.to_datetime(ev["exit"])
    last, t0 = entry.max(), entry.iloc[min_train]
    out = []
    while t0 <= last:
        t1 = t0 + pd.Timedelta(days=test_days)
        te = ev[(entry >= t0) & (entry < t1)]
        tr = ev[exit_ < (t0 - pd.Timedelta(days=embargo))]
        if len(tr) >= min_train and len(te) >= 30:
            o = te[["code", "entry", "exit", "net"]].copy()
            for k in ks:
                o[f"檢索 k={k}"] = retrieve(tr, te, feats, k)
            if not quiet:
                m = HistGradientBoostingRegressor(
                    max_depth=4, max_iter=300, learning_rate=0.05,
                    min_samples_leaf=40, early_stopping=False, random_state=42)
                m.fit(tr[feats].to_numpy(), tr["net"].to_numpy())
                o["樹模型"] = m.predict(te[feats].to_numpy())
                # 百分位只在同一個測試窗內排，不用到其他窗的分數
                o["檢索＋樹"] = (o[f"檢索 k={ks[0]}"].rank(pct=True)
                               + o["樹模型"].rank(pct=True)) / 2
                print(f"    fold{len(out) + 1}: 檢索庫 {len(tr):,}　測試 {len(te):,}"
                      f"（{t0:%Y-%m} ~ {t1:%Y-%m}）", flush=True)
            out.append(o)
        t0 = t1
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="相似事件檢索 vs 樹模型 vs 照單全收")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--topn", type=int, default=120)
    ap.add_argument("--out", default="results/rag_event")
    a = ap.parse_args(argv)

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        print("找不到事件快取，請先執行 event_model.py")
        return 1
    ev, feats = load_events(a.start, a.end, a.topn)
    print(f"事件 {len(ev):,} 筆　特徵 {len(feats)} 個\n")

    P = walk_forward(ev, feats)
    if P.empty:
        print("樣本不足")
        return 1
    P.to_csv(out_dir / "scored.csv", index=False, encoding="utf-8-sig")

    base_m, base_t, base_n = A.cluster_t(P)
    print(f"\n{'=' * 92}")
    print(f"照單全收：{len(P):,} 筆　{pd.to_datetime(P.entry).min():%Y-%m} ~ "
          f"{pd.to_datetime(P.entry).max():%Y-%m}　平均 {base_m:+.2%}"
          f"　群集 t {base_t:.2f}（{base_n} 個月）")
    print("=" * 92)
    print(f"  {'評分方式':<12}{'選取':<10}{'筆數':>7}{'勝率':>8}{'平均':>9}"
          f"{'群集t':>8}{'vs全收':>9}{'差異t':>8}")
    rows = []
    for col in [f"檢索 k={k}" for k in KS] + ["樹模型", "檢索＋樹"]:
        rho = P[col].rank().corr(P["net"].rank())
        for q, label in ((0.5, "前 50%"), (0.2, "前 20%"), (-0.2, "後 20%")):
            msk = (P[col] >= P[col].quantile(1 - q)) if q > 0 \
                else (P[col] <= P[col].quantile(-q))
            sel = P[msk]
            m, t, _ = A.cluster_t(sel)
            d, dt, _ = E._diff(P, msk)
            rows.append({"評分": col, "選取": label, "筆數": len(sel),
                         "勝率": (sel.net > 0).mean(), "平均": m, "群集t": t,
                         "vs全收": d, "差異t": dt, "Spearman": rho})
            print(f"  {col:<12}{label:<10}{len(sel):>7,}{(sel.net > 0).mean():>8.1%}"
                  f"{m:>+9.2%}{t:>8.2f}{d:>+9.2%}{dt:>8.2f}")
        print(f"  {'':<12}Spearman(分數, 淨報酬) = {rho:+.4f}\n")
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print("  判讀：前 20% 的差異 t > 2、且後 20% 明顯較差，檢索才算有挑選能力。")

    print(f"\n{'=' * 92}\n敏感度（事後補做）：換 k、切前後兩段\n{'=' * 92}")
    print(f"  {'k':<6}{'前50%差異t':>11}{'後20%差異':>11}{'後20%差異t':>11}"
          f"{'前段後20%t':>12}{'後段後20%t':>12}")
    S = walk_forward(ev, feats, ks=KS_SENS, quiet=True)
    early = pd.to_datetime(S["entry"]) < SPLIT
    sens = []
    for k in KS_SENS:
        c = f"檢索 k={k}"
        _, t50, _ = E._diff(S, S[c] >= S[c].quantile(0.5))
        d, tb, _ = E._diff(S, S[c] <= S[c].quantile(0.2))
        part = [E._diff(g, g[c] <= g[c].quantile(0.2))[1] for g in (S[early], S[~early])]
        sens.append({"k": k, "前50%差異t": t50, "後20%差異": d, "後20%差異t": tb,
                     "前段後20%差異t": part[0], "後段後20%差異t": part[1]})
        print(f"  {k:<6}{t50:>11.2f}{d:>+11.2%}{tb:>11.2f}{part[0]:>12.2f}{part[1]:>12.2f}")
    pd.DataFrame(sens).to_csv(out_dir / "sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"\n  輸出：{out_dir / 'summary.csv'}、{out_dir / 'sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
