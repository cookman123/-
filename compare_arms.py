#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兩組訓練結果的配對比較（Diebold–Mariano）。

用途：判斷某一個資料軸（例如 PTT）到底有沒有增益。

為什麼要配對而不是比報酬率
--------------------------
組合報酬的檢定力取決於「有幾次獨立的換倉」，PTT 只有 2019–2024，
測試期約 3 年、h=60 只換倉十幾次，比報酬率根本分辨不出差異。
配對比較則是同標的、同折、同一天、同一個真實答案，只差有沒有那組特徵，
逐日比較預測誤差，樣本數是「交易日數 × 標的數」，檢定力高出兩個數量級。

H0：兩組預測的期望損失相同。stat < 0 表示 A 組較佳、> 0 表示 B 組較佳。
含 Harvey–Leybourne–Newbold 小樣本修正；h > 1 時目標重疊，用 Newey–West 修自相關。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from davis_tw_train import diebold_mariano  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent


def load(run: Path, code: str, h: int):
    f = run / f"predictions_{code}_h{h}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    return d[["ret_next", "pred"]].dropna()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="兩組訓練結果的配對 DM 比較")
    ap.add_argument("--a", required=True, help="對照組輸出目錄")
    ap.add_argument("--b", required=True, help="實驗組輸出目錄")
    ap.add_argument("--horizon", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)

    A, B = ROOT / args.a, ROOT / args.b
    la, lb = args.label_a, args.label_b

    for h in args.horizon:
        codes = sorted({f.name[len("predictions_"):-len(f"_h{h}.csv")]
                        for f in A.glob(f"predictions_*_h{h}.csv")}
                       & {f.name[len("predictions_"):-len(f"_h{h}.csv")]
                          for f in B.glob(f"predictions_*_h{h}.csv")})
        rows = []
        for code in codes:
            da, db = load(A, code, h), load(B, code, h)
            if da is None or db is None:
                continue
            idx = da.index.intersection(db.index)
            if len(idx) < 60:
                continue
            y = da.loc[idx, "ret_next"].to_numpy()
            p1 = da.loc[idx, "pred"].to_numpy()
            p2 = db.loc[idx, "pred"].to_numpy()
            if not np.allclose(y, db.loc[idx, "ret_next"].to_numpy(), equal_nan=True):
                continue                      # 兩組的真實答案必須一致才算配對
            out = {"code": code, "n": len(idx)}
            for loss in ("mse", "sign"):
                st, pv = diebold_mariano(y, p1, p2, loss=loss, h=h)
                out[f"stat_{loss}"], out[f"p_{loss}"] = st, pv
            # 方向準確率直接對照
            out["DA_a"] = float(((p1 > 0) == (y > 0)).mean())
            out["DA_b"] = float(((p2 > 0) == (y > 0)).mean())
            out["mse_a"] = float(np.mean((y - p1) ** 2))
            out["mse_b"] = float(np.mean((y - p2) ** 2))
            rows.append(out)

        if not rows:
            print(f"h={h}：沒有可配對的標的")
            continue
        d = pd.DataFrame(rows)
        print(f"\n{'=' * 74}\nh={h}　可配對標的 {len(d)} 個"
              f"（平均 {d['n'].mean():.0f} 個測試日）\n{'=' * 74}")
        print(f"  {la}：平均 DA {d.DA_a.mean():.4f}　平均 MSE {d.mse_a.mean():.6f}")
        print(f"  {lb}：平均 DA {d.DA_b.mean():.4f}　平均 MSE {d.mse_b.mean():.6f}")
        print(f"  DA 差（{lb} − {la}）平均 {100 * (d.DA_b - d.DA_a).mean():+.3f} 個百分點"
              f"，{lb} 較高的標的 {int((d.DA_b > d.DA_a).sum())}/{len(d)}")

        for loss, name in (("mse", "點預測 MSE"), ("sign", "方向")):
            sig = d[d[f"p_{loss}"] < 0.05]
            a_better = int((sig[f"stat_{loss}"] < 0).sum())
            b_better = int((sig[f"stat_{loss}"] > 0).sum())
            exp = round(len(d) * 0.025)
            verdict = ("持平" if abs(a_better - b_better) <= max(3, 0.3 * (a_better + b_better))
                       else (f"{la} 較佳" if a_better > b_better else f"{lb} 較佳"))
            print(f"  DM（{name}）p<0.05：{la} 顯著較佳 {a_better}　"
                  f"{lb} 顯著較佳 {b_better}　（純機率下各約 {exp}）→ {verdict}")

        d.to_csv(ROOT / f"results/compare_{la}_vs_{lb}_h{h}.csv",
                 index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
