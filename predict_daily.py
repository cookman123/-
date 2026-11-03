#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日預測：把 1／5／20 日模型的預測接到最新一天，結果和 nextday_horizon.py 的 walk-forward 完全相同。

nextday_horizon.py 的做法是：交易日由舊到新排好，從第 1000 天起每 252 天一折，每一折用「這一折開始前
（再往前切掉 h+1 天）的全部資料」訓練一次，預測這一折的 252 天。所以新的一天只要找出它屬於哪一折、
用那一折的模型預測即可；每一折的模型訓練一次就存到 data/models/，大約一年才需要訓練新的一折。
模型參數、特徵篩選、隨機種子都和 nextday_horizon.py 相同，預測值應該一模一樣（--check 會驗證）。

用法（要在含下市股的母體下跑，和研究的 _surv 版本一致）：
    python tool/run_surv.py predict_daily --tag _surv
    python tool/run_surv.py predict_daily --tag _surv --check 20     # 重算最近 20 個已有預測的日子並比對

每天預測最近 RECENT 個交易日（含新的日子）；當天爬蟲只跑一部分時那天的預測不完整，隔天會再重算補齊。
輸出直接更新 results/nextday/pred_h{h}{tag}.csv，接著跑 build_local_db.py --tables predictions 寫進 DB。
"""
from __future__ import annotations

import sys
import time
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
from nextday_horizon import HORIZONS  # noqa: E402
from nextday_v2 import OUT, STORE  # noqa: E402

MIN_DAYS, STEP = 1000, 252                 # 和 nextday_horizon.walk_forward 的預設值相同
RECENT = 10
MODELS = ROOT / "data" / "models"


def fold_start(i: int) -> int:
    return MIN_DAYS + (i - MIN_DAYS) // STEP * STEP


def fold_model(sub: pd.DataFrame, feats: list, label: str, purge: int, days: pd.DatetimeIndex, k: int, tag: str):
    """第 k 折的模型（訓練資料＝days[:k − purge]）；存過就直接讀。"""
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    MODELS.mkdir(parents=True, exist_ok=True)
    f = MODELS / f"{label}{tag}_fold{days[k]:%Y%m%d}.joblib"
    if f.exists():
        return joblib.load(f)
    t0 = time.time()
    tr = sub[sub.index.get_level_values("date").isin(days[:max(0, k - purge)])].dropna(subset=[label])
    use = [c for c in feats if tr[c].nunique(dropna=True) >= 2]
    m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                      early_stopping=False, random_state=42)
    m.fit(tr[use].to_numpy("float32"), tr[label].to_numpy("float32"))
    joblib.dump((m, use), f)
    print(f"    訓練 {label} 第 {days[k]:%Y-%m-%d} 起的一折：{len(tr):,} 列，{time.time() - t0:.0f} 秒", flush=True)
    return m, use


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="每日預測（接續 nextday_horizon.py 的 walk-forward）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", type=int, default=0, help="重算最近 N 個已有預測的日子，和既有預測比對")
    a = ap.parse_args(argv)

    t0 = time.time()
    df = pd.read_pickle(STORE.with_name(f"panel{a.tag}.pkl"))
    feats = [c for c in df.columns if not c.startswith("y_")]
    O = FE.load_panels("2010-01-01", "2026-12-31", quiet=True)["open"]
    sub = df.dropna(subset=feats, how="all")
    days = sub.index.get_level_values("date").unique().sort_values()
    dl = sub.index.get_level_values("date")
    print(f"特徵庫 {len(sub):,} 列、{len(days):,} 個交易日（到 {days[-1]:%Y-%m-%d}），讀取 {time.time() - t0:.0f} 秒")

    bad = 0
    for h in HORIZONS:
        label = f"y_h{h}"
        lab = (O.shift(-(1 + h)) / O.shift(-1) - 1).stack(future_stack=True)
        lab.index.names = ["date", "code"]
        sub[label] = lab.reindex(sub.index).astype("float32")
        f = OUT / f"pred_h{h}{a.tag}.csv"
        old = pd.read_csv(f, dtype={"code": str}, parse_dates=["date"]) if f.exists() else \
            pd.DataFrame(columns=["date", "code", "pred"])
        have = pd.DatetimeIndex(old["date"].unique()).sort_values()
        if a.check:
            target = have[-a.check:]
        else:
            # 最近幾天重算（補齊當天只爬到一部分的日子）＋還沒有預測的新日子
            target = days[(days > have[-1 - RECENT]) if len(have) > RECENT else (days >= days[MIN_DAYS])]
        idx = days.get_indexer(target)
        target, idx = target[idx >= MIN_DAYS], idx[idx >= MIN_DAYS]
        parts = []
        for k in sorted({fold_start(i) for i in idx}):
            m, use = fold_model(sub, feats, label, h + 1, days, k, a.tag)
            ds = target[[fold_start(i) == k for i in idx]]
            te = sub[dl.isin(ds)]
            parts.append(pd.Series(m.predict(te[use].to_numpy("float32")), index=te.index, name="pred"))
        new = pd.concat(parts).reset_index() if parts else pd.DataFrame(columns=["date", "code", "pred"])
        if a.check:
            mg = old.merge(new, on=["date", "code"], suffixes=("_old", "_new"))
            diff = (mg["pred_old"] - mg["pred_new"]).abs().max()
            ok = len(mg) == len(old[old["date"].isin(target)]) and diff < 1e-6
            bad += not ok
            print(f"  h={h}：比對 {len(mg):,} 列（{len(target)} 天），最大差 {diff:.2e}　{'一致' if ok else '不一致'}")
            continue
        out = pd.concat([old[~old["date"].isin(target)], new]).sort_values(["date", "code"])
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(f, index=False, encoding="utf-8-sig")
        n_new = len(set(target) - set(have))
        print(f"  h={h}：重算 {len(target)} 天、其中新的 {n_new} 天，最新 {target[-1]:%Y-%m-%d}"
              f"（{(new['date'] == target[-1]).sum()} 檔）→ {f.name}")
    print(f"完成，{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
