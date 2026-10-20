#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預測期間對齊持有期間：用 1／5／20 日報酬當標籤訓練，緩衝帶持有，台指期對沖 beta。

為什麼做
--------
`nextday_buffer.py` 的結果：隔日模型的選股能力是真的（IC +0.069，多空零成本
年化 +70%），但只撐一兩天。加緩衝帶把換手壓下來後，持有期拉長到數天，預測力
跟著衰退，最後剩下的只有 beta（多單 30 檔、m=5：beta 1.05，beta 中性 t 0.54）。
模型學的是「明天」，卻被拿去抱好幾天，兩者對不上。
這裡直接用持有期間的報酬當標籤，讓模型學「接下來一週／一個月誰會贏」。

事前固定的規格（跑之前寫死，跑完不改、不加）
------------------------------------------
    標籤期間 h ∈ {1, 5, 20} 交易日，標籤 = 開盤(t+1+h) ÷ 開盤(t+1) − 1
    組合：純做多，檔數 n ∈ {10, 30}，緩衝 m ∈ {1, 5}
    共 3 × 2 × 2 = 12 組，全部送 Reality Check／SPA
    **評估指標是對沖後報酬**：
        日報酬 − 事前 beta × 大盤開盤→開盤報酬
        beta 用滾動 250 日、shift 1 日估（不含當日）
        對沖用台指期，成本約 0.01%／來回，相對個股 0.39% 可忽略，不計
    檢定對象是「對沖後報酬是否 > 0」，不是對母體等權的超額。
    前兩次（季頻、日頻緩衝帶）都是在這一關被拆穿的，所以直接以它為主指標。

其他設定沿用 nextday_v2／nextday_buffer
--------------------------------------
    特徵：panel.pkl 的 105 個特徵；模型 HistGradientBoosting，參數不變
    walk-forward：最少 1000 個交易日起訓、每 252 日重訓；
        訓練集尾端切掉 h + 1 日（標籤跨到 t+1+h 開盤，切少了會讓訓練集的答案
        由測試期的價格決定）
    實際損益一律用逐日開盤→開盤報酬（y_oo），與標籤期間無關
    漲跌停鎖死的不能新買、已持有的不能賣；成本 0.39%／來回
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
from nextday_buffer import run  # noqa: E402
from nextday_v2 import COST, OUT, STORE  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from reality_check import reality_check  # noqa: E402

HORIZONS = (1, 5, 20)
SIZES = (10, 30)
MULTS = (1, 5)


def walk_forward(df: pd.DataFrame, feats: list, label: str, purge: int,
                 min_days: int = 1000, step_days: int = 252) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingRegressor

    days = df.index.get_level_values("date").unique().sort_values()
    dl = df.index.get_level_values("date")
    out = []
    for k in range(min_days, len(days), step_days):
        tr_days = days[:max(0, k - purge)]
        te_days = days[k:min(k + step_days, len(days))]
        if len(te_days) == 0:
            break
        tr = df[dl.isin(tr_days)].dropna(subset=[label])
        te = df[dl.isin(te_days)]
        if len(tr) < 20000 or te.empty:
            continue
        use = [c for c in feats if tr[c].nunique(dropna=True) >= 2]
        m = HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=200, early_stopping=False, random_state=42)
        m.fit(tr[use].to_numpy("float32"), tr[label].to_numpy("float32"))
        out.append(pd.Series(m.predict(te[use].to_numpy("float32")), index=te.index))
        print(f"    {label}　{te_days[0]:%Y-%m} ~ {te_days[-1]:%Y-%m}　訓練 {len(tr):,}", flush=True)
    return pd.concat(out) if out else pd.Series(dtype=float)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="預測期間對齊持有期間")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    df = pd.read_pickle(STORE.with_name(f"panel{a.tag}.pkl"))
    feats = [c for c in df.columns if not c.startswith("y_")]
    O = FE.load_panels("2010-01-01", "2026-12-31", quiet=True)["open"]
    for h in HORIZONS:
        lab = (O.shift(-(1 + h)) / O.shift(-1) - 1).stack(future_stack=True)
        lab.index.names = ["date", "code"]
        df[f"y_h{h}"] = lab.reindex(df.index).astype("float32")
    # 驗證：h=1 必須與特徵庫原本的 y_oo 一致
    chk = (df["y_h1"] - df["y_oo"]).abs().max()
    print(f"標籤 h=1 與 y_oo 最大差 {chk:.2e}（應為 0）")

    y = df["y_oo"]
    locked = df["trd_locked"] >= 0.5
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]) \
        .set_index("date")["open"].sort_index()
    tw = tw[~tw.index.duplicated(keep="last")]
    mk_all = tw.shift(-2) / tw.shift(-1) - 1

    sub = df.dropna(subset=feats, how="all")
    rows, F = [], {}
    for h in HORIZONS:
        f = OUT / f"pred_h{h}{a.tag}.csv"
        if f.exists():
            p = pd.read_csv(f, dtype={"code": str}, parse_dates=["date"])
            pred = p.set_index(["date", "code"])["pred"]
            print(f"\n讀取既有預測 {f.name}")
        else:
            print(f"\n【h = {h}】walk-forward（purge {h + 1} 日）", flush=True)
            pred = walk_forward(sub, feats, f"y_h{h}", purge=h + 1)
            pred.rename("pred").reset_index().to_csv(f, index=False, encoding="utf-8-sig")
        ic_own = pd.concat([pred.rename("p"), df[f"y_h{h}"].reindex(pred.index).rename("y")], axis=1) \
            .groupby(level="date").apply(lambda g: g["p"].corr(g["y"], method="spearman"),
                                         include_groups=False).dropna()
        ic_1 = pd.concat([pred.rename("p"), y.reindex(pred.index).rename("y")], axis=1) \
            .groupby(level="date").apply(lambda g: g["p"].corr(g["y"], method="spearman"),
                                         include_groups=False).dropna()
        print(f"  IC（自己的 {h} 日標籤）{ic_own.mean():+.4f}　IC（隔日報酬）{ic_1.mean():+.4f}")
        lk = locked.reindex(pred.index).fillna(False)
        for n in SIZES:
            for m in MULTS:
                r, g, turn = run(pred, y, lk, n, m, "多單")
                mk = mk_all.reindex(r.index)
                beta = (r.rolling(250).cov(mk) / mk.rolling(250).var()).shift(1)
                hed = (r - beta * mk).dropna()
                t_h, _ = nw_tstat(hed)
                yrs = len(hed) / 252
                b_full = float(np.cov(r.loc[hed.index], mk.loc[hed.index])[0, 1] / mk.loc[hed.index].var())
                half = len(hed) // 2
                t1, _ = nw_tstat(hed.iloc[:half])
                t2, _ = nw_tstat(hed.iloc[half:])
                rows.append({"h": h, "檔數": n, "m": m, "換手": turn,
                             "毛報酬_日": g.mean(), "扣成本年化": (1 + r).prod() ** (252 / len(r)) - 1,
                             "beta": b_full, "對沖後年化": (1 + hed).prod() ** (1 / yrs) - 1,
                             "對沖後t": t_h, "前半t": t1, "後半t": t2,
                             "IC_自身": ic_own.mean(), "IC_隔日": ic_1.mean()})
                F[f"h{h}_n{n}_m{m}"] = hed
                o = rows[-1]
                print(f"  n={n:<3}m={m}　換手 {turn:.2f}　扣成本 {o['扣成本年化']:+.2%}"
                      f"　beta {b_full:.2f}　對沖後 {o['對沖後年化']:+.2%}（t {t_h:.2f}；"
                      f"前半 {t1:.2f}、後半 {t2:.2f}）", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"horizon{a.tag}.csv", index=False, encoding="utf-8-sig")
    X = pd.DataFrame(F).dropna()
    rc = reality_check(X.to_numpy(), mean_block=10, n_boot=2000, seed=42)
    best = X.columns[rc["最佳策略序號"]]
    print(f"\n{'=' * 90}\n多重比較：12 組對沖後報酬，{X.shape[0]} 日\n{'=' * 90}")
    print(f"  最佳 {best}　RC p = {rc['RC_p']:.4f}　SPA p = {rc['SPA_p']:.4f}")
    print(f"\n判讀\n  對沖後 t > 1.96 的有 {int((res['對沖後t'] > 1.96).sum())}/12 組；"
          f"RC 與 SPA 都要 < 0.05 才算有可交易的 alpha。")
    print(f"\n輸出：{OUT / f'horizon{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
