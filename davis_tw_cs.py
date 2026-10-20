#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
橫斷面排序模型：把訓練目標對準實際使用方式。

為什麼要重寫一支
----------------
davis_tw_train.py 是逐檔回歸：對 383 檔各自學「這檔未來 h 天漲幅是多少」，
等於 383 個獨立問題、每個只有 600 列訓練樣本。
但策略實際用的是「今天這 383 檔裡哪 20 檔最強」——一個橫斷面排序問題。

兩者不一致造成三個具體損失：

1. 目標錯置：逐檔回歸把大量參數用在預測「這檔的波動水準」，那對排序毫無幫助。
2. 波動偏誤：高波動股的預測值天生較大，排序時系統性佔據前段——等於在選波動度。
   （先前的展開式 z 分數是事後補救，治標；在標籤端解決才是治本。）
3. 樣本浪費：逐檔 600 列 vs 橫斷面 600 天 × 383 檔 ≈ 23 萬列，差三個數量級。

而「橫斷面排序」正是目前唯一還沒被否證的能力：大盤擇時已證明有害
（超額 t = −2.73）、空方無訊號（t = −1.97）、三個資料軸皆無增益。

做法
----
- 標籤：當日未來 h 日報酬的**橫斷面百分位**，置中到 [-0.5, +0.5]。
  用排名而非報酬率本身，模型才不會把力氣花在擬合波動水準；
  置中後 pred > 0 直接代表「預期優於當日中位數」。
- 特徵：每個特徵每天在所有股票之間做百分位轉換。
  這一步把「台積電的 20 日報酬是 +8%」變成「台積電今天排在第 87 百分位」，
  跨股票才可比較，也自動移除了波動度偏誤。
- 切分：walk-forward，訓練／驗證窗尾端各切掉 h 列（purging）。
  標籤是未來 h 天的報酬，不切的話驗證窗的答案會由測試期價格決定，
  而元模型正是用驗證窗擬合的。（這個洩漏曾讓逐檔版的 t 值從 1.26 灌水到 3.94。）

與原論文的差異（必須在論文中說明）
----------------------------------
基礎模型由 8 個改為 5 個：拿掉 SVR 與 KNN。兩者的複雜度對樣本數是超線性的，
在 23 萬列的橫斷面面板上不可行；改用 HistGradientBoosting 取代原本的 GBRT。
堆疊結構（基礎模型 → 驗證窗 → Huber 元模型）維持不變。

輸出格式刻意與 davis_tw_train.py 一致（predictions_{代號}_h{期間}.csv），
因此 portfolio_backtest.py 與 reality_check.py 可以直接套用，不必改動。
"""
from __future__ import annotations

import logging
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import davis_tw_train as T  # noqa: E402

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("cs")

TRAIN_DAYS, VAL_DAYS, TEST_DAYS, STEP_DAYS = 600, 120, 120, 120
MIN_STOCKS_PER_DAY = 50


# ============================== 面板建構 ==============================


def build_panel(codes: list, start: str, end: str, horizon: int,
                use_chip: bool = False) -> pd.DataFrame:
    """
    組出長表面板：索引 (date, code)，欄位為特徵 + 未來報酬。

    特徵沿用 davis_tw_train 的價量技術面（與逐檔版完全相同，確保比較公平），
    差別只在後續的橫斷面轉換與標籤定義。
    """
    frames = []
    warm = pd.Timestamp(start) - pd.Timedelta(days=T.WARMUP_DAYS)
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= warm) & (px.index <= pd.Timestamp(end))]
        if len(px) < TRAIN_DAYS // 2:
            continue
        f = T.step2_price_features(px, quiet=True)
        if use_chip:
            ch = T.load_chips(code)
            f = pd.concat([f, T.step3_chip_features(ch, px, quiet=True)], axis=1)

        c = px["close"]
        f["_fwd"] = c.shift(-horizon) / c - 1
        f["_close"] = c
        f["_close_fwd"] = c.shift(-horizon)
        f["code"] = code
        f = f[f.index >= pd.Timestamp(start)]
        frames.append(f.reset_index().rename(columns={"index": "date"}))
        if i % 50 == 0:
            log.info(f"  已載入 {i}/{len(codes)} 檔")

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={panel.columns[0]: "date"})
    return panel


def cross_sectional_transform(panel: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    每個交易日，把每個特徵在當日所有股票之間轉成百分位（0~1），標籤轉成置中百分位。

    這是本模型的核心：轉換後「特徵值」的意思從絕對水準變成相對排名，
    跨股票可比，且高波動股不再天生佔據預測值的前段。
    """
    g = panel.groupby("date")
    out = panel[["date", "code", "_fwd", "_close", "_close_fwd"]].copy()
    ranked = g[feat_cols].rank(pct=True)
    out[feat_cols] = ranked.astype(np.float32)
    # 標籤：當日未來報酬的百分位，置中到 [-0.5, +0.5]
    out["y"] = (g["_fwd"].rank(pct=True) - 0.5).astype(np.float32)
    # 當日可用股票數太少的日子丟掉（排名沒有意義）
    n = g["code"].transform("size")
    out = out[n >= MIN_STOCKS_PER_DAY]
    return out.dropna(subset=["y"] + feat_cols)


# ============================== 模型 ==============================


class CSEnsemble:
    """
    五個基礎模型 → Huber 元模型，沿用原論文的三段式堆疊。

    基礎模型只用訓練窗擬合、對驗證窗預測；元模型只用驗證窗的預測擬合，
    再對測試窗預測。刻意不用交叉驗證的 out-of-fold，因為時間序列上
    那等於讓模型看到未來。
    """

    def __init__(self):
        from sklearn.ensemble import (ExtraTreesRegressor,
                                      HistGradientBoostingRegressor,
                                      RandomForestRegressor)
        from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        sc = lambda est: Pipeline([("s", StandardScaler()), ("m", est)])
        self.base = {
            "ridge": sc(Ridge(alpha=10.0)),
            "elasticnet": sc(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),
            "hgb": HistGradientBoostingRegressor(
                max_depth=4, max_iter=200, learning_rate=0.05,
                early_stopping=False, random_state=42),
            "rf": RandomForestRegressor(
                n_estimators=120, max_depth=8, min_samples_leaf=200,
                n_jobs=-1, random_state=42),
            "et": ExtraTreesRegressor(
                n_estimators=120, max_depth=10, min_samples_leaf=200,
                n_jobs=-1, random_state=42),
        }
        self.meta = Pipeline([("s", StandardScaler()),
                              ("m", HuberRegressor(alpha=1e-4, max_iter=1000))])
        self.ok = []

    def fit(self, Xtr, ytr, Xva, yva):
        P = []
        for name, m in self.base.items():
            try:
                t0 = time.time()
                m.fit(Xtr, ytr)
                P.append(m.predict(Xva))
                self.ok.append(name)
                log.info(f"    {name:<12} 擬合完成（{time.time() - t0:.0f} 秒）")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"    {name} 失敗：{type(exc).__name__}: {exc}")
        if not P:
            raise RuntimeError("所有基礎模型都失敗")
        self.meta.fit(np.column_stack(P), yva)
        return self

    def predict(self, X):
        P = [self.base[n].predict(X) for n in self.ok]
        return self.meta.predict(np.column_stack(P))

    def weights(self) -> dict:
        try:
            return dict(zip(self.ok, self.meta.named_steps["m"].coef_))
        except Exception:  # noqa: BLE001
            return {}


# ============================== 主流程 ==============================


def rank_ic(df: pd.DataFrame) -> tuple:
    """逐日的橫斷面 Spearman 相關（rank IC）與其 t 值——排序模型的標準指標。"""
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 20:
            continue
        a, b = g["pred"].rank(), g["_fwd"].rank()
        if a.std() == 0 or b.std() == 0:
            continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    if len(ics) < 10:
        return np.nan, np.nan, 0
    ics = np.array(ics)
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))), len(ics)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="橫斷面排序模型")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--horizon", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--chip", action="store_true", help="加入籌碼面特徵")
    ap.add_argument("--out", default="results/cs")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout, force=True)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = [c for c in T.load_industry_map()] if False else None
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    log.info(f"標的 {len(codes)} 檔　期間 {args.start} ~ {args.end}")

    summary = []
    for horizon in args.horizon:
        log.info(f"\n{'=' * 74}\n持有期 h={horizon}\n{'=' * 74}")
        t0 = time.time()
        panel = build_panel(codes, args.start, args.end, horizon, args.chip)
        if panel.empty:
            log.warning("  面板為空")
            continue
        feat_cols = [c for c in panel.columns
                     if c not in ("date", "code", "_fwd", "_close", "_close_fwd")]
        log.info(f"  原始面板 {len(panel):,} 列 × {len(feat_cols)} 特徵"
                 f"（{time.time() - t0:.0f} 秒）")

        data = cross_sectional_transform(panel, feat_cols)
        del panel
        log.info(f"  橫斷面轉換後 {len(data):,} 列，"
                 f"交易日 {data['date'].nunique():,} 天、標的 {data['code'].nunique()} 檔")

        dates = pd.DatetimeIndex(sorted(data["date"].unique()))
        folds = T.step5_split(dates, quiet=True, horizon=horizon)
        log.info(f"  walk-forward {len(folds)} 折（訓練／驗證窗已各切掉尾端 {horizon} 列）")
        if not folds:
            continue

        preds = []
        for fold in folds:
            tr = data[data["date"].isin(fold.train)]
            va = data[data["date"].isin(fold.val)]
            te = data[data["date"].isin(fold.test)]
            if len(tr) < 5000 or len(va) < 500 or len(te) < 500:
                log.warning(f"  fold{fold.idx} 樣本不足，跳過")
                continue
            log.info(f"  fold{fold.idx}｜訓練 {len(tr):,} 列（{fold.train[0]:%Y-%m-%d}"
                     f"~{fold.train[-1]:%Y-%m-%d}）｜驗證 {len(va):,}｜測試 {len(te):,}"
                     f"（{fold.test[0]:%Y-%m-%d}~{fold.test[-1]:%Y-%m-%d}）")
            model = CSEnsemble().fit(tr[feat_cols].to_numpy(), tr["y"].to_numpy(),
                                     va[feat_cols].to_numpy(), va["y"].to_numpy())
            o = te[["date", "code", "_fwd", "_close", "_close_fwd"]].copy()
            o["pred"] = model.predict(te[feat_cols].to_numpy())
            o["fold"] = fold.idx
            preds.append(o)
            ic, tstat, nd = rank_ic(o)
            log.info(f"    測試窗 rank IC {ic:+.4f}（t={tstat:.2f}, {nd} 天）"
                     f"　元模型權重 {', '.join(f'{k} {v:+.3f}' for k, v in model.weights().items())}")

        if not preds:
            continue
        allp = pd.concat(preds, ignore_index=True)
        ic, tstat, nd = rank_ic(allp)
        log.info(f"\n  h={horizon} 整體 rank IC {ic:+.4f}（t={tstat:.2f}，{nd} 個交易日）")
        summary.append({"horizon": horizon, "rank_IC": ic, "IC_t": tstat,
                        "n_days": nd, "n_rows": len(allp)})

        # 輸出成與逐檔版相同的格式，讓 portfolio_backtest.py 直接可用
        for code, g in allp.groupby("code"):
            g = g.sort_values("date")
            pd.DataFrame({
                "date": g["date"], "ret_next": g["_fwd"], "close_t": g["_close"],
                "close_fwd": g["_close_fwd"], "pred": g["pred"], "fold": g["fold"],
            }).to_csv(out_dir / f"predictions_{code}_h{horizon}.csv",
                      index=False, encoding="utf-8-sig")
        log.info(f"  已輸出 {allp['code'].nunique()} 檔的預測至 {out_dir}")

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(out_dir / "rank_ic.csv", index=False, encoding="utf-8-sig")
        log.info(f"\n{'=' * 74}\n彙總\n{'=' * 74}")
        log.info(s.to_string(index=False))
        log.info("\nrank IC 是排序模型的標準指標：每天算「預測排名 vs 實際排名」的相關，"
                 "再看跨日平均。IC 0.02~0.05 已屬實務上可用的水準，但仍需經"
                 "組合回測與 Reality Check 確認扣掉成本後站得住。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
