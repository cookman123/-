# -*- coding: utf-8 -*-
"""
Walk-forward 滾動評估（對應斷點 1 的修正）。

切分順序嚴格依時序：訓練 → 驗證 → 測試，且測試窗永遠在最後。
原論文把測試期（2017/06–10）放在驗證期（2017/10–2018/02）之前，
調參時等於看過測試期之後的資料；本模組在結構上排除這種可能。

每一折內部的流程：
    1. 以「訓練窗結束日」為界，重新擬合文本模型（TF-IDF/PCA/Ward）
    2. 基礎模型以訓練窗擬合
    3. Huber meta 以驗證窗擬合
    4. 於測試窗預測，並同步產生基準預測（naive/MA/drift）
折與折之間不共用任何擬合物件。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import config as C
from . import baselines as bl
from .dataset import StockData, assemble
from .models import DavisStackedEnsemble

logger = logging.getLogger(__name__)


# ============================== 切分 ==============================


@dataclass
class Fold:
    idx: int
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def describe(self) -> str:
        f = lambda i: f"{i[0].date()}~{i[-1].date()}"
        return f"fold{self.idx}: train {f(self.train)} | val {f(self.val)} | test {f(self.test)}"


def make_folds(index: pd.DatetimeIndex, cfg: C.WalkForwardConfig = C.WALK_FORWARD) -> list[Fold]:
    """以交易日數（非日曆日）切窗，確保每折樣本數一致。"""
    n = len(index)
    span = cfg.train_days + cfg.val_days + cfg.test_days
    if n < span:
        logger.warning("樣本 %d 日 < 單折所需 %d 日，無法切折", n, span)
        return []

    folds, i, start = [], 0, 0
    while start + span <= n:
        tr_end = start + cfg.train_days
        va_end = tr_end + cfg.val_days
        te_end = min(va_end + cfg.test_days, n)
        train_start = 0 if cfg.expanding else start
        folds.append(Fold(i, index[train_start:tr_end], index[tr_end:va_end], index[va_end:te_end]))
        i += 1
        start += cfg.step_days
    return folds


# ============================== 單股單實驗的執行 ==============================


@dataclass
class RunResult:
    code: str
    experiment: str
    predictions: pd.DataFrame                 # 逐日：pred / ret_next / close_t / fold
    baseline_predictions: pd.DataFrame        # 逐日：各基準的預測報酬率
    base_model_predictions: pd.DataFrame      # 逐日：8 個基礎模型各自的預測
    meta_weights: pd.DataFrame                # 每折的 Huber 權重
    feature_importance: pd.Series = field(default_factory=pd.Series)
    n_features: int = 0
    folds: list = field(default_factory=list)


def run_stock_experiment(stock: StockData, experiment: str, groups=None,
                         wf: C.WalkForwardConfig = C.WALK_FORWARD,
                         model_cfg: C.ModelConfig = C.MODEL,
                         target: str = C.TARGET,
                         restrict_index: pd.DatetimeIndex | None = None,
                         tokenizer=None, horizon: int = 1) -> RunResult | None:
    """對單一股票跑完一個實驗設定（E0/E1/.../E3）的所有 walk-forward 折。"""
    groups = groups or C.EXPERIMENTS[experiment]
    needs_text = any(g in C.TEXT_GROUPS for g in groups)

    # 先用「全期間」組一次，取得可用日期索引與特徵欄位（文本欄位稍後每折重算）
    text_all = None
    if needs_text:
        # 這裡只是為了取得欄位與日期範圍，實際模型用的是折內重擬合的版本
        text_all = stock.build_text_features(stock.px.index[-1], sources=[g for g in groups if g in C.TEXT_GROUPS],
                                             tokenizer=tokenizer)
    try:
        panel = assemble(stock, text_all, groups, target, horizon)
    except ValueError as exc:
        logger.warning("[%s/%s] %s", stock.code, experiment, exc)
        return None

    if restrict_index is not None:
        panel = panel.loc[panel.index.intersection(restrict_index)]

    folds = make_folds(panel.index, wf)
    if len(folds) < wf.min_folds:
        logger.warning("[%s/%s] 可用折數 %d < %d，略過", stock.code, experiment, len(folds), wf.min_folds)
        return None

    preds, base_preds, base_rows, weight_rows, importances = [], [], [], [], []
    close_full = stock.px["close"]

    for fold in folds:
        # --- 文本模型每折重新擬合（防洩漏的關鍵） ---
        if needs_text:
            text_fold = stock.build_text_features(
                fold.train[-1], sources=[g for g in groups if g in C.TEXT_GROUPS], tokenizer=tokenizer)
            fold_panel = assemble(stock, text_fold, groups, target, horizon)
            if restrict_index is not None:
                fold_panel = fold_panel.loc[fold_panel.index.intersection(restrict_index)]
        else:
            fold_panel = panel

        cols = fold_panel.attrs["feature_columns"]
        tr = fold_panel.loc[fold_panel.index.intersection(fold.train)]
        va = fold_panel.loc[fold_panel.index.intersection(fold.val)]
        te = fold_panel.loc[fold_panel.index.intersection(fold.test)]
        if len(tr) < 100 or len(va) < 20 or len(te) < 10:
            logger.info("[%s/%s] %s 樣本不足（%d/%d/%d），跳過此折",
                        stock.code, experiment, fold.idx, len(tr), len(va), len(te))
            continue

        model = DavisStackedEnsemble(model_cfg)
        try:
            model.fit(tr[cols], tr["y"], va[cols], va["y"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s/%s] fold%d 擬合失敗：%s", stock.code, experiment, fold.idx, exc)
            continue

        yhat = model.predict(te[cols])
        out = te[["ret_next", "close_t", "close_next", "y", "y_dir"]].copy()
        out["pred"] = yhat
        out["fold"] = fold.idx
        out["code"] = stock.code
        preds.append(out)

        # 基準：drift 用訓練窗均值，不碰測試窗
        bp = pd.DataFrame(bl.build_baselines(te, close_full, float(tr["ret_next"].mean())), index=te.index)
        bp["fold"] = fold.idx
        base_preds.append(bp)

        # 各基礎模型的個別預測（供「集成 vs 單一模型」DM 檢定）
        bm = model.predict_base(te[cols])
        bm.index = te.index
        bm["fold"] = fold.idx
        base_rows.append(bm)

        w = model.meta_weights()
        weight_rows.append(pd.DataFrame({"fold": fold.idx, "model": w.index, "weight": w.to_numpy()}))
        imp = model.feature_importance()
        if not imp.empty:
            importances.append(imp)

    if not preds:
        return None

    fi = (pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
          if importances else pd.Series(dtype=float))
    return RunResult(
        code=stock.code,
        experiment=experiment,
        predictions=pd.concat(preds),
        baseline_predictions=pd.concat(base_preds),
        base_model_predictions=pd.concat(base_rows),
        meta_weights=pd.concat(weight_rows) if weight_rows else pd.DataFrame(),
        feature_importance=fi,
        n_features=len(panel.attrs["feature_columns"]),
        folds=[f.describe() for f in folds],
    )


# ============================== 實驗可比性：共同日期索引 ==============================


def common_evaluation_index(stock: StockData, experiments, target: str = C.TARGET,
                            tokenizer=None, horizon: int = 1) -> pd.DatetimeIndex | None:
    """
    取各實驗特徵皆非缺值的日期交集。

    必要性：籌碼面 T86 只到 2024/12，文本回溯亦不完整；若不對齊，
    E0 會比 E2/E3 多跑好幾年，實驗間的 DA/MAPE 差異就分不清是
    「特徵有效」還是「期間不同」。對齊後所有實驗共用同一段測試期。
    """
    idx = None
    for exp in experiments:
        groups = C.EXPERIMENTS[exp]
        text = None
        if any(g in C.TEXT_GROUPS for g in groups):
            text = stock.build_text_features(stock.px.index[-1],
                                             sources=[g for g in groups if g in C.TEXT_GROUPS],
                                             tokenizer=tokenizer)
        try:
            panel = assemble(stock, text, groups, target, horizon)
        except ValueError:
            return None
        idx = panel.index if idx is None else idx.intersection(panel.index)
        if len(idx) == 0:
            return None
    return idx
