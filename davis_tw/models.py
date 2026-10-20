# -*- coding: utf-8 -*-
"""
模型層（DAViS-A′）：8 個基礎迴歸模型 + Huber 堆疊集成。

沿用原論文架構，只改預測目標（價位 → 報酬率）。堆疊方式刻意採
「訓練窗 fit base → 驗證窗產生 meta 訓練資料 → 測試窗評估」的時序三段式，
而非交叉驗證的 out-of-fold，理由是後者在時序資料上會讓 meta 看到未來樣本
（原論文斷點 1 的同類問題）。

Huber meta-regressor 的係數即各基礎模型的貢獻權重，
對應簡報 §1「可解釋性是框架的實務賣點」。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C

logger = logging.getLogger(__name__)


# ============================== 基礎模型 ==============================


def _make_model(name: str, seed: int, n_jobs: int):
    """單一基礎模型；需要標準化的模型自帶 StandardScaler（在折內 fit，無洩漏）。"""
    scaled = lambda est: Pipeline([("scaler", StandardScaler()), ("est", est)])

    if name == "linear":
        return scaled(LinearRegression())
    if name == "ridge":
        return scaled(Ridge(alpha=1.0, random_state=seed))
    if name == "lasso":
        return scaled(Lasso(alpha=1e-4, max_iter=20000, random_state=seed))
    if name == "elasticnet":
        return scaled(ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=20000, random_state=seed))
    if name == "svr":
        return scaled(SVR(kernel="rbf", C=1.0, epsilon=0.005, gamma="scale"))
    if name == "knn":
        return scaled(KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=n_jobs))
    if name == "mlp":
        return scaled(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600,
                                   early_stopping=True, random_state=seed))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     max_features="sqrt", n_jobs=n_jobs, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                   max_features="sqrt", n_jobs=n_jobs, random_state=seed)
    if name == "gbrt":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                         max_depth=3, subsample=0.8, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor  # type: ignore

        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            n_jobs=n_jobs, random_state=seed, tree_method="hist")
    if name == "lightgbm":
        from lightgbm import LGBMRegressor  # type: ignore

        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.8,
                             n_jobs=n_jobs, random_state=seed, verbose=-1)
    raise ValueError(f"未知的基礎模型：{name}")


def build_base_models(cfg: C.ModelConfig = C.MODEL) -> dict:
    """依設定建立基礎模型字典；缺套件的模型會被跳過並記錄。"""
    models = {}
    for name in cfg.base_models:
        try:
            models[name] = _make_model(name, cfg.random_state, cfg.n_jobs)
        except ImportError as exc:
            logger.warning("跳過基礎模型 %s（未安裝：%s）", name, exc)
    if not models:
        raise RuntimeError("沒有任何可用的基礎模型")
    return models


# ============================== 堆疊集成 ==============================


class DavisStackedEnsemble:
    """
    三段式堆疊：
        fit(X_tr, y_tr, X_va, y_va)
            1. 各 base model 以訓練窗擬合
            2. 於驗證窗產生預測矩陣 P_va
            3. Huber meta 以 P_va → y_va 擬合
        predict(X_te) = meta(base_preds(X_te))
    """

    def __init__(self, cfg: C.ModelConfig = C.MODEL):
        self.cfg = cfg
        self.base_models_ = build_base_models(cfg)
        self.meta_ = None
        self.feature_names_: list[str] = []
        self.base_names_: list[str] = []

    # ---------- 內部 ----------

    def _meta_estimator(self):
        if self.cfg.meta_model == "huber":
            # 報酬率量級小（1e-2），epsilon 取 1.35 為 Huber 標準值，
            # 交由前置的標準化（下方 _scale_preds）處理量綱。
            return HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
        if self.cfg.meta_model == "ridge":
            return Ridge(alpha=1.0)
        if self.cfg.meta_model == "linear":
            return LinearRegression()
        raise ValueError(f"未知的 meta 模型：{self.cfg.meta_model}")

    def _base_matrix(self, X) -> np.ndarray:
        return np.column_stack([self.base_models_[n].predict(X) for n in self.base_names_])

    # ---------- 對外 ----------

    def fit(self, X_tr, y_tr, X_va, y_va) -> "DavisStackedEnsemble":
        self.feature_names_ = list(X_tr.columns) if isinstance(X_tr, pd.DataFrame) else []
        Xtr = np.asarray(X_tr, dtype=float)
        Xva = np.asarray(X_va, dtype=float)

        fitted = []
        for name, model in self.base_models_.items():
            try:
                model.fit(Xtr, np.asarray(y_tr, dtype=float))
                fitted.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("基礎模型 %s 擬合失敗：%s", name, exc)
        if not fitted:
            raise RuntimeError("所有基礎模型皆擬合失敗")
        self.base_names_ = fitted

        P_va = self._base_matrix(Xva)
        self.meta_ = Pipeline([("scaler", StandardScaler()), ("est", self._meta_estimator())])
        self.meta_.fit(P_va, np.asarray(y_va, dtype=float))

        if self.cfg.refit_base_on_val:
            # 需要時可用 train+val 重訓 base（預測分佈會與 meta 校準時略有偏移，預設關閉）
            X_all = np.vstack([Xtr, Xva])
            y_all = np.concatenate([np.asarray(y_tr, float), np.asarray(y_va, float)])
            for name in self.base_names_:
                self.base_models_[name].fit(X_all, y_all)
        return self

    def predict(self, X) -> np.ndarray:
        return self.meta_.predict(self._base_matrix(np.asarray(X, dtype=float)))

    def predict_base(self, X) -> pd.DataFrame:
        """各基礎模型的個別預測，用於「集成 vs 單一模型」的 DM 檢定（斷點 3）。"""
        P = self._base_matrix(np.asarray(X, dtype=float))
        return pd.DataFrame(P, columns=self.base_names_)

    def meta_weights(self) -> pd.Series:
        """Huber meta 的標準化係數 = 各基礎模型的貢獻權重（可解釋性產出）。"""
        est = self.meta_.named_steps["est"]
        coef = getattr(est, "coef_", None)
        if coef is None:
            return pd.Series(dtype=float)
        return pd.Series(np.ravel(coef), index=self.base_names_).sort_values(key=np.abs, ascending=False)

    def feature_importance(self) -> pd.Series:
        """樹模型的平均特徵重要度（僅供描述性解讀，非因果宣稱）。"""
        if not self.feature_names_:
            return pd.Series(dtype=float)
        acc, n = None, 0
        for name in self.base_names_:
            imp = getattr(self.base_models_[name], "feature_importances_", None)
            if imp is None or len(imp) != len(self.feature_names_):
                continue
            acc = imp.astype(float) if acc is None else acc + imp
            n += 1
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(acc / n, index=self.feature_names_).sort_values(ascending=False)
