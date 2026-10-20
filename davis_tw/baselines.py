# -*- coding: utf-8 -*-
"""
天真基準（對應斷點 2 的修正）。

原論文沒有列出 naive baseline，導致 MAPE 0.93% 無從判讀。
本模組提供三組必列基準，所有實驗的報表都會強制帶上：

    naive       : p̂_{t+1} = p_t，等價於「預測隔日報酬率為 0」
    always_long : 恆predict 上漲 —— DA 的正確對照組（= 樣本期的上漲日比例）
    ma3/ma5 : 以近 k 日均價作為隔日預測，換算成報酬率
    drift   : 以訓練窗平均報酬率作為常數預測（含漂移的 naive）
    buy_hold: 交易層基準，永遠持有

注意：naive 的預測報酬率恆為 0，其方向準確率在本專案的定義下為 0
（sign(0) != sign(y)），這是刻意的 —— 它凸顯「零預測」不提供方向資訊，
價位 MAPE 卻可以很低，正是斷點 2 想揭露的現象。因此：
    比價位誤差（MAPE/MSE）→ 對照 naive
    比方向準確率（DA）    → 對照 always_long
拿 naive 當 DA 的對照組會得到「模型永遠顯著較佳」的空洞結論。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_return(df: pd.DataFrame) -> np.ndarray:
    """p̂_{t+1} = p_t → 報酬率預測恆為 0。"""
    return np.zeros(len(df))


def ma_return(df: pd.DataFrame, close_full: pd.Series, window: int) -> np.ndarray:
    """p̂_{t+1} = MA_k(p_t) → r̂ = MA_k(p_t)/p_t - 1。用完整價格序列算 MA 再對齊。"""
    ma = close_full.rolling(window).mean().reindex(df.index)
    return (ma / df["close_t"] - 1).to_numpy(float)


def drift_return(df: pd.DataFrame, train_mean_ret: float) -> np.ndarray:
    """以訓練窗的平均報酬率作為常數預測（避免用到測試窗資訊）。"""
    return np.full(len(df), float(train_mean_ret))


def build_baselines(test_df: pd.DataFrame, close_full: pd.Series,
                    train_mean_ret: float) -> dict[str, np.ndarray]:
    """回傳 {基準名稱: 預測報酬率陣列}，索引與 test_df 對齊。"""
    return {
        "naive": naive_return(test_df),
        "always_long": np.full(len(test_df), 1e-8),
        "ma3": ma_return(test_df, close_full, 3),
        "ma5": ma_return(test_df, close_full, 5),
        "drift": drift_return(test_df, train_mean_ret),
    }
