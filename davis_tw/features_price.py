# -*- coding: utf-8 -*-
"""
價量技術面特徵（特徵組 price，前綴 px_）。

所有特徵在時點 t 只使用 <= t 的資訊；目標為 t+1，故不存在前視偏誤。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_price_features(px: pd.DataFrame) -> pd.DataFrame:
    """由 OHLCV 建立技術面特徵，全部以 px_ 為前綴。"""
    c, h, l, o, v = (px["close"], px["high"], px["low"], px["open"], px["volume"])
    f = pd.DataFrame(index=px.index)

    # --- 報酬率與波動 ---
    ret1 = c.pct_change()
    f["px_ret1"] = ret1
    for k in (2, 3, 5, 10, 20):
        f[f"px_ret{k}"] = c.pct_change(k)
    for k in (5, 10, 20):
        f[f"px_vol{k}"] = ret1.rolling(k).std()
    f["px_ret1_lag1"] = ret1.shift(1)
    f["px_ret1_lag2"] = ret1.shift(2)
    f["px_ret1_lag3"] = ret1.shift(3)

    # --- 均線乖離（用比值而非絕對價位，避免非定態） ---
    for k in (3, 5, 10, 20, 60):
        ma = c.rolling(k).mean()
        f[f"px_ma{k}_dev"] = c / ma - 1
    f["px_ma5_ma20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1

    # --- 區間位置 ---
    for k in (5, 20, 60):
        hi, lo = h.rolling(k).max(), l.rolling(k).min()
        f[f"px_pos{k}"] = (c - lo) / (hi - lo).replace(0, np.nan)

    # --- 日內結構 ---
    f["px_hl_range"] = (h - l) / c.shift(1)
    f["px_oc_gap"] = o / c.shift(1) - 1
    f["px_close_loc"] = (c - l) / (h - l).replace(0, np.nan)

    # --- 量能 ---
    logv = np.log1p(v)
    f["px_logvol"] = logv
    for k in (5, 20):
        f[f"px_vol_ratio{k}"] = v / v.rolling(k).mean().replace(0, np.nan)
    f["px_dvol_ma5"] = logv - logv.rolling(5).mean()

    # --- 動能指標 ---
    f["px_rsi14"] = _rsi(c, 14) / 100
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["px_macd"] = macd / c
    f["px_macd_sig"] = (macd - macd.ewm(span=9).mean()) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["px_bb_z"] = (c - ma20) / sd20.replace(0, np.nan)

    # --- 漲跌幅限制相關（§5 RISKS：台股 ±10%） ---
    f["px_limit_up"] = (ret1 >= 0.095).astype(float)
    f["px_limit_dn"] = (ret1 <= -0.095).astype(float)

    # --- 日曆效應 ---
    f["px_dow"] = px.index.dayofweek.astype(float)
    f["px_month"] = px.index.month.astype(float)

    return f.replace([np.inf, -np.inf], np.nan)
