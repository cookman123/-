# -*- coding: utf-8 -*-
"""
籌碼面特徵（特徵組 chip，前綴 chip_）—— 本研究相對原論文的新增資料軸（RQ2）。

原始欄位是「當日」公開數值（三大法人買賣超、融資券餘額、借券、當沖），
盤後才揭露，因此在預測 t+1 時可使用 t 日數值，無前視問題。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_chip_features(chips: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """把籌碼面原始日資料轉成量綱穩定的特徵（以成交量／流通量為分母做標準化）。"""
    df = chips.reindex(px.index)  # 對齊交易日；缺日不前填，交由下游 dropna 處理
    vol = px["volume"].replace(0, np.nan)
    f = pd.DataFrame(index=px.index)

    inst_cols = [c for c in ("foreign_investor", "investment_trust", "dealer",
                             "dealer_self", "dealer_hedge") if c in df.columns]
    for col in inst_cols:
        x = df[col]
        f[f"chip_{col}_norm"] = x / vol                          # 買賣超佔當日成交量比重
        f[f"chip_{col}_ma5"] = (x / vol).rolling(5).mean()
        f[f"chip_{col}_ma20"] = (x / vol).rolling(20).mean()
        f[f"chip_{col}_z20"] = _rolling_z(x, 20)
        f[f"chip_{col}_streak"] = _sign_streak(x)                # 連買／連賣天數

    if inst_cols:
        # 有官方「三大法人買賣超」欄就直接用，避免自營商自行買賣／避險重複相加
        if "inst_total" in df.columns:
            total = df["inst_total"]
        else:
            total = df[[c for c in inst_cols if c in ("foreign_investor", "investment_trust", "dealer")]].sum(axis=1)
        f["chip_inst_total_norm"] = total / vol
        f["chip_inst_total_ma5"] = (total / vol).rolling(5).mean()
        if {"foreign_investor", "investment_trust"} <= set(inst_cols):
            # 外資與投信同向與否：籌碼共識訊號
            f["chip_fi_it_agree"] = np.sign(df["foreign_investor"]) * np.sign(df["investment_trust"])

    if "margin_purchase" in df.columns:
        m = df["margin_purchase"]
        f["chip_margin_chg"] = m.pct_change().replace([np.inf, -np.inf], np.nan)
        f["chip_margin_chg5"] = m.pct_change(5).replace([np.inf, -np.inf], np.nan)
        f["chip_margin_z20"] = _rolling_z(m, 20)

    if "short_sale" in df.columns:
        s = df["short_sale"]
        f["chip_short_chg"] = s.pct_change().replace([np.inf, -np.inf], np.nan)
        f["chip_short_z20"] = _rolling_z(s, 20)
        if "margin_purchase" in df.columns:
            # 券資比：融券餘額 / 融資餘額，軋空壓力的常用代理變數
            f["chip_short_margin_ratio"] = s / df["margin_purchase"].replace(0, np.nan)

    if "securities_lending" in df.columns:
        f["chip_lending_norm"] = df["securities_lending"] / vol
        f["chip_lending_ma5"] = (df["securities_lending"] / vol).rolling(5).mean()

    if "day_trade_volume" in df.columns:
        f["chip_daytrade_ratio"] = df["day_trade_volume"] / vol
        f["chip_daytrade_ma5"] = (df["day_trade_volume"] / vol).rolling(5).mean()

    return f.replace([np.inf, -np.inf], np.nan)


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    """
    20 日 z 分數。標準差為 0（例如投信連續 20 日完全沒進出）時，偏離度定義為 0，
    不是遺漏值 —— 否則整列會在 dropna 被剔除，冷門類股／小型股會因此損失大量樣本。
    暖機期（不足 window 日）仍維持 NaN，因為那是真的還算不出來。
    """
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    z = (x - mu) / sd.replace(0, np.nan)
    return z.mask(sd.eq(0), 0.0)


def _sign_streak(x: pd.Series) -> pd.Series:
    """連續同向天數（正為連買、負為連賣）。"""
    sign = np.sign(x.fillna(0))
    out = np.zeros(len(sign))
    run = 0.0
    prev = 0.0
    for i, s in enumerate(sign.to_numpy()):
        run = run + s if s != 0 and s == prev else s
        prev = s
        out[i] = run
    return pd.Series(out, index=x.index)
