#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徵家族的實作。由 feature_engineering.py 匯入，拆成兩檔只是為了讓每檔好讀。

全部函式的輸入是 OHLCV 面板（date × code），輸出是 {特徵名: 面板} 的字典。
所有 rolling 一律向後看，沒有任何 shift(-N)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIMIT = 0.095          # 台股漲跌幅 ±10%，留一點容差給撮合誤差
LIMIT_OLD = 0.065      # 2015-06-01 以前是 ±7%
LIMIT_CHANGE = "2015-06-01"


def _limit(idx: pd.DatetimeIndex) -> pd.Series:
    """
    逐日的漲跌幅門檻。原本全期都用 10%，2015-06 以前（7%）幾乎從不觸發，
    2010–2014 年的 trd_locked 只有 0.00%～0.02%，可交易性過濾等於沒做（2026-10-15 修正）。
    """
    return pd.Series(np.where(idx < pd.Timestamp(LIMIT_CHANGE), LIMIT_OLD, LIMIT), index=idx)


def _rsi(c: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _zs(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """自身歷史 z 分數（causal）。"""
    return (x - x.rolling(n).mean()) / x.rolling(n).std().replace(0, np.nan)


def _pairwise(frames: list, how: str) -> pd.DataFrame:
    """對齊的逐元素 max / min，避免 concat+stack 在大面板上爆記憶體。"""
    out = frames[0]
    for g in frames[1:]:
        out = np.maximum(out, g) if how == "max" else np.minimum(out, g)
    return out


# ============================================================================
# A. 時間序列（每檔自己的歷史）
# ============================================================================

def timeseries(P: dict) -> dict:
    """沿用原有 43 項的精神，另補波動結構與更有效率的波動估計量。"""
    O, H, L, C, V = P["open"], P["high"], P["low"], P["close"], P["volume"]
    r1 = C.pct_change()
    f = {}

    for k in (1, 2, 3, 5, 10, 20, 60):
        f[f"ts_ret{k}"] = C.pct_change(k)
    for lag in (1, 2, 3):
        f[f"ts_ret1_lag{lag}"] = r1.shift(lag)
    for k in (5, 10, 20, 60):
        f[f"ts_vol{k}"] = r1.rolling(k).std()
    for k in (5, 10, 20, 60, 120):
        f[f"ts_ma{k}_dev"] = C / C.rolling(k).mean() - 1
    f["ts_ma5_ma20"] = C.rolling(5).mean() / C.rolling(20).mean() - 1
    f["ts_ma20_ma60"] = C.rolling(20).mean() / C.rolling(60).mean() - 1
    for k in (20, 60, 120):
        hi, lo = H.rolling(k).max(), L.rolling(k).min()
        f[f"ts_pos{k}"] = (C - lo) / (hi - lo).replace(0, np.nan)

    # Parkinson 與 Garman-Klass：用到高低價，同樣樣本數下比收盤價波動更有效率
    hl = np.log(H / L.replace(0, np.nan))
    f["ts_parkinson20"] = np.sqrt((hl ** 2).rolling(20).mean() / (4 * np.log(2)))
    co = np.log(C / O.replace(0, np.nan))
    f["ts_gk20"] = np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)
                           .rolling(20).mean().clip(lower=0))

    # 上下行波動不對稱：下跌時的波動是否明顯大於上漲時。
    # where 之後窗內只剩約半數非缺值，所以 min_periods 必須放寬，
    # 否則 rolling(60) 要求滿 60 個非缺值，整欄會是 NaN。
    f["ts_updn_vol_ratio"] = (
        r1.where(r1 > 0).rolling(60, min_periods=15).std()
        / r1.where(r1 < 0).rolling(60, min_periods=15).std().replace(0, np.nan))
    f["ts_skew60"] = r1.rolling(60).skew()
    f["ts_kurt60"] = r1.rolling(60).kurt()
    f["ts_volofvol"] = r1.rolling(20).std().rolling(60).std()
    f["ts_vol_trend"] = (r1.rolling(20).std()
                         / r1.rolling(60).std().replace(0, np.nan))

    f["ts_rsi14"] = _rsi(C) / 100
    low9, high9 = L.rolling(9).min(), H.rolling(9).max()
    rsv = ((C - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    kl = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    dl = kl.ewm(alpha=1 / 3, adjust=False).mean()
    f["ts_k"], f["ts_d"], f["ts_kd_diff"] = kl / 100, dl / 100, (kl - dl) / 100
    macd = C.ewm(span=12).mean() - C.ewm(span=26).mean()
    f["ts_macd"] = macd / C
    f["ts_macd_sig"] = (macd - macd.ewm(span=9).mean()) / C
    f["ts_bb_z"] = ((C - C.rolling(20).mean())
                    / C.rolling(20).std().replace(0, np.nan))

    tr = _pairwise([H - L, (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], "max")
    atr = tr.rolling(14).mean()
    f["ts_atr14"] = atr / C
    f["ts_drop_in_atr"] = (C / C.shift(5) - 1) / (atr / C).replace(0, np.nan)

    f["ts_hl_range"] = (H - L) / C.shift(1)
    f["ts_oc_gap"] = O / C.shift(1) - 1
    f["ts_close_loc"] = (C - L) / (H - L).replace(0, np.nan)
    # 用成交金額而不是成交股數：Yahoo 的股數會因日後的配股、減資被往回調整，
    # 股數的「水準」因此帶有未來資訊；價 × 量不受分割調整影響（2026-10-15 修正）
    lv = np.log1p(C * V)
    f["ts_logvol"] = lv
    for k in (5, 20):
        f[f"ts_vol_ratio{k}"] = V / V.rolling(k).mean().replace(0, np.nan)
    f["ts_dvol_ma5"] = lv - lv.rolling(5).mean()
    obv = (np.sign(C.diff().fillna(0)) * V).cumsum()
    f["ts_obv_dev"] = ((obv - obv.rolling(20).mean())
                       / V.rolling(20).mean().replace(0, np.nan))
    # 量價背離：價格走強但量能萎縮，是常見的轉折前兆
    f["ts_px_vol_corr20"] = r1.rolling(20).corr(V.pct_change())
    return f


# ============================================================================
# B. 相對市場與產業
# ============================================================================

def relative(P: dict, M: pd.DataFrame, ind: dict, mask: pd.DataFrame) -> dict:
    C = P["close"]
    r1 = C.pct_change()
    tw = M["TWII"].reindex(C.index).ffill()
    mr = tw.pct_change()
    f = {}

    for k in (1, 5, 20, 60):
        f[f"rel_excess{k}"] = C.pct_change(k).sub(tw.pct_change(k), axis=0)

    beta = r1.rolling(60).cov(mr).div(mr.rolling(60).var(), axis=0)
    f["rel_beta60"] = beta
    f["rel_corr60"] = r1.rolling(60).corr(mr)
    resid = r1.sub(beta.mul(mr, axis=0), axis=0)
    f["rel_idio_vol60"] = resid.rolling(60).std()
    f["rel_idio_ret20"] = resid.rolling(20).sum()

    # 產業相對：同產業等權平均，採 leave-one-out 不把自己算進去
    grp = pd.Series({c: ind.get(c, "其他") for c in C.columns})
    for k in (1, 5, 20):
        rk = C.pct_change(k).where(mask)
        num = rk.T.groupby(grp).transform("sum").T
        cnt = rk.notna().T.groupby(grp).transform("sum").T
        loo = (num - rk) / (cnt - 1).replace(0, np.nan)
        f[f"rel_ind_excess{k}"] = C.pct_change(k) - loo
    return f


# ============================================================================
# C. 流動性與微結構
# ============================================================================

def liquidity(P: dict) -> dict:
    H, L, C, V = P["high"], P["low"], P["close"], P["volume"]
    r1 = C.pct_change()
    dv = C * V                      # 成交金額
    f = {}
    f["liq_logdvol20"] = np.log1p(dv.rolling(20).mean())
    f["liq_dvol_trend"] = (dv.rolling(5).mean()
                           / dv.rolling(60).mean().replace(0, np.nan))
    # Amihud 非流動性：單位成交金額推動的價格變動，越大越難成交
    f["liq_amihud20"] = (r1.abs() / dv.replace(0, np.nan)).rolling(20).mean() * 1e9
    # 零報酬日比例：流動性差的股票常整天沒成交而原價不動
    f["liq_zero_ret60"] = (r1.abs() < 1e-8).rolling(60).mean()
    # Corwin-Schultz 價差代理：由連續兩日的高低價推估買賣價差
    hi2 = _pairwise([H, H.shift(1)], "max")
    lo2 = _pairwise([L, L.shift(1)], "min")
    b = (np.log(H / L.replace(0, np.nan)) ** 2
         + np.log(H.shift(1) / L.shift(1).replace(0, np.nan)) ** 2)
    g = np.log(hi2 / lo2.replace(0, np.nan)) ** 2
    k = 3 - 2 * np.sqrt(2)
    a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    f["liq_cs_spread"] = (2 * (np.exp(a) - 1) / (1 + np.exp(a))).rolling(20).mean()
    f["liq_turnover_z"] = _zs(V, 60)
    return f


# ============================================================================
# D. 可交易性（做成明確欄位，用來排除而不是利用）
# ============================================================================

def tradability(P: dict) -> dict:
    H, L, C = P["high"], P["low"], P["close"]
    prev = C.shift(1)
    r1 = C.pct_change()
    lim = _limit(C.index)
    up = (C >= prev.mul(1 + lim, axis=0)) & (C >= H * 0.999)
    dn = (C <= prev.mul(1 - lim, axis=0)) & (C <= L * 1.001)
    f = {"trd_limit_up": up.astype("float64"),
         "trd_limit_dn": dn.astype("float64"),
         "trd_locked": (up | dn).astype("float64"),
         "trd_dist_up": (prev.mul(1 + lim, axis=0) - C) / C,   # 距漲停還有多少空間
         "trd_dist_dn": (C - prev.mul(1 - lim, axis=0)) / C,
         "trd_limit_cnt20": (up | dn).rolling(20).sum(),
         "trd_big_move20": r1.abs().rolling(20).max()}
    f["trd_up_streak"] = _streak(up)
    f["trd_dn_streak"] = _streak(dn)
    return f


def _streak(flag: pd.DataFrame) -> pd.DataFrame:
    """
    連續為真的天數（鎖死越多天越買不到）。
    每欄的分組邊界不同，groupby 在 DataFrame 上做不到，所以逐列遞推：
    只有 4000 列，每列是一個 383 寬的向量運算，成本可忽略。
    """
    a = flag.to_numpy()
    out = np.zeros(a.shape, dtype="float64")
    run = np.zeros(a.shape[1], dtype="float64")
    for i in range(a.shape[0]):
        run = np.where(a[i], run + 1.0, 0.0)
        out[i] = run
    return pd.DataFrame(out, index=flag.index, columns=flag.columns)


# ============================================================================
# E. 美股隔夜（台股開盤前已知，屬合法事前資訊）
# ============================================================================

def overnight(P: dict, M: pd.DataFrame) -> dict:
    """
    費半、那斯達克在台北時間凌晨收盤，台股開盤前就知道結果。
    全市場共同值本身沒有橫斷面差異，所以要乘上個股 beta 才有區辨力。
    """
    C = P["close"]
    r1 = C.pct_change()
    f = {}
    for name in ("SOX", "IXIC", "GSPC"):
        if name not in M.columns:
            continue
        s = M[name].reindex(C.index).ffill().pct_change()
        f[f"ovn_{name}"] = pd.DataFrame(
            np.repeat(s.to_numpy()[:, None], C.shape[1], axis=1),
            index=C.index, columns=C.columns)
        b = r1.rolling(60).cov(s).div(s.rolling(60).var(), axis=0)
        f[f"ovn_{name}_beta"] = b
        f[f"ovn_{name}_x_beta"] = b.mul(s, axis=0)
    return f


# ============================================================================
# F. 橫斷面百分位（這是原本完全缺的一類）
# ============================================================================

CS_SOURCES = ["ts_ret1", "ts_ret5", "ts_ret20", "ts_ret60", "ts_vol20",
              "ts_vol60", "ts_ma20_dev", "ts_ma60_dev", "ts_rsi14",
              "ts_pos60", "ts_atr14", "rel_excess5", "rel_excess20",
              "rel_beta60", "rel_idio_vol60", "rel_ind_excess5",
              "liq_logdvol20", "liq_amihud20", "liq_cs_spread",
              "ts_parkinson20", "ts_skew60"]


def cross_sectional(f: dict, mask: pd.DataFrame) -> dict:
    """
    每日把各訊號在母體內排成 0~1 百分位。
    模型要判斷的是「今天這幾百檔裡誰相對強」，單檔自己的指標水準回答不了這件事。
    """
    out = {}
    for src in CS_SOURCES:
        if src in f:
            out[f"cs_{src}"] = f[src].where(mask).rank(axis=1, pct=True)
    return out
