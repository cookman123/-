# -*- coding: utf-8 -*-
"""
評估指標與統計檢定（對應斷點 2、3 的修正）。

主指標：DA（方向準確率）；輔指標：MAPE / RMSE。
顯著性一律用 Diebold–Mariano 檢定（含 Harvey–Leybourne–Newbold 小樣本修正），
不用成對 t 檢定 —— 後者忽略預測誤差的序列相關，是原論文斷點 3 的來源。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


# ============================== 點預測指標 ==============================


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def directional_accuracy(y_true, y_pred) -> float:
    """
    方向準確率：以「實際報酬率」與「預測報酬率」的正負號比對。
    實際報酬率恰為 0 的日子（漲跌停鎖死、無量）不列入分母，避免灌水。
    """
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(yt) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask])))


def price_mape_from_returns(close_t, ret_true, ret_pred) -> float:
    """把報酬率預測還原成價位後計算 MAPE，供與原論文的 0.93% 對照。"""
    close_t = np.asarray(close_t, float)
    p_true = close_t * (1 + np.asarray(ret_true, float))
    p_pred = close_t * (1 + np.asarray(ret_pred, float))
    return mape(p_true, p_pred)


def evaluate_predictions(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """df 需含 ret_next（實際）、close_t 與 pred_col（預測報酬率）。"""
    yt, yp = df["ret_next"].to_numpy(float), df[pred_col].to_numpy(float)
    return {
        "n": int(len(df)),
        "DA": directional_accuracy(yt, yp),
        "MAPE_price": price_mape_from_returns(df["close_t"], yt, yp),
        "RMSE_ret": rmse(yt, yp),
        "MAE_ret": mae(yt, yp),
        "corr": float(np.corrcoef(yt, yp)[0, 1]) if len(df) > 2 and np.std(yp) > 0 else float("nan"),
    }


# ============================== Diebold–Mariano ==============================


@dataclass
class DMResult:
    stat: float
    p_value: float
    n: int
    loss: str
    better: str  # "model1" | "model2" | "tie"

    def as_dict(self) -> dict:
        return {"DM_stat": self.stat, "DM_p": self.p_value,
                "DM_n": self.n, "DM_loss": self.loss, "DM_better": self.better}


def diebold_mariano(y_true, pred1, pred2, h: int = 1, loss: str = "mse",
                    hln_correction: bool = True) -> DMResult:
    """
    H0：兩組預測的期望損失相同。
    stat < 0 表示 pred1 損失較低（較佳）。

    loss:
        "mse"   均方誤差（點預測比較）
        "mae"   絕對誤差
        "sign"  方向損失（1 - 命中），用於以 DA 為主指標時的顯著性檢定
    """
    yt = np.asarray(y_true, float)
    p1, p2 = np.asarray(pred1, float), np.asarray(pred2, float)
    mask = np.isfinite(yt) & np.isfinite(p1) & np.isfinite(p2)
    yt, p1, p2 = yt[mask], p1[mask], p2[mask]

    if loss == "mse":
        l1, l2 = (yt - p1) ** 2, (yt - p2) ** 2
    elif loss == "mae":
        l1, l2 = np.abs(yt - p1), np.abs(yt - p2)
    elif loss == "sign":
        keep = np.abs(yt) > 1e-12
        yt, p1, p2 = yt[keep], p1[keep], p2[keep]
        l1 = (np.sign(yt) != np.sign(p1)).astype(float)
        l2 = (np.sign(yt) != np.sign(p2)).astype(float)
    else:
        raise ValueError(f"未知的損失函數：{loss}")

    d = l1 - l2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return DMResult(float("nan"), float("nan"), n, loss, "tie")

    d_bar = float(np.mean(d))
    # Newey–West 長期變異數（自相關至 h-1 階）
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for k in range(1, h):
        gk = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var += 2 * gk
    if var <= 0:
        return DMResult(float("nan"), float("nan"), n, loss, "tie")

    stat = d_bar / np.sqrt(var / n)
    if hln_correction:
        # Harvey, Leybourne & Newbold (1997) 小樣本修正
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= adj
        p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))

    better = "tie"
    if p < 0.05:
        better = "model1" if stat < 0 else "model2"
    return DMResult(float(stat), float(p), n, loss, better)


# ============================== 交易績效指標 ==============================


def sharpe_ratio(returns, risk_free: float = 0.0,
                 periods: int = C.TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    excess = r - risk_free / periods
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods))


def sortino_ratio(returns, periods: int = C.TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(downside, ddof=1) * np.sqrt(periods))


def max_drawdown(returns) -> float:
    """回傳最大回撤（負值，例如 -0.23 表示 -23%）。"""
    r = np.asarray(returns, float)
    r = np.nan_to_num(r)
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else float("nan")


def annualized_return(returns, periods: int = C.TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    total = float(np.prod(1 + r))
    return float(total ** (periods / len(r)) - 1)
