# -*- coding: utf-8 -*-
"""
含成本交易回測（對應斷點 4 的修正：誤差小 ≠ 可獲利）。

策略設定（刻意簡單，讓「訊號價值」而非「策略工程」成為被檢驗的對象）：
    long_flat  : 預測報酬 > threshold 則隔日持有，否則空手（台股現股可行）
    long_short : 預測方向做多／做空（需信用交易，僅供對照）

成本：手續費 0.1425%（買賣各一次，可設折讓）＋ 證交稅 0.3%（賣出），
      僅在「部位變動」的當日計費，不做每日虛擬進出以免高估成本。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .metrics import annualized_return, max_drawdown, sharpe_ratio, sortino_ratio


@dataclass
class BacktestResult:
    equity: pd.Series
    net_returns: pd.Series
    positions: pd.Series
    stats: dict

    def to_dict(self) -> dict:
        return dict(self.stats)


def _position_from_signal(pred: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    if mode == "long_flat":
        return (pred > threshold).astype(float)
    if mode == "long_short":
        return np.sign(pred) * (np.abs(pred) > threshold)
    if mode == "buy_hold":
        return np.ones(len(pred))
    raise ValueError(f"未知的策略模式：{mode}")


def backtest(test_df: pd.DataFrame, pred: np.ndarray, mode: str = "long_flat",
             threshold: float = 0.0, cost: C.CostConfig = C.COST,
             horizon: int = 1) -> BacktestResult:
    """
    long_flat：訊號 > 門檻就進場，並至少持有 horizon 個交易日；最短持有期滿後
    訊號仍為真就續抱（不會多付一次來回成本），轉為假才出場。horizon=1 等同逐日再平衡。

    逐日損益用 close_t 自己算出的「隔日報酬」，不是 h 日累積報酬 ——
    後者會讓權益曲線重複計算同一段漲跌。
    """
    idx = test_df.index
    sig = np.asarray(pred, float)
    n = len(sig)
    fee = cost.fee_rate * cost.fee_discount
    daily = test_df["close_t"].pct_change().shift(-1).fillna(0.0).to_numpy(float)
    want_all = (np.ones(n) if mode == "buy_hold"
                else _position_from_signal(sig, mode, threshold))

    pos = np.zeros(n)
    entry_cost = np.zeros(n)
    exit_cost = np.zeros(n)
    in_pos, hold_until, n_trades = False, -1, 0

    for i in range(n):
        want = want_all[i]
        if not in_pos and want != 0:
            in_pos, hold_until = True, i + horizon
            entry_cost[i] += fee + cost.slippage
            n_trades += 1
        elif in_pos and i >= hold_until and want == 0:
            in_pos = False
            exit_cost[i] += fee + cost.tax_rate + cost.slippage
        pos[i] = want if in_pos else 0.0

    if in_pos:                                   # 最後一日強制平倉
        exit_cost[-1] += fee + cost.tax_rate + cost.slippage

    gross = pos * daily
    net = gross - entry_cost - exit_cost
    equity = pd.Series(np.cumprod(1 + net), index=idx, name="equity")

    stats = {
        "strategy": mode,
        "horizon": horizon,
        "n_days": int(n),
        "n_trades": n_trades,
        "exposure": float(np.mean(np.abs(pos))),
        "avg_hold_days": float(np.sum(np.abs(pos)) / n_trades) if n_trades else float("nan"),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else float("nan"),
        "ann_return": annualized_return(net),
        "sharpe": sharpe_ratio(net),
        "sortino": sortino_ratio(net),
        "max_drawdown": max_drawdown(net),
        "gross_ann_return": annualized_return(gross),
        "cost_drag": float(np.sum(entry_cost + exit_cost)),
        "win_rate": float(np.mean(net[pos != 0] > 0)) if np.any(pos != 0) else float("nan"),
    }
    return BacktestResult(equity, pd.Series(net, index=idx, name="net_ret"),
                          pd.Series(pos, index=idx, name="position"), stats)


def backtest_suite(test_df: pd.DataFrame, pred: np.ndarray,
                   baselines: dict[str, np.ndarray] | None = None,
                   threshold: float = 0.0, horizon: int = 1) -> pd.DataFrame:
    """模型策略 + 買進持有 + 各基準訊號策略的績效對照表。"""
    rows = []
    rows.append({"name": "model", **backtest(test_df, pred, "long_flat", threshold,
                                         horizon=horizon).to_dict()})
    rows.append({"name": "buy_hold", **backtest(test_df, np.ones(len(test_df)), "buy_hold").to_dict()})
    for bname, bpred in (baselines or {}).items():
        if bname == "always_long":
            continue  # 與 buy_hold 等價，不重複列
        if np.allclose(np.nan_to_num(bpred), 0):
            continue  # naive 恆為 0，策略上等同空手，無須列
        rows.append({"name": bname, **backtest(test_df, bpred, "long_flat", threshold,
                                        horizon=horizon).to_dict()})
    return pd.DataFrame(rows)
