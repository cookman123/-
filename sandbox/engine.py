# -*- coding: utf-8 -*-
"""
主迴圈：策略 → 撮合 → 記帳（計劃 §3）。

    t 日收盤後   策略只看得到 ≤ t 的 DataView，回傳目標權重（或 None：不調整）
    t+1 日       以開盤價（預設）或收盤價撮合
    t+1 日收盤   以收盤價評價淨值；停牌、下市後沒有收盤價的持股用最近一次收盤價

風控（§5）在 M3 加入；這裡只擋掉格式不合法的權重。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from .broker import Broker
from .data import DataView, Panel
from .portfolio import Portfolio

FILLS = ("next_open", "next_close")


class Strategy(Protocol):
    def generate(self, date: pd.Timestamp, view: DataView) -> dict[str, float] | None:
        """回傳 {代號: 目標權重}，未列出的持股目標為 0；回傳 None 表示不調整。"""


@dataclass(frozen=True)
class Config:
    initial_cash: float = 1_000_000.0
    fill: str = "next_open"
    broker: Broker = field(default_factory=Broker)

    def __post_init__(self):
        if self.fill not in FILLS:
            raise ValueError(f"fill 必須是 {FILLS} 之一")


@dataclass
class Result:
    nav: pd.DataFrame          # 每日收盤：cash, holdings, nav
    positions: pd.DataFrame    # 每日收盤持股：日期 × 代號
    ledger: pd.DataFrame
    config: Config


def run(strategy: Strategy, panel: Panel, start=None, end=None,
        config: Config | None = None) -> Result:
    config = config or Config()
    cal = panel.calendar
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    if cal.empty:
        raise ValueError("期間內沒有交易日")

    close_ff = panel.fields["close"].ffill()
    ref_field = "open" if config.fill == "next_open" else "close"
    pf = Portfolio(config.initial_cash)
    pending: dict[str, float] | None = None
    prev = None
    nav_rows, pos_rows = [], []

    for d in cal:
        if pending is not None:
            ref = panel.fields[ref_field].loc[d].dropna()
            # 決定目標金額用的淨值：當日參考價，沒有的用前一日收盤
            if ref_field == "open":
                mark = ref.reindex(close_ff.columns).fillna(close_ff.loc[prev])
            else:
                mark = close_ff.loc[d]
            config.broker.rebalance(pf, d, pending, ref, mark, panel.kinds)
            pending = None

        mark = close_ff.loc[d]
        nav = pf.value(mark)
        nav_rows.append({"date": d, "cash": pf.cash, "holdings": nav - pf.cash, "nav": nav})
        pos_rows.append(pd.Series(pf.positions, name=d, dtype="int64"))

        w = strategy.generate(d, DataView(panel, d))
        if w is not None:
            why = _invalid(w, panel)
            if why:
                pf.reject(d, "*", f"invalid_weights: {why}")
            else:
                pending = dict(w)
        prev = d

    nav_df = pd.DataFrame(nav_rows).set_index("date")
    positions = pd.DataFrame(pos_rows).reindex(cal).fillna(0).astype("int64")
    return Result(nav_df, positions, pf.ledger(), config)


def _invalid(w: dict[str, float], panel: Panel) -> str | None:
    """權重格式檢查（§5 最後一條）。合法時回傳 None。"""
    for code, x in w.items():
        if code not in panel.kinds:
            return f"未知代號 {code}"
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return f"{code} 權重不是有限數值"
        if x < 0:
            return f"{code} 權重為負"
    if sum(w.values()) > 1 + 1e-9:
        return f"權重總和 {sum(w.values()):.4f} > 1"
    return None
