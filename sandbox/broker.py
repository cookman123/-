# -*- coding: utf-8 -*-
"""
台股交易規則。

目前只有升降單位；撮合、漲跌停與成本在 M1、M2 加入（見 docs/SANDBOX_PLAN.md §4）。
"""
from __future__ import annotations

# (價格上限, 升降單位)，價格 < 上限時適用
_STOCK_TICKS = ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0))
_ETF_TICKS = ((50, 0.01),)


def tick_size(price: float, etf: bool = False) -> float:
    """證交所的升降單位。股票六級、ETF 兩級。"""
    for cap, tick in (_ETF_TICKS if etf else _STOCK_TICKS):
        if price < cap:
            return tick
    return 0.05 if etf else 5.0
