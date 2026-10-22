# -*- coding: utf-8 -*-
"""
驗證用的假策略：答案已知，用來確認沙盒本身沒算錯（計劃 §7）。不含任何實際訊號。
"""
from __future__ import annotations

import pandas as pd

from .data import DataView


class DoNothing:
    """從不下單。淨值應逐日恆等於初始資金。"""

    def generate(self, date: pd.Timestamp, view: DataView) -> None:
        return None


class BuyAndHold:
    """第一天送出目標權重，之後不再調整。"""

    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)
        self._sent = False

    def generate(self, date: pd.Timestamp, view: DataView) -> dict[str, float] | None:
        if self._sent:
            return None
        self._sent = True
        return dict(self.weights)


class Scripted:
    """照日期表送出權重，表上沒有的日期不調整。測試撮合細節用。"""

    def __init__(self, plan: dict):
        self.plan = {pd.Timestamp(d): w for d, w in plan.items()}

    def generate(self, date: pd.Timestamp, view: DataView) -> dict[str, float] | None:
        w = self.plan.get(date)
        return None if w is None else dict(w)
