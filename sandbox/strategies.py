# -*- coding: utf-8 -*-
"""
驗證用的假策略：答案已知，用來確認沙盒本身沒算錯（計劃 §7）。不含任何實際訊號。
"""
from __future__ import annotations

import numpy as np
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


class _Periodic:
    """每 every 個交易日換倉一次，其餘日子不調整。"""

    def __init__(self, every: int):
        if every < 1:
            raise ValueError("every 至少為 1")
        self.every = every
        self._i = 0

    def generate(self, date: pd.Timestamp, view: DataView) -> dict[str, float] | None:
        due = self._i % self.every == 0
        self._i += 1
        if not due:
            return None
        stocks = [c for c in view.universe() if view.kind(c) != "etf"]
        return self.pick(stocks) if stocks else None

    def pick(self, stocks: list[str]) -> dict[str, float]:
        raise NotImplementedError


class EqualWeight(_Periodic):
    """當日所有個股（含之後才下市的）等權。隨機策略的對照組。"""

    def pick(self, stocks):
        return {c: 1 / len(stocks) for c in stocks}


class RandomPick(_Periodic):
    """
    每次從當日個股隨機挑 n 檔等權持有。期望報酬約等於等權大盤減成本；
    跑出明顯的正超額報酬，代表沙盒有 bug（偷看未來、漏算成本）。
    """

    def __init__(self, n: int = 10, every: int = 20, seed: int = 0):
        super().__init__(every)
        self.n = n
        self.rng = np.random.default_rng(seed)

    def pick(self, stocks):
        chosen = self.rng.choice(stocks, size=min(self.n, len(stocks)), replace=False)
        return {str(c): 1 / len(chosen) for c in chosen}
