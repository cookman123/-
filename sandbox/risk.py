# -*- coding: utf-8 -*-
"""
風控（計劃 §5）。寫死在沙盒裡，策略無法關掉或繞過。

    權重格式     NaN、負值、未知代號、總和 > 1（不融資）→ 整筆拒絕
    個股上限     單一個股的目標權重超過上限 → 截到上限；ETF 不受此限
    單日虧損     收盤淨值較前一日跌幅超過門檻 → 之後 halt_days 個交易日只准減倉

每一次介入都寫進帳冊（type = risk），不靜默修改。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_stock_weight: float = 0.10
    max_daily_loss: float = 0.05
    halt_days: int = 5

    @classmethod
    def none(cls) -> "RiskLimits":
        """不設個股上限與停損。只留權重格式檢查，測試撮合細節用。"""
        return cls(max_stock_weight=1.0, max_daily_loss=math.inf, halt_days=0)

    def invalid(self, w: dict[str, float], known) -> str | None:
        """權重格式檢查。合法時回傳 None。"""
        for code, x in w.items():
            if code not in known:
                return f"未知代號 {code}"
            if not isinstance(x, (int, float)) or not math.isfinite(x):
                return f"{code} 權重不是有限數值"
            if x < 0:
                return f"{code} 權重為負"
        if sum(w.values()) > 1 + 1e-9:
            return f"權重總和 {sum(w.values()):.4f} > 1"
        return None

    def cap(self, w: dict[str, float], kinds: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
        """把個股權重截到上限。回傳 (截過的權重, {被截的代號: 原權重})。"""
        out, clipped = {}, {}
        for code, x in w.items():
            if kinds.get(code) != "etf" and x > self.max_stock_weight + 1e-12:
                clipped[code] = x
                x = self.max_stock_weight
            out[code] = x
        return out, clipped

    def breached(self, nav: float, prev_nav: float) -> bool:
        """單日淨值跌幅是否超過門檻。"""
        return prev_nav > 0 and nav / prev_nav - 1 < -self.max_daily_loss
