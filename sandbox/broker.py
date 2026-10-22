# -*- coding: utf-8 -*-
"""
台股交易規則：升降單位、成本、把目標權重換算成委託並撮合。

規格見 docs/SANDBOX_PLAN.md §4。漲跌停鎖死、公司行動在 M2 加入。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .portfolio import Portfolio

# (價格上限, 升降單位)，價格 < 上限時適用
_STOCK_TICKS = ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0))
_ETF_TICKS = ((50, 0.01),)


def tick_size(price: float, etf: bool = False) -> float:
    """證交所的升降單位。股票六級、ETF 兩級。"""
    for cap, tick in (_ETF_TICKS if etf else _STOCK_TICKS):
        if price < cap:
            return tick
    return 0.05 if etf else 5.0


# ============================== 成本 ==============================


@dataclass(frozen=True)
class Costs:
    """
    交易成本（§4.3）。手續費與證交稅照券商實務以元為單位無條件捨去，
    手續費再套下限。fee_min 的預設值 20 元還待確認（§8.2）。
    """

    fee_rate: float = 0.001425
    fee_discount: float = 0.6
    fee_min: float = 20.0
    tax_stock: float = 0.003
    tax_etf: float = 0.001
    slippage: float = 0.002

    @classmethod
    def zero(cls) -> "Costs":
        return cls(fee_rate=0.0, fee_min=0.0, tax_stock=0.0, tax_etf=0.0, slippage=0.0)

    def fee(self, gross: float) -> float:
        if gross <= 0:
            return 0.0
        return max(float(math.floor(gross * self.fee_rate * self.fee_discount)), self.fee_min)

    def tax(self, gross: float, etf: bool) -> float:
        return float(math.floor(gross * (self.tax_etf if etf else self.tax_stock)))

    def exec_price(self, ref: float, side: str) -> float:
        """滑價：買價上調、賣價下調。"""
        return ref * (1 + self.slippage) if side == "buy" else ref * (1 - self.slippage)


# ============================== 撮合 ==============================


@dataclass(frozen=True)
class Broker:
    """
    把目標權重換算成股數並成交（§4.1、§4.2）。

    lot：交易單位。整股 1000 股；odd_lot=True 時以 1 股為單位（零股）。
    """

    costs: Costs = Costs()
    lot: int = 1000
    odd_lot: bool = False

    @property
    def unit(self) -> int:
        return 1 if self.odd_lot else self.lot

    def rebalance(self, pf: Portfolio, date: pd.Timestamp, weights: dict[str, float],
                  ref: pd.Series, mark: pd.Series, kinds: dict[str, str]) -> None:
        """
        以 ref 價（開盤或收盤）把持股調到目標權重。

        weights：未列出的持股目標為 0。
        ref：當日可成交的參考價，缺值的標的拒單 no_price。
        mark：計算淨值用的價格（當日 ref，缺值時用最近收盤），決定目標金額。
        先賣後買；買進的錢不夠時依比例縮單，不允許負現金。
        """
        nav = pf.value(mark)
        codes = sorted(set(weights) | {c for c, n in pf.positions.items() if n})
        sells, buys = [], []
        for code in codes:
            have = pf.positions.get(code, 0)
            px = ref.get(code)
            w = weights.get(code, 0.0)
            if px is None or not px > 0:
                if have or w > 0:
                    pf.reject(date, code, "no_price")
                continue
            side_px = self.costs.exec_price(px, "buy")
            want = self._round(w * nav / side_px)
            if want < have:
                sells.append((code, have - want, px))
            elif want > have:
                buys.append((code, want - have, px))

        for code, n, px in sells:
            gross = n * self.costs.exec_price(px, "sell")
            etf = kinds.get(code) == "etf"
            pf.sell(date, code, n, px, self.costs.exec_price(px, "sell"),
                    self.costs.fee(gross), self.costs.tax(gross, etf))

        need = sum(self._buy_cost(n, px) for _, n, px in buys)
        if need > pf.cash and need > 0:
            scale = pf.cash / need
            buys = [(c, self._round(n * scale), px) for c, n, px in buys]
        for code, n, px in buys:
            # 縮單後仍可能因手續費下限、取整差一點，逐筆退到買得起為止
            while n > 0 and self._buy_cost(n, px) > pf.cash:
                n -= self.unit
            if n <= 0:
                pf.reject(date, code, "insufficient_cash")
                continue
            exec_px = self.costs.exec_price(px, "buy")
            pf.buy(date, code, n, px, exec_px, self.costs.fee(n * exec_px))

    def _round(self, shares: float) -> int:
        return int(math.floor(shares / self.unit + 1e-9)) * self.unit

    def _buy_cost(self, n: int, px: float) -> float:
        gross = n * self.costs.exec_price(px, "buy")
        return gross + self.costs.fee(gross)
