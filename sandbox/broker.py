# -*- coding: utf-8 -*-
"""
台股交易規則：升降單位、成本、把目標權重換算成委託並撮合。

規格見 docs/SANDBOX_PLAN.md §4。公司行動與下市在 engine.py 處理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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


# 漲跌幅限制：2015-06-01 起由 ±7% 放寬為 ±10%
LIMIT_CHANGE_DAY = pd.Timestamp("2015-06-01")


def limit_prices(base: float, date, etf: bool = False) -> tuple[float, float]:
    """
    漲停價、跌停價（§4.2）。base 是前日收盤，除權息、減資恢復買賣日是參考價。
    漲停往下、跌停往上取到升降單位；升降單位依取整前的價位決定。
    """
    up, down = limit_series(pd.Series([base], index=[pd.Timestamp(date)]), etf)
    return float(up.iloc[0]), float(down.iloc[0])


def limit_series(base: pd.Series, etf: bool = False) -> tuple[pd.Series, pd.Series]:
    """limit_prices 的向量版。base 以日期為索引，缺值的日期（例如上市首日）沒有漲跌停。"""
    r = np.where(base.index < LIMIT_CHANGE_DAY, 0.07, 0.10)
    up, down = base * (1 + r), base * (1 - r)
    tu, td = _tick_vec(up, etf), _tick_vec(down, etf)
    return (np.floor(up / tu + 1e-9) * tu).round(4), (np.ceil(down / td - 1e-9) * td).round(4)


def _tick_vec(price: pd.Series, etf: bool) -> pd.Series:
    table = _ETF_TICKS if etf else _STOCK_TICKS
    out = pd.Series(0.05 if etf else 5.0, index=price.index)
    for cap, tick in reversed(table):
        out[price < cap] = tick
    return out


def is_locked(price: float, extreme: float, limit: float, etf: bool = False) -> bool:
    """成交價與當日最高（或最低）都等於漲停（或跌停）價：整天鎖住，日資料下的近似。"""
    if pd.isna(limit) or pd.isna(price) or pd.isna(extreme):
        return False
    half = tick_size(limit, etf) / 2
    return abs(price - limit) < half and abs(extreme - limit) < half


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
                  ref: pd.Series, mark: pd.Series, kinds: dict[str, str],
                  no_buy: dict[str, str] | None = None, no_sell: dict[str, str] | None = None) -> None:
        """
        以 ref 價（開盤或收盤）把持股調到目標權重。

        weights：未列出的持股目標為 0。
        ref：當日可成交的參考價，缺值的標的拒單 no_price。
        mark：計算淨值用的價格（當日 ref，缺值時用最近收盤），決定目標金額。
        no_buy／no_sell：{代號: 原因}，當日不准買進／賣出的標的（漲跌停鎖死、風控停損），
            拒單時記下原因。
        先賣後買；買進的錢不夠時依比例縮單，不允許負現金。
        """
        no_buy, no_sell = no_buy or {}, no_sell or {}
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
                if code in no_sell:
                    pf.reject(date, code, no_sell[code])
                else:
                    sells.append((code, have - want, px))
            elif want > have:
                if code in no_buy:
                    pf.reject(date, code, no_buy[code])
                else:
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
