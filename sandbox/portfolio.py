# -*- coding: utf-8 -*-
"""
虛擬帳戶：現金、持股與帳冊。

帳冊是唯一的真相來源：每一筆成交、拒單、公司行動、下市都是一列，
現金與持股的任何變動都必須對得上帳冊，會計恆等式的測試就是逐列重算。

帳冊欄位
    date     成交日
    type     buy / sell / reject，以及 cash_dividend / stock_dividend / reduction / delisted
    code     代號
    shares   股數。買賣皆為正；其餘類型是持股的變動量（減少為負）
    ref      參考價（開盤或收盤，未含滑價；公司行動是參考價，下市是最後收盤）
    price    成交價（含滑價）
    gross    股數 × 成交價
    fee      手續費
    tax      證交稅
    cash     對現金的影響（買進為負）
    reason   拒單原因
"""
from __future__ import annotations

import math

import pandas as pd

LEDGER_COLS = ["date", "type", "code", "shares", "ref", "price", "gross",
               "fee", "tax", "cash", "reason"]


class Portfolio:
    def __init__(self, cash: float):
        if cash <= 0:
            raise ValueError("初始資金必須為正")
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.positions: dict[str, int] = {}
        self._ledger: list[dict] = []

    def buy(self, date, code: str, shares: int, ref: float, price: float, fee: float) -> None:
        gross = shares * price
        flow = -(gross + fee)
        if self.cash + flow < -1e-6:
            raise ValueError(f"[{code}] 現金不足：需要 {-flow:,.0f}，只有 {self.cash:,.0f}")
        self.cash += flow
        self.positions[code] = self.positions.get(code, 0) + shares
        self._record(date, "buy", code, shares, ref, price, gross, fee, 0.0, flow)

    def sell(self, date, code: str, shares: int, ref: float, price: float,
             fee: float, tax: float) -> None:
        have = self.positions.get(code, 0)
        if shares > have:
            raise ValueError(f"[{code}] 賣出 {shares} 股，但只持有 {have} 股")
        gross = shares * price
        flow = gross - fee - tax
        self.cash += flow
        self.positions[code] = have - shares
        if not self.positions[code]:
            del self.positions[code]
        self._record(date, "sell", code, shares, ref, price, gross, fee, tax, flow)

    def adjust(self, date, type_: str, code: str, shares: int, ref: float, cash: float) -> None:
        """公司行動與下市：持股變動 shares 股、現金變動 cash 元，不收手續費與稅。"""
        have = self.positions.get(code, 0)
        if have + shares < 0:
            raise ValueError(f"[{code}] {type_} 後持股為負")
        if not math.isfinite(cash):
            raise ValueError(f"[{code}] {type_} 的現金變動不是有限數值：{cash}")
        self.cash += cash
        self.positions[code] = have + shares
        if not self.positions[code]:
            del self.positions[code]
        self._record(date, type_, code, shares, ref, ref, 0.0, 0.0, 0.0, cash)

    def reject(self, date, code: str, reason: str) -> None:
        self._ledger.append({"date": date, "type": "reject", "code": code, "reason": reason})

    def value(self, prices: pd.Series) -> float:
        """現金 + 持股市值。持股缺價時報錯，由呼叫端決定用哪個價格補。"""
        held = 0.0
        for code, n in self.positions.items():
            px = prices.get(code)
            if px is None or pd.isna(px):
                raise ValueError(f"[{code}] 沒有評價用的價格")
            held += n * px
        return self.cash + held

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self._ledger, columns=LEDGER_COLS)

    def _record(self, date, type_, code, shares, ref, price, gross, fee, tax, flow) -> None:
        self._ledger.append({"date": date, "type": type_, "code": code, "shares": shares,
                             "ref": ref, "price": price, "gross": gross, "fee": fee,
                             "tax": tax, "cash": flow, "reason": None})
