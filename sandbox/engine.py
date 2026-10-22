# -*- coding: utf-8 -*-
"""
主迴圈：策略 → 撮合 → 記帳（計劃 §3）。

    t 日收盤後   策略只看得到 ≤ t 的 DataView，回傳目標權重（或 None：不調整）
    t+1 日開盤前 已下市的持股以最後收盤價轉成現金；處理除權息、減資（§4.4）
    t+1 日       以開盤價（預設）或收盤價撮合；漲跌停鎖死的拒單
    t+1 日收盤   以收盤價評價淨值；停牌中沒有收盤價的持股用最近一次收盤價

風控（§5）在 M3 加入；這裡只擋掉格式不合法的權重。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from .broker import Broker, is_locked
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
    acts = defaultdict(list)
    for a in panel.actions.itertuples(index=False):
        acts[a.date].append(a)

    for d in cal:
        _delist(pf, panel, d, close_ff, prev)
        for a in acts.get(d, ()):
            _corporate_action(pf, d, a, close_ff.at[prev, a.code] if prev is not None else math.nan)
        if pending is not None:
            ref = panel.fields[ref_field].loc[d].dropna()
            # 決定目標金額用的淨值：當日參考價，沒有的用前一日收盤
            if ref_field == "open":
                mark = ref.reindex(close_ff.columns).fillna(close_ff.loc[prev])
            else:
                mark = close_ff.loc[d]
            no_buy, no_sell = _locked(panel, d, ref_field)
            config.broker.rebalance(pf, d, pending, ref, mark, panel.kinds, no_buy, no_sell)
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


def _delist(pf: Portfolio, panel: Panel, d, close_ff: pd.DataFrame, prev) -> None:
    """下市股在最後交易日之後的第一個交易日，以最後收盤價轉成現金（§2.3）。"""
    for code in [c for c in pf.positions if c in panel.last_day]:
        if d > panel.last_day[code]:
            n = pf.positions[code]
            px = close_ff.at[prev, code]
            pf.adjust(d, "delisted", code, -n, px, n * px)


def _corporate_action(pf: Portfolio, d, a, prev_close: float) -> None:
    """
    除權息、減資（§4.4）。持股以前一日收盤後的股數為準，在撮合之前處理。
    配股、減資後不足 1 股的部分以參考價折成現金；資料缺參考價時退而用前日收盤。
    """
    n = pf.positions.get(a.code, 0)
    if not n:
        return
    ref = a.ref if pd.notna(a.ref) else prev_close
    if a.cash > 0:
        pf.adjust(d, "cash_dividend", a.code, 0, ref, n * a.cash)
    if a.stock > 0:
        extra = n * a.stock
        whole = int(math.floor(extra + 1e-9))
        pf.adjust(d, "stock_dividend", a.code, whole, ref, _fraction_cash(extra - whole, ref))
    if pd.notna(a.reduce):
        after = n * a.reduce
        whole = int(math.floor(after + 1e-9))
        pf.adjust(d, "reduction", a.code, whole - n, ref, _fraction_cash(after - whole, ref))


def _fraction_cash(frac: float, ref: float) -> float:
    return frac * ref if frac > 1e-9 else 0.0


def _locked(panel: Panel, d, ref_field: str) -> tuple[frozenset, frozenset]:
    """
    當日漲停鎖死（買不到）與跌停鎖死（賣不掉）的標的（§4.2）。
    開盤成交：開盤 = 最高 = 漲停價；收盤成交：收盤 = 最高 = 漲停價。跌停反之。
    """
    f = panel.fields
    px, hi, lo = f[ref_field].loc[d], f["high"].loc[d], f["low"].loc[d]
    up, down = f["limit_up"].loc[d], f["limit_down"].loc[d]
    no_buy, no_sell = set(), set()
    for code in px.index[px.notna()]:
        etf = panel.kinds.get(code) == "etf"
        if is_locked(px[code], hi[code], up[code], etf):
            no_buy.add(code)
        if is_locked(px[code], lo[code], down[code], etf):
            no_sell.add(code)
    return frozenset(no_buy), frozenset(no_sell)


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
