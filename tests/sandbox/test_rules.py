# -*- coding: utf-8 -*-
"""台股規則：漲跌停、停牌、下市、公司行動（計劃 M2）。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sandbox.broker import Broker, Costs, limit_prices
from sandbox.data import Panel, RepoData
from sandbox.engine import Config
from sandbox.strategies import BuyAndHold, Scripted

from .conftest import CASH_0050, CASH_DAY, DELIST_DAY, DIV_DAY, LOCK_DOWN, LOCK_UP, RED_DAY
from .test_engine import D0, check_accounting, run

ODD = Config(broker=Broker(odd_lot=True))


@pytest.fixture(scope="module")
def panel(data_dir) -> Panel:
    return Panel.build(RepoData(data_dir), "2019-12-01", "2020-03-31")


def prev_day(panel, d):
    return panel.calendar[panel.calendar.get_loc(pd.Timestamp(d)) - 1]


# ============================== 漲跌停 ==============================


@pytest.mark.parametrize("base, date, etf, want", [
    (100, "2020-01-02", False, (110.0, 90.0)),
    (100, "2015-05-29", False, (107.0, 93.0)),      # 2015-06-01 之前 ±7%
    (100, "2015-06-01", False, (110.0, 90.0)),
    (9.5, "2020-01-02", False, (10.45, 8.55)),      # 漲停價跨進 0.05 的級距
    (51.3, "2020-01-02", False, (56.4, 46.2)),      # 56.43 → 56.4、46.17 → 46.2
    (48, "2020-01-02", True, (52.8, 43.2)),
])
def test_limit_prices(base, date, etf, want):
    assert limit_prices(base, date, etf) == pytest.approx(want)


def test_limit_uses_ref_on_ex_date(panel):
    """除權息日的漲跌停以參考價為基準，不是前日收盤。"""
    ref = panel.actions.query("code == '1101'")["ref"].iloc[0]
    assert panel.fields["limit_up"].at[DIV_DAY, "1101"] == pytest.approx(limit_prices(ref, DIV_DAY)[0])
    prev_close = panel.fields["close"].at[prev_day(panel, DIV_DAY), "1101"]
    assert limit_prices(prev_close, DIV_DAY)[0] != limit_prices(ref, DIV_DAY)[0]


def test_limit_up_locked_rejects_buy(panel):
    res = run(Scripted({prev_day(panel, LOCK_UP): {"3008": 0.5}}), panel)
    led = res.ledger
    assert list(led["type"]) == ["reject"]
    assert (led.iloc[0]["date"], led.iloc[0]["reason"]) == (LOCK_UP, "limit_up_locked")


def test_limit_down_locked_rejects_sell(panel):
    res = run(Scripted({D0: {"3008": 0.2}, LOCK_UP: {}}), panel)
    rej = res.ledger[res.ledger["type"] == "reject"]
    assert list(rej["reason"]) == ["limit_down_locked"] and list(rej["date"]) == [LOCK_DOWN]
    assert res.positions.at[LOCK_DOWN, "3008"] > 0


def test_limit_lock_not_triggered_on_normal_day(panel):
    res = run(Scripted({D0: {"3008": 0.2}}), panel)
    assert list(res.ledger["type"]) == ["buy"]


# ============================== 下市 ==============================


def test_delisted_position_liquidated(panel):
    res = run(BuyAndHold({"9999": 0.5}), panel)
    led = res.ledger
    row = led[led["type"] == "delisted"].iloc[0]
    n = res.positions.at[DELIST_DAY, "9999"]
    last_close = panel.fields["close"].at[DELIST_DAY, "9999"]
    assert row["date"] == panel.calendar[panel.calendar > DELIST_DAY][0]
    assert (row["shares"], row["ref"]) == (-n, last_close)
    assert row["cash"] == pytest.approx(n * last_close)
    assert res.positions["9999"].iloc[-1] == 0
    check_accounting(res, panel)


# ============================== 公司行動 ==============================


def test_stock_dividend(panel):
    """配股每千股 100 股：整股持股剛好乘以 1.1，淨值在除權前後連續。"""
    res = run(BuyAndHold({"1101": 0.5}), panel)
    before = res.positions.at[prev_day(panel, DIV_DAY), "1101"]
    row = res.ledger.query("type == 'stock_dividend'").iloc[0]
    assert row["date"] == DIV_DAY
    assert row["shares"] == before // 10 and row["cash"] == 0
    assert res.positions.at[DIV_DAY, "1101"] == before + before // 10
    check_accounting(res, panel)


def test_stock_dividend_fraction_paid_in_cash(panel):
    res = run(BuyAndHold({"1101": 0.5}), panel, config=ODD)
    before = res.positions.at[prev_day(panel, DIV_DAY), "1101"]
    row = res.ledger.query("type == 'stock_dividend'").iloc[0]
    ref = panel.actions.query("code == '1101'")["ref"].iloc[0]
    assert row["shares"] == math.floor(before * 0.1)
    assert row["cash"] == pytest.approx((before * 0.1 - math.floor(before * 0.1)) * ref)
    check_accounting(res, panel)


def test_reduction(panel):
    res = run(BuyAndHold({"2330": 0.5}), panel, config=ODD)
    before = res.positions.at[prev_day(panel, RED_DAY), "2330"]
    a = panel.actions.query("code == '2330'").iloc[0]
    row = res.ledger.query("type == 'reduction'").iloc[0]
    after = before * a["reduce"]
    assert row["date"] == RED_DAY
    assert row["shares"] == math.floor(after) - before
    assert row["cash"] == pytest.approx((after - math.floor(after)) * a["ref"])
    # 減資前後市值相等（前日收盤 × 舊股數 = 參考價 × 新股數 + 零股現金）
    prev_close = panel.fields["close"].at[prev_day(panel, RED_DAY), "2330"]
    assert before * prev_close == pytest.approx(
        res.positions.at[RED_DAY, "2330"] * a["ref"] + row["cash"])
    check_accounting(res, panel)


def test_cash_dividend_and_0050_reconciliation(panel):
    """
    買進持有 0050、成本設為零（§7）：期末淨值 = 剩餘現金 + 股數 × 期末收盤 + 股數 × 每股股利，
    一分不差。這驗證的是沙盒的會計；資料本身的對錯要另外和外部公布的數字比。
    """
    cfg = Config(broker=Broker(costs=Costs.zero(), odd_lot=True))
    res = run(BuyAndHold({"0050": 1.0}), panel, config=cfg)
    buy = res.ledger.query("type == 'buy'").iloc[0]
    div = res.ledger.query("type == 'cash_dividend'").iloc[0]
    n = buy["shares"]
    assert div["date"] == CASH_DAY and div["cash"] == pytest.approx(n * CASH_0050)
    close = panel.fields["close"]["0050"]
    want = 1_000_000 - n * buy["price"] + n * close.iloc[-1] + n * CASH_0050
    assert res.nav["nav"].iloc[-1] == pytest.approx(want)
    # 除息日淨值不會因為股價機械性下跌而掉下來
    d = prev_day(panel, CASH_DAY)
    jump = res.nav.at[CASH_DAY, "nav"] - res.nav.at[d, "nav"]
    assert jump == pytest.approx(n * (close[CASH_DAY] - close[d] + CASH_0050))


def test_0050_with_costs_differs_by_buy_cost(panel):
    """成本照算：與零成本版的差距 = 一次買進的手續費 + 滑價，可逐項對帳（§7）。"""
    zero = run(BuyAndHold({"0050": 1.0}), panel,
               config=Config(broker=Broker(costs=Costs.zero(), odd_lot=True)))
    cost = run(BuyAndHold({"0050": 1.0}), panel, config=ODD)
    bz, bc = zero.ledger.iloc[0], cost.ledger.iloc[0]
    # 股數不同（成本吃掉一點現金），所以比較的是「同股數下」的差距
    n = bc["shares"]
    close_end = panel.fields["close"]["0050"].iloc[-1]
    want = 1_000_000 - n * bc["price"] - bc["fee"] + n * (close_end + CASH_0050)
    assert cost.nav["nav"].iloc[-1] == pytest.approx(want)
    assert bc["price"] - bz["price"] == pytest.approx(bz["price"] * 0.002)
    assert bc["fee"] == max(math.floor(bc["gross"] * 0.001425 * 0.6), 20)


def test_accounting_with_all_events(panel):
    """持有每一檔、每週換倉，跑過配股、減資、除息、下市、漲跌停，恆等式都要成立。"""
    codes = ["1101", "2330", "3008", "9999", "0050"]
    plan = {d: {c: 0.18 for c in codes} if i % 5 == 0 else ({"0050": 0.5} if i % 5 == 3 else None)
            for i, d in enumerate(panel.calendar)}
    plan = {d: w for d, w in plan.items() if w is not None}
    for cfg in (Config(), ODD):
        res = run(Scripted(plan), panel, config=cfg)
        assert set(res.ledger["type"]) >= {"buy", "sell"}
        check_accounting(res, panel)
