# -*- coding: utf-8 -*-
"""帳戶、撮合、成本（計劃 M1）。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sandbox.broker import Broker, Costs
from sandbox.data import Panel, RepoData
from sandbox.engine import Config, run
from sandbox.strategies import BuyAndHold, DoNothing, Scripted

from .conftest import CAL, CASH_0050, CASH_DAY, FLAT_DAY

D0 = pd.Timestamp("2020-01-06")


@pytest.fixture(scope="module")
def panel(data_dir) -> Panel:
    return Panel.build(RepoData(data_dir), "2019-12-01", "2020-03-31")


def nxt(panel, d):
    return panel.calendar[panel.calendar.get_loc(pd.Timestamp(d)) + 1]


def churn():
    """每天輪流換倉，盡量多產生買賣。"""
    targets = [{"1101": 0.45, "2330": 0.45}, {"0050": 0.9}, {"0050": 0.3, "2330": 0.3}, {}]
    return Scripted({d: targets[i % len(targets)] for i, d in enumerate(CAL)})


def check_accounting(res, panel):
    """會計恆等式（§7）：逐日、逐筆重算都要對得上。"""
    led = res.ledger
    moves = led[led["type"] != "reject"]
    trades = moves[moves["type"].isin(["buy", "sell"])]
    # 每筆的現金流
    buys, sells = trades[trades["type"] == "buy"], trades[trades["type"] == "sell"]
    np.testing.assert_allclose(buys["cash"], -(buys["gross"] + buys["fee"]))
    np.testing.assert_allclose(sells["cash"], sells["gross"] - sells["fee"] - sells["tax"])
    np.testing.assert_allclose(trades["gross"], trades["shares"] * trades["price"])
    # 期末現金 = 初始資金 + 帳冊現金流合計（含股利、零股折現、下市）
    assert res.nav["cash"].iloc[-1] == pytest.approx(res.config.initial_cash + moves["cash"].sum())
    # 逐日現金 + 持股市值 = 淨值
    close = panel.fields["close"].ffill().reindex(res.nav.index)
    held = (res.positions * close.reindex(columns=res.positions.columns)).sum(axis=1)
    np.testing.assert_allclose(res.nav["cash"] + held, res.nav["nav"])
    # 持股 = 帳冊股數累加（賣出為減少，公司行動與下市本身就是帶正負號的變動量）
    signed = moves.assign(n=np.where(moves["type"] == "sell", -moves["shares"], moves["shares"]))
    from_ledger = signed.pivot_table(index="date", columns="code", values="n", aggfunc="sum") \
        .reindex(res.nav.index).fillna(0).cumsum()
    pd.testing.assert_frame_equal(from_ledger.reindex(columns=res.positions.columns, fill_value=0)
                                  .astype("int64"), res.positions, check_names=False)
    assert (res.nav["cash"] >= -1e-6).all()


# ============================== 成本 ==============================


def test_costs():
    c = Costs()
    assert c.fee(100_000) == 85          # 100000 × 0.1425% × 0.6 = 85.5，捨去
    assert c.fee(10_000) == 20           # 8.55 → 下限 20
    assert c.fee(0) == 0
    assert c.tax(100_000, etf=False) == 300
    assert c.tax(100_000, etf=True) == 100
    assert c.exec_price(100, "buy") == pytest.approx(100.2)
    assert c.exec_price(100, "sell") == pytest.approx(99.8)
    z = Costs.zero()
    assert (z.fee(1e6), z.tax(1e6, False), z.exec_price(50, "buy")) == (0, 0, 50)


# ============================== 引擎 ==============================


def test_do_nothing(panel):
    res = run(DoNothing(), panel)
    assert (res.nav["nav"] == 1_000_000).all()
    assert res.ledger.empty


def test_fill_next_open(panel):
    res = run(Scripted({D0: {"0050": 0.5}}), panel)
    b = res.ledger.iloc[0]
    d1 = nxt(panel, D0)
    assert (b["type"], b["date"], b["code"]) == ("buy", d1, "0050")
    assert b["ref"] == panel.fields["open"].at[d1, "0050"]
    assert b["price"] == pytest.approx(b["ref"] * 1.002)
    assert b["shares"] % 1000 == 0 and b["shares"] > 0
    assert b["gross"] <= 0.5 * 1_000_000
    assert res.positions.loc[D0].sum() == 0          # 決策當天還沒成交


def test_fill_next_close(panel):
    res = run(Scripted({D0: {"0050": 0.5}}), panel, config=Config(fill="next_close"))
    b = res.ledger.iloc[0]
    assert b["ref"] == panel.fields["close"].at[nxt(panel, D0), "0050"]


def test_buy_and_hold_tracks_price(panel):
    cfg = Config(broker=Broker(costs=Costs.zero(), odd_lot=True))
    res = run(BuyAndHold({"0050": 1.0}), panel, config=cfg)
    n = res.ledger.iloc[0]["shares"]
    close = panel.fields["close"]["0050"]
    d1 = res.ledger.iloc[0]["date"]
    after = res.nav.loc[d1:, "nav"]
    div = np.where(close.loc[d1:].index >= CASH_DAY, n * CASH_0050, 0.0)
    np.testing.assert_allclose(after - after.iloc[0], n * (close.loc[d1:] - close.loc[d1]) + div)
    assert res.nav.loc[d1, "cash"] < close.loc[d1]     # 零股、零成本：剩不到一股的錢


def test_accounting_identity_with_churn(panel):
    res = run(churn(), panel)
    assert (res.ledger["type"] == "sell").sum() > 10
    check_accounting(res, panel)


def test_accounting_identity_odd_lot_full_investment(panel):
    """權重合計 1、零股、照算成本：買進金額一定超過現金，必須縮單而不是變負數。"""
    plan = {d: ({"1101": 0.5, "0050": 0.5} if i % 2 else {"2330": 1.0}) for i, d in enumerate(CAL)}
    res = run(Scripted(plan), panel, config=Config(broker=Broker(odd_lot=True)))
    check_accounting(res, panel)
    assert res.nav["cash"].min() >= 0


def test_min_fee_applied(panel):
    res = run(Scripted({D0: {"0050": 0.1}}), panel, config=Config(broker=Broker(odd_lot=True)))
    b = res.ledger.iloc[0]
    assert b["fee"] == max(math.floor(b["gross"] * 0.001425 * 0.6), 20)


def test_no_price_rejected(panel):
    """停牌日沒有價格：賣出拒單，持股以最近收盤價評價。"""
    res = run(Scripted({D0: {"1101": 0.2}, FLAT_DAY - pd.offsets.BDay(1): {}}), panel)
    rej = res.ledger[res.ledger["type"] == "reject"]
    assert list(rej["code"]) == ["1101"] and list(rej["reason"]) == ["no_price"]
    assert list(rej["date"]) == [FLAT_DAY]
    assert res.positions.at[FLAT_DAY, "1101"] > 0


@pytest.mark.parametrize("w, why", [
    ({"0050": float("nan")}, "有限數值"),
    ({"0050": -0.1}, "為負"),
    ({"0050": 0.6, "1101": 0.6}, "總和"),
    ({"XXXX": 0.1}, "未知代號"),
])
def test_invalid_weights_rejected(panel, w, why):
    res = run(Scripted({D0: w}), panel)
    assert list(res.ledger["type"]) == ["reject"]
    assert why in res.ledger.iloc[0]["reason"]
    assert (res.nav["nav"] == 1_000_000).all()


def test_bad_fill_mode():
    with pytest.raises(ValueError):
        Config(fill="same_close")
