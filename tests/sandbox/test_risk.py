# -*- coding: utf-8 -*-
"""風控（計劃 M3）。"""
from __future__ import annotations

import pandas as pd
import pytest

from sandbox.broker import Broker
from sandbox.data import Panel, RepoData
from sandbox.engine import Config, run
from sandbox.risk import RiskLimits
from sandbox.strategies import Scripted

from .conftest import LOCK_DOWN
from .test_engine import D0, check_accounting, churn

ODD = Broker(odd_lot=True)


@pytest.fixture(scope="module")
def panel(data_dir) -> Panel:
    return Panel.build(RepoData(data_dir), "2019-12-01", "2020-03-31")


def shift(panel, d, k):
    return panel.calendar[panel.calendar.get_loc(pd.Timestamp(d)) + k]


def test_stock_weight_capped(panel):
    res = run(Scripted({D0: {"1101": 0.5, "2330": 0.05}}), panel, config=Config(broker=ODD))
    notes = res.ledger.query("type == 'risk'")
    assert list(notes["code"]) == ["1101"]
    assert notes.iloc[0]["date"] == D0 and "0.5000 → 0.1000" in notes.iloc[0]["reason"]
    buy = res.ledger.query("type == 'buy' and code == '1101'").iloc[0]
    assert buy["gross"] <= 0.10 * 1_000_000


def test_etf_not_capped(panel):
    res = run(Scripted({D0: {"0050": 0.9}}), panel)
    assert "risk" not in set(res.ledger["type"])
    assert res.ledger.iloc[0]["gross"] > 0.8 * 1_000_000


def test_custom_cap(panel):
    cfg = Config(broker=ODD, risk=RiskLimits(max_stock_weight=0.3))
    res = run(Scripted({D0: {"1101": 0.5}}), panel, config=cfg)
    assert "→ 0.3000" in res.ledger.query("type == 'risk'").iloc[0]["reason"]


def test_daily_loss_halts_buys_but_allows_sells(panel):
    """
    3008 在 LOCK_DOWN 跌停約 −10%，持有 95% 的帳戶單日跌幅超過 5%：
    之後 3 個交易日的買進拒單（risk_halt），賣出照常；第 4 個交易日恢復。
    """
    d = LOCK_DOWN
    risk = RiskLimits(max_stock_weight=1.0, max_daily_loss=0.05, halt_days=3)
    plan = {
        D0: {"3008": 0.95},
        d: {"3008": 0.5, "0050": 0.4},             # d+1 成交：賣 3008 可以、買 0050 拒單
        shift(panel, d, 2): {"3008": 0.5, "0050": 0.4},   # d+3 成交：仍在停損期
        shift(panel, d, 3): {"3008": 0.5, "0050": 0.4},   # d+4 成交：恢復
    }
    res = run(Scripted(plan), panel, config=Config(broker=ODD, risk=risk))
    led = res.ledger

    halt = led.query("type == 'risk'")
    assert list(halt["date"]) == [d] and halt.iloc[0]["reason"].startswith("daily_loss_halt")
    assert res.nav.at[d, "nav"] / res.nav.at[shift(panel, d, -1), "nav"] - 1 < -0.05

    rej = led.query("type == 'reject'")
    assert list(rej["date"]) == [shift(panel, d, 1), shift(panel, d, 3)]
    assert set(rej["reason"]) == {"risk_halt"} and set(rej["code"]) == {"0050"}
    assert not led.query("type == 'sell' and code == '3008'").empty
    first_0050 = led.query("type == 'buy' and code == '0050'")["date"].min()
    assert first_0050 == shift(panel, d, 4)
    check_accounting(res, panel)


def test_no_halt_when_threshold_not_hit(panel):
    risk = RiskLimits(max_stock_weight=1.0, max_daily_loss=0.15, halt_days=3)
    res = run(Scripted({D0: {"3008": 0.95}}), panel, config=Config(risk=risk))
    assert "risk" not in set(res.ledger["type"])


def test_accounting_with_default_risk(panel):
    res = run(churn(), panel)
    assert (res.ledger["type"] == "risk").any()
    check_accounting(res, panel)
