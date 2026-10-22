# -*- coding: utf-8 -*-
"""資料轉接器與防偷看資料視圖（計劃 M0）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sandbox.broker import tick_size
from sandbox.data import DataView, LookaheadError, Panel, RepoData, audit, unadjust

from .conftest import CAL, DELIST_DAY, DIV_DAY, FLAT_DAY, LIST_DAY

ROOT = Path(__file__).resolve().parents[2]
PRICE_COLS = ["open", "high", "low", "close"]


@pytest.fixture(scope="module")
def data(data_dir) -> RepoData:
    return RepoData(data_dir)


@pytest.fixture(scope="module")
def panel(data) -> Panel:
    return Panel.build(data, "2019-12-01", "2020-03-31")


# ============================== 轉接器 ==============================


def test_codes(data):
    assert data.codes() == {"1101": "stock", "2000": "stock", "2330": "stock",
                            "9999": "delisted", "0050": "etf"}


def test_unadjust_inverts_yahoo_adjust():
    idx = pd.bdate_range("2020-01-01", periods=10)
    raw = pd.DataFrame({c: np.linspace(10, 20, 10) for c in PRICE_COLS} | {"volume": 1000.0}, index=idx)
    ev = pd.DataFrame({"date": [idx[3], idx[7]], "s": [1.1, 0.5]})
    adj = raw.copy()
    for _, e in ev.iterrows():
        before = adj.index < e["date"]
        adj.loc[before, PRICE_COLS] /= e["s"]
        adj.loc[before, "volume"] *= e["s"]
    pd.testing.assert_frame_equal(unadjust(adj, ev), raw)


def test_unadjust_ignores_events_outside_data():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({c: 10.0 for c in PRICE_COLS} | {"volume": 1000.0}, index=idx)
    ev = pd.DataFrame({"date": [idx[0], idx[-1] + pd.Timedelta(days=3)], "s": [2.0, 2.0]})
    pd.testing.assert_frame_equal(unadjust(px, ev), px)


@pytest.mark.parametrize("code", ["1101", "2330", "2000", "0050"])
def test_load_returns_raw_prices(data, raw_prices, code):
    kind = data.codes()[code]
    got = data.load(code, kind)
    want = raw_prices[code].drop(FLAT_DAY, errors="ignore") if code == "1101" else raw_prices[code]
    pd.testing.assert_frame_equal(got[PRICE_COLS], want[PRICE_COLS], atol=1e-6, check_names=False, check_freq=False)
    np.testing.assert_allclose(got["volume"], want["volume"])


def test_flat_row_dropped(data):
    assert FLAT_DAY not in data.load("1101", "stock").index


def test_delisted_not_unadjusted(data, raw_prices):
    got = data.load("9999", "delisted")
    pd.testing.assert_frame_equal(got[PRICE_COLS], raw_prices["9999"][PRICE_COLS], check_names=False, check_freq=False)


def test_audit_clean(data):
    assert audit(data).empty


def test_audit_catches_missing_event(data_dir, tmp_path):
    """exright.csv 少了一筆配股時，還原不回去，稽核要抓到。"""
    import shutil

    d = tmp_path / "data"
    shutil.copytree(data_dir, d)
    ex = pd.read_csv(d / "corp_actions" / "exright.csv", dtype={"code": str})
    ex.assign(split=1.0).to_csv(d / "corp_actions" / "exright.csv", index=False, encoding="utf-8-sig")
    bad = audit(RepoData(d), {"1101": "stock"})
    assert (bad["check"] == "prev_close").any()


def test_tick_size():
    assert [tick_size(p) for p in (9.99, 10, 49.9, 50, 99, 100, 499, 500, 999, 1000)] == \
        [0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 5.0]
    assert [tick_size(p, etf=True) for p in (49.9, 50, 200)] == [0.01, 0.05, 0.05]


# ============================== 資料視圖 ==============================


def test_view_stops_at_asof(panel):
    asof = pd.Timestamp("2020-02-10")
    close = DataView(panel, asof).field("close")
    assert close.index.max() == asof
    assert DataView(panel, asof).calendar().max() == asof


def test_cheat_strategy_raises(panel):
    """作弊策略：在 t 日讀 t+1 的價格。必須報錯，不能回傳空值。"""
    asof = pd.Timestamp("2020-02-10")
    nxt = panel.calendar[panel.calendar.get_loc(asof) + 1]
    view = DataView(panel, asof)
    with pytest.raises(LookaheadError):
        view.field("open", start=nxt, end=nxt)
    with pytest.raises(LookaheadError):
        view.field("close", end="2020-12-31")


def test_view_returns_copy(panel):
    asof = pd.Timestamp("2020-02-10")
    close = DataView(panel, asof).field("close")
    close.loc[:, :] = -1
    assert (DataView(panel, asof).field("close").fillna(0) >= 0).all().all()


def test_universe_point_in_time(panel):
    before_list = DataView(panel, LIST_DAY - pd.offsets.BDay(1)).universe()
    assert "2000" not in before_list and "9999" in before_list
    assert "2000" in DataView(panel, LIST_DAY).universe()
    after_delist = DataView(panel, DELIST_DAY + pd.offsets.BDay(1)).universe()
    assert "9999" not in after_delist


def test_view_rejects_non_trading_day(panel):
    with pytest.raises(ValueError):
        DataView(panel, pd.Timestamp("2020-02-08"))   # 星期六


def test_panel_on_calendar(panel):
    assert panel.calendar.equals(pd.DatetimeIndex(CAL))
    assert DIV_DAY in panel.calendar


# ============================== 與 load_price 對照（需要本機資料） ==============================


def test_matches_load_price_on_real_data():
    """轉接器還原前的價格要與 davis_tw_train.load_price 完全一致（計劃 §1）。"""
    stocks = ROOT / "data" / "stocks"
    if not stocks.exists():
        pytest.skip("本機沒有 data/stocks/")
    sys.path.insert(0, str(ROOT))
    T = pytest.importorskip("davis_tw_train")
    data = RepoData(ROOT / "data")
    codes = [c for c, k in data.codes().items() if k == "stock"][:30]
    for code in codes:
        want = T.load_price(code)
        got = data.load_adjusted(code, "stock")
        pd.testing.assert_frame_equal(got, want, check_names=False, check_freq=False)
