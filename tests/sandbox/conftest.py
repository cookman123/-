# -*- coding: utf-8 -*-
"""
在暫存資料夾產生小型合成 data/，格式與本 repo 爬蟲輸出相同。

原始資料不在 repo 裡，所以單元測試全部用這份合成資料。每一檔先定義「真實的原始價」，
再照 Yahoo 的慣例做分割調整後寫檔；轉接器讀回來應該得到原本的原始價。

    1101  存活股，2020-02-03 配股每千股 100 股（分割比 1.1），2019、2020 兩個逐年檔，
          另有一列停牌補值的假列
    2330  存活股，2020-03-02 減資（前日收盤 50 → 參考價 100），2020-01-15 有一筆
          Yahoo 漏調整的錯價，靠 price_fixes.csv 修正
    2000  存活股，2020-02-03 才上市
    9999  下市股（證交所原始價），2020-02-14 為最後交易日
    0050  ETF，沒有公司行動
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CAL = pd.bdate_range("2019-12-02", "2020-03-31")
DIV_DAY = pd.Timestamp("2020-02-03")
RED_DAY = pd.Timestamp("2020-03-02")
FIX_DAY = pd.Timestamp("2020-01-15")
FLAT_DAY = pd.Timestamp("2020-01-08")
LIST_DAY = pd.Timestamp("2020-02-03")
DELIST_DAY = pd.Timestamp("2020-02-14")


def _walk(dates, start, seed) -> pd.DataFrame:
    """日漲跌在 ±3% 內的隨機價格，OHLC 自洽。"""
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.uniform(-0.03, 0.03, len(dates)))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, len(dates)))
    return pd.DataFrame({
        "open": open_.round(2),
        "high": (np.maximum(open_, close) * 1.01).round(2),
        "low": (np.minimum(open_, close) * 0.99).round(2),
        "close": close.round(2),
        "volume": rng.integers(1_000, 50_000, len(dates)) * 1000.0,
    }, index=pd.DatetimeIndex(dates, name="date"))


def _with_event(px: pd.DataFrame, day: pd.Timestamp, ratio: float) -> pd.DataFrame:
    """事件日起價格乘以 ratio（原始價在事件日的機械性跳動）。"""
    px = px.copy()
    after = px.index >= day
    px.loc[after, ["open", "high", "low", "close"]] *= ratio
    return px.round({"open": 2, "high": 2, "low": 2, "close": 2})


def _yahoo_adjust(px: pd.DataFrame, day: pd.Timestamp, s: float) -> pd.DataFrame:
    """Yahoo 的分割調整：事件日之前價格 ÷ s、成交量 × s（adjust_delisted.adjust）。"""
    px = px.copy()
    before = px.index < day
    px.loc[before, ["open", "high", "low", "close"]] /= s
    px.loc[before, "volume"] *= s
    return px


def _write(px: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    px.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")) \
        .to_csv(path, index=False, encoding="utf-8-sig")


@pytest.fixture(scope="session")
def raw_prices() -> dict[str, pd.DataFrame]:
    """每檔「真實的原始價」，測試用來比對轉接器的輸出。"""
    p1101 = _with_event(_walk(CAL, 100, 1), DIV_DAY, 1 / 1.1)
    p2330 = _walk(CAL, 50, 2)
    p2330 = _with_event(p2330, RED_DAY, 100 / p2330.loc[p2330.index < RED_DAY, "close"].iloc[-1])
    return {
        "1101": p1101,
        "2330": p2330,
        "2000": _walk(CAL[CAL >= LIST_DAY], 30, 3),
        "9999": _walk(CAL[CAL <= DELIST_DAY], 8, 4),
        "0050": _walk(CAL, 95, 5),
    }


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, raw_prices) -> Path:
    d = tmp_path_factory.mktemp("data")
    raw = raw_prices

    _write(_walk(CAL, 12000, 0), d / "market" / "TWII.csv")

    # 1101：配股。Yahoo 的分割比就是 1 + 每千股配股 / 1000
    prev = raw["1101"].loc[raw["1101"].index < DIV_DAY, "close"].iloc[-1]
    y1101 = _yahoo_adjust(raw["1101"], DIV_DAY, 1.1)
    flat = pd.DataFrame({"open": 77.0, "high": 77.0, "low": 77.0, "close": 77.0, "volume": 0.0},
                        index=pd.DatetimeIndex([FLAT_DAY], name="date"))
    y1101 = pd.concat([y1101.drop(FLAT_DAY), flat]).sort_index()
    for year in (2019, 2020):
        _write(y1101[y1101.index.year == year], d / "stocks" / "1101" / "price" / f"yahoo_1101_{year}.csv")

    # 2330：減資，外加一筆 Yahoo 漏調整（FIX_DAY 之前整段被除以 1.25）
    red_prev = raw["2330"].loc[raw["2330"].index < RED_DAY, "close"].iloc[-1]
    y2330 = _yahoo_adjust(raw["2330"], RED_DAY, red_prev / 100)
    y2330.loc[y2330.index < FIX_DAY, ["open", "high", "low", "close"]] /= 1.25
    y2330.loc[y2330.index < FIX_DAY, "volume"] *= 1.25
    _write(y2330, d / "stocks" / "2330" / "price" / "yahoo_2330_2020.csv")

    _write(raw["2000"], d / "stocks" / "2000" / "price" / "yahoo_2000_2020.csv")
    _write(raw["0050"], d / "market" / "0050.csv")
    _write(raw["9999"].assign(turnover=1.0), d / "delisted" / "price" / "9999.csv")

    pd.DataFrame([{"code": "2330", "date": FIX_DAY.strftime("%Y-%m-%d"), "factor": 1.25,
                   "ret": 0.25, "why": "測試"}]) \
        .to_csv(d / "price_fixes.csv", index=False, encoding="utf-8-sig")

    ca = d / "corp_actions"
    ca.mkdir()
    pd.DataFrame([
        {"date": DIV_DAY.strftime("%Y-%m-%d"), "code": "1101", "kind": "權", "prev_close": prev,
         "ref": round(prev / 1.1, 2), "cash": 0.0, "stock_per_1000": 100.0, "split": 1.1},
        # 下市股的事件：原始價不應被還原
        {"date": "2020-01-20", "code": "9999", "kind": "權", "prev_close": 0.0,
         "ref": 0.0, "cash": 0.0, "stock_per_1000": 50.0, "split": 1.05},
    ]).to_csv(ca / "exright.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"date": RED_DAY.strftime("%Y-%m-%d"), "code": "2330", "prev_close": red_prev,
         "ref": 100.0, "reason": "彌補虧損"},
    ]).to_csv(ca / "reduction.csv", index=False, encoding="utf-8-sig")
    return d
