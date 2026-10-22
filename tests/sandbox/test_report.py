# -*- coding: utf-8 -*-
"""績效報表、隨機策略與命令列（計劃 M4）。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sandbox import run as cli
from sandbox.broker import Broker, Costs
from sandbox.data import Panel, RepoData
from sandbox.engine import Config, run
from sandbox.report import metrics, write
from sandbox.strategies import BuyAndHold, DoNothing, EqualWeight, RandomPick

from .test_engine import churn

ZERO = Config(broker=Broker(costs=Costs.zero(), odd_lot=True))


@pytest.fixture(scope="module")
def panel(data_dir) -> Panel:
    return Panel.build(RepoData(data_dir), "2019-12-01", "2020-03-31")


def test_metrics_do_nothing(panel):
    m = metrics(run(DoNothing(), panel))
    assert m["總報酬"] == 0 and m["最大回撤"] == 0 and m["成本合計"] == 0
    assert m["成交筆數"] == 0 and m["拒單"] == {} and np.isnan(m["夏普"])


def test_metrics_by_hand(panel):
    """指標逐項以淨值序列和帳冊手算對照。"""
    res = run(churn(), panel)
    m = metrics(res)
    nav = res.nav["nav"]
    r = nav.pct_change()
    r.iloc[0] = nav.iloc[0] / 1_000_000 - 1
    years = len(nav) / 252
    assert m["總報酬"] == pytest.approx(nav.iloc[-1] / 1e6 - 1)
    assert m["年化報酬"] == pytest.approx((nav.iloc[-1] / 1e6) ** (1 / years) - 1)
    assert m["夏普"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    peak = np.maximum.accumulate(np.r_[1e6, nav.to_numpy()])[1:]
    assert m["最大回撤"] == pytest.approx((nav.to_numpy() / peak - 1).min())
    t = res.ledger.query("type in ['buy', 'sell']")
    assert m["手續費"] == t["fee"].sum() and m["證交稅"] == t["tax"].sum()
    assert m["滑價"] == pytest.approx(((t["price"] - t["ref"]).abs() * t["shares"]).sum())
    assert m["年化換手率"] == pytest.approx(t["gross"].sum() / 2 / nav.mean() / years)
    assert sum(m["風控介入"].values()) == (res.ledger["type"] == "risk").sum()


def test_beta_of_benchmark_is_one(panel):
    b = run(BuyAndHold({"0050": 1.0}), panel, config=ZERO).nav["nav"]
    res = run(BuyAndHold({"0050": 1.0}), panel, config=ZERO)
    m = metrics(res, b)
    assert m["beta"] == pytest.approx(1.0)
    assert m["年化 alpha"] == pytest.approx(0.0, abs=1e-12)
    assert m["超額年化報酬"] == pytest.approx(0.0, abs=1e-12)


def test_cash_has_zero_beta(panel):
    b = run(BuyAndHold({"0050": 1.0}), panel, config=ZERO).nav["nav"]
    m = metrics(run(DoNothing(), panel), b)
    assert m["beta"] == pytest.approx(0.0, abs=1e-12)


def test_random_is_deterministic_and_costs_hurt(panel):
    def go(seed, cfg):
        return run(RandomPick(n=3, every=5, seed=seed), panel, config=cfg)

    a, b = go(7, ZERO), go(7, ZERO)
    pd.testing.assert_frame_equal(a.ledger, b.ledger)
    costly = go(7, Config(broker=Broker(odd_lot=True)))
    # 同一個 seed 挑到的股票相同，差別只在成本
    assert set(costly.ledger.query("type == 'buy'")["code"]) == set(a.ledger.query("type == 'buy'")["code"])
    assert costly.nav["nav"].iloc[-1] < a.nav["nav"].iloc[-1]
    assert metrics(costly)["成本合計"] > 0


def test_random_close_to_equal_weight(panel):
    """
    零成本下，隨機挑股的平均結果應接近等權持有全部個股（§7）。
    合成資料只有 5 檔，用多個 seed 平均，容許 3 個百分點。
    """
    ew = run(EqualWeight(every=5), panel, config=ZERO).nav["nav"].iloc[-1]
    rnd = np.mean([run(RandomPick(n=2, every=5, seed=s), panel, config=ZERO).nav["nav"].iloc[-1]
                   for s in range(40)])
    assert abs(rnd / ew - 1) < 0.03


def test_strategies_only_pick_stocks(panel):
    res = run(EqualWeight(every=5), panel, config=ZERO)
    assert "0050" not in set(res.ledger["code"])


def test_write_outputs(panel, tmp_path):
    b = run(BuyAndHold({"0050": 1.0}), panel).nav["nav"]
    res = run(churn(), panel)
    m = write(res, tmp_path / "churn", "churn", b, "0050", notes=["測試附註"])
    out = tmp_path / "churn"
    assert {p.name for p in out.iterdir()} == {"report.md", "config.json", "summary.csv",
                                               "nav.csv", "positions.csv", "ledger.csv"}
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "beta" in md and "測試附註" in md and "不得被引用" in md
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["broker"]["costs"]["fee_discount"] == 0.6
    s = pd.read_csv(out / "summary.csv", encoding="utf-8-sig")
    assert s.loc[0, "期末淨值"] == pytest.approx(m["期末淨值"])


def test_cli_end_to_end(data_dir, tmp_path, capsys):
    rc = cli.main(["--strategy", "random", "--n", "2", "--every", "5", "--data", str(data_dir),
                   "--out", str(tmp_path), "--start", "2019-12-01", "--end", "2020-03-31"])
    assert rc == 0
    md = (tmp_path / "random" / "report.md").read_text(encoding="utf-8")
    assert "買進持有 0050" in md and "資料稽核" in md
    assert "beta" in capsys.readouterr().out


def test_cli_without_data(tmp_path, capsys):
    assert cli.main(["--data", str(tmp_path)]) == 1
