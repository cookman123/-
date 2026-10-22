# -*- coding: utf-8 -*-
"""
沙盒命令列入口：跑一個驗證用策略，並以買進持有 0050（同一套成本與規則）為基準出報表。

    python -m sandbox.run --strategy random --start 2011-01-01 --end 2024-12-31
    python -m sandbox.run --help

產物寫到 results/sandbox/<名稱>/：report.md、config.json、summary.csv 會上傳，
nav.csv、positions.csv、ledger.csv 只留本機（見 .gitignore）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from .broker import Broker, Costs
from .data import ROOT, Panel, RepoData, audit
from .engine import Config, run
from .report import write
from .risk import RiskLimits
from .strategies import BuyAndHold, DoNothing, EqualWeight, RandomPick

BENCH = "0050"


def make_strategy(a):
    if a.strategy == "do_nothing":
        return DoNothing()
    if a.strategy == "buy_and_hold":
        return BuyAndHold({BENCH: 1.0})
    if a.strategy == "equal_weight":
        return EqualWeight(every=a.every)
    if a.strategy == "random":
        return RandomPick(n=a.n, every=a.every, seed=a.seed)
    raise ValueError(a.strategy)


def main(argv=None) -> int:
    import argparse

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="模擬交易沙盒：跑驗證用策略並出報表")
    ap.add_argument("--strategy", default="random",
                    choices=["do_nothing", "buy_and_hold", "equal_weight", "random"])
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--name", default=None, help="執行名稱（預設為策略名稱）")
    ap.add_argument("--data", default=None, help="資料夾（預設 repo 的 data/）")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 results/sandbox/）")
    ap.add_argument("--cash", type=float, default=1_000_000)
    ap.add_argument("--fill", default="next_open", choices=["next_open", "next_close"])
    ap.add_argument("--odd-lot", action="store_true", help="以零股為單位（預設整股 1000 股）")
    ap.add_argument("--fee-discount", type=float, default=0.6)
    ap.add_argument("--fee-min", type=float, default=20.0)
    ap.add_argument("--slippage", type=float, default=0.002)
    ap.add_argument("--zero-cost", action="store_true", help="手續費、稅、滑價全部設為 0")
    ap.add_argument("--max-stock-weight", type=float, default=0.10)
    ap.add_argument("--max-daily-loss", type=float, default=0.05)
    ap.add_argument("--halt-days", type=int, default=5)
    ap.add_argument("--n", type=int, default=10, help="random：每次挑幾檔")
    ap.add_argument("--every", type=int, default=20, help="equal_weight／random：每幾個交易日換倉")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-audit", action="store_true", help="略過資料稽核（§2.2）")
    a = ap.parse_args(argv)

    data = RepoData(a.data)
    if not data.codes():
        print(f"{data.dir} 底下沒有任何價格檔；請先照 README 建好資料")
        return 1
    costs = Costs.zero() if a.zero_cost else Costs(
        fee_discount=a.fee_discount, fee_min=a.fee_min, slippage=a.slippage)
    config = Config(initial_cash=a.cash, fill=a.fill,
                    broker=Broker(costs=costs, odd_lot=a.odd_lot),
                    risk=RiskLimits(a.max_stock_weight, a.max_daily_loss, a.halt_days))

    print(f"建立面板 {a.start} ～ {a.end} …", flush=True)
    panel = Panel.build(data, a.start, a.end)
    print(f"  {len(panel.calendar):,} 個交易日 × {len(panel.kinds):,} 檔", flush=True)

    notes = []
    if not a.no_audit:
        bad = audit(data, panel.kinds)
        notes.append(f"資料稽核（§2.2）：前日收盤不符 {int((bad['check'] == 'prev_close').sum())} 筆、"
                     f"非事件日漲跌超過 ±10.5% {int((bad['check'] == 'jump').sum())} 筆。"
                     + ("" if bad.empty else "未處理前，報酬可能受資料錯誤影響。"))
    else:
        notes.append("本次略過資料稽核。")
    if a.strategy == "random":
        notes.append(f"隨機策略：每 {a.every} 個交易日從當日個股隨機挑 {a.n} 檔等權，seed {a.seed}。"
                     "明顯的正 alpha 代表沙盒有 bug，不代表策略有效。")

    name = a.name or a.strategy
    out = Path(a.out) if a.out else ROOT / "results" / "sandbox"
    print(f"執行 {name} …", flush=True)
    res = run(make_strategy(a), panel, config=config)
    bench = None
    if BENCH in panel.kinds and a.strategy != "buy_and_hold":
        bench = run(BuyAndHold({BENCH: 1.0}), panel, config=config).nav["nav"]
    m = write(res, out / name, name, bench, f"買進持有 {BENCH}（同一套成本與規則）", notes)

    print(f"  年化報酬 {m['年化報酬']:+.2%}、最大回撤 {m['最大回撤']:+.2%}、夏普 {m['夏普']:.2f}")
    if "beta" in m:
        print(f"  相對 {BENCH}：超額 {m['超額年化報酬']:+.2%}、beta {m['beta']:.2f}、"
              f"年化 alpha {m['年化 alpha']:+.2%}")
    print(f"輸出：{out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
