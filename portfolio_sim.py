#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多標的資金曲線模擬：回答「交易多但每筆弱」與「交易少但每筆強」哪個實際賺比較多。

為什麼需要這一支
----------------
先前在大盤（單一標的）上看到：勝率 70%、每筆 +1.84%，但總報酬只有 +37.97%，
輸給買進持有的 +180.65%——因為只有一個標的，沒訊號時只能抱現金，
15 年只進場 20 次，空手期間把報酬稀釋掉了。

但多標的的情況完全不同：120 檔股票的訊號可以同時持有多個部位，
「每筆報酬」乘上「一年能做幾輪」才是真正的年化報酬。
篩選條件把交易數從 1,923 砍到 411，每筆從 +1.69% 提高到 +2.43%，
究竟是賺還是賠，必須把資金限制放進來算。

模擬規則
--------
    資金分成 K 個等額部位（同時最多持有 K 檔）
    訊號依時間順序到達；有空位就進場，投入「當時總權益 / K」
    沒空位就放棄該訊號（記錄放棄數——這是「交易太多」的真正代價）
    到出場日結算，本金加報酬回到現金

報酬已是扣除手續費與證交稅後的淨值（來自 news_reversal 的交易明細）。
未計滑價，且期間內不做逐日評價（只在出場日結算），
因此最大回撤會被低估——這一點在解讀時要記得。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent


def simulate(trades: pd.DataFrame, k_slots: int,
             calendar: pd.DatetimeIndex,
             idle_ret: pd.Series | None = None,
            park_cost: float = 0.0) -> dict:
    """
    依時間順序模擬 K 個部位的資金曲線。

    idle_ret 給定時，閒置現金會投入該標的（例如 0050）賺市場報酬。
    這是本策略能否實用的關鍵：篩選後每一塊錢在場上時的報酬很高，
    但資金投入率只有一到二成，九成的錢閒置會把年化拉垮。
    把閒置資金放進大盤 ETF，沒訊號時賺市場報酬、有訊號時賺策略報酬，
    才是這種「機會稀少但品質高」的策略正確的用法。
    """
    tr = trades.sort_values("entry").reset_index(drop=True)
    cash = 1.0
    open_pos = []          # (exit_date, 投入金額, 淨報酬)
    taken = skipped = 0
    equity_curve = {}
    deployed_days = 0
    deployed_frac = []      # 每日「投入金額 / 總權益」——這才是真正的資金利用率

    all_dates = calendar
    ti = 0
    for day in all_dates:
        # 閒置現金先按大盤當日報酬滾動（等同把現金停泊在 0050）
        if idle_ret is not None and cash > 0:
            cash *= (1 + float(idle_ret.get(day, 0.0)))
        # 先結算今天到期的部位
        still = []
        for cid_, exit_d, amt, net in open_pos:
            if exit_d <= day:
                back = amt * (1 + net)
                # 資金回到現金池後要買回 0050，付一次買進成本
                cash += back * (1 - park_cost / 2) if idle_ret is not None else back
            else:
                still.append((cid_, exit_d, amt, net))
        open_pos = still

        # 再處理今天的新訊號
        while ti < len(tr) and tr.loc[ti, "entry"] <= day:
            row = tr.loc[ti]
            # 同一個操作回合可能拆成多批，各批各自進場、各自到期釋放資金。
            # 佔用的格子數以「回合」計，不是以「批次」計。
            # campaign 欄是進場日期（例如 20110810），不同股票同一天進場會共用同一個值。
            # 只用它當鍵，會讓同一天的幾十檔擠進同一格（實測最多 52 檔），格數上限形同虛設。
            # 回合必須以（代號, campaign）識別（2026-10-15 修正）。
            cid = (row.get("code"), row.get("campaign", ti))
            open_cids = {c for c, _, _, _ in open_pos}
            if cid in open_cids or len(open_cids) < k_slots:
                equity = cash + sum(a for _, _, a, _ in open_pos)
                # weight 是該批佔整個回合倉位的比重；size_mult 是新聞加碼倍數
                w = float(row.get("weight", 1.0)) * float(row.get("size_mult", 1.0))
                size = min(equity / k_slots * w, cash)
                if size > 1e-9:
                    # 先賣掉 0050 換現金，付一次賣出成本（含 ETF 證交稅）
                    if idle_ret is not None:
                        cash -= size * (park_cost / 2)
                    cash -= size
                    open_pos.append((cid, row["exit"], size, row["net"]))
                    taken += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            ti += 1

        invested = sum(a for _, _, a, _ in open_pos)
        eq = cash + invested
        equity_curve[day] = eq
        if open_pos:
            deployed_days += 1
        deployed_frac.append(invested / eq if eq > 0 else 0.0)

    # 結算未平倉部位
    for _, _, amt, net in open_pos:
        cash += amt * (1 + net)
    final = cash

    s = pd.Series(equity_curve).sort_index()
    yrs = len(all_dates) / 252
    dd = float((s / s.cummax() - 1).min())
    return {
        "部位數上限": k_slots,
        "採用訊號": taken, "放棄訊號": skipped,
        "採用比例": taken / max(taken + skipped, 1),
        "總報酬": final - 1,
        "年化": final ** (1 / yrs) - 1 if final > 0 else np.nan,
        "有部位天數比例": deployed_days / len(all_dates),
        "平均資金投入率": float(np.mean(deployed_frac)) if deployed_frac else 0.0,
        "最大回撤(低估)": dd,
        "_curve": s,
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="多標的資金曲線模擬")
    ap.add_argument("--trades", default="results/integrated/trades.csv")
    ap.add_argument("--slots", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--park", action="store_true",
                    help="閒置資金停泊於對照標的（而非閒置現金）")
    ap.add_argument("--bench", default="0050",
                    help="對照標的（data/market/ 下的檔名）")
    args = ap.parse_args(argv)

    tr = pd.read_csv(ROOT / args.trades, parse_dates=["entry", "exit", "drop_day"])
    lo, hi = tr["entry"].min(), tr["exit"].max()

    bench = pd.read_csv(ROOT / "data" / "market" / f"{args.bench}.csv",
                        parse_dates=["date"]).set_index("date").sort_index()
    bench = bench[(bench.index >= lo) & (bench.index <= hi)]
    cal = bench.index
    yrs = len(cal) / 252
    b_tot = float(bench["close"].iloc[-1] / bench["close"].iloc[0]) - 1
    b_cagr = (1 + b_tot) ** (1 / yrs) - 1
    b_dd = float((bench["close"] / bench["close"].cummax() - 1).min())

    # 兩種規則：全部交易 vs 只取結構健康的
    healthy = ((tr["px_obv_dev"] > 0) & (tr["px_ma_align"] > 0)
               & (tr["px_ma60_dev"] > 0)).fillna(False)
    variants = {
        f"主規則（全部 {len(tr):,} 筆）": tr,
        f"結構健康篩選（{int(healthy.sum()):,} 筆）": tr[healthy],
    }

    idle = bench["close"].pct_change().fillna(0.0) if args.park else None
    print(f"期間 {lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}（{yrs:.1f} 年，{len(cal)} 個交易日）")
    print(f"閒置資金：{'停泊於 ' + args.bench if args.park else '持有現金（零報酬）'}")
    print(f"對照：{args.bench} 買進持有　總報酬 {b_tot:+.2%}　年化 {b_cagr:+.2%}"
          f"　最大回撤 {b_dd:.2%}\n")

    rows = []
    for name, t in variants.items():
        print("=" * 92)
        print(f"{name}　平均每筆淨報酬 {t['net'].mean():+.2%}　"
              f"勝率 {(t['net'] > 0).mean():.1%}　平均持有 {t['hold'].mean():.1f} 日")
        print("=" * 92)
        print(f"  {'部位上限':>8}{'採用':>8}{'放棄':>8}{'採用率':>8}"
              f"{'資金投入率':>10}{'總報酬':>11}{'年化':>10}{'超額年化':>10}"
              f"{'投入期間年化':>12}")
        for k in args.slots:
            r = simulate(t, k, cal, idle)
            rows.append({**{x: y for x, y in r.items() if x != "_curve"},
                         "規則": name})
            # 把資金利用率還原掉，才看得出「錢真正在場上時」的報酬水準
            util = max(r["平均資金投入率"], 1e-6)
            adj = (1 + r["總報酬"]) ** (1 / (yrs * util)) - 1 if r["總報酬"] > -1 else np.nan
            print(f"  {k:>8}{r['採用訊號']:>8,}{r['放棄訊號']:>8,}"
                  f"{r['採用比例']:>8.1%}{r['平均資金投入率']:>10.1%}"
                  f"{r['總報酬']:>+11.2%}{r['年化']:>+10.2%}"
                  f"{r['年化'] - b_cagr:>+10.2%}{adj:>+12.2%}")
        print()

    pd.DataFrame(rows).to_csv(ROOT / "results/integrated/portfolio_sim.csv",
                              index=False, encoding="utf-8-sig")
    print("判讀：『放棄訊號』是交易太密集的真正代價——部位上限不夠時，"
          "多出來的訊號根本吃不下。")
    print("因此『每筆報酬高但次數少』不一定輸：在部位上限的限制下，"
          "它反而可能讓每一塊錢都投在較好的機會上。")
    print("\n注意：期間內不做逐日評價，只在出場日結算，最大回撤被低估；"
          "亦未計入滑價。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
