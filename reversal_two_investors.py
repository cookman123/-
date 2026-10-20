#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跌深反彈的實用性測試：兩位投資人，同一筆錢、同一段時間。

    投資人 A：期初全部買 0050，一路持有到期末。
    投資人 B：期初也全部買 0050。跌深反彈出現訊號時，賣掉一格的 0050 去買那檔股票；
              策略出場後，錢立刻買回 0050。也就是「沒有策略在跑的錢，永遠放在 0050」。

問 B 最後有沒有比 A 多。這比「策略每筆平均賺多少」更貼近實際：策略只有在有訊號時才用錢，
其餘時間的機會成本就是 0050，而 0050 這段期間本身漲很多。

和舊版（run_long.py 的組合層級）差在哪
------------------------------------
    一、兩邊都含現金股利：0050 用證交所除息資料把股利在除息日再投入（Yahoo 價格不含息）。
        舊版的 0050 不含息，A 被低估。
    二、0050 與個股之間的切換要付成本：賣 0050 付手續費＋ETF 證交稅 0.1%，買回付手續費。
        舊版 park_cost = 0。
    三、格數上限修正：舊版用 campaign（進場日期）當回合鍵，同一天進場的不同股票共用同一格，
        最多 52 檔擠一格。改成（代號, campaign）。
    四、A 期初買、期末賣也付成本，兩人站在同一個起跑點。

策略本身（訊號、進出場、每筆淨報酬）沿用 run_long.py 的 parts.csv，不重新調參；
格數 k ∈ {5, 10, 20} 也是原本就有的三個設定，全部列出，不挑最好的。
個股部位在持有期間的現金股利沒有計入（持有期多為數天到一個月，影響小，對 B 偏保守）。
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT))
import portfolio_sim as PS  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402

OUT = ROOT / "results" / "long_rule"
FEE = 0.001425 * 0.3
ETF_TAX = 0.001
PARK = 2 * FEE + ETF_TAX          # 0050 來回一次的成本


def etf_total_return() -> pd.Series:
    """0050 含息日報酬：Yahoo 收盤（已處理分割）＋除息日的現金殖利率。"""
    b = pd.read_csv(ROOT / "data" / "market" / "0050.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    b = b[~b.index.duplicated(keep="last")]
    r = b["close"].pct_change().fillna(0.0)
    dy = {}
    for f in (ROOT / "data" / "corp_actions" / "raw").glob("TWT49U_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("stat") != "OK":
            continue
        for x in d["data"]:
            if x[1].strip() != "0050":
                continue
            t = "".join(c if c.isdigit() else " " for c in x[0]).split()
            day = pd.Timestamp(int(t[0]) + 1911, int(t[1]), int(t[2]))
            prev, ref = float(x[3].replace(",", "")), float(x[4].replace(",", ""))
            dy[day] = (prev - ref) / prev
    dy = pd.Series(dy).sort_index()
    i = r.index.searchsorted(dy.index)
    ok = i < len(r)
    add = pd.Series(dy.values[ok], index=r.index[i[ok]]).groupby(level=0).sum()
    return r.add(add, fill_value=0.0), dy


def stats(curve: pd.Series) -> dict:
    d = curve.pct_change().dropna()
    yrs = len(curve) / 252
    return {"終值": float(curve.iloc[-1]), "年化": float(curve.iloc[-1]) ** (1 / yrs) - 1,
            "夏普": d.mean() / d.std() * np.sqrt(252),
            "最大回撤": float((curve / curve.cummax() - 1).min())}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="跌深反彈：兩位投資人對照")
    ap.add_argument("--parts", default=str(OUT / "parts.csv"))
    a = ap.parse_args(argv)

    parts = pd.read_csv(a.parts, dtype={"code": str, "campaign": str},
                        parse_dates=["entry", "exit"])
    etf, dy = etf_total_return()
    cal = etf[(etf.index >= parts["entry"].min()) & (etf.index <= parts["exit"].max())].index
    r_etf = etf.reindex(cal).fillna(0.0)
    yrs = len(cal) / 252
    print(f"期間 {cal[0]:%Y-%m-%d} ~ {cal[-1]:%Y-%m-%d}（{yrs:.1f} 年）　訊號批次 {len(parts):,}"
          f"　回合 {parts.groupby(['code', 'campaign']).ngroups:,}")
    print(f"0050 現金股利 {len(dy)} 次，平均每年殖利率約 {dy.groupby(dy.index.year).sum().mean():.2%}")

    # 投資人 A：期初買、期末賣
    A = (1 + r_etf).cumprod() * (1 - FEE)
    A.iloc[-1] *= (1 - FEE - ETF_TAX)
    curves = {"A：0050 買進持有": A}
    res = []
    sa = stats(A)
    res.append({"投資人": "A：0050 買進持有", **sa, "投入率": 0.0, "採用訊號": 0})

    for k in (5, 10, 20):
        r = PS.simulate(parts, k, cal, r_etf, park_cost=PARK)
        c = r["_curve"] * (1 - FEE)                 # 期初買 0050 的成本
        c.iloc[-1] *= (1 - FEE - ETF_TAX)            # 期末全部賣出
        name = f"B：跌深反彈 {k} 格＋閒置買 0050"
        curves[name] = c
        res.append({"投資人": name, **stats(c), "投入率": r["平均資金投入率"], "採用訊號": r["採用訊號"]})

    print(f"\n{'投資人':<26}{'1 元變成':>9}{'年化':>9}{'夏普':>7}{'最大回撤':>10}{'投入率':>8}"
          f"{'B 減 A 年化':>12}{'超額 t':>8}")
    for row in res:
        c = curves[row["投資人"]]
        if row["投資人"].startswith("B"):
            ex = (c.pct_change() - A.pct_change()).dropna()
            t, p = nw_tstat(ex)
            diff = row["年化"] - sa["年化"]
            row.update({"B減A年化": diff, "超額t": t, "p": p})
            tail = f"{diff:>+12.2%}{t:>8.2f}"
        else:
            tail = ""
        print(f"{row['投資人']:<26}{row['終值']:>9.2f}{row['年化']:>+9.2%}{row['夏普']:>7.2f}"
              f"{row['最大回撤']:>+10.1%}{row['投入率']:>8.1%}{tail}")

    # 逐年：B（10 格）對 A
    cB = curves["B：跌深反彈 10 格＋閒置買 0050"]
    yA = A.groupby(A.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    yB = cB.groupby(cB.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    print("\n逐年（B 10 格 對 A）")
    print("  " + "　".join(f"{y}:{b - a_:+.1%}" for y, a_, b in zip(yA.index, yA, yB)))
    print(f"  B 贏的年數 {int((yB > yA).sum())}/{len(yA)}")

    # 對照：舊版的算法（0050 不含息、不付切換成本、回合鍵用 campaign）
    old_parts = parts.copy()
    old_parts["code"] = None                         # 讓回合鍵退化成只有 campaign，重現舊 bug
    b = pd.read_csv(ROOT / "data" / "market" / "0050.csv", parse_dates=["date"]).set_index("date").sort_index()
    idle_old = b["close"].pct_change().fillna(0.0).reindex(cal).fillna(0.0)
    print("\n對照：舊算法（0050 不含息、切換零成本、同日進場共用一格）")
    for k in (5, 10, 20):
        r = PS.simulate(old_parts, k, cal, idle_old)
        print(f"  {k:>2} 格：年化 {r['年化']:+.2%}　投入率 {r['平均資金投入率']:.1%}")
    print(f"  0050 不含息：年化 {(1 + idle_old).prod() ** (1 / yrs) - 1:+.2%}")

    pd.DataFrame(res).to_csv(OUT / "two_investors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(OUT / "two_investors_curves.csv", encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Microsoft JhengHei", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    col = {"A：0050 買進持有": ("#e8833a", "-", 2.0)}
    for i, k in enumerate((5, 10, 20)):
        col[f"B：跌深反彈 {k} 格＋閒置買 0050"] = ("#2a6fdb", ["-", "--", ":"][i], 1.3)
    for n_, c in curves.items():
        ax.plot(c.index, c.values, color=col[n_][0], ls=col[n_][1], lw=col[n_][2], label=n_)
    ax.set_ylabel("1 元累積（含息、扣成本）")
    ax.set_title("跌深反彈：B 把閒置資金放 0050，最後有沒有贏過只買 0050 的 A", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "two_investors.png", dpi=200)
    print(f"\n輸出：{OUT / 'two_investors.csv'}、two_investors_curves.csv、two_investors.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
