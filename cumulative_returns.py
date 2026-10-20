#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 日標籤＋緩衝帶策略的累積報酬，與大盤對照。

    策略：nextday_horizon.py 的 h=5、m=5，10 檔與 30 檔，扣交易成本 0.39%／來回，
          不含現金股利（Yahoo 價格）
    大盤（價格）：加權指數開盤→開盤，與策略同一個持有區間，不含股利、不含費用
    大盤（含息、扣費）：近似 0050 實際持有 —— 加權指數 + 3.5%／年股利 − 0.43%／年內扣費用，
          期初買進、期末賣出各付一次成本
輸出 results/nextday/cumulative.csv 與 cumulative.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nextday_buffer import run  # noqa: E402
from nextday_v2 import COST, OUT, STORE  # noqa: E402

DIV, EXP = 0.035, 0.0043


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    df = pd.read_pickle(STORE.with_name(f"panel{a.tag}.pkl"))[["y_oo", "trd_locked"]]
    p = pd.read_csv(OUT / f"pred_h5{a.tag}.csv", dtype={"code": str}, parse_dates=["date"])
    pred = p.set_index(["date", "code"])["pred"]
    lk = (df["trd_locked"] >= 0.5).reindex(pred.index).fillna(False)
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]) \
        .set_index("date")["open"].sort_index()
    tw = tw[~tw.index.duplicated(keep="last")]
    mk = tw.shift(-2) / tw.shift(-1) - 1

    R = {}
    for n in (10, 30):
        r, _, _ = run(pred, df["y_oo"], lk, n, 5, "多單")
        R[f"策略 {n} 檔"] = r
    R = pd.DataFrame(R)
    R["大盤（價格指數）"] = mk.reindex(R.index)
    R["大盤（含息扣費，近似 0050）"] = R["大盤（價格指數）"] + (DIV - EXP) / 252
    R = R.dropna()
    R.iloc[0, R.columns.get_loc("大盤（含息扣費，近似 0050）")] -= 0.001425 * 0.3
    R.iloc[-1, R.columns.get_loc("大盤（含息扣費，近似 0050）")] -= 0.001425 * 0.3 + 0.001
    eq = (1 + R).cumprod()
    eq.to_csv(OUT / f"cumulative{a.tag}.csv", encoding="utf-8-sig")

    yrs = len(R) / 252
    print(f"期間 {R.index[0]:%Y-%m-%d} ~ {R.index[-1]:%Y-%m-%d}（{yrs:.1f} 年），1 元變成：")
    for c in R.columns:
        dd = (eq[c] / eq[c].cummax() - 1).min()
        print(f"  {c:<22}{eq[c].iloc[-1]:>8.2f} 倍　累積 {eq[c].iloc[-1] - 1:>+9.0%}"
              f"　年化 {eq[c].iloc[-1] ** (1 / yrs) - 1:+.2%}　最大回撤 {dd:+.1%}")
    print("\n逐年報酬")
    y = R.groupby(R.index.year).apply(lambda g: (1 + g).prod() - 1)
    print((y * 100).round(1).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Microsoft JhengHei", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sty = {"策略 10 檔": ("#2a6fdb", "-"), "策略 30 檔": ("#2a6fdb", "--"),
           "大盤（價格指數）": ("#9aa3ad", "-"), "大盤（含息扣費，近似 0050）": ("#e8833a", "-")}
    for c in eq.columns:
        ax.plot(eq.index, eq[c], color=sty[c][0], ls=sty[c][1], lw=1.4, label=c)
    ax.set_yscale("log")
    ax.set_ylabel("1 元累積（對數刻度）")
    which = "含下市股" if a.tag == "_surv" else "只有存活股"
    ax.set_title(f"5 日標籤＋緩衝帶的累積報酬（扣交易成本、{which}）", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"cumulative{a.tag}.png", dpi=200)
    print(f"\n輸出：{OUT / f'cumulative{a.tag}.csv'}、cumulative{a.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
