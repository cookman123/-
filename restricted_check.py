#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
處置／注意／全額交割股在候選池裡占多少、之後表現如何（restricted.py 的規則，取自 tw-stock-quant）。

為什麼要看：5 日模型偏好短期大漲、高波動的股票，正好是最容易被列為注意或處置的一群。
處置股要預收款券、分盤撮合，研究裡「隔日開盤成交」的假設對它們不成立（第七節「成交不到的價格」）。

規格
----
候選池  5 日模型（含下市股版，db/local.db 的 predictions）每 5 個交易日一期、排名前 12 名，2014 年起
        （和 llm_select_eval.py 同一個池子，只是期間拉長到全部歷史）
標記    訊號日收盤後、隔日開盤成交時是否為處置／注意／全額交割（restricted.flags，lag 1）
結果    隔日開盤買、第 6 個交易日開盤賣，扣成本（同 llm_eval.outcome）
比較    被標記的與沒被標記的平均報酬；每期「排除後的前 12 名等權」對「不排除」的差，對期數做 t 檢定
注意    處置股實際上多半買不到隔日開盤價，所以被標記那組的報酬本來就不一定拿得到；
        這裡看的是「如果硬算」會讓回測多算或少算多少

輸出 results/restricted_check/summary.csv、periods.csv
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
sys.path.insert(0, str(ROOT))
import forecast as F  # noqa: E402
import llm_eval as E  # noqa: E402
import restricted as R  # noqa: E402

OUT = ROOT / "results" / "restricted_check"
N, STEP = 12, 5


def main() -> int:
    ev = R.load()
    if ev.empty:
        return 1
    con = F.connect()
    cal = pd.DatetimeIndex(pd.to_datetime([d for (d,) in con.execute(
        "SELECT DISTINCT date FROM prices ORDER BY date")]))
    days = [d for d in F.complete_days(con, E.H).index if d >= "2014-01-01"][::STEP]
    cover_to = ev["announce_date"].max()
    days = [d for d in days if pd.Timestamp(d) <= cover_to]
    p = pd.read_sql("SELECT date, code, pred FROM predictions WHERE h = ?", con, params=[E.H])
    p = p[p["date"].isin(days)]
    rows = []
    for d, g in p.groupby("date"):
        g = g.nlargest(N * 2, "pred").reset_index(drop=True)
        g["flag"] = R.flags(pd.DataFrame({"date": pd.Timestamp(d), "code": g["code"].astype(str)}), cal, 1, ev).to_numpy()
        g["net"] = [E.outcome(con, c, d) for c in g["code"]]
        g["date"] = d
        g["rank"] = np.arange(1, len(g) + 1)
        rows.append(g)
    c = pd.concat(rows, ignore_index=True)
    top = c[c["rank"] <= N]
    per = pd.DataFrame({
        "all": top.groupby("date")["net"].mean(),
        "excl": c[~c["flag"]].groupby("date").head(N).groupby("date")["net"].mean(),
        "n_flag": top.groupby("date")["flag"].sum(),
    }).dropna(subset=["all", "excl"])
    diff = per["excl"] - per["all"]
    t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
    f, nf = top[top["flag"]]["net"].dropna(), top[~top["flag"]]["net"].dropna()
    summ = {
        "期間": f"{per.index.min()} ~ {per.index.max()}", "期數": len(per), "候選筆數": len(top),
        "被標記比例": float(top["flag"].mean()), "有標記的期數比例": float((per["n_flag"] > 0).mean()),
        "被標記的平均報酬": float(f.mean()), "沒被標記的平均報酬": float(nf.mean()),
        "被標記的勝率": float((f > 0).mean()), "沒被標記的勝率": float((nf > 0).mean()),
        "不排除每期": float(per["all"].mean()), "排除後每期": float(per["excl"].mean()),
        "差（排除 − 不排除）": float(diff.mean()), "差 t": float(t),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.Series(summ).to_csv(OUT / "summary.csv", header=["值"], encoding="utf-8-sig")
    per.to_csv(OUT / "periods.csv", encoding="utf-8-sig")
    print(f"5 日模型前 {N} 名的處置／注意／全額交割股（資料到 {cover_to:%Y-%m-%d}）")
    for k, v in summ.items():
        if not isinstance(v, float):
            s = str(v)
        elif "比例" in k or "勝率" in k:
            s = f"{v:.1%}"
        elif k.endswith(" t"):
            s = f"{v:.2f}"
        else:
            s = f"{v:+.3%}"
        print(f"  {k:<14}{s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
