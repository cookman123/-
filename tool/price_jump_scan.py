#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掃描存活股 Yahoo 價格裡「不可能的單日跳動」。

台股漲跌幅限制 2015-06-01 前 7%、之後 10%。相鄰兩個交易日之間（中間沒有停牌）
收盤變動超過這個限制，只有幾種合理原因：新上市前 5 日、除權息／減資恢復買賣
（參考價重設），或資料錯誤。Yahoo 對配股、減資理應做分割調整，所以除權息、
減資日之後若仍有超限跳動，代表 Yahoo 沒有調整，那一跳是假的。

輸出 results/data_fix_2011/price_jumps.csv：每一筆超限跳動與它最可能的原因。
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import davis_tw_train as T  # noqa: E402

CA = ROOT / "data" / "corp_actions"


def main() -> int:
    ex = pd.read_csv(CA / "exright.csv", dtype={"code": str}, parse_dates=["date"])
    rd = pd.read_csv(CA / "reduction.csv", dtype={"code": str}, parse_dates=["date"])
    ev = pd.concat([ex[["date", "code"]].assign(kind="除權息"), rd[["date", "code"]].assign(kind="減資")])
    ev = ev.set_index(["code", "date"])["kind"]
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]).set_index("date").index
    rows = []
    for d in sorted((ROOT / "data" / "stocks").iterdir()):
        px = T.load_price(d.name)
        if px is None or len(px) < 10:
            continue
        px = px[px.index >= "2010-01-01"]
        c = px["close"]
        r = c.pct_change()
        # 只看相鄰交易日（以大盤交易日曆判斷中間有沒有停牌）
        pos = tw.searchsorted(c.index)
        gap = pd.Series(np.r_[np.nan, np.diff(pos)], index=c.index)
        lim = np.where(c.index < "2015-06-01", 0.075, 0.105)
        bad = (r.abs() > lim) & (gap == 1)
        first5 = c.index[:5]
        for day in r[bad].index:
            if day in first5:
                why = "上市前 5 日"
            elif (d.name, day) in ev.index:
                why = "事件日：" + str(ev.loc[(d.name, day)] if not isinstance(ev.loc[(d.name, day)], pd.Series)
                                    else ev.loc[(d.name, day)].iloc[0]) + "（Yahoo 未調整）"
            else:
                why = "無對應事件（疑似資料錯誤）"
            rows.append((d.name, day.date(), round(float(r[day]), 4), why))
    out = pd.DataFrame(rows, columns=["code", "date", "ret", "why"])
    p = ROOT / "results" / "data_fix_2011" / "price_jumps.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"超限跳動 {len(out)} 筆，涉及 {out['code'].nunique()} 檔")
    print(out["why"].str.replace(r"：.*", "", regex=True).value_counts().to_string())
    print(out.groupby(out["why"].str[:3])["ret"].describe()[["count", "mean", "min", "max"]].round(3))
    print(out.reindex(out["ret"].abs().sort_values(ascending=False).index).head(25).to_string(index=False))
    print(f"輸出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
