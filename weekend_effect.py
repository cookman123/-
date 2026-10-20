#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
休市長度效應：停得越久，累積的未定價資訊越多，開盤後的消化過程是否也越長？

為什麼這個角度可能是新聞文字唯一有用的地方
------------------------------------------
平日美股收盤價已經把美國新聞概括掉了，所以先前測出「新聞文字沒有增量」是合理的
—— 文字只是價格的更嘈雜版本。

但週末不同：週六日全球主要市場都休市，這段時間發生的事（政策宣布、地緣衝突、
產業消息）沒有任何價格載體，只存在於文字裡。台股週一開盤要一次消化兩到三天的
資訊量，而費半的「最後收盤」還停在週五。

若「休市越久 → 開盤後的盤中延續越強」，就代表資訊沒有在開盤瞬間消化完，
那才有空間讓新聞文字發揮作用。若週一跟平常沒兩樣，爬新聞就不必做了。

三個檢定
--------
  一、依星期幾分組：跳空與盤中報酬的規模、以及費半的解釋力
  二、依休市天數分組（含農曆年等長假）：把週末當成連續變數的一般化
  三、費半訊號的盤中價差是否在週一（或長假後）特別大
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_lead as O  # noqa: E402

WD = ["週一", "週二", "週三", "週四", "週五"]


def cluster_t(v: pd.Series, dates: pd.Series) -> tuple:
    d = pd.DataFrame({"v": v, "m": pd.to_datetime(dates).dt.to_period("M")}).dropna()
    if len(d) < 40:
        return np.nan, np.nan
    g = d.groupby("m")["v"].mean()
    if len(g) < 6 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan
    return float(d["v"].mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))


def spread_t(d: pd.DataFrame) -> tuple:
    """強弱兩組盤中報酬差的群集 t 值。"""
    sub = d[d["grp"] != "mid"]
    if sub.empty:
        return np.nan, np.nan, 0
    per = (sub.groupby([pd.to_datetime(sub["date"]).dt.to_period("M"), sub["grp"]])
              ["intraday"].mean().unstack())
    if not {"strong", "weak"} <= set(per.columns):
        return np.nan, np.nan, 0
    per = (per["strong"] - per["weak"]).dropna()
    if len(per) < 6:
        return float(per.mean()) if len(per) else np.nan, np.nan, len(per)
    return (float(per.mean()),
            float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), len(per))


def main() -> int:
    tw = O.load("TWII")
    tw = tw[(tw["date"] >= "2011-01-01") & (tw["date"] <= "2024-12-31")].copy()
    tw["gap"] = tw["open"] / tw["close"].shift(1) - 1
    tw["intraday"] = tw["close"] / tw["open"] - 1
    tw["closed_days"] = (tw["date"] - tw["date"].shift(1)).dt.days
    tw["wd"] = tw["date"].dt.weekday
    tw = tw.dropna(subset=["gap", "intraday", "closed_days"])

    sox = O.load("SOX")[["date", "close"]].copy()
    sox["lead"] = sox["close"].pct_change()
    m = pd.merge_asof(tw[["date", "gap", "intraday", "closed_days", "wd"]],
                      sox[["date", "lead"]].dropna(), on="date",
                      direction="backward", allow_exact_matches=False).dropna()
    q = m["lead"].quantile([0.2, 0.8])
    m["grp"] = np.where(m["lead"] >= q.iloc[1], "strong",
                        np.where(m["lead"] <= q.iloc[0], "weak", "mid"))
    print(f"台股 {len(m):,} 個交易日　{m['date'].min():%Y-%m-%d} ~ {m['date'].max():%Y-%m-%d}\n")

    print("=" * 94)
    print("一、依星期幾分組")
    print("=" * 94)
    print(f"  {'星期':<6}{'天數':>6}{'平均休市天':>11}{'跳空絕對值':>12}"
          f"{'盤中絕對值':>12}{'費半-盤中相關':>14}{'強弱價差':>11}{'群集t':>8}")
    for w in range(5):
        d = m[m["wd"] == w]
        if len(d) < 100:
            continue
        sp, t, _ = spread_t(d)
        print(f"  {WD[w]:<6}{len(d):>6,}{d['closed_days'].mean():>11.2f}"
              f"{d['gap'].abs().mean():>12.3%}{d['intraday'].abs().mean():>12.3%}"
              f"{d['lead'].corr(d['intraday']):>14.3f}{sp:>+11.3%}{t:>8.2f}")

    print("\n" + "=" * 94)
    print("二、依休市天數分組（把週末一般化：停越久，累積資訊越多）")
    print("=" * 94)
    bins = [(1, 1, "1 天（一般日）"), (3, 3, "3 天（週末）"),
            (4, 5, "4-5 天（含國定假）"), (6, 30, "6 天以上（農曆年等長假）")]
    print(f"  {'休市':<18}{'天數':>6}{'跳空絕對值':>12}{'盤中絕對值':>12}"
          f"{'費半-盤中相關':>14}{'強弱價差':>11}{'群集t':>8}")
    for lo, hi, label in bins:
        d = m[(m["closed_days"] >= lo) & (m["closed_days"] <= hi)]
        if len(d) < 60:
            continue
        sp, t, _ = spread_t(d)
        print(f"  {label:<18}{len(d):>6,}{d['gap'].abs().mean():>12.3%}"
              f"{d['intraday'].abs().mean():>12.3%}"
              f"{d['lead'].corr(d['intraday']):>14.3f}{sp:>+11.3%}{t:>8.2f}")

    print("\n" + "=" * 94)
    print("三、週一 vs 其他日（僅 2018 年後——效應是這段期間才明顯的）")
    print("=" * 94)
    recent = m[m["date"] >= "2018-01-01"]
    for label, d in (("週一", recent[recent["wd"] == 0]),
                     ("週二至週五", recent[recent["wd"] != 0])):
        sp, t, n = spread_t(d)
        mi, ti = cluster_t(d["intraday"], d["date"])
        print(f"  {label:<10}{len(d):>6,} 天　費半-盤中相關 {d['lead'].corr(d['intraday']):>6.3f}"
              f"　強弱價差 {sp:>+.3%}（群集 t={t:.2f}，{n} 個月）"
              f"　平均盤中 {mi:>+.3%}")

    print("\n判讀：若週一（或長假後）的『盤中』規模與費半解釋力明顯更高，")
    print("代表資訊沒在開盤瞬間消化完，才有空間讓新聞文字補足價格看不到的部分。")
    print("若週一與平常無異，爬國際新聞就沒有必要。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
