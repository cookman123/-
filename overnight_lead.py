#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海外隔夜訊號對台股的領先性檢定。

為什麼先測價格、不先爬新聞
--------------------------
「川普宣布了什麼」「韓國／日本產業如何」這些新聞的結論，已經完整反映在
費半、那斯達克、標普、KOSPI、日經、台積電 ADR 的價格裡了 —— 而且價格有
精確時戳、沒有斷詞問題、沒有情緒判讀誤差。若連價格通道都沒有預測力，
新聞文字更不可能有，因為文字是同一份資訊的更嘈雜版本。

時間結構（台北時間）
--------------------
    美股收盤     凌晨 4:00     → 台股開盤前就已知，是真正的「領先」
    韓股 / 日股  08:00~14:30  → 與台股盤中重疊，其「前一日收盤」才算領先
    台股         09:00~13:30

三個要分開的問題
----------------
    1. 隔夜訊號能不能預測台股「開盤跳空」？（幾乎一定可以，但吃不到）
    2. 能不能預測「開盤後到收盤」的走勢？（這才是能交易的部分）
    3. 扣掉跳空後還剩多少？—— 若資訊已在開盤價反映完畢，第 2 項會是 0

顯著性用以月份為群集的 t 值；隔夜報酬有明顯的波動群聚，
把每天當獨立樣本會高估顯著性。
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
MKT = ROOT / "data" / "market"

# 領先程度不同：美股是真隔夜領先，亞股與台股盤中重疊
LEADERS = {
    "SOX": "費城半導體（美，隔夜）",
    "SOXX": "iShares半導體（美，隔夜）",
    "SMH": "VanEck半導體（美，隔夜）",
    "IXIC": "那斯達克（美，隔夜）",
    "GSPC": "標普500（美，隔夜）",
    "TSM": "台積電ADR（美，隔夜）",
    "KS11": "韓國KOSPI（亞，前日收盤）",
    "N225": "日經225（亞，前日收盤）",
}


def load(name: str) -> pd.DataFrame:
    f = MKT / f"{name}.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    return d.dropna(subset=["close"]).drop_duplicates("date")


def cluster_stat(x: pd.Series, dates: pd.Series) -> tuple:
    d = pd.DataFrame({"v": x, "m": pd.to_datetime(dates).dt.to_period("M")}).dropna()
    if len(d) < 60:
        return np.nan, np.nan
    g = d.groupby("m")["v"].mean()
    if len(g) < 6 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan
    return float(d["v"].mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))


def main() -> int:
    tw = load("TWII")
    if tw.empty:
        print("找不到 TWII")
        return 1
    tw = tw[(tw["date"] >= "2011-01-01") & (tw["date"] <= "2024-12-31")].copy()
    tw["gap"] = tw["open"] / tw["close"].shift(1) - 1        # 開盤跳空
    tw["intraday"] = tw["close"] / tw["open"] - 1            # 開盤到收盤（可交易）
    tw["full"] = tw["close"] / tw["close"].shift(1) - 1      # 全日
    tw = tw.dropna(subset=["gap", "intraday", "full"])
    print(f"台股 {len(tw):,} 個交易日　{tw['date'].min():%Y-%m-%d} ~ {tw['date'].max():%Y-%m-%d}")
    print(f"全日報酬 = 跳空 + 盤中：平均跳空 {tw['gap'].mean():+.4%}／"
          f"平均盤中 {tw['intraday'].mean():+.4%}\n")

    print("=" * 96)
    print("一、海外隔夜報酬與台股次日的相關係數")
    print("=" * 96)
    print(f"  {'海外標的':<26}{'樣本':>7}{'vs 跳空':>11}{'vs 盤中':>11}{'vs 全日':>11}")
    merged = {}
    for name, label in LEADERS.items():
        d = load(name)
        if d.empty:
            continue
        d = d[["date", "close"]].copy()
        d["lead"] = d["close"].pct_change()
        d = d.dropna(subset=["lead"])
        # 取「嚴格早於台股當日」的最後一筆海外收盤
        m = pd.merge_asof(tw[["date", "gap", "intraday", "full"]], d[["date", "lead"]],
                          on="date", direction="backward", allow_exact_matches=False)
        m = m.dropna(subset=["lead"])
        if len(m) < 500:
            continue
        merged[name] = m
        c = [m["lead"].corr(m[k]) for k in ("gap", "intraday", "full")]
        print(f"  {label:<26}{len(m):>7,}{c[0]:>11.3f}{c[1]:>11.3f}{c[2]:>11.3f}")

    print("\n  判讀：對『跳空』的相關高、對『盤中』接近 0，")
    print("  代表資訊在開盤瞬間就反映完畢 —— 知道了也吃不到。")

    print("\n" + "=" * 96)
    print("二、能不能交易？依費半隔夜漲跌分組，看台股當日各段報酬")
    print("=" * 96)
    if "SOX" in merged:
        m = merged["SOX"].copy()
        q = m["lead"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        m["grp"] = np.digitize(m["lead"], q)
        names = ["最弱 20%", "20-40%", "40-60%", "60-80%", "最強 20%"]
        print(f"  {'費半隔夜':<12}{'樣本':>7}{'台股跳空':>20}{'台股盤中（可交易）':>24}")
        for g, nm in enumerate(names):
            s = m[m["grp"] == g]
            if len(s) < 60:
                continue
            mg, tg = cluster_stat(s["gap"], s["date"])
            mi, ti = cluster_stat(s["intraday"], s["date"])
            print(f"  {nm:<12}{len(s):>7,}"
                  f"{mg:>+13.3%}(t={tg:>5.2f}){mi:>+15.3%}(t={ti:>5.2f})")

        hi = m[m["grp"] == 4]["intraday"]
        lo = m[m["grp"] == 0]["intraday"]
        diff = hi.mean() - lo.mean()
        both = pd.concat([hi.rename("v").to_frame().assign(s=1),
                          lo.rename("v").to_frame().assign(s=0)])
        both["date"] = pd.concat([m[m["grp"] == 4]["date"], m[m["grp"] == 0]["date"]])
        per = both.groupby([pd.to_datetime(both["date"]).dt.to_period("M")]).apply(
            lambda g: (g.loc[g["s"] == 1, "v"].mean() - g.loc[g["s"] == 0, "v"].mean())
            if (g["s"] == 1).any() and (g["s"] == 0).any() else np.nan,
            include_groups=False).dropna()
        t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 5 else np.nan
        print(f"\n  最強 20% 與最弱 20% 的『盤中』報酬差：{diff:+.3%}"
              f"（群集 t={t:.2f}，{len(per)} 個月）")
        print(f"  台股現股來回成本約 0.39%、期貨約 0.01%。"
              f"差距要大於成本才有交易價值。")

    print("\n" + "=" * 96)
    print("三、對跌深反彈策略的可能用途：費半隔夜走勢與台股跌深後的反彈")
    print("=" * 96)
    print("  （下一步：把費半隔夜訊號加入事件模型的特徵，")
    print("   測它能否幫忙判斷『這個跌該不該買』）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
