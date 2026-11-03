#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月營收公告的事件研究：公告後 5／20／60 個交易日，對四種基準的超額報酬。

做法取自 jameschen108/tw-stock-quant 的 event_study.py，改成本專案的資料與慣例：
    - 事件日 = 營收 avail_date（次月 10 日，法定期限）；隔一個交易日「開盤」進場
      （本專案一律用開盤成交，訊號與成交價才不是同一個數字）
    - 持有 h 個交易日後的開盤出場；出場日停牌就用持有期最後一個有效開盤價
      （與 quarterly_factors.forward_returns 相同，不事先剔除日後停牌的股票）
    - 只看進場當天屬於時點正確成交金額前 250 名的股票（universe_pit）
    - 扣成本用 quarterly_factors.COST（來回，手續費 3 折＋證交稅）

為什麼要做：季頻檢定裡營收年增率是唯一「不是 beta」的候選（第十五節），
但季末才換倉，看不到公告前後的反應，也沒排除「其實只是小型股／高波動／某些產業」。
所以把超額換成四種基準一起看：
    母體        同進場日、母體等權
    成交值×波動  同進場日、母體內 60 日均成交金額五分位 × 60 日波動五分位（25 格）的等權
                （本專案沒有時點正確的市值，用成交金額代替；這是和原版最大的差別）
    產業        同進場日、母體內同產業（≥3 檔）的等權；產業用 industry_stock_map.json（現行分類）
    大盤        同期加權指數開盤對開盤（價格指數，不含股利，對大盤的超額會略為高估）

t 值：先把事件依「進場月份」平均成一條月序列，再做 Newey-West（落後期 = ceil(h/21)）。
事件都擠在每月 11 日前後，直接對事件算 t 會把同一天的市場波動當成獨立樣本；
60 日持有又會讓相鄰月份重疊，所以比原版的單純月份分群 t 再保守一點。

事件（事先固定，不看結果再增減）
----
  全部公告                 母體內所有營收公告
  創歷史新高               單月營收高於過去所有月份（至少要有 24 個月歷史）
  年增≥30% 且 累計年增≥20%
  創新高 且 年增≥30%       原版最強的事件
  年增≥30% 但未創新高
  年增≤−20%
  營收驚喜／壞消息         取自 tw-stock-quant 的 surprise.py（2026-10 加入，規則照原版、不調參數）：
                           SUR =（單月營收 − 去年同月 − 前 24 個月這個差的平均）÷ 前 24 個月的標準差
                           （至少 12 個月，截在 ±10）；門檻用公布月份之前 12 個月全部上市櫃公司的公告分 10 組，
                           最高一組 = 驚喜、最低一組 = 壞消息。比「年增 30%」多考慮了這家公司平常的波動
                           這兩個是看過前六個事件的結果之後才加的，算事後補充

輸出 results/revenue_event/：events.csv（逐事件）、summary.csv、bench.csv、yearly.csv
"""
from __future__ import annotations

import json
import math
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
import feature_engineering as FE  # noqa: E402
import fundamental_factors as FF  # noqa: E402
import quarterly_factors as QF  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402

OUT = ROOT / "results" / "revenue_event"
HORIZONS = (5, 20, 60)
BENCH = {"uni": "母體", "sv": "成交值×波動", "ind": "產業", "mkt": "大盤"}
PERIODS = {"全期": ("2011-01-01", "2100-01-01"), "2011–2018": ("2011-01-01", "2019-01-01"),
           "2019–": ("2019-01-01", "2100-01-01"), "2024–": ("2024-01-01", "2100-01-01")}


def revenue_events() -> pd.DataFrame:
    """每筆月營收加上事件需要的欄位：年增、累計年增、是否創歷史新高。"""
    r = FF.revenue_table()
    g = r.groupby("code")["revenue"]
    r["prior_max"] = g.transform(lambda x: x.shift(1).cummax())
    r["n_hist"] = r.groupby("code").cumcount()
    r["is_high"] = (r["revenue"] > r["prior_max"]) & (r["n_hist"] >= 24)
    # 累計年增：今年 1 月到當月合計 vs 去年同期（中間缺月就不算）
    r["year"] = r["p"].dt.year
    r["ytd"] = r.groupby(["code", "year"])["revenue"].cumsum()
    r["ytd_n"] = r.groupby(["code", "year"]).cumcount() + 1
    key = r.set_index(["code", "p"])
    ly = key.reindex(list(zip(r["code"], r["p"] - 12)))
    full = (r["ytd_n"].to_numpy() == r["p"].dt.month.to_numpy()) & (ly["ytd_n"].to_numpy() == r["p"].dt.month.to_numpy())
    r["cum_yoy"] = FF._growth(r["ytd"], pd.Series(ly["ytd"].to_numpy(), index=r.index)).where(full)

    # 營收驚喜 SUR（tw-stock-quant surprise.py）：期數對齊，中間缺月不硬接
    d = (r["revenue"] - r["rev_ly"]).set_axis(pd.MultiIndex.from_arrays([r["code"], r["p"]]))
    sur = []
    for code, x in d.groupby(level=0):
        x = x.droplevel(0)
        x = x.reindex(pd.period_range(x.index.min(), x.index.max(), freq="M"))
        prev = x.shift(1).rolling(24, min_periods=12)
        z = ((x - prev.mean()) / prev.std().where(lambda v: v > 0)).clip(-10, 10)
        sur.append(pd.Series(z.to_numpy(), index=pd.MultiIndex.from_product([[code], z.index])))
    sur = pd.concat(sur)
    r["sur"] = sur.reindex(pd.MultiIndex.from_arrays([r["code"], r["p"]])).to_numpy()
    # 十分位門檻：公布月份之前 12 個月的全部公告，至少 2,000 筆才給
    r["am"] = r["avail_date"].dt.to_period("M")
    r["sur_dec"] = np.nan
    for m in r["am"].dropna().unique():
        past = r.loc[(r["am"] < m) & (r["am"] >= m - 12), "sur"].dropna()
        if len(past) < 2000:
            continue
        cur = (r["am"] == m) & r["sur"].notna()
        r.loc[cur, "sur_dec"] = np.searchsorted(np.quantile(past, np.linspace(0.1, 0.9, 9)),
                                                r.loc[cur, "sur"], side="right") + 1
    return r.drop(columns="am")


def bench_returns(O: pd.DataFrame, C: pd.DataFrame, turn: pd.DataFrame, mask: pd.DataFrame,
                  ind: pd.Series, tw: pd.Series, entries: list) -> dict:
    """每個進場日：個股 h 日報酬，以及四種基準下每檔對應的基準報酬。"""
    idx = O.index
    vol = C.pct_change(fill_method=None).rolling(60, min_periods=40).std().shift(1)
    out = {h: {} for h in HORIZONS}
    for d in entries:
        i = idx.get_loc(d)
        pool = mask.loc[d] & O.iloc[i].notna()
        codes = pool[pool].index
        if len(codes) < 50:
            continue
        q = lambda x: np.ceil(x.reindex(codes).rank(pct=True) * 5)
        cell = q(turn.loc[d]) * 10 + q(vol.loc[d])
        grp = ind.reindex(codes)
        for h in HORIZONS:
            if i + h >= len(idx):
                continue
            r = O.iloc[i:i + h + 1, :][codes].ffill().iloc[-1] / O.iloc[i][codes] - 1
            m = tw.reindex(idx[[i, i + h]])
            n_ind = grp.map(grp.value_counts())
            out[h][d] = pd.DataFrame({
                "ret": r,
                "uni": r.mean(),
                "sv": r.groupby(cell).transform("mean").where(cell.notna()),
                "ind": r.groupby(grp).transform("mean").where(n_ind >= 3),
                "mkt": float(m.iloc[1] / m.iloc[0] - 1) if m.notna().all() else np.nan,
            })
    return out


def summarize(ev: pd.DataFrame, h: int, b: str) -> dict:
    x = ev[f"{b}{h}"].dropna()
    if len(x) < 30:
        return {"事件數": len(x)}
    by_m = x.groupby(ev.loc[x.index, "entry"].dt.to_period("M")).mean().sort_index()
    t, p = nw_tstat(by_m, lags=math.ceil(h / 21)) if len(by_m) >= 30 else (np.nan, np.nan)
    return {"事件數": len(x), "月數": len(by_m), "平均超額": x.mean(), "中位超額": x.median(),
            "扣成本": x.mean() - QF.COST, "勝率": float((x > 0).mean()), "t": t, "p": p}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="月營收公告的事件研究")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--uni", type=int, default=250)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    print("載入價格面板…", flush=True)
    P = FE.load_panels(a.start, a.end, quiet=True)
    O, C = P["open"], P["close"]
    idx, codes = O.index, list(O.columns)
    turn = pd.read_csv(U.CACHE, index_col=0, parse_dates=True).reindex(index=idx, columns=codes)
    mask = U.top_n_mask(turn, a.uni).fillna(False)
    ind = pd.Series(FE.industry_map())
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]).set_index("date")["open"]
    tw = tw[~tw.index.duplicated()]

    print("建立事件…", flush=True)
    ev = revenue_events()
    ev = ev[ev["code"].isin(codes) & ev["yoy"].notna()].copy()
    pos = idx.searchsorted(ev["avail_date"].to_numpy(), side="right")   # avail_date 之後第一個交易日
    ev = ev[pos < len(idx)].copy()
    ev["entry"] = idx[pos[pos < len(idx)]]
    entries = list(pd.DatetimeIndex(ev["entry"].unique()).sort_values())
    print(f"  {len(ev):,} 筆營收公告、{len(entries)} 個進場日", flush=True)

    print("計算各基準下的超額報酬…", flush=True)
    br = bench_returns(O, C, turn, mask, ind, tw, entries)
    for h in HORIZONS:
        L = pd.concat(br[h], names=["entry", "code"])
        for b in BENCH:
            L[f"{b}{h}"] = L["ret"] - L[b]
        ev = ev.join(L[[f"{b}{h}" for b in BENCH]], on=["entry", "code"])
    ev = ev[ev[f"uni{HORIZONS[0]}"].notna()]          # 只留進場日在母體內、有開盤價的
    print(f"  母體內事件 {len(ev):,} 筆", flush=True)

    hi30 = ev["is_high"] & (ev["yoy"] >= 0.30)
    events = {
        "全部公告": pd.Series(True, index=ev.index),
        "創歷史新高": ev["is_high"],
        "年增≥30% 且 累計年增≥20%": (ev["yoy"] >= 0.30) & (ev["cum_yoy"] >= 0.20),
        "創新高 且 年增≥30%": hi30,
        "年增≥30% 但未創新高": ~ev["is_high"] & (ev["yoy"] >= 0.30),
        "年增≤−20%": ev["yoy"] <= -0.20,
        "營收驚喜（SUR 最高一成）": ev["sur_dec"] == 10,
        "營收壞消息（SUR 最低一成）": ev["sur_dec"] == 1,
    }

    rows = []
    for name, m in events.items():
        for tag, (lo, hi) in PERIODS.items():
            sel = ev[m & (ev["entry"] >= lo) & (ev["entry"] < hi)]
            for h in HORIZONS:
                for b, bn in BENCH.items():
                    rows.append({"事件": name, "區間": tag, "持有": h, "基準": bn, **summarize(sel, h, b)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "bench.csv", index=False, encoding="utf-8-sig")

    def show(df: pd.DataFrame) -> None:
        print(f"  {'事件':<22}{'區間':<11}{'持有':>4}{'基準':>10}{'事件數':>7}{'平均':>8}"
              f"{'中位':>8}{'扣成本':>8}{'勝率':>6}{'t':>7}")
        for _, r in df.iterrows():
            if pd.isna(r.get("平均超額")):
                print(f"  {r['事件']:<22}{r['區間']:<11}{r['持有']:>4}{r['基準']:>10}{int(r['事件數']):>7}　樣本不足")
                continue
            print(f"  {r['事件']:<22}{r['區間']:<11}{r['持有']:>4}{r['基準']:>10}{int(r['事件數']):>7}"
                  f"{r['平均超額']:>+8.2%}{r['中位超額']:>+8.2%}{r['扣成本']:>+8.2%}"
                  f"{r['勝率']:>6.0%}{r['t']:>7.2f}")

    print(f"\n{'=' * 104}\n超額報酬對母體（前 {a.uni} 名等權），各事件 × 區間 × 持有期\n{'=' * 104}")
    s = res[res["基準"] == "母體"]
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    show(s)

    print(f"\n{'=' * 104}\n基準比較：超額是不是只來自成交值／波動或產業（60 日）\n{'=' * 104}")
    show(res[(res["持有"] == 60) & res["事件"].isin(["全部公告", "創新高 且 年增≥30%", "年增≥30% 但未創新高"])])

    yr = []
    for y in sorted(ev["entry"].dt.year.unique()):
        sel = ev[hi30 & (ev["entry"].dt.year == y)]
        for b, bn in BENCH.items():
            x = sel[f"{b}60"].dropna()
            yr.append({"年": y, "基準": bn, "事件數": len(x), "平均超額": x.mean() if len(x) else np.nan})
    yr = pd.DataFrame(yr)
    yr.to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    w = yr.pivot(index="年", columns="基準", values="平均超額")[list(BENCH.values())]
    n = yr[yr["基準"] == "母體"].set_index("年")["事件數"]
    print(f"\n逐年：創新高且年增≥30%，60 日平均超額")
    print("  年     事件數" + "".join(f"{c:>12}" for c in w.columns))
    for y, r in w.iterrows():
        print(f"  {y}  {int(n[y]):>6}" + "".join(f"{v:>+12.2%}" if pd.notna(v) else f"{'':>12}" for v in r))

    keep = ["code", "ym", "avail_date", "entry", "revenue", "yoy", "cum_yoy", "is_high",
            *[f"{b}{h}" for h in HORIZONS for b in BENCH]]
    ev[keep].to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
