#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
週期性邏輯檢驗：最近這一期的表現，差到「不能用運氣解釋」了嗎？

為什麼不能只看最近兩個月
------------------------
策略每 42 個交易日（約兩個月）換倉一次，所以檢驗也跟著每 42 天做一次。
但 42 天的報酬雜訊極大：高波動組合的 beta 大於 1，大盤跌兩個月策略一定跟著跌，
那不代表選股邏輯壞了。所以這裡比的是「策略 − 大盤」的超額報酬，
並且拿 20 年回測裡所有 42 天窗口的超額報酬當參考分布，問「這次落在哪裡」。

三條規則（事先固定，不看結果再調）
----------------------------------
  A 單期極端   本期 42 天累積超額 < 參考分布第 5 百分位        → 注意
  B 連續偏弱   最近兩個「不重疊」的 42 天窗口都 < 第 10 百分位   → 顯著
               若各期獨立，純運氣連續兩次的機率約 1%
  C 長期漂移   最近 252 天平均日超額，顯著低於參考期平均
               （單尾 Newey-West t 檢定，p < 0.05）                → 顯著
  只有 A 成立時只觀察、不換股；B 或 C 成立才進入換股流程（成本感知換手）。

參考分布只用「被檢驗那段之前」的資料，避免拿未來比現在。
另外做一次逐期回放：從第 5 年起每 42 天檢驗一次，算出歷史上的誤報頻率，
這個數字就是規則的代價 —— 報告裡要一起交代。

輸入
----
  results/spec/daily.csv          strategy_spec.py 產生的逐日報酬（策略、大盤）
  --live 檔案（可選）             實盤逐日報酬，欄位 date、策略、大盤；
                                  有給就接在回測後面檢驗實盤，沒給就檢驗回測最後一期
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
from portfolio_backtest import nw_tstat  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "significance"
WIN = 42              # 與 strategy_spec.REBAL 一致
DRIFT_WIN = 252
P_A, P_B, ALPHA_C = 0.05, 0.10, 0.05
MIN_REF_YEARS = 5


def window_excess(ex: pd.Series) -> pd.Series:
    """逐日滾動的 42 天累積超額（以窗口結束日為索引）。"""
    return np.log1p(ex).rolling(WIN).sum().pipe(np.expm1).dropna()


def check(ex: pd.Series, end: int) -> dict:
    """
    在第 end 個交易日（含）收盤時做檢驗。
    參考分布只用「本期開始之前就已經結束」的窗口。
    """
    cur_start = end - WIN + 1
    prev_start = cur_start - WIN
    ref = window_excess(ex.iloc[:cur_start])
    cur = float(np.expm1(np.log1p(ex.iloc[cur_start:end + 1]).sum()))
    prev = float(np.expm1(np.log1p(ex.iloc[prev_start:cur_start]).sum()))
    # 前一期也要和「它之前」的分布比，才不會用到本期資訊
    ref_prev = window_excess(ex.iloc[:max(prev_start, 0)])

    pct = float((ref < cur).mean())
    pct_prev = float((ref_prev < prev).mean()) if len(ref_prev) else np.nan
    q5, q10 = ref.quantile(P_A), ref.quantile(P_B)
    rule_a = cur < q5
    rule_b = cur < q10 and pct_prev < P_B

    recent = ex.iloc[end - DRIFT_WIN + 1:end + 1]
    base = ex.iloc[:end - DRIFT_WIN + 1]
    t, p2 = nw_tstat(recent - base.mean())
    p1 = p2 / 2 if t < 0 else 1 - p2 / 2          # 單尾：只在乎「變差」
    rule_c = bool(p1 < ALPHA_C)

    level = "顯著" if (rule_b or rule_c) else ("注意" if rule_a else "正常")
    return {"日期": ex.index[end], "本期超額": cur, "百分位": pct,
            "第5百分位": q5, "第10百分位": q10, "前期超額": prev,
            "前期百分位": pct_prev, "近一年日超額": recent.mean(),
            "參考期日超額": base.mean(), "漂移t": t, "漂移p": p1,
            "A單期極端": rule_a, "B連續偏弱": rule_b, "C長期漂移": rule_c,
            "結論": level}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="週期性邏輯檢驗：最近一期是否顯著變差")
    ap.add_argument("--daily", default=str(ROOT / "results" / "spec" / "daily.csv"))
    ap.add_argument("--live", help="實盤逐日報酬 CSV（date、策略、大盤）")
    ap.add_argument("--col", default="策略", help="檢驗哪一欄：策略（含過濾）或 無過濾")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.daily, parse_dates=["date"], index_col="date")
    ex = (df[a.col] - df["大盤"]).dropna()
    if a.live:
        lv = pd.read_csv(a.live, parse_dates=["date"], index_col="date")
        lv_ex = (lv["策略"] - lv["大盤"]).dropna()
        ex = pd.concat([ex[ex.index < lv_ex.index[0]], lv_ex])
        print(f"接上實盤 {len(lv_ex)} 天（{lv_ex.index[0]:%Y-%m-%d} 起）")

    print(f"{'=' * 80}\n一、本期檢驗（{a.col} − 大盤，{WIN} 天窗口）\n{'=' * 80}")
    r = check(ex, len(ex) - 1)
    print(f"  檢驗日          {r['日期']:%Y-%m-%d}")
    print(f"  本期累積超額    {r['本期超額']:+.2%}（歷史第 {r['百分位']:.0%} 百分位）")
    print(f"  門檻            第5百分位 {r['第5百分位']:+.2%}　第10百分位 {r['第10百分位']:+.2%}")
    print(f"  前一期          {r['前期超額']:+.2%}（第 {r['前期百分位']:.0%} 百分位）")
    print(f"  近一年日超額    {r['近一年日超額']:+.4%}　參考期 {r['參考期日超額']:+.4%}"
          f"　t {r['漂移t']:.2f}　單尾 p {r['漂移p']:.3f}")
    for k in ("A單期極端", "B連續偏弱", "C長期漂移"):
        print(f"  規則 {k:<8}  {'成立' if r[k] else '—'}")
    print(f"  結論            {r['結論']}"
          + ("　→ 進入換股流程" if r["結論"] == "顯著" else ""))

    print(f"\n{'=' * 80}\n二、逐期回放：規則在歷史上多常觸發（誤報率）\n{'=' * 80}")
    start = MIN_REF_YEARS * 252
    ends = list(range(start + WIN - 1, len(ex), WIN))
    hist = pd.DataFrame([check(ex, e) for e in ends])
    n = len(hist)
    for k in ("A單期極端", "B連續偏弱", "C長期漂移"):
        print(f"  {k:<10} {int(hist[k].sum()):>3}/{n} 期（{hist[k].mean():.1%}）")
    sig = hist["結論"] == "顯著"
    print(f"  判為顯著     {int(sig.sum()):>3}/{n} 期（{sig.mean():.1%}）"
          f"　約每 {n / max(sig.sum(), 1) * WIN / 252:.1f} 年一次")
    # 觸發之後下一期是否真的繼續變差：若和沒觸發時差不多，規則只是在追雜訊
    nxt = hist["本期超額"].shift(-1)
    print(f"  觸發後下一期平均超額 {nxt[sig].mean():+.2%}"
          f"　未觸發後 {nxt[~sig].mean():+.2%}")
    print("\n  判為顯著的期別：")
    for _, h in hist[sig].iterrows():
        why = "、".join(k[0] for k in ("A單期極端", "B連續偏弱", "C長期漂移") if h[k])
        print(f"    {h['日期']:%Y-%m-%d}  本期 {h['本期超額']:+7.2%}"
              f"（第 {h['百分位']:>4.0%}）  規則 {why}")

    pd.DataFrame([r]).to_csv(OUT / "latest.csv", index=False, encoding="utf-8-sig")
    hist.to_csv(OUT / "history.csv", index=False, encoding="utf-8-sig")
    print(f"\n  輸出：{OUT / 'latest.csv'}、{OUT / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
