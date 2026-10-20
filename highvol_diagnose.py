#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高波動因子的兩個必要檢查：產業集中度，以及趨勢過濾能否壓低回撤。

背景
----
2003–2024、21.4 年的檢定：
    高波動/前250/持30檔   年化 +16.43%、夏普 0.69、最大回撤 −72.8%、超額 t = 2.14
    大盤 TWII             年化  +7.90%、夏普 0.50、最大回撤 −58.3%
Reality Check p = 0.037（拒絕），但 Hansen SPA p = 0.119（不拒絕），結果在邊緣。

兩個問題決定它是不是真的可用：

第一　它是不是只是變相重押電子？
      台股的高波動股集中在電子與小型成長股。若持股長期都是同一批電子股，
      那買的是「產業集中投資」，不是波動度因子 —— 結論會完全不同。
      檢查：持股的產業分布、集中度（HHI）、與選股母體的分布差異、名單換手率。

第二　−72.8% 的回撤能不能壓下來？
      夏普只看波動，不反映毀滅性的尾部風險。資產剩四分之一時，
      多數人會停損出場，然後拿不到後面的反彈，所以年化 16% 是拿不到的。
      檢查：以大盤均線當趨勢過濾（跌破則空手或轉持大盤），
      看回撤能壓多少、報酬要付出多少代價。

趨勢訊號一律用「前一日收盤」計算後才在次日執行，不使用當日資訊。
每次進出場依實際換手收取手續費與證交稅。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factor_test as F  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from reality_check import reality_check  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "highvol"
FEE, TAX = 0.001425 * 0.3, 0.003


def holdings_log(score, C, mask, top_n, rebal=63):
    """回傳每個換倉日選到的名單，用來看產業分布與換手。"""
    dates, picks_all = C.index, []
    for i0 in range(0, len(dates), rebal):
        i1 = min(i0 + rebal, len(dates) - 1)
        if i1 <= i0:
            continue
        d0 = dates[i0]
        if d0 not in score.index:
            continue
        s = score.loc[d0].dropna()
        cand = [c for c in s.index if c in C.columns]
        if d0 in mask.index:
            cand = [c for c in cand if c in mask.columns and bool(mask.at[d0, c])]
        seg = dates[i0:i1 + 1]
        if not cand:
            continue
        sub = C.loc[seg, cand]
        ok = sub.notna().all() & (sub > 0).all()
        cand = [c for c in cand if bool(ok.get(c, False))]
        if len(cand) < top_n:
            continue
        picks_all.append((d0, list(s[cand].nlargest(top_n).index), cand))
    return picks_all


def trend_overlay(strat: pd.Series, bench_px: pd.Series, ma: int,
                  fallback: pd.Series | None) -> pd.Series:
    """
    趨勢過濾：大盤收在 ma 日均線之上才持有策略，否則空手或轉持大盤。

    訊號用「前一日」的收盤與均線判斷，次日才執行，不使用當日資訊。
    每次切換依整個部位收取一次來回成本（賣出付手續費＋證交稅、買回付手續費）。
    """
    b = bench_px.reindex(strat.index.union(bench_px.index)).ffill()
    sig = (b > b.rolling(ma).mean()).shift(1).reindex(strat.index).fillna(False)
    alt = fallback.reindex(strat.index).fillna(0.0) if fallback is not None \
        else pd.Series(0.0, index=strat.index)
    out, prev = [], False
    for d, on in sig.items():
        r = float(strat.get(d, 0.0)) if on else float(alt.get(d, 0.0))
        if bool(on) != prev:
            r -= (FEE + TAX) + FEE          # 全數換手一次
            prev = bool(on)
        out.append(r)
    return pd.Series(out, index=strat.index)


def stats(r: pd.Series, br: pd.Series) -> dict:
    idx = r.index.intersection(br.index)
    r, br = r.reindex(idx), br.reindex(idx)
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    t, p = nw_tstat(r - br)
    return {
        "年化": (1 + r).prod() ** (1 / yrs) - 1,
        "夏普": r.mean() / r.std() * np.sqrt(252),
        "最大回撤": float((eq / eq.cummax() - 1).min()),
        "超額": (1 + r).prod() ** (1 / yrs) - 1 - ((1 + br).prod() ** (1 / yrs) - 1),
        "超額t": t, "超額p": p,
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="高波動因子的產業集中度與趨勢過濾")
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--uni", type=int, default=250)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"載入 {len(codes)} 檔股價…")
    C = F.load_panel(codes, args.start, args.end)
    fs = F.factors(C, pd.DataFrame())
    mask = U.get_mask(args.start, args.end, args.uni)
    score = fs["高波動"]

    # ---------- 一、產業集中度 ----------
    sec = {r["code"]: r["category"] for r in
           json.load(open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))}
    logs = holdings_log(score, C, mask, args.top)
    print(f"\n{'=' * 92}\n一、產業集中度（{len(logs)} 個換倉日）\n{'=' * 92}")

    held, univ = [], []
    prev, turns = set(), []
    for d0, picks, cand in logs:
        held += [sec.get(c, "未分類") for c in picks]
        univ += [sec.get(c, "未分類") for c in cand]
        cur = set(picks)
        if prev:
            turns.append(1 - len(cur & prev) / len(cur))
        prev = cur
    hs = pd.Series(held).value_counts(normalize=True)
    us = pd.Series(univ).value_counts(normalize=True)
    cmp = pd.DataFrame({"持股佔比": hs, "母體佔比": us}).fillna(0.0)
    cmp["差距"] = cmp["持股佔比"] - cmp["母體佔比"]
    cmp = cmp.sort_values("持股佔比", ascending=False)
    print(f"  {'產業':<12}{'持股佔比':>10}{'母體佔比':>10}{'差距':>10}")
    for name, row in cmp.head(10).iterrows():
        print(f"  {name:<12}{row['持股佔比']:>10.1%}{row['母體佔比']:>10.1%}{row['差距']:>+10.1%}")
    hhi = float((hs ** 2).sum())
    hhi_u = float((us ** 2).sum())
    print(f"\n  集中度 HHI：持股 {hhi:.3f}　母體 {hhi_u:.3f}"
          f"　（1/28 = {1/28:.3f} 為完全分散）")
    print(f"  每季名單換手率 平均 {np.mean(turns):.1%}"
          f"　—— 越高代表越不是長期重押同一批股票")
    elec = [k for k in hs.index if any(x in k for x in
            ("半導體", "電子", "光電", "資訊", "通信", "電腦", "數位"))]
    print(f"  電子相關合計：持股 {hs[elec].sum():.1%}　母體 {us[elec].sum():.1%}")
    cmp.to_csv(OUT / "sector_mix.csv", encoding="utf-8-sig")

    # ---------- 二、趨勢過濾 ----------
    print(f"\n{'=' * 92}\n二、趨勢過濾能否壓低回撤\n{'=' * 92}")
    strat = F.backtest(score, C, mask, args.top)
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv",
                     parse_dates=["date"]).set_index("date")["close"]
    br = tw.pct_change().reindex(strat.index).fillna(0.0)

    rows, series = [], {}
    base = stats(strat, br)
    rows.append({"設定": "高波動（無過濾）", **base})
    series["高波動（無過濾）"] = (strat - br).dropna()
    for ma in (120, 200):
        for fb, lab in ((None, "空手"), (br, "轉持大盤")):
            r = trend_overlay(strat, tw, ma, fb)
            name = f"高波動＋{ma}日均線過濾／{lab}"
            rows.append({"設定": name, **stats(r, br)})
            series[name] = (r - br).dropna()
    rows.append({"設定": "大盤買進持有", **stats(br, br)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "trend_filter.csv", index=False, encoding="utf-8-sig")
    print(f"  {'設定':<28}{'年化':>9}{'夏普':>7}{'最大回撤':>10}{'超額':>9}{'超額t':>7}")
    for _, x in res.iterrows():
        print(f"  {x['設定']:<28}{x['年化']:>+9.2%}{x['夏普']:>7.2f}"
              f"{x['最大回撤']:>+10.2%}{x['超額']:>+9.2%}{x['超額t']:>7.2f}")

    print(f"\n{'=' * 92}\nReality Check（含過濾變體）\n{'=' * 92}")
    names = [n for n in series if len(series[n]) > 500]
    common = None
    for n in names:
        common = series[n].index if common is None else common.intersection(series[n].index)
    F_ = np.column_stack([series[n].reindex(common).to_numpy() for n in names])
    print(f"  策略族 {F_.shape[1]} 組，共同樣本 {F_.shape[0]:,} 天")
    for mb in (63, 126):
        r = reality_check(F_, mean_block=mb, n_boot=3000)
        print(f"  平均區塊 {mb:>3} 日：最佳＝{names[r['最佳策略序號']]}"
              f"　單獨 t {r['最佳策略單獨t值']:.2f}")
        print(f"    Reality Check p = {r['RC_p']:.4f}　Hansen SPA p = {r['SPA_p']:.4f}")
    print(f"\n輸出：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
