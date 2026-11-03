#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可執行規格：高波動選股 ＋ 大盤趨勢過濾。

規則（每一條都只用當下已知的資料）
----------------------------------
選股範圍   每日以「近 60 個交易日平均成交金額」排名，取前 250 名
           用成交金額而非市值：成交金額當天就知道，且直接代表能不能成交
選股       每 42 個交易日（約兩個月）第一天，在範圍內以「近 60 日報酬標準差」排名
           排名用前一日為止的資料，當日收盤才建立部位
緩衝帶     舊持股排名仍在前 40 名就續抱，掉出前 40 名才賣；空出的位置由排名最高的
           非持股補滿 30 檔
權重       第一次等權；之後續抱的部位不拉回等權，賣出所得平均分給新買進的股票
           期間內不再調整，讓權重隨價格漂移
           （舊規則是每 63 天、只留前 30 名、每次重設等權，回測一併列出對照）
趨勢過濾   大盤收在 120 日均線之上才持有；跌破則全數出場、空手等待
           訊號用前一日收盤判斷，次日執行
成本       手續費 0.1425% 的 3 折，買賣各一次；賣出加證交稅 0.3%
           趨勢切換時整個部位進出，各收一次

本程式做兩件事
--------------
    一、用完整歷史回測這套規則，給出年化、夏普、最大回撤與逐年表現
    二、印出「若今天換倉，會買哪 30 檔」以及目前的趨勢訊號狀態

產生的清單是規則的輸出，不是投資建議；使用前請看最後的風險說明。
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
import davis_tw_train as T  # noqa: E402
import factor_test as F  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "spec"
FEE, TAX = 0.001425 * 0.3, 0.003
UNIVERSE_N, TOP_N, REBAL, SELL_N, VOL_WIN, MA = 250, 30, 42, 40, 60, 120
OLD_REBAL = 63
# 目前實際持股（欄位「代號」），有這個檔才能算緩衝帶的續抱／賣出；沒有就當成空手
HOLDINGS = OUT / "holdings.csv"


def backtest_buffer(score: pd.DataFrame, C: pd.DataFrame, mask: pd.DataFrame,
                    top_n: int, rebal: int, sell_n: int) -> pd.Series:
    """
    緩衝帶版的 F.backtest：舊持股排名在前 sell_n 名內就續抱、保留漂移後的權重，
    掉出才賣；賣出所得平均分給補進來的新股。成本計法與 F.backtest 相同。

    2006 起回測（含成本）：42 天＋前 40 名賣出線 年化 +23.4%（t 3.15），
    舊規則 63 天＋前 30 名＋重設等權 +19.0%（t 2.41）。成本只省約 0.2 個百分點，
    多出來的主要來自「不重設等權」讓漲多的股票續抱，且集中在 2016 年後，
    前半段（2006–2015）超額 t 只有 1.1。另外這是從約 16 種組合中挑出的，有挑選偏誤。
    """
    dates = C.index
    rebs = list(range(0, len(dates), rebal))
    nav, nav_d, prev = [1.0], [], pd.Series(dtype=float)

    for bi, i0 in enumerate(rebs):
        i1 = rebs[bi + 1] if bi + 1 < len(rebs) else len(dates) - 1
        if i1 <= i0:
            continue
        d0 = dates[i0]
        s = score.loc[d0].dropna() if d0 in score.index else pd.Series(dtype=float)
        if s.empty:
            continue
        seg = dates[i0:i1 + 1]
        cand = [c for c in s.index if c in C.columns and c in mask.columns
                and bool(mask.at[d0, c])] if d0 in mask.index else []
        p0 = C.loc[d0, cand]
        cand = [c for c in cand if pd.notna(p0[c]) and p0[c] > 0]
        if len(cand) < min(top_n, 10):
            continue
        rank = s[cand].rank(ascending=False)
        keep = sorted((c for c in prev.index if c in rank.index and rank[c] <= sell_n),
                      key=lambda c: rank[c])[:top_n]
        new = [c for c in rank.sort_values().index if c not in keep][:top_n - len(keep)]
        if prev.empty:
            w = pd.Series(1.0 / len(new), index=new)
        elif new:
            kw = prev.reindex(keep)
            w = pd.concat([kw, pd.Series((1 - kw.sum()) / len(new), index=new)])
        else:
            w = prev.reindex(keep) / prev.reindex(keep).sum()

        allc = prev.index.union(w.index)
        diff = w.reindex(allc).fillna(0.0) - prev.reindex(allc).fillna(0.0)
        sell, buy = float(-diff[diff < 0].sum()), float(diff[diff > 0].sum())
        nav[-1] *= (1 - sell * (FEE + TAX) - buy * FEE)

        px = C.loc[seg, w.index].ffill()
        rel = px.div(px.iloc[0], axis=1)
        val = (rel * w).sum(axis=1)
        val = val / val.iloc[0]
        for j in range(1, len(seg)):
            nav.append(nav[-1] * float(val.iloc[j] / val.iloc[j - 1]))
            nav_d.append(seg[j])
        last = rel.iloc[-1] * w
        prev = last / last.sum()

    if not nav_d:
        return pd.Series(dtype=float)
    return pd.Series(nav[1:], index=pd.DatetimeIndex(nav_d)).pct_change().dropna()


def apply_filter(strat: pd.Series, bench_px: pd.Series, ma: int) -> tuple:
    """大盤跌破均線就空手；切換當日扣一次整體進出成本。"""
    b = bench_px.reindex(strat.index.union(bench_px.index)).ffill()
    sig = (b > b.rolling(ma).mean()).shift(1).reindex(strat.index).fillna(False)
    out, prev, sw = [], False, 0
    for d, on in sig.items():
        r = float(strat.get(d, 0.0)) if on else 0.0
        if bool(on) != prev:
            r -= (FEE + TAX) + FEE
            sw += 1
            prev = bool(on)
        out.append(r)
    return pd.Series(out, index=strat.index), sig, sw


def perf(r: pd.Series, base: pd.Series | None = None) -> dict:
    r = r.dropna()
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    d = {"年化": cagr, "夏普": r.mean() / r.std() * np.sqrt(252),
         "最大回撤": dd, "報酬回撤比": cagr / abs(dd) if dd else np.nan, "年數": yrs}
    if base is not None:
        idx = r.index.intersection(base.index)
        t, p = nw_tstat(r.reindex(idx) - base.reindex(idx))
        byrs = len(idx) / 252
        d["超額"] = cagr - ((1 + base.reindex(idx)).prod() ** (1 / byrs) - 1)
        d["超額t"], d["超額p"] = t, p
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="高波動＋趨勢過濾：規格與當前清單")
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"載入 {len(codes)} 檔股價…")
    C = F.load_panel(codes, a.start, a.end)
    mask = U.get_mask(a.start, a.end, UNIVERSE_N, refresh=True)
    vol = C.pct_change().rolling(VOL_WIN).std().shift(1)      # 前一日為止

    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv",
                     parse_dates=["date"]).set_index("date")["close"]
    strat = backtest_buffer(vol, C, mask, TOP_N, REBAL, SELL_N)
    filt, sig, sw = apply_filter(strat, tw, MA)
    bh = tw.pct_change().reindex(strat.index).fillna(0.0)
    old = F.backtest(vol, C, mask, TOP_N, OLD_REBAL)
    old_filt, _, _ = apply_filter(old, tw, MA)
    old_bh = tw.pct_change().reindex(old.index).fillna(0.0)

    print(f"\n{'=' * 92}\n一、回測結果\n{'=' * 92}")
    rows = [("大盤買進持有", perf(bh)),
            ("高波動（無過濾）", perf(strat, bh)),
            (f"高波動 ＋ {MA} 日均線過濾", perf(filt, bh)),
            (f"舊規則：{OLD_REBAL} 天、重設等權（無過濾）", perf(old, old_bh)),
            (f"舊規則：{OLD_REBAL} 天、重設等權 ＋ 過濾", perf(old_filt, old_bh))]
    print(f"  {'設定':<24}{'年化':>9}{'夏普':>7}{'最大回撤':>10}"
          f"{'報酬/回撤':>10}{'超額':>9}{'超額t':>7}")
    for name, s in rows:
        print(f"  {name:<24}{s['年化']:>+9.2%}{s['夏普']:>7.2f}"
              f"{s['最大回撤']:>+10.2%}{s['報酬回撤比']:>10.2f}"
              f"{s.get('超額', float('nan')):>+9.2%}"
              f"{s.get('超額t', float('nan')):>7.2f}")
    print(f"  在場比例 {sig.mean():.0%}　年均切換 {sw / (len(strat)/252):.1f} 次")

    print(f"\n{'=' * 92}\n二、逐年表現（過濾版 vs 大盤）\n{'=' * 92}")
    df = pd.DataFrame({"策略": filt, "大盤": bh}).dropna()
    yr = df.groupby(df.index.year).apply(
        lambda g: pd.Series({"策略": (1 + g["策略"]).prod() - 1,
                             "大盤": (1 + g["大盤"]).prod() - 1}), include_groups=False)
    yr["差"] = yr["策略"] - yr["大盤"]
    win = int((yr["差"] > 0).sum())
    print(f"  {'年':<6}{'策略':>10}{'大盤':>10}{'差':>10}")
    for y, r in yr.iterrows():
        print(f"  {y:<6}{r['策略']:>+10.1%}{r['大盤']:>+10.1%}{r['差']:>+10.1%}")
    print(f"  勝過大盤 {win}/{len(yr)} 年")
    yr.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")
    # 逐日報酬留給 significance_check.py 當參考分布
    pd.DataFrame({"策略": filt, "無過濾": strat, "大盤": bh, "在場": sig.astype(int)}
                 ).to_csv(OUT / "daily.csv", index_label="date", encoding="utf-8-sig")
    pd.DataFrame([{"設定": n, **s} for n, s in rows]).to_csv(
        OUT / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 92}\n三、目前狀態與清單\n{'=' * 92}")
    # 選股基準日必須是「各檔資料都到齊」的那一天。
    # 股價若不是同一天抓的，直接用面板最後一日選股，等於從一個碰巧被重爬的
    # 子集裡挑 —— 實測這個錯誤讓清單有一半是錯的（兩份只重疊 15/30 檔）。
    n_valid = C.notna().sum(axis=1)
    last = n_valid[n_valid >= n_valid.max() * 0.9].index[-1]
    if last != C.index[-1]:
        print(f"  ⚠ 資料未對齊：面板最後日 {C.index[-1]:%Y-%m-%d} 只有 "
              f"{int(n_valid.iloc[-1])} 檔有資料，改用最後一個全覆蓋日")
        print("    建議先跑 python refresh_prices.py 把股價重抓到同一天")
    ma_now = float(tw.rolling(MA).mean().iloc[-1])
    tw_now = float(tw.iloc[-1])
    on = tw_now > ma_now
    print(f"  選股基準日　　{last:%Y-%m-%d}（{int(n_valid[last])} 檔有資料）")
    print(f"  大盤資料至　　{tw.index[-1]:%Y-%m-%d}")
    print(f"  大盤 {tw_now:,.0f}　{MA} 日均線 {ma_now:,.0f}"
          f"　→ {'均線之上，規則為持有' if on else '跌破均線，規則為空手'}")

    s = vol.loc[last].dropna()
    cand = [c for c in s.index if c in mask.columns and bool(mask.at[last, c])] \
        if last in mask.index else list(s.index)
    # 換倉排程跟回測一樣，從面板第一天起每 REBAL 個交易日一次
    i_last = C.index.get_loc(last)
    prev_reb = C.index[i_last // REBAL * REBAL]
    left = (i_last // REBAL + 1) * REBAL - i_last
    print(f"  換倉排程　　　每 {REBAL} 個交易日；上次換倉日 {prev_reb:%Y-%m-%d}，"
          f"約 {left} 個交易日後再換")

    held = []
    if HOLDINGS.exists():
        held = pd.read_csv(HOLDINGS, dtype={"代號": str}, encoding="utf-8-sig")["代號"].tolist()
    rank = s[cand].rank(ascending=False)
    keep = sorted((c for c in held if c in rank.index and rank[c] <= SELL_N),
                  key=lambda c: rank[c])[:TOP_N]
    new = [c for c in rank.sort_values().index if c not in keep][:TOP_N - len(keep)]
    sold = [c for c in held if c not in keep]
    picks = s[keep + new].sort_values(ascending=False)
    sec = {r["code"]: (r["name"], r["category"]) for r in
           json.load(open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))}
    if not held:
        print(f"  （沒有 {HOLDINGS.name}，視同空手：直接取前 {TOP_N} 名等權買進）")
    print(f"\n  若今日換倉，會持有的 30 檔（依近 {VOL_WIN} 日波動度排序；"
          f"舊持股在前 {SELL_N} 名內續抱，賣出所得平均分給新買進）：")
    print(f"  {'代號':<8}{'名稱':<12}{'產業':<12}{'排名':>6}{'年化波動':>10}  動作")
    out_rows = []
    for code, v in picks.items():
        nm, cat = sec.get(code, ("？", "？"))
        av = v * np.sqrt(252)
        act = "續抱" if code in keep else "買進"
        print(f"  {code:<8}{nm:<12}{cat:<12}{int(rank[code]):>6}{av:>10.1%}  {act}")
        out_rows.append({"代號": code, "名稱": nm, "產業": cat, "排名": int(rank[code]),
                         "年化波動": av, "動作": act})
    for code in sold:
        nm, cat = sec.get(code, ("？", "？"))
        r_ = f"{int(rank[code])}" if code in rank.index else "範圍外"
        print(f"  {code:<8}{nm:<12}{cat:<12}{r_:>6}{'':>10}  賣出")
        out_rows.append({"代號": code, "名稱": nm, "產業": cat,
                         "排名": int(rank[code]) if code in rank.index else np.nan,
                         "年化波動": np.nan, "動作": "賣出"})
    # 處置／注意／全額交割股（restricted.py，取自 tw-stock-quant）：隔日開盤不能正常買。
    # 只標示、不改清單 —— 回測的規格沒有這條，換掉會讓清單和回測對不上；要不要暫緩買進由人決定
    import restricted as RS
    out = pd.DataFrame(out_rows)
    flag = RS.flags(pd.DataFrame({"date": last, "code": out["代號"]}), C.index, 1)
    out["交易限制"] = np.where(flag & (out["動作"] == "買進"), "處置／注意，隔日不能正常買", "")
    if flag.any():
        print(f"\n  ⚠ 處置／注意／全額交割：{'、'.join(out.loc[flag, '代號'])}"
              "（隔日不能正常買；清單照規格不變）")
    out.to_csv(OUT / "current_picks.csv", index=False, encoding="utf-8-sig")
    cats = pd.Series([sec.get(c, ("", "？"))[1] for c in picks.index]).value_counts()
    print(f"\n  產業分布：{'、'.join(f'{k} {v}' for k, v in cats.head(6).items())}")
    print(f"  輸出：{OUT / 'current_picks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
