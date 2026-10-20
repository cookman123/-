#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
進場時點檢定：跌幅成立後，該在隔日開盤還是隔日收盤進場？

由來
----
news_timing.py 測出：開盤前（前一交易日 13:30 之後至當日 09:00）發布的負面新聞，
會預測當日「開盤→收盤」再跌 −0.158%（群集 t = −4.59）。
幅度小於來回成本 0.39%，所以不能單獨拿來做空；
但它不必增加任何交易就能當成過濾條件——只是把既有的進場往後挪幾小時。

四種進場方式（出場規則完全相同，只差進場價）
--------------------------------------------
    A 隔日開盤          現行做法
    B 隔日收盤          一律延後
    C 有負面新聞才延後   開盤前有負面新聞 → 延到收盤，否則照常開盤進
    D 有任何新聞才延後   開盤前有任何新聞 → 延到收盤

C 是本次要驗的規則。若 C 優於 A 且優於 B，代表「延後」本身沒有用，
有用的是「依新聞決定要不要延後」。若 B 就等於 C，那新聞條件是多餘的。

沒有偷看答案
------------
新聞的可用時點以 13:30 收盤與 09:00 開盤切分，只採計進場日開盤前已發布者。
跌幅在前一日收盤成立，進場價一律取進場日當天的開盤價或收盤價，不取未來價格。
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
import davis_tw_train as T  # noqa: E402
import news_reversal as NR  # noqa: E402
import universe_pit as U  # noqa: E402

ROOT = Path(__file__).resolve().parent
CLOSE_H, CLOSE_M, OPEN_H = 13, 30, 9


def preopen_news(news: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    每個交易日「開盤前已發布」的新聞則數與負面則數。

    開盤前 = 前一交易日 13:30 之後，到當日 09:00 之前。
    盤中（09:00~13:30）發布的不算，因為進場決定在開盤那一刻就要做。
    """
    out = pd.DataFrame(0.0, index=calendar, columns=["n", "neg"])
    if news is None or len(news) == 0:
        return out
    ts = pd.to_datetime(news["ts"])
    h, m = ts.dt.hour, ts.dt.minute
    after_close = (h > CLOSE_H) | ((h == CLOSE_H) & (m >= CLOSE_M))
    before_open = h < OPEN_H
    keep = after_close | before_open
    if not keep.any():
        return out
    # 盤後發布者歸到下一個交易日；凌晨至開盤前發布者歸到當日
    eff = ts.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    idx = calendar.searchsorted(eff.to_numpy(), side="left")
    ok = keep.to_numpy() & (idx < len(calendar))
    if not ok.any():
        return out
    days = calendar[np.clip(idx, 0, len(calendar) - 1)][ok]
    neg = news["title"].fillna("").str.contains(
        "|".join(NR.NEGATIVE_WORDS), case=False, regex=True, na=False).to_numpy()[ok]
    d = pd.DataFrame({"day": days, "neg": neg.astype(float)})
    g = d.groupby("day").agg(n=("neg", "size"), neg=("neg", "sum"))
    out["n"] = g["n"].reindex(calendar).fillna(0.0)
    out["neg"] = g["neg"].reindex(calendar).fillna(0.0)
    return out


def run_stock(px, ns, pre, watch: int, drop_pct: float, max_hold: int) -> list:
    """對單一標的產生事件，並同時記下四種進場方式各自的報酬。"""
    idx = px.index
    o, c = px["open"].to_numpy(float), px["close"].to_numpy(float)
    n = len(idx)
    pre_n, pre_neg = pre["n"].to_numpy(), pre["neg"].to_numpy()
    out, busy = [], -1

    for i0 in range(n):
        if i0 <= busy:
            continue
        ref = c[i0]
        if not np.isfinite(ref) or ref <= 0:
            continue
        hit = None
        for k in range(1, watch + 1):
            i = i0 + k
            if i >= n - 1:
                break
            if np.isfinite(c[i]) and c[i] / ref - 1 <= -abs(drop_pct):
                hit = i
                break
        if hit is None:
            continue

        e = hit + 1                       # 進場日
        if e >= n - 1:
            continue
        p_open, p_close = o[e], c[e]
        if not (np.isfinite(p_open) and np.isfinite(p_close) and p_open > 0 and p_close > 0):
            continue

        # 出場：自進場日起算，收盤回到基準價即次日開盤出場；逾期強制出場
        exit_i, reason = None, "時間到"
        for j in range(e, min(e + max_hold, n - 1)):
            if np.isfinite(c[j]) and c[j] >= ref:
                exit_i, reason = j + 1, "回到基準"
                break
        if exit_i is None:
            exit_i = min(e + max_hold, n - 1)
        px_out = o[exit_i]
        if not np.isfinite(px_out) or px_out <= 0:
            continue

        def net(entry):
            return (px_out * (1 - NR.COST_SELL)) / (entry * (1 + NR.COST_BUY)) - 1

        has_neg = pre_neg[e] >= 1
        has_any = pre_n[e] >= 1
        out.append({
            "drop_day": idx[hit], "entry": idx[e], "exit": idx[exit_i],
            "hold": exit_i - e, "reason": reason, "drop": c[hit] / ref - 1,
            "pre_neg": bool(has_neg), "pre_any": bool(has_any),
            "A_open": net(p_open),
            "B_close": net(p_close),
            "C_news": net(p_close if has_neg else p_open),
            "D_any": net(p_close if has_any else p_open),
            "open_to_close": p_close / p_open - 1,
        })
        busy = exit_i
    return out


def cluster_t(v: pd.Series, dates: pd.Series) -> tuple:
    d = pd.DataFrame({"v": v.to_numpy(),
                      "m": pd.to_datetime(dates).dt.to_period("M")}).dropna()
    if len(d) < 30:
        return np.nan, np.nan, 0
    g = d.groupby("m")["v"].mean()
    if len(g) < 5 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan, len(g)
    return (float(d["v"].mean()),
            float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), len(g))


def paired_t(a: pd.Series, b: pd.Series, dates: pd.Series) -> tuple:
    """同一批事件的兩種進場方式相減，再以月為群集檢定差異。"""
    return cluster_t(a - b, dates)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="進場時點：開盤 vs 收盤 vs 依新聞決定")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--watch", type=int, default=5)
    ap.add_argument("--drop", type=float, default=0.08)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--topn", type=int, default=120)
    args = ap.parse_args(argv)

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"掃描 {len(codes)} 檔　期間 {args.start} ~ {args.end}")
    rows = []
    for i, code in enumerate(codes, 1):
        px, news = NR.load_stock(code, args.start, args.end)
        if px is None:
            continue
        pre = preopen_news(news, px.index)
        r = run_stock(px, pre, pre, args.watch, args.drop, args.hold) if False else \
            run_stock(px, None, pre, args.watch, args.drop, args.hold)
        for x in r:
            x["code"] = code
        rows.extend(r)
        if i % 80 == 0:
            print(f"  {i}/{len(codes)}")
    ev = pd.DataFrame(rows)
    if ev.empty:
        print("沒有事件")
        return 1

    ev = U.filter_trades(ev, U.get_mask(args.start, args.end, args.topn), "drop_day")
    print(f"時點正確篩選後 {len(ev):,} 筆　"
          f"其中開盤前有負面新聞 {int(ev['pre_neg'].sum()):,} 筆"
          f"（{ev['pre_neg'].mean():.1%}）\n")

    print("=" * 88)
    print("四種進場方式（出場規則相同，只差進場價）")
    print("=" * 88)
    print(f"  {'方式':<26}{'平均淨報酬':>12}{'群集t':>9}{'延後筆數':>10}")
    base = None
    for col, label, nlate in (
            ("A_open", "A 隔日開盤（現行）", 0),
            ("B_close", "B 隔日收盤（一律延後）", len(ev)),
            ("C_news", "C 有負面新聞才延後", int(ev["pre_neg"].sum())),
            ("D_any", "D 有任何新聞才延後", int(ev["pre_any"].sum()))):
        m, t, _ = cluster_t(ev[col], ev["entry"])
        if base is None:
            base = m
        print(f"  {label:<26}{m:>+12.2%}{t:>9.2f}{nlate:>10,}")

    print("\n" + "=" * 88)
    print("與現行做法（A）的配對差異")
    print("=" * 88)
    print(f"  {'比較':<26}{'差異':>10}{'群集t':>9}{'月數':>7}")
    for col, label in (("B_close", "B 一律延後 − A"),
                       ("C_news", "C 依負面新聞 − A"),
                       ("D_any", "D 依任何新聞 − A")):
        d, t, n = paired_t(ev[col], ev["A_open"], ev["entry"])
        print(f"  {label:<26}{d:>+10.3%}{t:>9.2f}{n:>7}")

    print("\n" + "=" * 88)
    print("只看「開盤前有負面新聞」的那批事件（延後才會改變的部分）")
    print("=" * 88)
    sub = ev[ev["pre_neg"]]
    if len(sub) > 50:
        mo, to, _ = cluster_t(sub["A_open"], sub["entry"])
        mc, tc, _ = cluster_t(sub["B_close"], sub["entry"])
        d, t, n = paired_t(sub["B_close"], sub["A_open"], sub["entry"])
        moc, toc, _ = cluster_t(sub["open_to_close"], sub["entry"])
        print(f"  筆數 {len(sub):,}")
        print(f"    開盤進場　{mo:+.2%}（t={to:.2f}）")
        print(f"    收盤進場　{mc:+.2%}（t={tc:.2f}）")
        print(f"    差異　　　{d:+.3%}（t={t:.2f}，{n} 個月）")
        print(f"    這些日子的開盤→收盤　{moc:+.3%}（t={toc:.2f}）"
              f"　—— 為負代表延後進場買得更便宜")
    else:
        print("  樣本不足")

    ev.to_csv(ROOT / "results" / "entry_timing.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出：results/entry_timing.csv")
    print("判讀：C 要同時優於 A 與 B，才代表『依新聞決定』本身有價值；"
          "若 B 就等於 C，新聞條件是多餘的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
