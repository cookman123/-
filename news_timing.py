#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新聞發布時點與可交易區間的對應檢定。

現在的做法（features_text._to_trading_day）
-------------------------------------------
    13:30 前發布 → 歸當日　／　13:30 後發布 → 順延隔日
然後所有歸到同一交易日的新聞，一律用來預測「該日收盤起算」的報酬。

沒處理到的區別
--------------
兩種新聞能動手的最早時點其實不同：

    昨日盤後（13:30 後）  最早可在今日 09:00 開盤進場 → 吃得到「開盤→收盤」
    今日盤中（09:00~13:30）最早只能在今日收盤進場     → 只吃得到「收盤→次日」

現在一律當成「收盤才進場」，等於放棄了盤後新聞可捕捉的開盤到收盤那一段。
但費半的分析已顯示隔夜資訊有很大一部分在開盤瞬間就被消化，個股新聞是否相同未知。

三個要分開量的區間
------------------
    跳空      前一日收盤 → 當日開盤   （知道了也吃不到，開盤價已反映）
    盤中      當日開盤 → 當日收盤     （盤後新聞可捕捉的部分）
    隔日      當日收盤 → 次日收盤     （盤中新聞可捕捉的部分）

顯著性用以月份為群集的 t 值：重大消息會同時影響許多股票與連續多天。
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
from news_reversal import NEGATIVE_WORDS  # noqa: E402

ROOT = Path(__file__).resolve().parent
CLOSE_H, CLOSE_M = 13, 30
OPEN_H = 9


def build(codes: list, start: str, end: str) -> pd.DataFrame:
    """
    每個 (股票, 交易日) 一列，帶三段報酬與兩類新聞計數。

    兩類新聞：
        pre_open  前一交易日 13:30 之後 ~ 當日 09:00 之前發布（開盤前已知）
        intraday  當日 09:00 ~ 13:30 之間發布（盤中才知道）
    """
    neg_pat = "|".join(NEGATIVE_WORDS)
    rows = []
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        if len(px) < 250:
            continue
        news = T.load_news(code)
        if news is None or len(news) == 0:
            continue

        cal = px.index
        ts = pd.to_datetime(news["ts"])
        title = news["title"].fillna("")
        is_neg = title.str.contains(neg_pat, case=False, regex=True, na=False)

        # 每則新聞能最早動手的交易日，以及屬於哪一類
        h, m = ts.dt.hour, ts.dt.minute
        after_close = (h > CLOSE_H) | ((h == CLOSE_H) & (m >= CLOSE_M))
        before_open = h < OPEN_H
        # 盤後發布（含午夜到開盤前）→ 下一個交易日的「開盤前」可用
        nxt = ts.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
        idx = cal.searchsorted(nxt.to_numpy(), side="left")
        ok = idx < len(cal)
        day = pd.Series(pd.NaT, index=ts.index)
        day[ok] = cal[np.clip(idx, 0, len(cal) - 1)][ok]
        kind = np.where(after_close | before_open, "pre_open", "intraday")

        d = pd.DataFrame({"day": day, "kind": kind, "neg": is_neg.astype(float)}).dropna(subset=["day"])
        if d.empty:
            continue
        g = d.groupby(["day", "kind"]).agg(n=("neg", "size"), neg=("neg", "sum")).unstack(fill_value=0)
        out = pd.DataFrame(index=cal)
        for k in ("pre_open", "intraday"):
            out[f"n_{k}"] = g[("n", k)].reindex(cal).fillna(0.0) if ("n", k) in g.columns else 0.0
            out[f"neg_{k}"] = g[("neg", k)].reindex(cal).fillna(0.0) if ("neg", k) in g.columns else 0.0

        o, c = px["open"], px["close"]
        out["gap"] = o / c.shift(1) - 1            # 前收 → 開盤（吃不到）
        out["intra"] = c / o - 1                   # 開盤 → 收盤（盤後新聞可吃）
        out["next"] = c.shift(-1) / c - 1          # 收盤 → 次日收（盤中新聞可吃）
        out["next5"] = c.shift(-5) / c - 1
        out["code"] = code
        out = out.reset_index().rename(columns={"index": "date", px.index.name or "index": "date"})
        rows.append(out)
        if i % 80 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cluster_t(v: pd.Series, dates: pd.Series) -> tuple:
    d = pd.DataFrame({"v": v.to_numpy(), "m": pd.to_datetime(dates).dt.to_period("M")}).dropna()
    if len(d) < 60:
        return np.nan, np.nan
    g = d.groupby("m")["v"].mean()
    if len(g) < 6 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan
    return float(d["v"].mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="新聞時點 × 可交易區間")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--top-n", type=int, default=150)
    args = ap.parse_args(argv)

    import json
    mp = json.load(open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))
    rank = {r["code"]: r.get("market_cap") or 0 for r in mp}
    codes = sorted((x.name for x in (ROOT / "data" / "stocks").iterdir()
                    if x.is_dir() and any((x / "price").glob("*.csv"))),
                   key=lambda c: -rank.get(c, 0))[:args.top_n]
    print(f"標的 {len(codes)} 檔　期間 {args.start} ~ {args.end}")

    A = build(codes, args.start, args.end).dropna(subset=["gap", "intra", "next"])
    print(f"樣本 {len(A):,} 個股票-交易日\n")

    print("=" * 96)
    print("一、兩類新聞各自出現後，三段報酬分別是多少")
    print("=" * 96)
    base = {k: cluster_t(A[k], A["date"])[0] for k in ("gap", "intra", "next")}
    print(f"  {'分組':<28}{'樣本':>9}" + "".join(f"{c:>19}" for c in
          ("跳空(吃不到)", "盤中(盤後新聞可吃)", "隔日(盤中新聞可吃)")))
    print(f"  {'全部交易日':<28}{len(A):>9,}"
          + "".join(f"{base[k]:>+18.3%} " for k in ("gap", "intra", "next")))

    groups = [
        ("開盤前有新聞", A[A["n_pre_open"] > 0]),
        ("開盤前有負面新聞", A[A["neg_pre_open"] > 0]),
        ("盤中有新聞", A[A["n_intraday"] > 0]),
        ("盤中有負面新聞", A[A["neg_intraday"] > 0]),
    ]
    for name, g in groups:
        if len(g) < 200:
            continue
        cells = []
        for k in ("gap", "intra", "next"):
            m, t = cluster_t(g[k], g["date"])
            cells.append(f"{m - base[k]:>+10.3%}(t={t:>5.2f})")
        print(f"  {name:<28}{len(g):>9,}" + "".join(f"{x:>19}" for x in cells))
    print("\n  數字是「該組報酬 − 全部交易日平均」，即超額。")

    print("\n" + "=" * 96)
    print("二、關鍵問題：開盤前就知道的新聞，開盤後還吃得到嗎")
    print("=" * 96)
    pre = A[A["neg_pre_open"] > 0]
    if len(pre) > 200:
        mg, tg = cluster_t(pre["gap"], pre["date"])
        mi, ti = cluster_t(pre["intra"], pre["date"])
        print(f"  開盤前有負面新聞（{len(pre):,} 筆）")
        print(f"    跳空　{mg:+.3%}（t={tg:.2f}）　—— 開盤瞬間反映的部分，吃不到")
        print(f"    盤中　{mi:+.3%}（t={ti:.2f}）　—— 開盤進場才吃得到的部分")
        # 跳空與盤中若方向相反，相除沒有意義；要看的是盤中那一段的絕對值能否蓋過成本。
        same = (mg > 0) == (mi > 0)
        print(f"    兩段方向{'相同' if same else '相反'}"
              f"　—— {'開盤已反映一部分，盤中延續' if same else '開盤反應過頭，盤中反向修正'}")
        print(f"    盤中可捕捉的幅度 {abs(mi):.3%}，來回成本 0.39%"
              f" → {'足以交易' if abs(mi) > 0.0039 else '不足以單獨交易'}")
    print("\n  台股現股來回成本約 0.39%（手續費6折＋證交稅）。")
    print("  盤中那一段的絕對值要大於成本，開盤進場才有交易價值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
