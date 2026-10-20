#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分批進出的自適應版本：把「全有全無」的進出場換成倉位管理。

與前一版（news_reversal.py）的差別
----------------------------------
前一版是二元的：跌幅達標就全額買進，回到基準就全額賣出。
實際操作不是這樣——基於新聞的跌可以押重一點，價格回到不錯的位置可以先出一部分，
而且不必去接絕對低點，等止跌跡象出現再進場即可。

四個可獨立開關的機制（每一個都單獨測，才知道哪一個真的有用）
------------------------------------------------------------
  news_size    跌勢中有負面新聞 → 倉位加倍。
               依據：先前測得負面新聞在 −5% 門檻有 +1.21% 的增量（p=0.043），
               雖不足以當主訊號，但作為「押多押少」的依據是合理的用法。
  wait_rebound 跌幅達標後不立刻進場，等止跌訊號（收盤高於前一日，或站回 5 日均線）。
               依據：先前測得「結構健康」的跌優於「極度超賣」的跌，
               代表接刀不如等回穩。
  scale_out    分批出場：回補跌幅一半出 1/3、回到基準出 1/3、超過基準再出剩下。
  early_trim   預警減碼：RSI 超買或逼近 20 日高點時，先減掉一半。

每個部位的報酬是各批次的加權平均，並記錄倉位倍數，
讓後續的資金曲線模擬能反映「押重」實際佔用的資金。

成本：每一批進出都各自計算手續費與證交稅——分批出場會增加成本，
這是這個做法的代價，必須照實算進去。
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

ROOT = Path(__file__).resolve().parent


def _stabilized(mode: str, j: int, a: dict) -> bool:
    """
    止跌訊號的四種寬嚴程度。

    使用者要求「不必等絕對低點，相對止跌就可以進」，所以除了明確的反轉訊號，
    也提供「跌勢趨緩」這種更早、更寬鬆的判斷。寬鬆的進得早、成本低，
    但可能還沒跌完；嚴格的進得晚、較安全但價格較高。哪個好是實證問題，四種都測。
    """
    c, o, hi, lo, ma5 = a["c"], a["o"], a["hi"], a["lo"], a["ma5"]
    if mode == "strict":        # 明確反轉：收高於前一日「且」站回 5 日均線
        return bool(c[j] > c[j - 1] and np.isfinite(ma5[j]) and c[j] > ma5[j])
    if mode == "normal":        # 收高於前一日「或」站回 5 日均線
        return bool(c[j] > c[j - 1] or (np.isfinite(ma5[j]) and c[j] > ma5[j]))
    if mode == "red_k":         # 收紅 K：收盤高於開盤，盤中有承接
        return bool(np.isfinite(o[j]) and c[j] > o[j])
    if mode == "relative":      # 相對止跌：跌勢減速，或收在當日高低中位之上
        slowing = (c[j] / c[j - 1] - 1) > (c[j - 1] / c[j - 2] - 1)
        mid = ((hi[j] + lo[j]) / 2 if np.isfinite(hi[j]) and np.isfinite(lo[j])
               else np.nan)
        return bool(slowing or (np.isfinite(mid) and c[j] > mid))
    return True


def run_stock(px: pd.DataFrame, ns: pd.DataFrame, tech: pd.DataFrame,
              watch: int, drop_pct: float, max_hold: int,
              news_size: bool, wait_rebound: bool,
              scale_out: bool, early_trim: bool,
              rebound_limit: int = 5, rebound_mode: str = "normal") -> list:
    """對單一標的產生所有「操作回合」，每回合是一次完整的進場到清空。"""
    idx = px.index
    o = px["open"].to_numpy(float)
    c = px["close"].to_numpy(float)
    ma5 = px["close"].rolling(5).mean().to_numpy(float)
    px_arr = {"c": c, "o": o, "hi": px["high"].to_numpy(float),
              "lo": px["low"].to_numpy(float), "ma5": ma5}
    rsi = tech["px_rsi14"].to_numpy(float) if "px_rsi14" in tech else np.full(len(c), .5)
    hi20 = px["close"].rolling(20).max().to_numpy(float)
    neg = ns["neg"].to_numpy(float)
    n = len(idx)

    out, busy_until = [], -1
    for i0 in range(n):
        if i0 <= busy_until:
            continue
        ref = c[i0]
        if not np.isfinite(ref) or ref <= 0:
            continue

        # 1) 等跌幅出現
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

        # 跌勢中是否有負面新聞（決定押多押少）
        had_neg = float(neg[i0:hit + 1].sum()) >= 1
        mult = 2.0 if (news_size and had_neg) else 1.0

        # 2) 進場時點：立刻，或等止跌訊號
        entry_i = hit + 1
        if wait_rebound:
            entry_i = None
            for j in range(hit + 1, min(hit + 1 + rebound_limit, n - 1)):
                if j < 2 or not np.isfinite(c[j]):
                    continue
                if _stabilized(rebound_mode, j, px_arr):
                    entry_i = j + 1
                    break
            if entry_i is None or entry_i >= n:
                continue
        if entry_i >= n:
            continue
        entry = o[entry_i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        # 3) 分批出場的目標價
        half = entry + (ref - entry) * 0.5 if ref > entry else ref
        ladder = ([(half, 1 / 3), (ref, 1 / 3), (ref * 1.03, 1 / 3)]
                  if scale_out else [(ref, 1.0)])

        remaining = 1.0
        parts = []          # (出場索引, 出場價, 比重)
        li = 0
        for j in range(entry_i, min(entry_i + max_hold, n - 1)):
            if remaining <= 1e-9:
                break
            # 預警減碼：超買或逼近前高，先出一半
            if early_trim and remaining > 0.5 and np.isfinite(c[j]) and (
                    rsi[j] > 0.7 or (np.isfinite(hi20[j]) and c[j] >= hi20[j] * 0.99)):
                parts.append((j + 1, o[j + 1], remaining * 0.5))
                remaining *= 0.5
            # 達到階梯目標價就出一批
            while li < len(ladder) and np.isfinite(c[j]) and c[j] >= ladder[li][0]:
                w = min(ladder[li][1], remaining)
                if w > 1e-9:
                    parts.append((j + 1, o[j + 1], w))
                    remaining -= w
                li += 1
        if remaining > 1e-9:            # 時間到，剩下的全出
            last = min(entry_i + max_hold, n - 1)
            parts.append((last, o[last], remaining))
            remaining = 0.0

        parts = [(j, p, w) for j, p, w in parts
                 if j < n and np.isfinite(p) and p > 0 and w > 1e-9]
        if not parts:
            continue

        # 每一批各自成為一列。這很重要：分批出場（或預警減碼）釋出的資金
        # 必須當場回到可用現金去等下一個機會，不能被鎖在原部位裡直到最後一批出完。
        # 每一批各自計費，分批會多付手續費，照實算。
        exit_i = max(j for j, _, _ in parts)
        cid = f"{idx[entry_i]:%Y%m%d}"
        for pi, (j, p, w) in enumerate(sorted(parts)):
            out.append({
                "campaign": cid, "part": pi, "n_parts": len(parts),
                "news_day": idx[i0], "drop_day": idx[hit], "entry": idx[entry_i],
                "exit": idx[j], "hold": j - entry_i,
                "weight": w,                       # 佔該回合總倉位的比重
                "size_mult": mult, "had_neg": had_neg,
                "drop": c[hit] / ref - 1,
                "gross": p / entry - 1,
                "net": (p * (1 - NR.COST_SELL)) / (entry * (1 + NR.COST_BUY)) - 1,
            })
        busy_until = exit_i
    return out


def build(codes, start, end, watch, drop_pct, max_hold, **flags) -> pd.DataFrame:
    rows = []
    for i, code in enumerate(codes, 1):
        px, news = NR.load_stock(code, start, end)
        if px is None:
            continue
        ns = NR.daily_news_stats(news, px.index)
        tech = T.step2_price_features(px, quiet=True)
        r = run_stock(px, ns, tech, watch, drop_pct, max_hold, **flags)
        for x in r:
            x["code"] = code
        rows.extend(r)
        if i % 80 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔")
    return pd.DataFrame(rows)


def to_campaigns(tr: pd.DataFrame) -> pd.DataFrame:
    """把逐批次的列彙總回「每個操作回合一列」，供報表與顯著性檢定使用。"""
    if tr.empty:
        return tr
    keys = [k for k in ("code", "campaign") if k in tr.columns]
    if not keys:
        return tr
    # 直接走訪群組，不用 groupby.apply —— include_groups 的行為在各版本 pandas
    # 不一致，曾造成分組欄位（code／campaign）與其他欄位一起被丟掉。
    rows = []
    for _, d in tr.groupby(keys, sort=False):
        w = d["weight"].to_numpy(float)
        sw = w.sum() if w.sum() > 0 else 1.0
        rec = {k: d[k].iloc[0] for k in keys}
        for col in ("news_day", "drop_day", "entry", "size_mult", "had_neg",
                    "drop", "n_parts"):
            if col in d.columns:
                rec[col] = d[col].iloc[0]
        rec["exit"] = d["exit"].max()
        rec["hold"] = float((d["hold"].to_numpy(float) * w).sum() / sw)
        for col in ("net", "gross"):
            rec[col] = float((d[col].to_numpy(float) * w).sum() / sw)
        rows.append(rec)
    return pd.DataFrame(rows)


def cluster_t(tr: pd.DataFrame, col: str = "net") -> tuple:
    if tr.empty:
        return np.nan, np.nan, 0
    g = tr.groupby(pd.to_datetime(tr["entry"]).dt.to_period("M"))[col].mean()
    if len(g) < 3 or g.std(ddof=1) == 0:
        return float(tr[col].mean()), np.nan, len(g)
    return (float(tr[col].mean()),
            float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), len(g))


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="分批進出的自適應策略")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--watch", type=int, default=5)
    ap.add_argument("--drop", type=float, default=0.08)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--out", default="results/adaptive")
    args = ap.parse_args(argv)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    if args.top_n:
        import json
        mp = json.load(open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))
        rank = {r["code"]: r.get("market_cap") or 0 for r in mp}
        codes = sorted(codes, key=lambda c: -rank.get(c, 0))[:args.top_n]

    print(f"主規則：盯 {args.watch} 天、跌 {args.drop:.0%}、最長持有 {args.hold} 日")
    print(f"標的 {len(codes)} 檔　期間 {args.start} ~ {args.end}\n")

    variants = {
        "A 基準（立刻進場、全額、單次出場）":
            dict(news_size=False, wait_rebound=False, scale_out=False, early_trim=False),
        "B ＋新聞加碼":
            dict(news_size=True, wait_rebound=False, scale_out=False, early_trim=False),
        "C ＋等反彈進場":
            dict(news_size=False, wait_rebound=True, scale_out=False, early_trim=False),
        "D ＋分批出場":
            dict(news_size=False, wait_rebound=False, scale_out=True, early_trim=False),
        "E ＋預警減碼":
            dict(news_size=False, wait_rebound=False, scale_out=False, early_trim=True),
        "F 全部開啟":
            dict(news_size=True, wait_rebound=True, scale_out=True, early_trim=True),
    }

    print(f"  {'版本':<26}{'回合數':>7}{'勝率':>8}{'平均倉位':>9}"
          f"{'平均持有':>9}{'批次':>6}{'淨報酬':>10}{'群集t':>8}")
    results = {}
    for name, flags in variants.items():
        tr = build(codes, args.start, args.end, args.watch, args.drop,
                   args.hold, **flags)
        if tr.empty:
            print(f"  {name:<26} 無交易")
            continue
        tr.to_csv(out_dir / f"trades_{name[0]}.csv", index=False,
                  encoding="utf-8-sig")
        m, t, nclu = cluster_t(tr)
        results[name] = tr
        print(f"  {name:<26}{len(tr):>7,}{(tr['net'] > 0).mean():>8.1%}"
              f"{tr['size_mult'].mean():>9.2f}{tr['hold'].mean():>9.1f}"
              f"{tr['n_parts'].mean():>6.2f}{m:>+10.2%}{t:>8.2f}")

    # 新聞加碼是否有效：直接比較加碼組與未加碼組
    if "B ＋新聞加碼" in results:
        tr = results["B ＋新聞加碼"]
        a = tr[tr["had_neg"]]
        b = tr[~tr["had_neg"]]
        if len(a) > 50 and len(b) > 50:
            ma, ta, _ = cluster_t(a)
            mb, tb, _ = cluster_t(b)
            print(f"\n  新聞加碼的依據檢查：")
            print(f"    跌勢中有負面新聞　{len(a):>6,} 筆　平均 {ma:+.2%}（t={ta:.2f}）")
            print(f"    跌勢中無負面新聞　{len(b):>6,} 筆　平均 {mb:+.2%}（t={tb:.2f}）")
            print(f"    差異 {ma - mb:+.2%}　—— 差異為正才支持「有新聞就押重」")

    print("\n  註：分批出場會多付手續費，已逐批計入；"
          "『批次』是平均出場次數，越大成本越高。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
