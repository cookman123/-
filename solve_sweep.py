#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系統性掃描：有沒有任何一組參數，能讓「跌深反彈 + 閒置資金停泊 0050」贏過單純買 0050？

問題定義
--------
每筆交易的邊際是正的（14 年 4,809 筆、每筆 +1.06%、群集 t = 4.53），
換算成「資金在場時」的年化約 +17%，高於 0050 的 +11.5%。
但實際組合年化只有 +9%。差距全在兩件事：
    資金投入率不足（沒訊號時錢閒著）
    進出 0050 的成本（每做一筆就要賣一次、買回一次）

所以要調的不是訊號，是這三個決定投入率與周轉的參數：
    標的範圍 topn   放寬 → 訊號變多
    跌幅門檻 drop   降低 → 訊號變多
    最長持有 hold   縮短 → 資金周轉快
再加上部位上限 slots（決定單筆佔多少資金）。

誠實性設計
----------
掃描 N 組參數再挑最好的一組，必然會挑到運氣好的那組——本研究已因此誤判五次。
因此最後一律對整個參數族做 White Reality Check：
    H0：整族參數中最好的那一組，相對 0050 的期望超額 ≤ 0
不能拒絕 H0，就代表「找不到有效配置」，而不是「找到了最好的一組」。

選股範圍一律用時點正確的成交金額排名（當日已知），不用當期市值。
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
import news_reversal as NR  # noqa: E402
import portfolio_sim as PS  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from reality_check import reality_check  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "solve"
PARK = 0.001425 * 0.3 * 2 + 0.001      # 0050 來回：手續費3折×2 + ETF 證交稅 0.1%


def events_for(px, watch: int, drop: float, hold: int) -> list:
    """單一標的、單一參數組的事件。出場規則：收盤回到基準價 → 次日開盤出。"""
    idx = px.index
    o, c = px["open"].to_numpy(float), px["close"].to_numpy(float)
    n = len(idx)
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
            if np.isfinite(c[i]) and c[i] / ref - 1 <= -drop:
                hit = i
                break
        if hit is None:
            continue
        e = hit + 1
        if e >= n - 1 or not np.isfinite(o[e]) or o[e] <= 0:
            continue
        exit_i = None
        for j in range(e, min(e + hold, n - 1)):
            if np.isfinite(c[j]) and c[j] >= ref:
                exit_i = j + 1
                break
        if exit_i is None:
            exit_i = min(e + hold, n - 1)
        if not np.isfinite(o[exit_i]) or o[exit_i] <= 0:
            continue
        out.append({
            "drop_day": idx[hit], "entry": idx[e], "exit": idx[exit_i],
            "hold": exit_i - e,
            "net": (o[exit_i] * (1 - NR.COST_SELL)) / (o[e] * (1 + NR.COST_BUY)) - 1,
        })
        busy = exit_i
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="參數掃描 + Reality Check")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--drops", type=float, nargs="+", default=[0.05, 0.08, 0.12])
    ap.add_argument("--holds", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--topns", type=int, nargs="+", default=[120, 250])
    ap.add_argument("--slots", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--watch", type=int, default=5)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"標的池 {len(codes)} 檔　期間 {args.start} ~ {args.end}")
    n_combo = len(args.drops) * len(args.holds) * len(args.topns) * len(args.slots)
    print(f"參數組合 {n_combo} 組"
          f"（跌幅 {len(args.drops)} × 持有 {len(args.holds)} × 範圍 {len(args.topns)}"
          f" × 部位 {len(args.slots)}）\n")

    # 每檔只載入一次價格，跑完所有 (drop, hold) 組合
    print("產生事件…")
    raw: dict = {}
    for i, code in enumerate(codes, 1):
        px, _ = NR.load_stock(code, args.start, args.end)
        if px is None:
            continue
        for d in args.drops:
            for h in args.holds:
                ev = events_for(px, args.watch, d, h)
                for e in ev:
                    e["code"] = code
                raw.setdefault((d, h), []).extend(ev)
        if i % 60 == 0:
            print(f"  {i}/{len(codes)}", flush=True)
    for k, v in raw.items():
        print(f"  跌幅{k[0]:.0%} 持有{k[1]:>2}日：{len(v):,} 筆原始事件")

    masks = {n: U.get_mask(args.start, args.end, n) for n in args.topns}
    b = pd.read_csv(ROOT / "data" / "market" / "0050.csv",
                    parse_dates=["date"]).set_index("date").sort_index()

    rows, series = [], {}
    print("\n模擬資金曲線…")
    for (d, h), evs in raw.items():
        ev_all = pd.DataFrame(evs)
        for topn in args.topns:
            ev = U.filter_trades(ev_all, masks[topn], "drop_day")
            if len(ev) < 200:
                continue
            bb = b[(b.index >= ev.entry.min()) & (b.index <= ev.exit.max())]
            cal, idle = bb.index, bb.close.pct_change().fillna(0.0)
            yrs = len(cal) / 252
            cagr = (1 + idle).prod() ** (1 / yrs) - 1
            for k in args.slots:
                r = PS.simulate(ev, k, cal, idle, park_cost=PARK)
                c = r["_curve"]
                dd = c.pct_change().dropna()
                exc = (dd - idle.reindex(dd.index)).dropna()
                t, p = nw_tstat(exc)
                name = f"跌{d:.0%}/持{h}日/前{topn}/{k}格"
                series[name] = exc
                rows.append({
                    "設定": name, "跌幅": d, "持有": h, "範圍": topn, "部位": k,
                    "事件數": len(ev), "投入率": r["平均資金投入率"],
                    "年化": r["年化"], "基準": cagr, "超額": r["年化"] - cagr,
                    "夏普": dd.mean() / dd.std() * np.sqrt(252),
                    "最大回撤": float((c / c.cummax() - 1).min()),
                    "超額t": t, "超額p": p,
                })
        print(f"  跌幅{d:.0%} 持有{h}日 完成", flush=True)

    res = pd.DataFrame(rows).sort_values("超額", ascending=False)
    res.to_csv(OUT / "sweep.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print(f"掃描結果　前 12 名（基準 0050 年化 {res['基準'].iloc[0]:+.2%}）")
    print("=" * 104)
    v = res.head(12)[["設定", "事件數", "投入率", "年化", "超額", "夏普", "最大回撤", "超額t", "超額p"]].copy()
    for c in ("投入率",):
        v[c] = v[c].map(lambda x: f"{x:.0%}")
    for c in ("年化", "超額", "最大回撤"):
        v[c] = v[c].map(lambda x: f"{x:+.2%}")
    for c in ("超額t",):
        v[c] = v[c].map(lambda x: f"{x:.2f}")
    v["超額p"] = v["超額p"].map(lambda x: f"{x:.3f}")
    v["事件數"] = v["事件數"].map(lambda x: f"{x:,}")
    print(v.to_string(index=False))

    print("\n" + "=" * 104)
    print("Reality Check：校正「掃了這麼多組再挑最好」的資料窺探")
    print("=" * 104)
    names = list(series)
    common = None
    for n in names:
        common = series[n].index if common is None else common.intersection(series[n].index)
    F = np.column_stack([series[n].reindex(common).to_numpy() for n in names])
    print(f"  策略族 {F.shape[1]} 組，共同樣本 {F.shape[0]:,} 天"
          f"（{common.min():%Y-%m} ~ {common.max():%Y-%m}）")
    for mb in (20, 60):
        r = reality_check(F, mean_block=mb, n_boot=3000)
        best = names[r["最佳策略序號"]]
        print(f"  平均區塊 {mb:>2} 日：最佳＝{best}")
        print(f"    單獨 t 值 {r['最佳策略單獨t值']:.3f}（未校正）")
        print(f"    Reality Check p = {r['RC_p']:.4f}　"
              f"{'拒絕虛無：確實存在有效配置' if r['RC_p'] < 0.05 else '無法拒絕：找不到有效配置'}")
        print(f"    Hansen SPA   p = {r['SPA_p']:.4f}")

    print(f"\n完整結果：{OUT / 'sweep.csv'}")
    print("判讀：RC p ≥ 0.05 時，即使表格第一名看起來很好，也只是掃描的產物。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
