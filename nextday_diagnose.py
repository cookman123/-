#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔日個股模型的可信度診斷：那組數字好到不合理，先證明它是假的。

為什麼要先懷疑
--------------
全檔個股版跑出「多空 10 檔、扣成本後年化 +161.65%、夏普 3.82、
每日橫斷面 IC t = 31.82」。日頻橫斷面能有這種水準，等於宣稱台股
存在一個每天都能收割、規模可觀又沒人套利掉的訊號。先驗上這不可能。

本程式測四件事，任何一件成立都足以推翻那組數字：

一、它是不是只是買賣價差造成的假反轉？
    日收盤價在買價與賣價之間跳動，產生的「昨跌今漲」完全無法成交。
    對照：直接用「昨日報酬的相反數」排序（不訓練任何模型）。
    若天真反轉就能拿到同等級的績效，那模型學到的就是這個雜訊。

二、漲跌停的部位根本買不到也賣不掉。
    台股日漲跌幅 ±10%。漲停鎖死時掛買單成交不了，跌停時賣不掉。
    對照：把第 t 日觸及漲跌停的個股從可交易清單剔除。

三、是不是因為訊號與成交用了同一個收盤價？
    對照：改成隔日開盤成交（訊號 t 日收盤 → t+1 開盤進場 → t+2 開盤出場）。

四、是不是只在流動性差的小型股上成立？
    對照：把母體縮到成交金額前 100 名，價差相對小很多。
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
sys.path.insert(0, str(ROOT))
import davis_tw_train as T  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402

OUT = ROOT / "results" / "nextday"
FEE, TAX = 0.001425 * 0.3, 0.003


def panels(codes: list, start: str, end: str) -> dict:
    """收盤、開盤、最高、最低的寬表。"""
    cols = {k: {} for k in ("close", "open", "high", "low")}
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        if len(px) < 400:
            continue
        for k in cols:
            cols[k][code] = px[k]
        if i % 100 == 0:
            print(f"  {i}/{len(codes)}", flush=True)
    return {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}


def perf(r: pd.Series, base: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 60 or r.std() == 0:
        return {}
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    d = {"年化": (1 + r).prod() ** (1 / yrs) - 1,
         "夏普": r.mean() / r.std() * np.sqrt(252),
         "最大回撤": float((eq / eq.cummax() - 1).min())}
    if base is not None:
        idx = r.index.intersection(base.index)
        t, _ = nw_tstat(r.reindex(idx) - base.reindex(idx))
        d["超額t"] = t
    return d


def run_rank(score: pd.DataFrame, ret: pd.DataFrame, ok: pd.DataFrame,
             n: int, side: str, cost: float) -> pd.Series:
    """每日依 score 排序取頭尾各 n 檔，實現 ret，換手依名單差異計費。"""
    out, prev_l, prev_s = [], set(), set()
    for d in score.index:
        s = score.loc[d].dropna()
        m = ok.loc[d] if d in ok.index else None
        if m is not None:
            s = s[[c for c in s.index if bool(m.get(c, False))]]
        if len(s) < n * 2:
            continue
        s = s.sort_values(ascending=False)
        L, S = list(s.index[:n]), (list(s.index[-n:]) if side == "多空" else [])
        rd = ret.loc[d] if d in ret.index else pd.Series(dtype=float)
        rl = float(rd.reindex(L).fillna(0.0).mean())
        rs = float(rd.reindex(S).fillna(0.0).mean()) if S else 0.0
        r = rl - rs if S else rl
        tl = 1 - len(set(L) & prev_l) / n if prev_l else 1.0
        ts = (1 - len(set(S) & prev_s) / n) if (S and prev_s) else (1.0 if S else 0.0)
        r -= ((tl + ts) / (2 if S else 1)) * cost
        out.append((d, r))
        prev_l, prev_s = set(L), set(S)
    return pd.Series(dict(out)).sort_index()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="隔日個股模型的可信度診斷")
    ap.add_argument("--start", default="2014-12-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"載入 {len(codes)} 檔…", flush=True)
    P = panels(codes, a.start, a.end)
    C, O, H, L = P["close"], P["open"], P["high"], P["low"]

    mask250 = U.get_mask(a.start, a.end, 250).reindex(C.index).fillna(False)
    mask100 = U.get_mask(a.start, a.end, 100).reindex(C.index).fillna(False)
    mask250 = mask250.reindex(columns=C.columns, fill_value=False)
    mask100 = mask100.reindex(columns=C.columns, fill_value=False)

    r1 = C.pct_change()                      # r(t) = C(t)/C(t-1) - 1
    fwd_cc = r1.shift(-1)                    # 收盤→收盤：t 日收盤進、t+1 收盤出
    fwd_oo = (O.shift(-2) / O.shift(-1) - 1)  # 開盤→開盤：t+1 開盤進、t+2 開盤出

    # 漲跌停：當日振幅貼著 ±10% 且收在極端 → 視為不可成交
    prev_c = C.shift(1)
    lim_up = (C >= prev_c * 1.095) & (C >= H * 0.999)
    lim_dn = (C <= prev_c * 0.905) & (C <= L * 1.001)
    tradable = mask250 & ~lim_up & ~lim_dn

    ew = fwd_cc[mask250].mean(axis=1)
    cost = 2 * FEE + TAX
    print(f"\n{'=' * 104}")
    print(f"對照基準：母體等權（前 250 檔）年化 {perf(ew)['年化']:+.2%}"
          f"　每次換手成本 {cost:.2%}")
    print("=" * 104)

    # 天真反轉：完全不訓練模型，只用「昨日報酬的相反數」
    naive = -r1
    rows = []
    cases = [
        ("天真反轉／收盤進出／前250", naive, fwd_cc, mask250),
        ("天真反轉／收盤進出／剔除漲跌停", naive, fwd_cc, tradable),
        ("天真反轉／開盤進出／前250", naive, fwd_oo, mask250),
        ("天真反轉／開盤進出／剔除漲跌停", naive, fwd_oo, tradable),
        ("天真反轉／開盤進出／前100", naive, fwd_oo, mask100),
    ]
    print(f"  {'情境':<34}{'方式':<6}{'零成本':>11}{'扣成本':>11}{'夏普':>8}{'超額t':>8}")
    for name, sc, fwd, msk in cases:
        for side in ("多空", "多單"):
            r0 = run_rank(sc, fwd, msk, a.n, side, 0.0)
            r1c = run_rank(sc, fwd, msk, a.n, side, cost)
            s0, s1 = perf(r0), perf(r1c, ew)
            if not s0 or not s1:
                continue
            rows.append({"情境": name, "方式": side, "年化_零成本": s0["年化"],
                         "年化_扣成本": s1["年化"], "夏普": s1["夏普"],
                         "超額t": s1["超額t"]})
            print(f"  {name:<34}{side:<6}{s0['年化']:>+11.2%}"
                  f"{s1['年化']:>+11.2%}{s1['夏普']:>8.2f}{s1['超額t']:>8.2f}")

    pd.DataFrame(rows).to_csv(OUT / "diagnose.csv", index=False, encoding="utf-8-sig")

    # 訊號本身的衰減：反轉強度隨執行延遲的變化
    print(f"\n{'=' * 104}\n反轉訊號隨執行延遲的衰減\n{'=' * 104}")
    print(f"  {'延遲':<28}{'多空年化（零成本）':>22}")
    for lag, lab in ((0, "t 收盤進、t+1 收盤出"),
                     (1, "t+1 收盤進、t+2 收盤出"),
                     (2, "t+2 收盤進、t+3 收盤出")):
        f = r1.shift(-(1 + lag))
        r = run_rank(naive, f, mask250, a.n, "多空", 0.0)
        s = perf(r)
        if s:
            print(f"  {lab:<28}{s['年化']:>+22.2%}")

    print(f"\n判讀")
    print(f"  若『天真反轉』不訓練任何模型就能拿到與模型相近的績效，")
    print(f"  那模型學到的就是這個效應，而不是什麼預測能力。")
    print(f"  若效應在『開盤進出』或『剔除漲跌停』後大幅消失，代表它成交不到。")
    print(f"  若延遲一天就衰減殆盡，代表它是收盤價的微結構雜訊，不是資訊。")
    print(f"\n輸出：{OUT / 'diagnose.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
