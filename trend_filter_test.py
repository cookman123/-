#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
純趨勢過濾：把均線規則直接套在大盤／0050 上，不依賴任何因子。

為什麼值得單獨測
----------------
高波動因子的檢定停在 Reality Check p ≈ 0.06，仍在邊緣，而且最大回撤 −72.8%。
但在那個測試裡，120 日均線過濾把回撤從 −72.8% 壓到 −36.3%，夏普反而升到全場最高。

那個改善**不依賴「高波動溢酬是真的」**。若把同一條規則直接套在指數上，
就變成一個完全不需要選股、不需要因子的風險管理工具：
    指數收在均線之上 → 持有
    跌破 → 空手
問題只有一個：付出多少報酬，換到多少回撤的降低。

實作上的現實條件
----------------
    訊號用前一日收盤與均線判斷，次日開盤才執行（不用當日資訊）
    每次進出依 ETF 費率收費：手續費 3 折買賣各一次 + 賣出證交稅 0.1%
    記錄每年切換次數 —— 盤整期反覆停損是這類規則的主要代價
    對整族均線長度做 White Reality Check，校正「試了多條均線再挑最好」
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
from reality_check import reality_check  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "trend"
FEE = 0.001425 * 0.3
TAX_ETF = 0.001


def load(name: str, start: str, end: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "market" / f"{name}.csv",
                    parse_dates=["date"]).set_index("date").sort_index()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    return d.dropna(subset=["close"])


def run(px: pd.DataFrame, ma: int) -> tuple:
    """
    訊號：前一日收盤 > 前一日的 ma 日均線 → 今日持有。
    進出場都在開盤執行，因此今日的報酬取「開盤→收盤」與次日的「收盤→開盤」串接，
    等同於以開盤價切換部位。為簡化，以收盤到收盤報酬計算，並在切換當日扣一次成本。
    """
    c = px["close"]
    sig = (c > c.rolling(ma).mean()).shift(1).fillna(False)
    ret = c.pct_change().fillna(0.0)
    out, prev, switches = [], False, 0
    for d in ret.index:
        on = bool(sig.loc[d])
        r = float(ret.loc[d]) if on else 0.0
        if on != prev:
            r -= (FEE + TAX_ETF) + FEE     # 賣出付手續費＋證交稅，買進付手續費
            switches += 1
            prev = on
        out.append(r)
    return pd.Series(out, index=ret.index), sig, switches


def stats(r: pd.Series, base: pd.Series | None = None) -> dict:
    r = r.dropna()
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    d = {"年化": cagr, "夏普": r.mean() / r.std() * np.sqrt(252),
         "最大回撤": dd, "報酬回撤比": cagr / abs(dd) if dd else np.nan}
    if base is not None:
        idx = r.index.intersection(base.index)
        t, p = nw_tstat(r.reindex(idx) - base.reindex(idx))
        byrs = len(idx) / 252
        d["超額"] = cagr - ((1 + base.reindex(idx)).prod() ** (1 / byrs) - 1)
        d["超額t"], d["超額p"] = t, p
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="純趨勢過濾")
    ap.add_argument("--mas", type=int, nargs="+", default=[20, 60, 120, 200, 250])
    ap.add_argument("--targets", nargs="+", default=["TWII", "0050"])
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2024-12-31")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    allrows, series = [], {}

    for tgt in a.targets:
        px = load(tgt, a.start, a.end)
        if len(px) < 500:
            print(f"{tgt} 資料不足，略過")
            continue
        bh = px["close"].pct_change().fillna(0.0)
        yrs = len(bh) / 252
        s0 = stats(bh)
        print(f"\n{'=' * 92}")
        print(f"{tgt}　{px.index.min():%Y-%m-%d} ~ {px.index.max():%Y-%m-%d}"
              f"（{yrs:.1f} 年）")
        print("=" * 92)
        print(f"  {'設定':<22}{'年化':>9}{'夏普':>7}{'最大回撤':>10}"
              f"{'報酬/回撤':>10}{'在場比例':>9}{'年切換':>8}{'超額t':>7}")
        print(f"  {'買進持有':<22}{s0['年化']:>+9.2%}{s0['夏普']:>7.2f}"
              f"{s0['最大回撤']:>+10.2%}{s0['報酬回撤比']:>10.2f}"
              f"{1.0:>9.0%}{0:>8.1f}{'':>7}")
        allrows.append({"標的": tgt, "設定": "買進持有", **s0})

        for ma in a.mas:
            r, sig, sw = run(px, ma)
            st = stats(r, bh)
            name = f"{tgt}／{ma}日均線"
            series[name] = (r - bh).dropna()
            allrows.append({"標的": tgt, "設定": f"{ma}日均線", **st,
                            "在場比例": float(sig.mean()), "年切換": sw / yrs})
            print(f"  {f'{ma} 日均線':<22}{st['年化']:>+9.2%}{st['夏普']:>7.2f}"
                  f"{st['最大回撤']:>+10.2%}{st['報酬回撤比']:>10.2f}"
                  f"{sig.mean():>9.0%}{sw/yrs:>8.1f}{st['超額t']:>7.2f}")

    pd.DataFrame(allrows).to_csv(OUT / "trend.csv", index=False, encoding="utf-8-sig")

    if len(series) >= 3:
        print(f"\n{'=' * 92}\nReality Check：校正「試了多條均線再挑最好」\n{'=' * 92}")
        names = list(series)
        common = None
        for n in names:
            common = series[n].index if common is None else common.intersection(series[n].index)
        F = np.column_stack([series[n].reindex(common).to_numpy() for n in names])
        print(f"  策略族 {F.shape[1]} 組，共同樣本 {F.shape[0]:,} 天")
        for mb in (60, 120):
            r = reality_check(F, mean_block=mb, n_boot=3000)
            print(f"  平均區塊 {mb:>3} 日：最佳＝{names[r['最佳策略序號']]}"
                  f"　單獨 t {r['最佳策略單獨t值']:.2f}")
            print(f"    Reality Check p = {r['RC_p']:.4f}　"
                  f"Hansen SPA p = {r['SPA_p']:.4f}")

    print("\n判讀：趨勢過濾的目的是降低回撤，不是提高報酬。")
    print("『報酬/回撤』比買進持有高，就代表同樣的報酬承擔了較少的下檔風險；")
    print("『年切換』越高，盤整期被反覆停損的代價越大。")
    print(f"\n輸出：{OUT / 'trend.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
