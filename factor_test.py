#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子組合檢定：動能、低波動、價值、反轉，季度換倉。

為什麼測這個
------------
先前所有測試都圍繞同一個異象（短期反轉／跌深買進），54 組參數掃描的
Reality Check p = 0.93，可以確定那個方向窮盡了。

但學術上有跨市場、跨年代重複驗證的異象至少五類，本研究一類都沒測。
而且它們的成本結構完全不同：

    跌深反轉   持有 17 天、年換手約 15 次、年成本 6~8%
    因子組合   季度換倉、年換手約 2 次、年成本約 0.8%

同樣大小的邊際，低換手能留下來的多得多 —— 這才是先前策略真正的死因。

測的四個因子（都只用當日已知的資料計算）
----------------------------------------
    動能    過去 12 個月報酬，跳過最近 1 個月（避開短期反轉的干擾）
    低波動  過去 60 個交易日的報酬標準差，取最低者
    價值    股價淨值比（證交所每日揭露，2015 年起）
    反轉    過去 1 個月報酬取最低者（對照組，與先前的跌深買同向）

誠實性設計
----------
    選股範圍用時點正確的成交金額排名，不用當期市值
    因子值一律用換倉日當天為止的資料，不含當日之後
    成本：買進手續費 3 折、賣出加證交稅 0.3%，依實際換手率計算
    最後對整族（4 因子 × 3 持股數 × 2 範圍）做 White Reality Check，
    校正「測了多組再挑最好」的資料窺探
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
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from reality_check import reality_check  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "factor"
FEE = 0.001425 * 0.3
TAX = 0.003
# backtest() 原本只選「整段持有期都有報價」的股票，是前視（2026-10-15 修正）。
# 設成 True 可重現舊結果做對照
LOOKAHEAD_FILTER = False


def load_panel(codes: list, start: str, end: str) -> pd.DataFrame:
    """收盤價寬表（date × code）。"""
    cols = {}
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        s = px["close"]
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        if len(s) > 250:
            cols[code] = s
        if i % 100 == 0:
            print(f"  {i}/{len(codes)}")
    return pd.DataFrame(cols).sort_index()


def load_pb() -> pd.DataFrame:
    """證交所每日股價淨值比（2015 年起）。"""
    files = sorted((ROOT / "data" / "fundamentals").glob("bwibbu_*.csv"))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    d["code"] = d["code"].astype(str).str.strip()
    d = d.dropna(subset=["pb"])
    return d.pivot_table(index="date", columns="code", values="pb", aggfunc="last")


def factors(C: pd.DataFrame, pb: pd.DataFrame) -> dict:
    """每個因子回傳一張 date × code 的分數表，分數越高越該買。"""
    r1 = C.pct_change()
    out = {}
    # 動能：過去 12 個月報酬，跳過最近 1 個月
    out["動能"] = (C.shift(21) / C.shift(252) - 1)
    # 低波動：標準差取負號，越低分數越高
    out["低波動"] = -r1.rolling(60).std()
    # 反轉：過去 1 個月報酬取負號（跌越多分數越高）
    out["反轉"] = -(C / C.shift(21) - 1)
    # 價值：股價淨值比取負號（越低越便宜）
    if not pb.empty:
        p = pb.reindex(index=C.index, columns=C.columns).ffill(limit=5)
        out["價值"] = -p

    # ---- 高波動與動能的組合 ----
    # 低波動在 2003-2024 測出六組全部顯著為負（t = -2.00 ~ -2.07），
    # 代表台股的波動度溢酬是正的：波動高的股票長期報酬較高。
    # 反過來用就是高波動。但這等於承擔更多風險，不是免費的超額，
    # 所以判讀時要看夏普有沒有改善，而不是只看年化數字。
    vol = r1.rolling(60).std()
    out["高波動"] = vol

    # 複合分數：兩個因子各自做橫斷面百分位後相加，量綱才可比
    mom_r = out["動能"].rank(axis=1, pct=True)
    vol_r = vol.rank(axis=1, pct=True)
    out["動能+高波動"] = mom_r + vol_r

    # 雙重排序：先取波動度前 50%，再在其中選動能最強者
    hi_vol = vol.rank(axis=1, pct=True) >= 0.5
    out["高波動中選動能"] = out["動能"].where(hi_vol)

    # 對照組：低波動中選動能（若組合有效，這一組應明顯較差）
    out["低波動中選動能"] = out["動能"].where(~hi_vol)
    return out


def backtest(score: pd.DataFrame, C: pd.DataFrame, mask: pd.DataFrame,
             top_n: int, rebal: int = 63) -> pd.Series:
    """
    季度換倉的等權多頭組合，回傳逐日報酬。

    rebal=63 約一季。換倉日取分數最高的 top_n 檔，期間內持有不動（權重隨價格漂移），
    成本依實際換手率計算：賣出付手續費＋證交稅，買進付手續費。
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
        # 只保留當日屬於選股範圍、且整段期間有正報價的標的
        seg = dates[i0:i1 + 1]
        cand = [c for c in s.index if c in C.columns]
        if d0 in mask.index:
            cand = [c for c in cand if c in mask.columns and bool(mask.at[d0, c])]
        if not cand:
            continue
        sub = C.loc[seg, cand]
        if LOOKAHEAD_FILTER:
            # 舊做法（有前視）：要求整段持有期每天都有報價，等於換倉當天就知道
            # 誰會在這一季停牌或下市，而高波動股正是最容易出事的一類
            ok = sub.notna().all() & (sub > 0).all()
        else:
            # 只要求換倉日買得到；之後停牌報酬記 0，下市則停在最後收盤（視同出清）
            ok = sub.iloc[0].notna() & (sub.iloc[0] > 0)
        cand = [c for c in cand if bool(ok.get(c, False))]
        # top_n 可以大於可用檔數（例如要做「母體全持」的等權基準），
        # 此時就持有全部候選；只有候選數低於下限才跳過該次換倉。
        n_pick = min(top_n, len(cand))
        if n_pick < min(top_n, 10):
            continue
        picks = s[cand].nlargest(n_pick).index
        w = pd.Series(1.0 / n_pick, index=picks)

        allc = prev.index.union(w.index)
        diff = w.reindex(allc).fillna(0.0) - prev.reindex(allc).fillna(0.0)
        sell, buy = float(-diff[diff < 0].sum()), float(diff[diff > 0].sum())
        nav[-1] *= (1 - sell * (FEE + TAX) - buy * FEE)

        px = C.loc[seg, picks].ffill()
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


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="因子組合檢定")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--tops", type=int, nargs="+", default=[20, 30, 50])
    ap.add_argument("--universes", type=int, nargs="+", default=[120, 250])
    ap.add_argument("--no-value", action="store_true",
                    help="排除價值因子。它只有 2015 年起的資料，會把整族的共同樣本"
                         "拖到最短，壓低 Reality Check 的檢定力")
    ap.add_argument("--bench", default="0050",
                    help="基準標的；0050 自 2003-06 才成立，更早的期間要用 TWII")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"載入 {len(codes)} 檔股價…")
    C = load_panel(codes, args.start, args.end)
    print(f"  價格面板 {C.shape[0]:,} 天 × {C.shape[1]} 檔")
    pb = pd.DataFrame() if args.no_value else load_pb()
    print(f"  股價淨值比 {pb.shape[0]:,} 天 × {pb.shape[1]} 檔"
          if not pb.empty else "  無基本面資料，略過價值因子")

    fs = factors(C, pb)
    masks = {n: U.get_mask(args.start, args.end, n) for n in args.universes}

    b = pd.read_csv(ROOT / "data" / "market" / f"{args.bench}.csv",
                    parse_dates=["date"]).set_index("date").sort_index()
    b = b[(b.index >= C.index.min()) & (b.index <= C.index.max())]
    bret = b["close"].pct_change().fillna(0.0)

    rows, series = [], {}
    print("\n回測…")
    for fname, score in fs.items():
        for uni in args.universes:
            for top in args.tops:
                r = backtest(score, C, masks[uni], top)
                if len(r) < 500:
                    continue
                idx = r.index.intersection(bret.index)
                r, br = r.reindex(idx), bret.reindex(idx)
                yrs = len(r) / 252
                cagr = (1 + r).prod() ** (1 / yrs) - 1
                bc = (1 + br).prod() ** (1 / yrs) - 1
                exc = r - br
                t, p = nw_tstat(exc)
                eq = (1 + r).cumprod()
                name = f"{fname}/前{uni}/持{top}檔"
                series[name] = exc
                rows.append({"設定": name, "因子": fname, "範圍": uni, "持股": top,
                             "年化": cagr, "基準": bc, "超額": cagr - bc,
                             "夏普": r.mean() / r.std() * np.sqrt(252),
                             "最大回撤": float((eq / eq.cummax() - 1).min()),
                             "超額t": t, "超額p": p, "天數": len(r)})
        print(f"  {fname} 完成", flush=True)

    if not rows:
        print("沒有結果")
        return 1
    res = pd.DataFrame(rows).sort_values("超額", ascending=False)
    res.to_csv(OUT / "factors.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f"因子組合　全部結果（基準 {args.bench} 年化 {res['基準'].iloc[0]:+.2%}，"
          f"{res['天數'].iloc[0]/252:.1f} 年）")
    print("=" * 100)
    v = res[["設定", "年化", "超額", "夏普", "最大回撤", "超額t", "超額p"]].copy()
    for c in ("年化", "超額", "最大回撤"):
        v[c] = v[c].map(lambda x: f"{x:+.2%}")
    v["夏普"] = v["夏普"].map(lambda x: f"{x:.2f}")
    v["超額t"] = v["超額t"].map(lambda x: f"{x:.2f}")
    v["超額p"] = v["超額p"].map(lambda x: f"{x:.3f}")
    print(v.to_string(index=False))

    print("\n" + "=" * 100)
    print("Reality Check：校正「測了多個因子再挑最好」")
    print("=" * 100)
    names = list(series)
    common = None
    for n in names:
        common = series[n].index if common is None else common.intersection(series[n].index)
    F = np.column_stack([series[n].reindex(common).to_numpy() for n in names])
    print(f"  策略族 {F.shape[1]} 組，共同樣本 {F.shape[0]:,} 天")
    for mb in (63, 126):
        r = reality_check(F, mean_block=mb, n_boot=3000)
        print(f"  平均區塊 {mb:>3} 日：最佳＝{names[r['最佳策略序號']]}"
              f"　單獨 t {r['最佳策略單獨t值']:.2f}")
        print(f"    Reality Check p = {r['RC_p']:.4f}　"
              f"{'拒絕虛無：確實存在有效因子' if r['RC_p'] < 0.05 else '無法拒絕'}")
        print(f"    Hansen SPA   p = {r['SPA_p']:.4f}")

    print(f"\n完整結果：{OUT / 'factors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
