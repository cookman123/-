#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驗證 nextday_horizon.py 的勝出者（h=5、緩衝 m=5）：對沖後 alpha 是真的，還是又一個陷阱。

本研究前面每一個「有效」的結果，都是被一個沒注意到的假設撐起來的
（purging、時點清單、收盤成交、beta）。這個結果第一次在沖掉 beta 後仍顯著，
所以用已知的每一種失敗模式逐一檢查：

    一、滑價：單邊 0.05%／0.10%／0.20%
    二、買不到：進場日（t+1）開盤就漲停的股票，實際上買不到。
        剔除這些進場（該檔當天不持有，報酬記 0），看還剩多少
    三、假跳動：Yahoo 有少數減資沒調整（例如 3557、1418 恢復買賣日 +99%、+141%），
        單日開盤→開盤報酬超過 ±11% 在台股不可能（漲跌幅 10%，2015-06 前 7%），
        把個股日報酬截在 ±11% 後重算；並列出貢獻最大的個股日
    四、逐年：對沖後報酬按年
    五、已知因子：對沖後報酬對「高波動減低波動」日報酬、「母體等權減大盤」日報酬
        迴歸，看 alpha 是否只是波動度溢酬或小型股效應換個樣子
    六、含息對沖：放空台指期等於放空「含息」大盤（期貨價 ≈ 現貨 − 預期股利），
        對沖腿每年要多付約 大盤殖利率 × beta；持股的現金股利則要加回。
        持股股利用證交所逐筆除息資料，大盤殖利率用 3%／3.5%／4.5% 三種假設，
        另扣期貨轉倉成本 0.2%／年（期交稅＋每月轉倉滑價）
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
import feature_engineering as FE  # noqa: E402
from nextday_buffer import _pick  # noqa: E402
from nextday_v2 import COST, OUT, STORE  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402


def simulate(pred, y, locked, n, m, unbuyable=None):
    """回傳 (毛報酬, 換手, 每日持股)。unbuyable：進場當天開盤漲停、買不到的 (date, code) 集合。"""
    H, gross, turns, hold = [], {}, {}, {}
    for d, g in pred.groupby(level="date"):
        g = g.droplevel("date").dropna()
        if len(g) < n * 2:
            continue
        lk = locked.xs(d, level="date")
        lk = set(lk[lk].index)
        desc = g.sort_values(ascending=False)
        rk = pd.Series(np.arange(len(desc)), index=desc.index)
        new = _pick(rk, H, lk, n, m)
        if unbuyable is not None:
            # 新進場但隔日開盤漲停 → 買不到，該格空著（報酬 0）
            new = [c for c in new if c in H or (d, c) not in unbuyable]
        yd = y.xs(d, level="date").reindex(new).fillna(0.0)
        gross[d] = float(yd.sum() / n)
        turns[d] = (len(set(new) - set(H)) / n) if H else 1.0
        hold[d] = new
        H = new
    return pd.Series(gross).sort_index(), pd.Series(turns).sort_index(), hold


def hedge(r, mk):
    mk = mk.reindex(r.index)
    beta = (r.rolling(250).cov(mk) / mk.rolling(250).var()).shift(1)
    return (r - beta * mk).dropna()


def ann(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


def main() -> int:
    h = 5
    p = pd.read_csv(OUT / f"pred_h{h}.csv", dtype={"code": str}, parse_dates=["date"])
    pred = p.set_index(["date", "code"])["pred"]
    df = pd.read_pickle(STORE)[["y_oo", "trd_locked", "cs_ts_vol60"]]
    y = df["y_oo"].reindex(pred.index)
    locked = (df["trd_locked"] >= 0.5).reindex(pred.index).fillna(False)
    P = FE.load_panels("2010-01-01", "2026-12-31", quiet=True)
    O, C = P["open"], P["close"]
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"]) \
        .set_index("date")["open"].sort_index()
    tw = tw[~tw.index.duplicated(keep="last")]
    mk = tw.shift(-2) / tw.shift(-1) - 1

    # 進場日 t+1 開盤相對 t 收盤的漲幅；超過漲停門檻 95% 視為開盤漲停買不到
    gap = (O.shift(-1) / C - 1).stack(future_stack=True)
    gap.index.names = ["date", "code"]
    lim = pd.Series(np.where(gap.index.get_level_values("date") < "2015-06-01", 0.065, 0.095),
                    index=gap.index)
    unbuy = set(gap[gap >= lim].index)

    for n in (10, 30):
        print(f"\n{'=' * 92}\nh={h}　n={n}　m=5\n{'=' * 92}")
        g, turn, hold = simulate(pred, y, locked, n, 5)
        net = g - turn * COST
        hd = hedge(net, mk)
        t0, _ = nw_tstat(hd)
        print(f"  基準：換手 {turn.mean():.2f}　扣成本 {ann(net):+.2%}　對沖後 {ann(hd):+.2%}（t {t0:.2f}）")

        print("  一、滑價（單邊）")
        for s in (0.0005, 0.001, 0.002):
            hs = hedge(g - turn * (COST + 2 * s), mk)
            t, _ = nw_tstat(hs)
            print(f"      {s:.2%}：對沖後 {ann(hs):+.2%}（t {t:.2f}）")

        g2, turn2, _ = simulate(pred, y, locked, n, 5, unbuyable=unbuy)
        h2 = hedge(g2 - turn2 * COST, mk)
        t2, _ = nw_tstat(h2)
        n_new = sum(1 for d in hold for c in hold[d])
        print(f"  二、剔除開盤漲停買不到的進場：對沖後 {ann(h2):+.2%}（t {t2:.2f}）")

        yc = y.clip(-0.11, 0.11)
        g3, turn3, _ = simulate(pred, yc, locked, n, 5)
        h3 = hedge(g3 - turn3 * COST, mk)
        t3, _ = nw_tstat(h3)
        print(f"  三、個股日報酬截在 ±11%：對沖後 {ann(h3):+.2%}（t {t3:.2f}）")
        contrib = []
        for d, cs in hold.items():
            yd = y.xs(d, level="date").reindex(cs)
            for c, v in yd.items():
                if pd.notna(v):
                    contrib.append((d.date(), c, v))
        cdf = pd.DataFrame(contrib, columns=["date", "code", "ret"])
        print(f"      持股日 {len(cdf):,} 筆，|報酬| > 11% 的 {int((cdf['ret'].abs() > 0.11).sum())} 筆；最大五筆：")
        for _, r in cdf.reindex(cdf["ret"].abs().sort_values(ascending=False).index).head(5).iterrows():
            print(f"        {r['date']} {r['code']} {r['ret']:+.1%}")

        print("  四、逐年（對沖後）")
        print("      " + "　".join(f"{yy}:{(1 + x).prod() - 1:+.0%}" for yy, x in hd.groupby(hd.index.year)))

        # 五、已知因子：每日高波動 30 檔減低波動 30 檔、母體等權減大盤
        vol = df["cs_ts_vol60"].reindex(pred.index)
        hv = {}
        for d, v in vol.groupby(level="date"):
            v = v.droplevel("date").dropna()
            if len(v) < 60:
                continue
            yd = y.xs(d, level="date")
            hv[d] = float(yd.reindex(v.nlargest(30).index).mean() - yd.reindex(v.nsmallest(30).index).mean())
        hv = pd.Series(hv)
        ew = y.groupby(level="date").mean() - mk.reindex(y.index.get_level_values("date").unique())
        X = pd.DataFrame({"r": hd, "hv": hv, "size": ew}).dropna()
        A = np.column_stack([np.ones(len(X)), X["hv"], X["size"]])
        coef, *_ = np.linalg.lstsq(A, X["r"].to_numpy(), rcond=None)
        resid_alpha = X["r"] - coef[1] * X["hv"] - coef[2] * X["size"]
        ta, _ = nw_tstat(resid_alpha)
        print(f"  五、對已知因子迴歸：高波動因子係數 {coef[1]:+.2f}、等權減大盤係數 {coef[2]:+.2f}　"
              f"剩餘 alpha 年化 {coef[0] * 252:+.2%}（t {ta:.2f}）")

        # 六、含息對沖。y_oo 在日 t 是 t+1 開盤→t+2 開盤，所以除息日落在 t+2 的股利屬於日 t 的持股
        dates = O.index
        pos = {d_: i for i, d_ in enumerate(dates)}
        ex = pd.read_csv(ROOT / "data" / "corp_actions" / "exright.csv",
                         dtype={"code": str}, parse_dates=["date"])
        ex = ex[(ex["cash"] > 0) & (ex["prev_close"] > 0)]
        dv = {}
        for _, e in ex.iterrows():
            i = dates.searchsorted(e["date"])
            if 2 <= i < len(dates):
                dv[(dates[i - 2], e["code"])] = dv.get((dates[i - 2], e["code"]), 0.0) + e["cash"] / e["prev_close"]
        div_ret = pd.Series({d_: sum(dv.get((d_, c), 0.0) for c in cs) / n for d_, cs in hold.items()})
        net_tr = (net + div_ret.reindex(net.index).fillna(0.0))
        print(f"  六、含息對沖：持股現金股利年化約 {div_ret.mean() * 252:.2%}")
        mkr = mk.reindex(net_tr.index)
        beta = (net_tr.rolling(250).cov(mkr) / mkr.rolling(250).var()).shift(1)
        for dy in (0.03, 0.035, 0.045):
            mk_tr = mkr + dy / 252
            h6 = (net_tr - beta * mk_tr - 0.002 / 252).dropna()
            t6, _ = nw_tstat(h6)
            print(f"      大盤殖利率 {dy:.1%}：對沖後 {ann(h6):+.2%}（t {t6:.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
