#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔日預測加緩衝帶：預測力不變，只改換手規則，看成本能不能被壓到毛價差以下。

為什麼做這個
------------
`nextday_v2.py` 的結論是：毛價差 +0.426%／日打不過兩腿合計成本 0.62%／日
（0.39% × 1.60 次換手）。差距不到 0.2%／日，而且**擋住它的是換手，不是預測力**。
每天把排名掉出前 n 名的股票全部換掉，持股平均只放一兩天，成本自然吃光毛利。

緩衝帶（hysteresis）是最直接的降換手規則：
    進場：排名進入前 n 名才買
    出場：排名掉出前 m × n 名才賣（m ≥ 1）
m = 1 就是原本的規則。m 越大持有越久、換手越低，但持股的平均預測值也越差。
這是一個「預測力換成本」的取捨，要看哪一邊的斜率比較陡。

不多做規格搜尋
--------------
變體在跑之前就固定：m ∈ {1, 2, 3, 5} × 檔數 {10, 30} × {多單, 多空}，共 16 組，
全部一起送 Reality Check／SPA。不看結果再加 m，不另外挑門檻。

其他設定沿用 v2
--------------
    預測：nextday_v2.py 的 walk-forward 預測（y_oo，t+1 開盤進、t+2 開盤出）
    可交易性：當日漲跌停鎖死者不能新買；**已持有而鎖死的不能賣，強制續抱**
              （v2 直接把鎖死股從清單剔除，等於假設賣得掉；這裡比較誠實，
              所以 m = 1 的數字和 v2 會有些微差異）
    成本：個股來回 0.39%，多空兩腿各自計換手
    基準：多單對母體等權，多空對 0
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
from nextday_v2 import COST, OUT, STORE, perf  # noqa: E402
from reality_check import reality_check  # noqa: E402

MULTS = (1, 2, 3, 5)
SIZES = (10, 30)
SIDES = ("多單", "多空")


def _pick(rank: pd.Series, held: list, locked: set, n: int, m: int) -> list:
    """rank：代號 → 名次（0 最好）。保留仍在前 m×n 名或鎖死賣不掉的持股，再由前段補滿。"""
    keep = [c for c in held if c in locked or rank.get(c, np.inf) < m * n]
    for c in rank.index:                       # rank 已按名次排序
        if len(keep) >= n:
            break
        if c not in keep and c not in locked:
            keep.append(c)
    return keep[:max(n, len(keep))]


def run(pred: pd.Series, y: pd.Series, locked: pd.Series,
        n: int, m: int, side: str) -> tuple:
    """回傳（扣成本日報酬, 零成本日報酬, 平均換手）。"""
    L, S = [], []
    out, gross, turns = {}, {}, []
    for d, g in pred.groupby(level="date"):
        g = g.droplevel("date").dropna()
        if len(g) < n * 2:
            continue
        lk = locked.xs(d, level="date")
        lk = set(lk[lk].index)
        desc = g.sort_values(ascending=False)
        rl = pd.Series(np.arange(len(desc)), index=desc.index)
        rs = pd.Series(np.arange(len(desc)), index=desc.index[::-1])
        nL = _pick(rl, L, lk, n, m)
        nS = _pick(rs, S, lk, n, m) if side == "多空" else []
        yd = y.xs(d, level="date")
        a = float(yd.reindex(nL).fillna(0.0).mean())
        tl = 1 - len(set(nL) & set(L)) / len(nL) if L else 1.0
        if nS:
            b = float(yd.reindex(nS).fillna(0.0).mean())
            ts = 1 - len(set(nS) & set(S)) / len(nS) if S else 1.0
            g0 = (a - b) / 2
            turn = (tl + ts) / 2
        else:
            g0, turn = a, tl
        gross[d] = g0
        out[d] = g0 - turn * COST
        turns.append(turn)
        L, S = nL, nS
    return (pd.Series(out).sort_index(), pd.Series(gross).sort_index(),
            float(np.mean(turns)) if turns else np.nan)


def verify(pred: pd.Series, y: pd.Series, locked: pd.Series, ew: pd.Series,
           n: int, m: int, side: str) -> None:
    """
    對單一變體做四道檢查。季頻那條線就是在這幾道裡被揭穿成 beta 的，
    所以這裡用同一套標準：
        一、beta 與 alpha：對大盤開盤→開盤報酬迴歸；另做事前 beta 中性
            （滾動 250 日 beta，shift 1 日，不含當日）
        二、滑價敏感度：成本模型 slippage = 0，逐檔加單邊 0.05%／0.10%／0.20%
        三、逐年：是不是一兩年撐起來的
        四、m 往外延（7、10）：**事前沒有列入，只當探索**，看 m=5 是不是剛好挑在峰值
    """
    from portfolio_backtest import nw_tstat

    r, g, turn = run(pred, y, locked, n, m, side)
    turn_s = (g - r) / COST                       # 每日換手（由成本反推）
    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv", parse_dates=["date"])         .set_index("date")["open"].sort_index()
    tw = tw[~tw.index.duplicated(keep="last")]
    mk = (tw.shift(-2) / tw.shift(-1) - 1).reindex(r.index)   # 與 y_oo 同一個持有區間
    d = pd.DataFrame({"r": r, "mk": mk, "ew": ew.reindex(r.index)}).dropna()
    b = float(np.cov(d["r"], d["mk"])[0, 1] / d["mk"].var())
    a_t, _ = nw_tstat(d["r"] - b * d["mk"])
    roll_b = (d["r"].rolling(250).cov(d["mk"]) / d["mk"].rolling(250).var()).shift(1)
    hedged = (d["r"] - roll_b * d["mk"]).dropna()
    h_t, _ = nw_tstat(hedged)
    yrs = len(d) / 252
    print(f"\n{'=' * 96}\n驗證：{side} {n} 檔 m={m}"
          f"（{d.index[0]:%Y-%m} ~ {d.index[-1]:%Y-%m}）\n{'=' * 96}")
    print(f"  一、beta {b:.2f}　alpha 年化 {(d['r'] - b * d['mk']).mean() * 252:+.2%}（NW t {a_t:.2f}）")
    print(f"      事前 beta 中性：年化 {(1 + hedged).prod() ** (252 / len(hedged)) - 1:+.2%}"
          f"（NW t {h_t:.2f}）　滾動 beta 中位數 {roll_b.median():.2f}")
    print(f"  二、滑價（單邊）　換手 {turn:.2f}／日")
    for slip in (0.0, 0.0005, 0.001, 0.002):
        rs = (g - turn_s * (COST + 2 * slip)).reindex(d.index)
        t, _ = nw_tstat(rs - d["ew"])
        print(f"      {slip:.2%}：扣成本年化 {(1 + rs).prod() ** (1 / yrs) - 1:+.2%}　對母體超額 t {t:.2f}")
    print(f"  三、逐年（扣成本，無滑價）")
    for yy, gg in d.groupby(d.index.year):
        s_, e_ = (1 + gg["r"]).prod() - 1, (1 + gg["ew"]).prod() - 1
        print(f"      {yy}  策略 {s_:+7.1%}　母體 {e_:+7.1%}　差 {s_ - e_:+7.1%}")
    print(f"  四、m 外延（探索，非事前規格）")
    for mm in (5, 7, 10):
        rr, _, tt = run(pred, y, locked, n, mm, side)
        t, _ = nw_tstat(rr - ew.reindex(rr.index))
        print(f"      m={mm:<3}換手 {tt:.2f}　扣成本年化 {(1 + rr).prod() ** (252 / len(rr)) - 1:+.2%}　超額 t {t:.2f}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="隔日預測＋緩衝帶")
    ap.add_argument("--tag", default="")
    ap.add_argument("--verify", default="",
                    help="只驗證單一變體，格式 方式,檔數,m，例如 多單,30,5")
    a = ap.parse_args(argv)

    pf = OUT / f"pred_v2{a.tag}_y_oo.csv"
    if not pf.exists():
        print(f"找不到 {pf}，請先跑 nextday_v2.py"); return 1
    p = pd.read_csv(pf, dtype={"code": str}, parse_dates=["date"])
    pred = p.set_index(["date", "code"])["pred"]
    df = pd.read_pickle(STORE.with_name(f"panel{a.tag}.pkl"))[["y_oo", "trd_locked"]]
    y = df["y_oo"]
    locked = (df["trd_locked"] >= 0.5).reindex(pred.index).fillna(False)
    ew = y.reindex(pred.index).groupby(level="date").mean()
    days = pred.index.get_level_values("date")
    print(f"預測 {len(pred):,} 列　{days.min():%Y-%m} ~ {days.max():%Y-%m}　成本 {COST:.4%}／來回")
    if a.verify:
        side, n, m = a.verify.split(",")
        verify(pred, y, locked, ew, int(n), int(m), side)
        return 0

    rows, F = [], {}
    print(f"\n  {'方式':<6}{'檔數':>5}{'m':>4}{'換手':>8}{'毛報酬/日':>11}{'成本/日':>10}"
          f"{'零成本年化':>12}{'扣成本年化':>12}{'夏普':>7}{'超額t':>7}")
    for side in SIDES:
        for n in SIZES:
            for m in MULTS:
                r, g, turn = run(pred, y, locked, n, m, side)
                base = ew if side == "多單" else pd.Series(0.0, index=r.index)
                s0, s1 = perf(g), perf(r, base)
                rows.append({"方式": side, "檔數": n, "m": m, "換手": turn,
                             "毛報酬_日": g.mean(), "成本_日": turn * COST,
                             "年化_零成本": s0["年化"], "年化_扣成本": s1["年化"],
                             "夏普": s1["夏普"], "超額t": s1["超額t"]})
                F[f"{side}{n}_m{m}"] = (r - base.reindex(r.index)).fillna(0.0)
                print(f"  {side:<6}{n:>5}{m:>4}{turn:>8.2f}{g.mean():>+11.3%}"
                      f"{turn * COST:>10.3%}{s0['年化']:>+12.2%}{s1['年化']:>+12.2%}"
                      f"{s1['夏普']:>7.2f}{s1['超額t']:>7.2f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"buffer{a.tag}.csv", index=False, encoding="utf-8-sig")

    X = pd.DataFrame(F).dropna()
    rc = reality_check(X.to_numpy(), mean_block=10, n_boot=2000, seed=42)
    best = X.columns[rc["最佳策略序號"]]
    print(f"\n{'=' * 96}\n多重比較（{X.shape[1]} 組變體，{X.shape[0]} 日）\n{'=' * 96}")
    print(f"  最佳：{best}　RC p = {rc['RC_p']:.4f}　SPA p = {rc['SPA_p']:.4f}")

    print(f"\n判讀")
    for side in SIDES:
        for n in SIZES:
            b = res[(res["方式"] == side) & (res["檔數"] == n)].set_index("m")
            print(f"  {side} {n} 檔：換手 {b.loc[1, '換手']:.2f} → {b.loc[5, '換手']:.2f}，"
                  f"毛報酬 {b.loc[1, '毛報酬_日']:+.3%} → {b.loc[5, '毛報酬_日']:+.3%}／日，"
                  f"扣成本年化 {b.loc[1, '年化_扣成本']:+.1%} → {b.loc[5, '年化_扣成本']:+.1%}")
    npos = int((res["年化_扣成本"] > 0).sum())
    print(f"  {len(res)} 組裡扣成本後為正的有 {npos} 組")
    print(f"\n輸出：{OUT / f'buffer{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
