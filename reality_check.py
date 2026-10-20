#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料窺探校正：White (2000) Reality Check 與 Hansen (2005) SPA 檢定。

要解決什麼問題
--------------
portfolio_backtest.py 每個持有期都測了 24 種設定（3 種持股數 × 2 種費率
× 2 種排序 × 多頭／多空），再從裡面挑出 t 值最大的那一格報告。這個「挑最大」
的動作本身就會製造顯著性：即使所有策略的真實超額都是 0，24 個相關的 t 值裡
最大的那個也常常會超過 2。單看那一格的 p 值是無效的。

Bonferroni（p × 24）則是另一個極端，它假設 24 個檢定互相獨立，
但這些策略高度重疊（前10 與前20 名單有一半一樣），校正會過度保守。

White 的 Reality Check 用 bootstrap 直接模擬「整族策略的最大值」在虛無假設下
的分布，自動處理策略之間的相關性，既不高估也不過度保守。
Hansen 的 SPA 進一步把「明顯很差的策略」從重新中心化中剔除，檢定力更高。

虛無假設
--------
H0：整族策略中，最好的那個相對大盤的期望超額報酬 ≤ 0
    （也就是「看起來有超額」完全是搜尋 24 種設定搜出來的）
拒絕 H0 才能說「這族策略裡真的存在打敗大盤的策略」。

Bootstrap 方法
--------------
Politis & Romano (1994) 的 stationary bootstrap：以幾何分布的區塊長度重抽，
保留報酬序列的自相關結構，也保留策略之間的同期相關（同一組日期一起重抽）。
平均區塊長度預設取持有期長度，因為超額報酬的自相關主要來自持有期內。
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


# ============================== Bootstrap ==============================


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap 的索引矩陣，形狀 (n_boot, n)。

    每一步以機率 1/mean_block 跳到新的隨機位置，否則接續前一個位置＋1（環狀）。
    區塊長度因此服從幾何分布，平均長度 = mean_block。
    """
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jump = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


# ============================== 檢定 ==============================


def reality_check(F: np.ndarray, mean_block: float, n_boot: int = 5000,
                  seed: int = 42) -> dict:
    """
    F：(n, K) 的逐日超額報酬矩陣，欄為策略。回傳 RC 與 SPA 的 p 值。

    RC  統計量：V  = max_k sqrt(n) * mean(f_k)
    SPA 統計量：T  = max(0, max_k sqrt(n) * mean(f_k) / omega_k)
    omega_k 取 bootstrap 標準差（已含自相關），比解析式的 Newey-West 穩健。
    """
    n, K = F.shape
    rng = np.random.default_rng(seed)
    fbar = F.mean(axis=0)
    sqn = np.sqrt(n)

    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    # (n_boot, K) 的 bootstrap 平均
    boot_mean = np.empty((n_boot, K))
    for b in range(n_boot):
        boot_mean[b] = F[idx[b]].mean(axis=0)

    # ---- White Reality Check ----
    V = sqn * fbar.max()
    V_star = (sqn * (boot_mean - fbar)).max(axis=1)
    p_rc = float((V_star >= V).mean())

    # ---- Hansen SPA（consistent 版本）----
    omega = (sqn * boot_mean).std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    t_k = sqn * fbar / omega
    T = float(max(0.0, np.nanmax(t_k)))

    # 把「明顯很差」的策略從中心化中剔除：門檻 -sqrt(2 log log n)
    thr = -np.sqrt(2.0 * np.log(max(np.log(n), 1.0001)))
    keep = (t_k >= thr).astype(float)          # 1 = 納入中心化，0 = 視為無用策略
    Z = sqn * (boot_mean - fbar * keep) / omega
    T_star = np.maximum(0.0, np.nanmax(Z, axis=1))
    p_spa = float((T_star >= T).mean())

    best = int(np.nanargmax(t_k))
    return {
        "K": K, "n": n, "平均區塊": mean_block,
        "最佳策略序號": best,
        "RC統計量": V, "RC_p": p_rc,
        "SPA統計量": T, "SPA_p": p_spa,
        "最佳策略單獨t值": float(t_k[best]),
    }


# ============================== 主程式 ==============================


def analyse(path: Path, horizon: int, family: str, n_boot: int) -> None:
    f = path / f"excess_returns_h{horizon}.csv"
    if not f.exists():
        print(f"  找不到 {f.name}，略過")
        return
    df = pd.read_csv(f, index_col=0, parse_dates=True)

    # 策略族：預設只含「我們真的會從中挑選」的主動策略，不含被動基準
    cols = [c for c in df.columns if "全市場等權" not in c]
    # 「重疊分批」也是多頭組合，只是進場時點分散，必須一起納入策略族，
    # 否則就漏掉了我實際搜尋過的設定，校正等於沒做。
    if family == "long":
        cols = [c for c in cols if ("多頭" in c or "重疊分批" in c)]
    elif family == "portfolio":
        cols = [c for c in cols if ("多頭" in c or "重疊分批" in c or "多空" in c)]
    df = df[cols].dropna(how="all")
    df = df.dropna()
    if df.shape[0] < 100 or df.shape[1] < 2:
        print(f"  h={horizon} 樣本或策略數不足（{df.shape}），略過")
        return

    F = df.to_numpy()
    print(f"\n  h={horizon}　策略族 {df.shape[1]} 個、樣本 {df.shape[0]} 天"
          f"（{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}）")

    # 平均區塊長度取持有期與其兩倍，看結論對這個選擇是否敏感
    for mb in (horizon, horizon * 2):
        r = reality_check(F, mean_block=mb, n_boot=n_boot)
        best_name = df.columns[r["最佳策略序號"]]
        star_rc = "✓ 拒絕虛無" if r["RC_p"] < 0.05 else "✗ 無法拒絕"
        star_spa = "✓ 拒絕虛無" if r["SPA_p"] < 0.05 else "✗ 無法拒絕"
        print(f"    平均區塊 {mb:>3} 日：最佳＝{best_name}")
        print(f"      單獨 t 值 {r['最佳策略單獨t值']:.3f}（未校正，不可直接引用）")
        print(f"      Reality Check p = {r['RC_p']:.4f}　{star_rc}")
        print(f"      Hansen SPA   p = {r['SPA_p']:.4f}　{star_spa}")
        from scipy import stats
        p_raw = 2 * (1 - stats.norm.cdf(abs(r["最佳策略單獨t值"])))
        print(f"      Bonferroni 參考 p = {min(1.0, p_raw * df.shape[1]):.4f}"
              f"（假設 {df.shape[1]} 個策略互相獨立，過度保守，僅供對照）")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="White Reality Check / Hansen SPA")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="portfolio_backtest.py 的輸出目錄（含 excess_returns_h*.csv）")
    ap.add_argument("--horizon", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--family", choices=["long", "portfolio", "all"], default="long",
                    help="納入檢定的策略族：long=只多頭組合（預設）、"
                         "portfolio=多頭+多空、all=再加大盤擇時")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(argv)

    print("=" * 78)
    print("資料窺探校正：White Reality Check 與 Hansen SPA")
    print("=" * 78)
    print("虛無假設 H0：整族策略中最好的那個，相對大盤的期望超額報酬 ≤ 0")
    print(f"策略族範圍：{a.family}　bootstrap 次數：{a.n_boot}")

    for d in a.dirs:
        p = ROOT / d if not Path(d).is_absolute() else Path(d)
        print(f"\n{'-' * 78}\n{p.name}\n{'-' * 78}")
        for h in a.horizon:
            analyse(p, h, a.family, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
