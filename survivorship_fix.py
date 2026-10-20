#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生存者偏誤修正：把 126 家下市公司放回母體，重跑波動度溢酬。

為什麼這一步才算真的修正
------------------------
階段一算出「區間內有 126 家下市，佔應有母體 24.8%」，但那是**家數**。
下市公司的報酬通常遠低於存活者，所以報酬偏誤會比 24.8% 更嚴重。
要知道真正的數字，必須把它們依時點正確地放回母體再回測。

下市那一刻怎麼計價 —— 這是本程式最關鍵的假設
---------------------------------------------
台股「終止上市」的原因不只一種：財務困難、合併、私有化。前者股東幾乎
血本無歸，後者則是拿到價金。證交所的清單沒有原因欄，所以無法逐檔區分。

本程式的基準假設是**以最後一個有成交的收盤價出清**。理由：
    下市前的崩跌已經反映在價格序列裡（財務困難的個股通常先跌掉八九成），
    所以「最後收盤價出清」已經吃到大部分傷害。
這個假設**偏寬鬆**（假設你真的賣得掉），所以另外做三檔壓力測試：
最後一季再扣 0% / 30% / 100%，涵蓋「賣不掉」到「全部歸零」。

結論要看的是修正後的數字落在哪個區間，而不是單一點估計。

不偷看未來
----------
    個股在「有成交資料的期間」才屬於母體，下市後自動退出
    母體遮罩仍用時點正確的成交金額排名（含下市股票一起排）
    季報酬一律隔日開盤進出，與前面所有回測一致
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
from portfolio_backtest import nw_tstat  # noqa: E402
from quarterly_factors import quarter_ends  # noqa: E402

# 下市股預設讀 adjust_delisted.py 換算過的版本：原始價和存活股（Yahoo 分割調整價）
# 基準不同，配股除權日會是假下跌、減資恢復日會是假上漲
DEL_RAW = ROOT / "data" / "delisted" / "price"
DEL_ADJ = ROOT / "data" / "delisted" / "price_adj"
DEL = DEL_ADJ if DEL_ADJ.exists() else DEL_RAW
OUT = ROOT / "results" / "quarterly"
FEE, TAX = 0.001425 * 0.3, 0.003
COST = 2 * FEE + TAX


def load_delisted(src: Path = None) -> dict:
    """下市公司的 OHLCV，欄位對齊主線面板。"""
    out = {}
    for f in sorted((src or DEL).glob("[0-9]*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"])
        d = d.dropna(subset=["date", "close"]).set_index("date").sort_index()
        if len(d) < 60:          # 不足一季的樣本無法算 60 日波動
            continue
        out[f.stem] = d
    return out


def merge_panels(P: dict, dl: dict) -> tuple:
    """把下市公司併進 close/open/high/low/volume 面板。"""
    idx = P["close"].index
    add = {k: {} for k in ("open", "high", "low", "close", "volume")}
    for code, d in dl.items():
        for k in add:
            if k in d.columns:
                add[k][code] = d[k].reindex(idx)
    merged = {}
    for k, v in P.items():
        extra = pd.DataFrame(add[k], index=idx) if add[k] else pd.DataFrame(index=idx)
        merged[k] = pd.concat([v, extra], axis=1)
    return merged, sorted(dl)


def build_signal(C: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    """60 日年化波動度 —— 與存活策略同一個訊號。"""
    return C.pct_change().rolling(win).std() * np.sqrt(252)


def pit_universe(C: pd.DataFrame, V: pd.DataFrame, n: int,
                 win: int = 60) -> pd.DataFrame:
    """時點正確的成交金額前 n 名（下市股票一起參與排名，下市後自動消失）。"""
    to = (C * V).rolling(win).mean().shift(1)
    return to.rank(axis=1, ascending=False, method="first") <= n


def quarter_returns(O: pd.DataFrame, C: pd.DataFrame, qe: list,
                    haircut: float) -> pd.DataFrame:
    """
    季報酬：t 季末隔日開盤進場，t+1 季末隔日開盤出場。

    下市的判準是「該檔在出場日之後還有沒有交易資料」，不是「出場日剛好
    沒有開盤價」。存活股也會有單日停牌或無成交，用後者判定會把大量存活股
    誤判成下市並套上折價 —— 實測會讓母體等權年化從 +11% 掉到 −4%，
    而當時母體裡只有 8 檔真的下市。

    所以分成兩種情況：
        出場日無報價但之後仍有交易 → 用區間內最後一個收盤價，不折價
        出場日之後完全沒有交易     → 下市，用最後收盤價並依 haircut 折價
    """
    idx = O.index
    pos = {d: i for i, d in enumerate(idx)}
    # 每檔最後一筆有效收盤的日期，只算一次
    last_valid = C.apply(lambda s: s.last_valid_index())
    rows = {}
    for a, b in zip(qe[:-1], qe[1:]):
        ia, ib = pos[pd.Timestamp(a)], pos[pd.Timestamp(b)]
        if ia + 1 >= len(idx) or ib + 1 >= len(idx):
            continue
        entry = O.iloc[ia + 1]
        exit_ = O.iloc[ib + 1]
        exit_date = idx[ib + 1]
        last_px = C.iloc[ia + 1:ib + 2].ffill().iloc[-1]
        # 出場日之後再無交易 → 視為期間內下市
        gone = (last_valid < exit_date) & entry.notna() & last_px.notna()
        px_out = exit_.fillna(last_px)                    # 單日無報價：用最後收盤
        px_out = px_out.where(~gone, last_px * (1 - haircut))
        rows[pd.Timestamp(a)] = px_out / entry - 1
    return pd.DataFrame(rows).T.sort_index()


def book(sig: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
         qd: list, n_fixed: int, dset: frozenset = frozenset()) -> tuple:
    """
    每季取波動度最高的 n 檔，等權持有；成本依實際換手計。

    ndel 是「被選進持股的下市股」累計檔次（每季每檔算一次）。它是 0 的話，
    三檔下市計價壓力測試對策略就完全不起作用，結果必然一模一樣。
    """
    rows, prev, turns, ndel = [], set(), [], 0
    for d in qd:
        s = sig.loc[d] if d in sig.index else None
        if s is None:
            continue
        m = mask.loc[d] if d in mask.index else None
        if m is not None:
            s = s.where(m)
        s = s.dropna()
        r = fwd.loc[d].reindex(s.index) if d in fwd.index else pd.Series(dtype=float)
        s = s[r.notna()]
        if len(s) < n_fixed * 2:
            continue
        pick = list(s.sort_values(ascending=False).index[:n_fixed])
        ndel += len(dset.intersection(pick))
        ret = float(fwd.loc[d].reindex(pick).mean())
        t = 1 - len(set(pick) & prev) / n_fixed if prev else 1.0
        rows.append((d, ret - t * COST))
        turns.append(t)
        prev = set(pick)
    return (pd.Series(dict(rows)).sort_index(),
            float(np.mean(turns)) if turns else np.nan, ndel)


def ann(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (4 / len(r)) - 1 if len(r) >= 8 else np.nan


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(4)) if len(r) > 4 and r.std() else np.nan


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="生存者偏誤修正後的波動度溢酬")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--uni", type=int, default=250)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--raw", action="store_true",
                    help="下市股用證交所原始價（未換算配股／減資），只做對照")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("載入存活股票面板…", flush=True)
    P = FE.load_panels(a.start, a.end, quiet=True)
    print(f"  存活 {P['close'].shape[1]} 檔", flush=True)

    src = DEL_RAW if a.raw else DEL
    print(f"下市股價格：{'原始價（對照用）' if src == DEL_RAW else '已換算配股／減資'}", flush=True)
    dl = load_delisted(src)
    print(f"載入下市股票 {len(dl)} 檔"
          f"（資料夾裡共 {len(list(src.glob('[0-9]*.csv')))} 個檔案）", flush=True)
    if not dl:
        print("還沒有下市股價資料，請先跑 delisted_prices.py --mode full")
        return 1

    M, dcodes = merge_panels(P, dl)
    C, O, V = M["close"], M["open"], M["volume"]
    print(f"  合併後母體 {C.shape[1]} 檔"
          f"（存活 {P['close'].shape[1]} ＋ 下市 {len(dcodes)}）\n", flush=True)

    qe = [pd.Timestamp(d) for d in quarter_ends(O.index)]
    sig = build_signal(C)
    rows = []
    for label, use in (("僅存活股（原結果）", P["close"].columns),
                       ("含下市股（修正後）", C.columns)):
        cs = [c for c in C.columns if c in set(use)]
        mask = pit_universe(C[cs], V[cs], a.uni)
        for hc, hname in ((0.0, "最後收盤價出清"), (0.3, "再扣 30%"),
                          (1.0, "歸零")):
            if label.startswith("僅存活") and hc > 0:
                continue        # 存活股沒有下市事件，壓力測試對它無意義
            fwd = quarter_returns(O[cs], C[cs], qe, hc)
            qd = [d for d in fwd.index if d in sig.index]
            r, turn, ndel = book(sig[cs], fwd, mask, qd, a.n, frozenset(dcodes))
            ew = fwd.where(mask).mean(axis=1).reindex(r.index)
            t, p = nw_tstat((r - ew).dropna())
            rows.append({"母體": label, "下市計價": hname,
                         "年化": ann(r), "夏普": sharpe(r),
                         "母體等權": ann(ew), "超額t": t, "超額p": p,
                         "季數": int(r.notna().sum()), "換手": turn,
                         "下市入選": ndel})
            print(f"  {label:<20}{hname:<16}年化 {ann(r):>+8.2%}"
                  f"　夏普 {sharpe(r):>5.2f}　母體等權 {ann(ew):>+8.2%}"
                  f"　超額t {t:>5.2f}　下市入選 {ndel:>3}", flush=True)

    res = pd.DataFrame(rows)
    out_name = "survivorship_raw.csv" if a.raw else "survivorship.csv"
    res.to_csv(OUT / out_name, index=False, encoding="utf-8-sig")

    base = res[res["母體"].str.startswith("僅存活")].iloc[0]
    fixed = res[res["母體"].str.startswith("含下市")]
    print(f"\n{'=' * 96}\n判讀\n{'=' * 96}")
    print(f"  原結果（僅存活股）　　年化 {base['年化']:+.2%}")
    for _, r in fixed.iterrows():
        print(f"  修正後（{r['下市計價']}）　年化 {r['年化']:+.2%}"
              f"　下降 {base['年化'] - r['年化']:+.2%}")
    print(f"\n  三檔壓力測試之間的差距，就是「下市計價假設」的不確定範圍。")
    print(f"  若連『歸零』情境下年化仍明顯高於母體等權，效應才算經得起"
          f"生存者偏誤。")
    print(f"\n輸出：{OUT / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
