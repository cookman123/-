#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件評分模型：不預測價格，只在「跌深事件」之間做取捨。

定位
----
規則負責產生候選事件（5 日內跌 8%），模型負責回答三個實務問題：
    同時出現 20 個訊號、資金只夠做 10 個 —— 該選哪 10 個？
    哪些該押重、哪些該押輕？
    哪些乾脆別做？
所以基準不是大盤，而是「照單全收」的規則本身（已知群集 t ≈ 5）。
模型只要贏過照單全收就有價值。

與原論文（DAViS）刻意不同的三件事
----------------------------------
1. 樣本：只取跌深事件，不是每檔每個交易日。
   先前每天都算特徵，但 87% 的日子什麼事都沒發生，等於用空白稀釋訊號。
2. 標籤：這一回合最終的淨報酬（含分批出場與成本），不是未來 h 日報酬率。
   直接對準要決策的事情。
3. 模型：單一梯度提升樹，不做 8 模型堆疊。
   先前的堆疊在 600 列訓練樣本上嚴重過擬合，且元模型是洩漏的主要入口。

防止偷看答案
------------
事件的答案要到出場日才知道，所以訓練集只收「出場日早於測試期起點」的事件，
中間再留 embargo 天數的緩衝。若只用進場日切分，訓練樣本的答案會由測試期的
價格決定 —— 這正是先前讓 t 值從 1.26 灌水到 3.94 的那個錯誤。

所有特徵都取自訊號日（跌幅成立當日）收盤，進場在隔日開盤，
因此特徵在決策當下全部已知。選股範圍用時點正確的成交金額排名，
不使用當期市值。

顯著性一律用以月份為群集的 t 值：全市場同時下跌會讓幾百檔同時觸發，
把每筆事件當獨立樣本會嚴重高估顯著性。
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
import adaptive_strategy as A  # noqa: E402
import davis_tw_train as T  # noqa: E402
import news_reversal as NR  # noqa: E402
import universe_pit as U  # noqa: E402

ROOT = Path(__file__).resolve().parent

TECH_COLS = ["px_k", "px_d", "px_kd_diff", "px_rsi14", "px_atr14", "px_drop_in_atr",
             "px_obv_dev", "px_ma_align", "px_ma60_dev", "px_bb_z", "px_ma5_ma20",
             "px_vol_ratio5", "px_vol_ratio20", "px_pos20", "px_pos60",
             "px_vol20", "px_ret5", "px_ret20"]


def build_events(codes: list, start: str, end: str, watch: int, drop: float,
                 hold: int) -> pd.DataFrame:
    """產生所有跌深事件（回合層級），並附掛決策當下已知的特徵。"""
    fund = _load_fund()
    rows = []
    for i, code in enumerate(codes, 1):
        px, news = NR.load_stock(code, start, end)
        if px is None:
            continue
        ns = NR.daily_news_stats(news, px.index)
        tech = T.step2_price_features(px, quiet=True)
        parts = A.run_stock(px, ns, tech, watch, drop, hold,
                            news_size=False, wait_rebound=True,
                            scale_out=True, early_trim=True,
                            rebound_mode="relative")
        if not parts:
            continue
        p = pd.DataFrame(parts)
        p["code"] = code
        cam = A.to_campaigns(p)
        if cam.empty:
            continue
        # 以訊號日（跌幅成立日）取特徵；該日收盤已知，隔日開盤才進場
        cam["signal"] = pd.to_datetime(cam["drop_day"])
        for c in TECH_COLS:
            if c in tech.columns:
                cam[c] = tech[c].reindex(cam["signal"]).to_numpy()

        # 事件本身的特徵
        c_ = px["close"]
        cam["drop_speed"] = cam["drop"] / watch
        cam["dist_hi60"] = (c_ / c_.rolling(60).max() - 1).reindex(cam["signal"]).to_numpy()
        cam["turnover_z"] = ((px["close"] * px["volume"]).pipe(
            lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std())
        ).reindex(cam["signal"]).to_numpy()

        # 籌碼
        chips = T.load_chips(code)
        if chips is not None and not chips.empty:
            vol = px["volume"].replace(0, np.nan)
            for nm, col in (("fi", "foreign_investor"), ("it", "investment_trust")):
                if col in chips.columns:
                    r = (chips[col].reindex(px.index) / vol).rolling(watch + 1).sum()
                    cam[f"chip_{nm}"] = r.reindex(cam["signal"]).to_numpy()

        # 新聞
        cam["news_neg"] = ns["neg"].rolling(watch + 1).sum().reindex(cam["signal"]).to_numpy()
        cam["news_n"] = ns["n"].rolling(watch + 1).sum().reindex(cam["signal"]).to_numpy()

        # 基本面（相對自身過去兩年的百分位，全為回看視窗）
        if fund is not None:
            fd = fund.get(code)
            if fd is not None and len(fd) > 250:
                for col in ("pe", "pb", "yield_pct"):
                    s = fd[col].reindex(px.index).ffill(limit=5)
                    cam[f"fund_{col}"] = s.rolling(504, min_periods=120).rank(
                        pct=True).reindex(cam["signal"]).to_numpy()
        rows.append(cam)
        if i % 80 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_fund():
    files = sorted((ROOT / "data" / "fundamentals").glob("bwibbu_*.csv"))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files],
                  ignore_index=True)
    d["code"] = d["code"].astype(str).str.strip()
    return {c: g.set_index("date").sort_index()
            for c, g in d.groupby("code")}


def walk_forward(ev: pd.DataFrame, feats: list, test_days: int = 252,
                 min_train: int = 1500, embargo: int = 30) -> pd.DataFrame:
    """
    依事件的「進場日」切測試窗，但訓練集只收「出場日早於測試窗起點 − embargo」的事件。

    只用進場日切分是不夠的：一個在測試窗前一天進場、持有 20 天的事件，
    它的答案是由測試期的價格決定的，放進訓練集等於讓模型看到測試期的結果。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    ev = ev.sort_values("entry").reset_index(drop=True)
    entry = pd.to_datetime(ev["entry"])
    exit_ = pd.to_datetime(ev["exit"])
    if len(ev) <= min_train:
        return pd.DataFrame()

    last = entry.max()
    t0 = entry.iloc[min_train]          # 累積滿 min_train 筆之後才開始測試
    out, n_fold = [], 0
    while t0 <= last:
        t1 = t0 + pd.Timedelta(days=test_days)
        te = ev[(entry >= t0) & (entry < t1)]
        # 訓練集只收「出場日早於測試窗起點 − embargo」的事件：
        # 事件的答案要到出場日才揭曉，用進場日切分會讓訓練標籤由測試期價格決定。
        tr = ev[exit_ < (t0 - pd.Timedelta(days=embargo))]
        if len(tr) >= min_train and len(te) >= 30:
            m = HistGradientBoostingRegressor(
                max_depth=4, max_iter=300, learning_rate=0.05,
                min_samples_leaf=40, early_stopping=False, random_state=42)
            m.fit(tr[feats].to_numpy(), tr["net"].to_numpy())
            o = te[["code", "entry", "exit", "net", "size_mult", "hold"]].copy()
            o["score"] = m.predict(te[feats].to_numpy())
            out.append(o)
            n_fold += 1
            print(f"    fold{n_fold}: 訓練 {len(tr):,}　測試 {len(te):,}"
                  f"（{t0:%Y-%m} ~ {t1:%Y-%m}）", flush=True)
        t0 = t1                          # 前進到下一個測試窗（先前漏了這一行，造成無窮迴圈）
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="事件評分模型")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--watch", type=int, default=5)
    ap.add_argument("--drop", type=float, default=0.08)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--topn", type=int, default=120)
    ap.add_argument("--use-fund", action="store_true",
                    help="納入基本面特徵（資料自 2015 年起，需搭配 --start 2015-01-01）")
    ap.add_argument("--out", default="results/event_model")
    args = ap.parse_args(argv)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    cache = out_dir / "events.csv"
    if cache.exists():
        ev = pd.read_csv(cache, parse_dates=["entry", "exit", "signal"])
        print(f"讀取快取事件 {len(ev):,} 筆")
    else:
        print(f"產生事件：{len(codes)} 檔，{args.start} ~ {args.end}")
        ev = build_events(codes, args.start, args.end, args.watch,
                          args.drop, args.hold)
        ev.to_csv(cache, index=False, encoding="utf-8-sig")
        print(f"事件 {len(ev):,} 筆，已快取")

    mask = U.get_mask(args.start, args.end, args.topn)
    ev = U.filter_trades(ev, mask, "signal")
    print(f"時點正確篩選後 {len(ev):,} 筆（當日成交金額前 {args.topn} 名）")

    # 基本面資料只從 2015 年起，若直接 dropna 會把 2011-2014 的事件全部刪掉，
    # 樣本從 14 年縮成 10 年。所以預設不用基本面（保留全期間），
    # 要測基本面的貢獻時另外限定 2015 年起跑一次，兩者對照才乾淨。
    # 注意排除 news_day：它是日期欄，卻剛好符合 news_ 前綴
    NOT_FEATURES = {"news_day", "drop_day", "entry", "exit", "signal", "campaign"}
    core = [c for c in ev.columns
            if (c.startswith(("px_", "chip_", "news_"))
                or c in ("drop", "drop_speed", "dist_hi60", "turnover_z"))
            and c not in NOT_FEATURES]
    fundc = [c for c in ev.columns if c.startswith("fund_")]
    feats = core + (fundc if args.use_fund else [])
    feats = [c for c in feats if ev[c].notna().mean() > 0.5]
    ev = ev.dropna(subset=feats + ["net"])
    print(f"特徵 {len(feats)} 個"
          f"（{'含' if args.use_fund else '不含'}基本面），可用事件 {len(ev):,} 筆"
          f"　{pd.to_datetime(ev['entry']).min():%Y-%m} ~ "
          f"{pd.to_datetime(ev['entry']).max():%Y-%m}\n")

    P = walk_forward(ev, feats)
    if P.empty:
        print("樣本不足，無法切出測試窗")
        return 1
    P.to_csv(out_dir / "scored.csv", index=False, encoding="utf-8-sig")

    base_m, base_t, base_n = A.cluster_t(P)
    print("=" * 84)
    print(f"基準（照單全收）：{len(P):,} 筆　勝率 {(P.net > 0).mean():.1%}　"
          f"平均 {base_m:+.2%}　群集 t {base_t:.2f}（{base_n} 個月）")
    print("=" * 84)
    print(f"  {'選取方式':<22}{'筆數':>8}{'勝率':>8}{'平均淨報酬':>12}"
          f"{'群集t':>8}{'vs照單全收':>12}{'差異t':>8}")
    for q, label in ((0.5, "模型分數前 50%"), (0.3, "模型分數前 30%"),
                     (0.2, "模型分數前 20%"), (0.1, "模型分數前 10%")):
        thr = P["score"].quantile(1 - q)
        sel = P[P["score"] >= thr]
        if len(sel) < 50:
            continue
        m, t, _ = A.cluster_t(sel)
        d, dt, _ = _diff(P, P["score"] >= thr)
        print(f"  {label:<22}{len(sel):>8,}{(sel.net > 0).mean():>8.1%}"
              f"{m:>+12.2%}{t:>8.2f}{d:>+12.2%}{dt:>8.2f}")
    worst = P[P["score"] <= P["score"].quantile(0.2)]
    mw, tw, _ = A.cluster_t(worst)
    print(f"  {'模型分數後 20%':<22}{len(worst):>8,}{(worst.net > 0).mean():>8.1%}"
          f"{mw:>+12.2%}{tw:>8.2f}")
    print("\n  判讀：『差異t』> 2 才代表模型的挑選真的優於照單全收。")
    print("  另看『後 20%』——若明顯較差，代表模型至少能辨識出不該做的事件。")
    return 0


def _diff(P: pd.DataFrame, mask) -> tuple:
    d = P.assign(_m=pd.to_datetime(P["entry"]).dt.to_period("M"), _g=mask)
    per = d.groupby("_m").apply(
        lambda g: (g.loc[g["_g"], "net"].mean() - g.loc[~g["_g"], "net"].mean())
        if g["_g"].any() and (~g["_g"]).any() else np.nan,
        include_groups=False).dropna()
    if len(per) < 5:
        return np.nan, np.nan, len(per)
    sd = per.std(ddof=1)
    return float(per.mean()), float(per.mean() / (sd / np.sqrt(len(per)))) if sd > 0 else np.nan, len(per)


if __name__ == "__main__":
    raise SystemExit(main())
