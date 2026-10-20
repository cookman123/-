#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔日預測（全檔個股版）：每天用一個共用模型預測每一檔明天的報酬，
今天收盤買進預測最高的一籃、放空預測最低的一籃。

與指數版的差別
--------------
指數版只有一條時間序列，樣本數 = 交易日數。個股版把 383 檔疊成一張
長表（日 × 檔），樣本數放大兩個數量級，模型學得到的是**橫斷面**的
相對強弱，而不是大盤方向——這正是「今天買哪幾檔」需要的東西。

不偷看答案
----------
    特徵只用到第 t 日收盤為止的資料（逐檔各自計算，不跨檔洩漏）
    標籤是該檔的 r(t+1) = C(t+1)/C(t) − 1
    walk-forward 依「日期」切分，不是依列切分 —— 同一天的所有個股
      必須整批落在同一側，否則同日其他檔的答案會洩漏給訓練集
    訓練集尾端再切掉 1 個交易日（purging）
    選股母體用時點正確的成交金額排名，與主策略一致

成本
----
個股來回約 0.39%（手續費 0.1425%×3 折 ×2 ＋ 賣出證交稅 0.3%）。
每天換一次手就要付掉這個數，所以一律同時報「扣成本前／後」。
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
COSTS = {"現股": (FEE, TAX), "零成本": (0.0, 0.0)}


def build_long(codes: list, start: str, end: str, mask: pd.DataFrame) -> pd.DataFrame:
    """把每檔的特徵與標籤疊成長表：index=(date, code)。"""
    frames = []
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        if len(px) < 400:
            continue
        f = T.step2_price_features(px, quiet=True)
        f = f.replace([np.inf, -np.inf], np.nan)
        f["_y"] = px["close"].pct_change().shift(-1)
        f["code"] = code
        frames.append(f.reset_index().rename(columns={"index": "date"}))
        if i % 50 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔", flush=True)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["_y"])
    # 只保留當日在選股母體內的列（時點正確）
    m = mask.stack()
    m.index.names = ["date", "code"]
    m = m[m].index
    df = df.set_index(["date", "code"])
    df = df[df.index.isin(m)]
    return df.sort_index()


def walk_forward(df: pd.DataFrame, feats: list, min_days: int,
                 step_days: int) -> pd.Series:
    """依日期切分的 walk-forward，訓練集尾端切掉 1 個交易日。"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    days = df.index.get_level_values("date").unique().sort_values()
    out = []
    for k in range(min_days, len(days), step_days):
        tr_days = days[:k - 1]                      # purge 最後 1 日
        te_days = days[k:min(k + step_days, len(days))]
        if len(te_days) == 0:
            break
        tr = df[df.index.get_level_values("date").isin(tr_days)]
        te = df[df.index.get_level_values("date").isin(te_days)]
        if len(tr) < 20000 or te.empty:
            continue
        m = HistGradientBoostingRegressor(
            max_depth=4, max_iter=250, learning_rate=0.05,
            min_samples_leaf=200, early_stopping=False, random_state=42)
        m.fit(tr[feats].to_numpy(), tr["_y"].to_numpy())
        out.append(pd.Series(m.predict(te[feats].to_numpy()), index=te.index))
        print(f"    {te_days[0]:%Y-%m} ~ {te_days[-1]:%Y-%m}"
              f"　訓練 {len(tr):,} 列　測試 {len(te):,} 列", flush=True)
    return pd.concat(out) if out else pd.Series(dtype=float)


def portfolio(pred: pd.Series, y: pd.Series, n_pick: int, side: str,
              fee: float, tax: float) -> tuple:
    """每天取預測最高／最低各 n_pick 檔，等權；成本依實際換手計。"""
    daily, prev_long, prev_short, turns = [], set(), set(), []
    for d, g in pred.groupby(level="date"):
        g = g.droplevel("date").sort_values(ascending=False)
        if len(g) < n_pick * 2:
            continue
        longs = list(g.index[:n_pick])
        shorts = list(g.index[-n_pick:]) if side == "多空" else []
        yd = y.xs(d, level="date")
        rl = float(yd.reindex(longs).fillna(0.0).mean())
        rs = float(yd.reindex(shorts).fillna(0.0).mean()) if shorts else 0.0
        r = rl - rs if side == "多空" else rl
        # 換手：與前一日名單的差異比例，買賣各收一次（賣出含稅）
        tl = 1 - len(set(longs) & prev_long) / n_pick if prev_long else 1.0
        ts = (1 - len(set(shorts) & prev_short) / n_pick) if (shorts and prev_short) else (1.0 if shorts else 0.0)
        turn = (tl + ts) / (2 if shorts else 1)
        r -= turn * (fee + (fee + tax))
        daily.append((d, r))
        turns.append(turn)
        prev_long, prev_short = set(longs), set(shorts)
    s = pd.Series(dict(daily)).sort_index()
    return s, float(np.mean(turns)) if turns else np.nan


def stats(r: pd.Series, base: pd.Series | None = None) -> dict:
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
        t, p = nw_tstat(r.reindex(idx) - base.reindex(idx))
        d["超額t"], d["超額p"] = t, p
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="隔日預測（全檔個股）")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--uni", type=int, default=250)
    ap.add_argument("--min-days", type=int, default=1000)
    ap.add_argument("--step-days", type=int, default=252)
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"載入 {len(codes)} 檔並建立特徵長表…", flush=True)
    mask = U.get_mask(a.start, a.end, a.uni)
    df = build_long(codes, a.start, a.end, mask)
    if df.empty:
        print("沒有可用資料"); return 1
    feats = [c for c in df.columns if c.startswith("px_")
             and df[c].notna().mean() > 0.9]
    df = df.dropna(subset=feats + ["_y"])
    days = df.index.get_level_values("date").unique()
    print(f"\n長表 {len(df):,} 列　{len(days):,} 個交易日"
          f"　特徵 {len(feats)} 個　平均每日 {len(df)/len(days):.0f} 檔\n")

    print("walk-forward（依日期切分，訓練集尾端切掉 1 日）…", flush=True)
    pred = walk_forward(df, feats, a.min_days, a.step_days)
    if pred.empty:
        print("樣本不足"); return 1
    pred.rename("pred").reset_index().to_csv(
        OUT / "pred_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"  預測已存檔：{OUT / 'pred_stocks.csv'}", flush=True)
    y = df["_y"]
    act = y.reindex(pred.index)
    da = float((np.sign(pred) == np.sign(act))[act != 0].mean())
    ic_daily = pd.concat([pred.rename("p"), act.rename("y")], axis=1) \
        .groupby(level="date").apply(
            lambda g: g["p"].corr(g["y"], method="spearman") if len(g) > 5 else np.nan,
            include_groups=False).dropna()
    icm, ict = float(ic_daily.mean()), float(
        ic_daily.mean() / (ic_daily.std(ddof=1) / np.sqrt(len(ic_daily))))
    print(f"\n{'=' * 100}")
    print(f"樣本外 {len(pred):,} 列　{pred.index.get_level_values('date').min():%Y-%m}"
          f" ~ {pred.index.get_level_values('date').max():%Y-%m}")
    print(f"  方向準確率 {da:.2%}（隨機＝50%）")
    print(f"  每日橫斷面 IC 平均 {icm:+.4f}　t = {ict:.2f}"
          f"（{len(ic_daily):,} 天）")
    print("=" * 100)

    # 等權持有全母體：橫斷面策略的正確對照
    ew = act.groupby(level="date").mean()
    s_ew = stats(ew)
    print(f"  母體等權對照　年化 {s_ew.get('年化', np.nan):+.2%}"
          f"　夏普 {s_ew.get('夏普', np.nan):.2f}")
    print(f"\n  {'方式':<8}{'檔數':>6}{'換手':>7}" +
          "".join(f"{k:>12}" for k in COSTS) + f"{'夏普':>8}{'超額t':>8}")
    rows = []
    for side in ("多單", "多空"):
        for n in (10, 30, 50):
            line = {}
            for cname, (fee, tax) in COSTS.items():
                r, turn = portfolio(pred, y, n, side, fee, tax)
                st = stats(r, ew)
                if not st:
                    line = {}; break
                line[cname] = st
                last_turn = turn
            if not line:
                continue
            rows.append({"方式": side, "檔數": n, "換手": last_turn, "DA": da,
                         "IC": icm, "IC_t": ict,
                         **{f"年化_{k}": v["年化"] for k, v in line.items()},
                         "夏普": line["現股"]["夏普"],
                         "超額t": line["現股"]["超額t"]})
            print(f"  {side:<8}{n:>6}{last_turn:>7.2f}" +
                  "".join(f"{line[k]['年化']:>+12.2%}" for k in COSTS) +
                  f"{line['現股']['夏普']:>8.2f}{line['現股']['超額t']:>8.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "nextday_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"\n判讀：『零成本』那一欄是訊號的理論上限；若只有它為正，")
    print(f"      代表訊號存在但付不起每天換手的 {2*FEE+TAX:.2%} 成本。")
    print(f"\n輸出：{OUT / 'nextday_stocks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
