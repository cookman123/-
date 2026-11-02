#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢驗「個股預測」頁裡還沒回測過的兩樣東西：停損／目標價出場，與歷史相似走勢。

事前固定的規格（跑之前寫死，跑完不改、不加）
------------------------------------------
一、停損／目標價（模型策略，h = 5 與 20）
    訊號    每天預測排名前 10 名（研究裡 n = 10 的買進名單），2014 年起全部
    進場    訊號日下一個交易日開盤
    固定    第 1 + h 個交易日開盤出場（就是模型的標籤，也是頁面上的建議出場日）
    停損    訊號日收盤 × exp(出場日的 80% 區間下緣)，區間＝預測中線 ± 1.2816 × 60 日波動 × √t
    目標    訊號日收盤 × exp(max(出場日中線, 出場日 50% 區間上緣 × 0.5))，和 forecast.py 完全相同
    觸發    持有期間開盤跳空越過就用開盤價，盤中碰到就用停損／目標價；同一天兩者都碰到算停損（保守）
    比較    固定、停損＋目標、只停損、只目標，每筆扣 0.3855%（手續費 3 折 × 2 ＋ 證交稅）
    檢定    每筆差異（變體 − 固定）先取月平均，再對月份做 t 檢定（同月訊號高度相關）

二、歷史相似走勢（L = 20、H = 20、k = 10，頁面預設值）
    母體    每個測試日在模型預測母體裡的股票（成交金額前 250 名，含下市股）
    測試日  2014 年起每 21 個交易日一次
    這檔自己  全部測試日 × 全部股票；只用段尾 + H 不晚於測試日的片段
    全市場  隨機抽 24 個測試日 × 40 檔（種子 0），太慢所以只抽樣
    指標    每個測試日的橫斷面 IC（預測中位數 vs 實際之後 H 日報酬）、方向準確率對「永遠猜漲」、
            前 20% 減後 20% 的實際報酬
    出場    偏多（中位數 > 1%）時，「中位數最高那天開盤出場」對「第 H 天開盤出場」，扣成本後比較

輸出：results/forecast_check/
"""
from __future__ import annotations

import sys
import time
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
import forecast as F  # noqa: E402
import paper_trade as P  # noqa: E402

OUT = ROOT / "results" / "forecast_check"
COST = 2 * P.FEE + P.TAX
TOPN, START = 10, "2014-01-01"
PRED = ROOT / "results" / "nextday"
L, H, K, STEP = 20, 20, 10, 21


def month_t(diff: pd.Series) -> tuple:
    m = diff.groupby(diff.index.to_period("M")).mean()
    return float(m.mean()), float(m.mean() / m.std() * np.sqrt(len(m))) if len(m) > 2 else np.nan, len(m)


def load_prices(con) -> dict:
    df = pd.read_sql("SELECT code, date, open, high, low, close FROM prices ORDER BY code, date", con,
                     parse_dates=["date"])
    return {c: g.set_index("date")[["open", "high", "low", "close"]] for c, g in df.groupby("code")}


# ---------------------------------------------------------------- 一、停損／目標

def stop_target(px: dict, h: int) -> pd.DataFrame:
    p = pd.read_csv(PRED / f"pred_h{h}_surv.csv", dtype={"code": str}, parse_dates=["date"])
    p = p[p["date"] >= START]
    sig = p.sort_values("pred", ascending=False).groupby("date").head(TOPN)
    rows = []
    for code, g in sig.groupby("code"):
        d = px.get(code)
        if d is None:
            continue
        o, hi, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        lr = np.diff(np.log(c), prepend=np.nan)
        pos = d.index.get_indexer(g["date"])
        for i, pred, dt in zip(pos, g["pred"], g["date"]):
            if i < 60 or i + h + 1 >= len(o) or not o[i + 1] > 0:
                continue
            sg = np.nanstd(lr[i - 59:i + 1], ddof=1)
            n = h + 1
            path = F._cone(n, np.log1p(pred), sg)
            stop = c[i] * np.exp(path["lo80"].iloc[n])
            tgt = c[i] * np.exp(max(path["mid"].iloc[n], path["hi50"].iloc[n] * 0.5))
            entry, fixed = o[i + 1], o[i + n]
            out = {}
            for name, use_s, use_t in (("停損＋目標", True, True), ("只停損", True, False), ("只目標", False, True)):
                ex = fixed
                for j in range(i + 1, i + n):
                    if j > i + 1 and use_s and o[j] <= stop:
                        ex = o[j]; break
                    if j > i + 1 and use_t and o[j] >= tgt:
                        ex = o[j]; break
                    if use_s and lo[j] <= stop:
                        ex = stop; break
                    if use_t and hi[j] >= tgt:
                        ex = tgt; break
                out[name] = ex / entry - 1 - COST
            rows.append(dict(date=dt, code=code, 固定=fixed / entry - 1 - COST, **out,
                             stop_pct=stop / c[i] - 1, tgt_pct=tgt / c[i] - 1))
    return pd.DataFrame(rows).set_index("date").sort_index()


# ---------------------------------------------------------------- 二、相似走勢

def _universe(h: int = 5) -> pd.DataFrame:
    p = pd.read_csv(PRED / f"pred_h{h}_surv.csv", dtype={"code": str}, parse_dates=["date"],
                    usecols=["date", "code"])
    return p[p["date"] >= START]


def analog_self(px: dict, uni: pd.DataFrame) -> pd.DataFrame:
    days = np.sort(uni["date"].unique())
    test = set(pd.DatetimeIndex(days[::STEP]))
    want = uni[uni["date"].isin(test)].groupby("code")["date"].apply(set)
    rows = []
    for code, dates in want.items():
        d = px.get(code)
        if d is None:
            continue
        lp, o = np.log(d["close"].to_numpy()), d["open"].to_numpy()
        w = F._windows(lp, L, H)
        if w is None:
            continue
        x, f, end = w
        for e0 in d.index.get_indexer(sorted(dates)):
            if e0 < L + 1 or e0 + H >= len(lp):
                continue
            known = end + H <= e0
            if known.sum() < K * 5:
                continue
            q = lp[e0 - L + 1:e0 + 1] - lp[e0 - L]
            xi, fi, ei = x[known], f[known], end[known]
            dist = np.sqrt(((xi - q) ** 2).sum(1))
            pick = F._greedy(np.argsort(dist)[:200], ei, K, L)
            med = np.median(np.column_stack([np.zeros(len(pick)), fi[pick]]), axis=0)
            rows.append(_row(d.index[e0], code, med, lp, o, e0))
    return pd.DataFrame(rows)


def _row(dt, code, med, lp, o, e0) -> dict:
    best = int(np.argmax(med[2:])) + 2
    ok = o[e0 + 1] > 0
    return dict(date=dt, code=code, pred=float(med[-1]), real=float(lp[e0 + H] - lp[e0]),
                best=best, fixed=o[e0 + H] / o[e0 + 1] - 1 - COST if ok else np.nan,
                argmax=o[e0 + best] / o[e0 + 1] - 1 - COST if ok else np.nan)


def analog_market(px: dict, uni: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    days = np.sort(uni["date"].unique())[::STEP]
    days = days[days <= closes.index[-H - 1]]
    rows = []
    for dt in rng.choice(days, size=min(24, len(days)), replace=False):
        codes = uni.loc[uni["date"] == dt, "code"].to_numpy()
        for code in rng.choice(codes, size=min(40, len(codes)), replace=False):
            d = px.get(code)
            a = F.analog_search(closes, code, pd.Timestamp(dt), L, H, K, "market")
            if a is None or d is None:
                continue
            e0 = d.index.get_loc(pd.Timestamp(dt))
            if e0 + H >= len(d):
                continue
            med = np.median(a["F"], axis=0)
            rows.append(_row(pd.Timestamp(dt), code, med, np.log(d["close"].to_numpy()),
                             d["open"].to_numpy(), e0))
        print(f"    全市場 {pd.Timestamp(dt):%Y-%m-%d}　累計 {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def analog_stats(r: pd.DataFrame, name: str) -> dict:
    r = r.dropna(subset=["pred", "real"])
    ic = r.groupby("date").apply(lambda g: g["pred"].corr(g["real"], method="spearman") if len(g) >= 10 else np.nan,
                                 include_groups=False).dropna()

    def spread(g):
        if len(g) < 10:
            return np.nan
        q = g["pred"].rank(pct=True)
        return np.expm1(g.loc[q >= 0.8, "real"]).mean() - np.expm1(g.loc[q <= 0.2, "real"]).mean()

    sp = r.groupby("date").apply(spread, include_groups=False).dropna()
    up = r[r["pred"] > 0.01].dropna(subset=["fixed", "argmax"])
    dm, dt_, nm = month_t((up["argmax"] - up["fixed"]).set_axis(pd.DatetimeIndex(up["date"])))
    return {"範圍": name, "筆數": len(r), "測試日": r["date"].nunique(),
            "IC": ic.mean(), "IC_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
            "方向準確率": (np.sign(r["pred"]) == np.sign(r["real"])).mean(), "永遠猜漲": (r["real"] > 0).mean(),
            "前20%減後20%": sp.mean(), "價差t": sp.mean() / sp.std() * np.sqrt(len(sp)),
            "偏多筆數": len(up), "偏多_固定H天": up["fixed"].mean(), "偏多_中位數高點": up["argmax"].mean(),
            "出場差": dm, "出場差t": dt_}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="停損／目標價與歷史相似走勢的檢驗")
    ap.add_argument("--skip-market", action="store_true", help="不跑全市場相似走勢（最慢）")
    ap.add_argument("--pred-dir", default=str(PRED))
    a = ap.parse_args(argv)
    globals()["PRED"] = Path(a.pred_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    con = F.connect()
    t0 = time.time()
    px = load_prices(con)
    print(f"載入 {len(px)} 檔價格　{time.time() - t0:.0f} 秒")

    print("\n一、停損／目標價")
    rows = []
    for h in (5, 20):
        r = stop_target(px, h)
        r.to_csv(OUT / f"stop_target_h{h}.csv", encoding="utf-8-sig")
        for v in ("固定", "停損＋目標", "只停損", "只目標"):
            m, t, n = month_t(r[v] - r["固定"]) if v != "固定" else (np.nan, np.nan, 0)
            rows.append({"h": h, "出場方式": v, "筆數": len(r), "平均淨報酬": r[v].mean(),
                         "勝率": (r[v] > 0).mean(), "減固定": m, "差異t": t, "月數": n})
        print(f"  h = {h}：{len(r):,} 筆　停損中位數 {r['stop_pct'].median():+.1%}　"
              f"目標中位數 {r['tgt_pct'].median():+.1%}")
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "stop_target_summary.csv", index=False, encoding="utf-8-sig")
    print(st.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n二、歷史相似走勢")
    uni = _universe()
    t0 = time.time()
    rs = analog_self(px, uni)
    rs.to_csv(OUT / "analog_self.csv", index=False, encoding="utf-8-sig")
    print(f"  這檔自己：{len(rs):,} 筆　{time.time() - t0:.0f} 秒")
    stats = [analog_stats(rs, "這檔自己")]
    if not a.skip_market:
        closes, _ = F.load_panel(con)
        t0 = time.time()
        rm = analog_market(px, uni, closes)
        rm.to_csv(OUT / "analog_market.csv", index=False, encoding="utf-8-sig")
        print(f"  全市場：{len(rm):,} 筆　{time.time() - t0:.0f} 秒")
        stats.append(analog_stats(rm, "全市場（抽樣）"))
    an = pd.DataFrame(stats)
    an.to_csv(OUT / "analog_summary.csv", index=False, encoding="utf-8-sig")
    print(an.T.to_string(header=False, float_format=lambda x: f"{x:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
