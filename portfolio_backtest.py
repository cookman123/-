#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
組合層級回測：把「逐檔獨立擇時」換成「橫斷面排序選股」，並把交易標的與費率
換成實務上真的拿得到的條件。

為什麼要另外寫這支
------------------
davis_tw_train.py 的回測是「每檔股票各自進出」，344 檔就付 344 份交易成本，
而且完全沒有用到模型唯一被證實為正的優勢 —— 橫斷面排序能力
（模型前 20 名每 5 日 +0.366%，全市場等權 +0.130%，差 0.236 個百分點）。
組合式回測只在「換手的那部分」付成本，付一份，這才是這個優勢該有的用法。

三個成本槓桿（都是實務事實，不是放寬假設）
------------------------------------------
1. 手續費折讓：0.1425% 是牌告價，電子下單普遍 6 折、大量交易到 2.8 折。
2. 交易標的：現股證交稅 0.3%，股票型 ETF 只有 0.1%，台指期約 0.002%×2。
   同一個訊號拿去做 0050 或期貨，成本差 3～60 倍。
3. 換手率：組合式回測下，前 N 名名單若只換掉 3 成，就只付 3 成的成本。

誠實註記：h=1 是「今日收盤買、明日收盤賣」，不是當日沖銷，
所以不適用當沖證交稅減半（0.15%）。本程式沒有偷用那個優惠。

防止事後挑選
------------
- 預測值全部來自 walk-forward 的測試折，沒有用到未來資料。
- 擇時門檻用「展開式中位數」（只看 t 之前的歷史），不是在測試期上調參。
- 不挑股票、不挑期間、不挑種子；每個策略都把全部標的、全部測試期跑完。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "results" / "full_v2"   # 可用 --pred-dir 指向別的訓練輸出
OUT_DIR = ROOT / "results" / "portfolio"
TWII = ROOT / "data" / "market" / "TWII.csv"
TRADING_DAYS = 252
HORIZONS = (1, 5, 20, 60)

# ============================== 成本設定 ==============================


@dataclass(frozen=True)
class Cost:
    """單邊費率；round_trip 為買進＋賣出合計。"""

    name: str
    fee: float           # 手續費（買賣各課一次）
    tax: float           # 交易稅（僅賣出課徵）
    borrow: float = 0.0  # 融券借券費（放空才有）

    @property
    def buy(self) -> float:
        return self.fee

    @property
    def sell(self) -> float:
        return self.fee + self.tax

    @property
    def round_trip(self) -> float:
        return 2 * self.fee + self.tax + self.borrow


# 牌告 vs 實務折讓；ETF 與期貨的稅率差異是最大的槓桿
COSTS = {
    "現股_全額": Cost("現股 手續費全額", fee=0.001425, tax=0.003),
    "現股_6折": Cost("現股 手續費6折", fee=0.001425 * 0.6, tax=0.003),
    "現股_3折": Cost("現股 手續費3折", fee=0.001425 * 0.3, tax=0.003),
    "ETF_3折": Cost("股票型ETF 手續費3折", fee=0.001425 * 0.3, tax=0.001),
    "台指期": Cost("台股期貨", fee=0.000030, tax=0.00002),
}
SHORT_COST = Cost("融券 手續費3折", fee=0.001425 * 0.3, tax=0.003, borrow=0.0008)


# ============================== 讀取預測 ==============================


def _is_stock(code: str) -> bool:
    """個股代號是純數字；產業類股是中文名稱，不納入橫斷面組合。"""
    return bool(re.fullmatch(r"\d{4,6}", code))


def load_predictions(horizon: int):
    """回傳 (pred 寬表 date×code, close 寬表 date×code)。close 取自 h=1 檔（日期最完整）。"""
    preds, closes = {}, {}
    tail = f"_h{horizon}.csv"
    for f in sorted(PRED_DIR.glob(f"predictions_*_h{horizon}.csv")):
        code = f.name[len("predictions_"):-len(tail)]
        if not _is_stock(code):
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        preds[code] = d["pred"]
        closes[code] = d["close_t"]
    # h=1 的檔案日期最完整（不必預留 h 天的未來），有的話用它補齊價格軸；
    # 沒跑 h=1 時就用本期間自己的 close_t，只會少掉最後 h 天的價格路徑。
    for f in sorted(PRED_DIR.glob("predictions_*_h1.csv")):
        code = f.name[len("predictions_"):-len("_h1.csv")]
        if not _is_stock(code) or code not in preds:
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        closes[code] = d["close_t"].combine_first(closes[code])
    P = pd.DataFrame(preds).sort_index()
    C = pd.DataFrame(closes).sort_index()
    return P, C


PRICE_DIR = ROOT / "data" / "stocks"


def load_price_panel(codes) -> pd.DataFrame:
    """
    從原始股價檔載入完整價格面板。

    不能用預測檔裡的 close_t：那只在「該檔有測試折預測」的日期才有值，
    walk-forward 的折邊界會造成缺口，於是「整段持有期都有報價」這個條件
    就變成在用未來的資料可得性篩選標的（實測剔除 15.76% 的候選）。
    原始股價檔沒有這個問題。
    """
    out = {}
    for code in codes:
        f = PRICE_DIR / code / "price" / f"yahoo_{code}_merged.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")["close"]
        out[code] = d[~d.index.duplicated(keep="last")]
    return pd.DataFrame(out).sort_index()


def load_twii(index: pd.DatetimeIndex) -> pd.Series:
    t = pd.read_csv(TWII, parse_dates=["date"]).set_index("date")["close"]
    return t.reindex(index.union(t.index)).ffill().reindex(index)


# ============================== 績效 ==============================


def perf(daily_ret: pd.Series, label: str) -> dict:
    r = daily_ret.dropna()
    if r.empty:
        return {"策略": label}
    eq = (1 + r).cumprod()
    yrs = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else np.nan
    sd = r.std()
    return {
        "策略": label,
        "年化報酬": cagr,
        "總報酬": eq.iloc[-1] - 1,
        "年化波動": sd * np.sqrt(TRADING_DAYS),
        "夏普": (r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "最大回撤": (eq / eq.cummax() - 1).min(),
        "交易日數": len(r),
    }


def nw_tstat(x: pd.Series, lags: int | None = None) -> tuple:
    """Newey-West 校正的平均值 t 檢定；超額報酬序列有自相關，OLS 標準誤會低估。"""
    from scipy import stats

    x = x.dropna().to_numpy()
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    g = float(e @ e) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        g += 2 * w * float(e[L:] @ e[:-L]) / n
    se = np.sqrt(max(g, 1e-18) / n)
    t = x.mean() / se
    return t, 2 * (1 - stats.t.cdf(abs(t), n - 1))


# ============================== 策略 ==============================


def rebalance_dates(dates: pd.DatetimeIndex, horizon: int) -> list:
    """每 horizon 個交易日換一次倉，持有期不重疊（避免重複計算同一段報酬）。"""
    return list(dates[::horizon])


def expanding_zscore(P: pd.DataFrame, min_periods: int = 120) -> pd.DataFrame:
    """
    逐檔對「自己過去的預測分布」標準化，只用 t 之前的資料（shift(1)），不涉及未來。

    為什麼需要：模型是逐檔訓練的，高波動股的預測報酬天生就比低波動股大一截，
    直接排序會讓前 N 名長期被同一批高波動股佔滿——那是在選波動度，不是在選訊號。
    標準化後比較的是「這檔相對於它自己平常的預測有多強」。
    """
    mu = P.expanding(min_periods=min_periods).mean().shift(1)
    sd = P.expanding(min_periods=min_periods).std().shift(1)
    return (P - mu) / sd.where(sd > 0)


def overlapping_portfolio(P, C, horizon, top_n, cost: Cost,
                          n_tranche=3, short_n=0, min_names=50):
    """
    重疊分批組合（Jegadeesh-Titman 式）：把資金切成 n_tranche 份，每
    horizon/n_tranche 天換一批，每批各持有 horizon 天。

    為什麼要這樣做：單一批次的組合一年只換倉 4 次（h=60），整段樣本只有 ~38 個
    進場決策，績效被「剛好在哪一天進場」的運氣主導，變異數很大、t 值上不來。
    分批後每 20 天就有一個新決策，進場時點的運氣被平均掉，平均報酬不變但
    波動下降 —— 這是動能文獻的標準做法，不是放寬假設。

    成本沒有變便宜：每一批仍各自付自己的換手成本，只是每次只動 1/n 的資金。
    """
    step = max(1, horizon // n_tranche)
    rets, infos = [], []
    for o in range(0, step * n_tranche, step):
        r, info = cross_sectional_portfolio(P, C, horizon, top_n, cost,
                                            short_n=short_n, min_names=min_names,
                                            offset=o)
        if not r.empty:
            rets.append(r)
            infos.append(info)
    if not rets:
        return pd.Series(dtype=float), {}
    # 各批期初等額投入後「各自漂移」，總淨值＝各批淨值之和。
    # 不可以直接對逐日報酬取平均——那等於每天把資金在各批之間重新配平，
    # 是一個會產生額外換手成本的策略，而那些成本沒有被計入，會憑空多出報酬。
    M = pd.concat(rets, axis=1).dropna()
    if M.empty:
        return pd.Series(dtype=float), {}
    navs = (1.0 + M).cumprod() / len(rets)
    R = navs.sum(axis=1).pct_change().dropna()
    agg = {"批數": len(rets),
           "平均換手率": float(np.mean([i.get("平均換手率", np.nan) for i in infos])),
           "累計成本": float(np.mean([i.get("累計成本", np.nan) for i in infos])),
           "換倉次數": int(np.sum([i.get("換倉次數", 0) for i in infos])),
           "平均持股數": float(np.mean([i.get("平均持股數", np.nan) for i in infos]))}
    return R, agg


def cross_sectional_portfolio(P, C, horizon, top_n, cost: Cost,
                              short_n=0, min_names=50, offset=0):
    """
    橫斷面排序組合：每 horizon 日取排序分數最高的 top_n 檔等權做多
    （short_n > 0 時同時做空最低的 short_n 檔），成本只對換手部分課徵。

    期間內各檔採「進場等權、之後不動」，與實際下單一致；權重漂移照實計算，
    換手率取新舊權重差的正負部分，賣出付手續費＋交易稅、買進付手續費。
    """
    dates = C.index
    # offset 讓重疊分批的每一批從不同的起始日開始排換倉日
    rebs = rebalance_dates(P.index[offset:], horizon)
    nav, nav_dates = [1.0], []
    prev_w = pd.Series(dtype=float)
    cost_paid, turnovers, n_held = 0.0, [], []

    for i, t in enumerate(rebs):
        end = rebs[i + 1] if i + 1 < len(rebs) else None
        seg = (dates[(dates >= t) & (dates <= end)] if end is not None
               else dates[dates >= t])
        if len(seg) < 2:
            continue
        p = P.loc[t].dropna() if t in P.index else pd.Series(dtype=float)
        if p.empty:
            continue
        # 選股濾網只能用「換倉當天」就知道的資訊：t 當日有正報價。
        # 不可以要求「整段持有期都有報價」——那是用未來的資料可得性篩選標的。
        cand = [c for c in p.index if c in C.columns]
        px_t = C.loc[t, cand] if t in C.index else pd.Series(dtype=float)
        ok = px_t.notna() & (px_t > 0)
        p = p[[c for c in cand if bool(ok.get(c, False))]]
        if len(p) < min_names:
            continue

        longs = p.nlargest(min(top_n, len(p))).index
        shorts = p.nsmallest(short_n).index if short_n else pd.Index([])
        gross = len(longs) + len(shorts)
        if gross == 0:
            continue
        w = pd.Series(0.0, index=p.index)
        w[longs] = 1.0 / gross
        if short_n:
            w[shorts] = -1.0 / gross

        # 換手成本：相對上一期「漂移後」的權重
        allc = prev_w.index.union(w.index)
        d = w.reindex(allc).fillna(0.0) - prev_w.reindex(allc).fillna(0.0)
        sell, buy = float(-d[d < 0].sum()), float(d[d > 0].sum())
        c = sell * cost.sell + buy * cost.buy
        if short_n:
            c += float(w[w < 0].abs().sum()) * cost.borrow
        cost_paid += c
        turnovers.append(buy)
        n_held.append(gross)
        nav[-1] *= (1 - c)

        # 期間內逐日：等權進場後持有，權重隨價格漂移
        held = w.index[w != 0]
        # 持有期內若停牌／缺值，以最後成交價延續（實務上就是賣不掉、帳面不動）
        px = C.loc[seg, held].ffill()
        rel = px.div(px.iloc[0], axis=1)
        w0 = w[held]
        # 未投入部位視為現金。現金權重要用「帶正負號」的權重和：
        # 多頭組合 Σw=1 → 現金 0；市場中立多空 Σw=0 → 現金 1，淨值從 1 起算。
        # 若誤用 Σ|w|，多空組合的起始淨值會是 0，之後每日相除就會噴出 inf。
        val = (rel * w0).sum(axis=1) + (1 - float(w0.sum()))
        val = val / val.iloc[0]
        for j in range(1, len(seg)):
            nav.append(nav[-1] * float(val.iloc[j] / val.iloc[j - 1]))
            nav_dates.append(seg[j])

        last = rel.iloc[-1] * w0
        denom = float(last.abs().sum())
        prev_w = last / denom if denom > 0 else last

    if not nav_dates:
        return pd.Series(dtype=float), {}
    s = pd.Series(nav[1:], index=pd.DatetimeIndex(nav_dates)).sort_index()
    return s.pct_change().dropna(), {
        "平均換手率": float(np.mean(turnovers)) if turnovers else np.nan,
        "累計成本": cost_paid,
        "換倉次數": len(turnovers),
        "平均持股數": float(np.mean(n_held)) if n_held else np.nan,
    }


def index_timing(P, twii, horizon, cost: Cost):
    """
    大盤擇時：用「橫斷面預測廣度」當訊號去做指數（0050／台指期）。

    訊號＝當日所有個股預測報酬的平均；門檻用展開式中位數（只看 t 之前），
    不在測試期上挑門檻。訊號 > 門檻才持有指數，否則空手。
    """
    breadth = P.mean(axis=1).dropna()
    thr = breadth.expanding(min_periods=60).median().shift(1)
    sig = breadth > thr
    rebs = rebalance_dates(breadth.index, horizon)
    idx = twii.reindex(breadth.index).ffill()

    nav, nav_dates = [1.0], []
    pos, cost_paid, trades, held_flags = 0.0, 0.0, 0, []
    for i, t in enumerate(rebs):
        if pd.isna(thr.get(t, np.nan)):
            continue
        end = rebs[i + 1] if i + 1 < len(rebs) else None
        seg = (breadth.index[(breadth.index >= t) & (breadth.index <= end)]
               if end is not None else breadth.index[breadth.index >= t])
        if len(seg) < 2:
            continue
        want = 1.0 if bool(sig.get(t, False)) else 0.0
        held_flags.append(want)
        d = want - pos
        c = (d * cost.buy) if d > 0 else (-d * cost.sell)
        if d != 0:
            trades += 1
        cost_paid += c
        nav[-1] *= (1 - c)
        pos = want
        r = idx.loc[seg].pct_change().fillna(0.0)
        for j in range(1, len(seg)):
            nav.append(nav[-1] * (1 + pos * float(r.iloc[j])))
            nav_dates.append(seg[j])

    if not nav_dates:
        return pd.Series(dtype=float), {}
    s = pd.Series(nav[1:], index=pd.DatetimeIndex(nav_dates)).sort_index()
    return s.pct_change().dropna(), {
        "進出次數": trades,
        "累計成本": cost_paid,
        "持有比例": float(np.mean(held_flags)) if held_flags else np.nan,
    }


# ============================== 主程式 ==============================


def main(argv=None) -> int:
    import argparse

    global PRED_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description="組合層級回測（橫斷面排序 + 實務費率）")
    ap.add_argument("--pred-dir", default=str(PRED_DIR),
                    help="訓練輸出目錄，需含 predictions_{代號}_h{期間}.csv")
    ap.add_argument("--out", default=None, help="結果輸出目錄（預設 results/portfolio）")
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS))
    a = ap.parse_args(argv)
    PRED_DIR = Path(a.pred_dir)
    OUT_DIR = Path(a.out) if a.out else OUT_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"預測來源：{PRED_DIR}")
    rows = []

    for horizon in a.horizon:
        print(f"\n{'=' * 76}\n持有期 h={horizon}\n{'=' * 76}", flush=True)
        P, C_pred = load_predictions(horizon)
        if P.empty:
            print("  無預測資料，略過")
            continue
        # 價格軸用原始股價檔；原始檔缺的標的才退回預測檔的 close_t
        C = load_price_panel(list(P.columns))
        C = C.reindex(C.index.union(C_pred.index))
        for c in C_pred.columns:
            if c not in C.columns or C[c].notna().sum() == 0:
                C[c] = C_pred[c].reindex(C.index)
        C = C.sort_index()
        print(f"  個股 {P.shape[1]} 檔，測試日 {P.shape[0]} 天"
              f"（{P.index.min():%Y-%m-%d} ~ {P.index.max():%Y-%m-%d}）", flush=True)
        twii = load_twii(C.index)

        bench_ret = twii.pct_change().reindex(P.index).dropna()
        base = perf(bench_ret, "大盤（加權指數，買進持有）")
        rows.append({**base, "持有期": horizon, "類別": "基準"})

        # 存下每個策略的逐日超額報酬序列，供 reality_check.py 做多重檢定校正。
        # 只看「最好的那一格」會嚴重高估顯著性，必須把整個搜尋過的策略族一起檢定。
        exc_series: dict = {}

        ew, ewi = cross_sectional_portfolio(P, C, horizon, 10 ** 9, COSTS["現股_3折"])
        if not ew.empty:
            m = perf(ew, "全市場等權（3折）")
            exc = (ew - bench_ret.reindex(ew.index)).dropna()
            t, p = nw_tstat(exc)
            exc_series[m["策略"]] = exc
            rows.append({**m, "持有期": horizon, "類別": "基準",
                         "超額年化": m["年化報酬"] - base["年化報酬"],
                         "超額t值": t, "超額p值": p, **ewi})

        # 兩種排序分數：原始預測報酬 vs 逐檔展開式 z 分數
        scores = {"原始": P, "z分數": expanding_zscore(P, min_periods=60)}

        for sname, S in scores.items():
            for top_n in (10, 20, 50):
                for ck in ("現股_全額", "現股_3折"):
                    r, info = cross_sectional_portfolio(S, C, horizon, top_n, COSTS[ck])
                    if r.empty:
                        continue
                    m = perf(r, f"橫斷面多頭 前{top_n}名／{sname}（{ck}）")
                    exc = (r - bench_ret.reindex(r.index)).dropna()
                    t, p = nw_tstat(exc)
                    exc_series[m["策略"]] = exc
                    rows.append({**m, "持有期": horizon, "類別": "多頭組合",
                                 "排序": sname,
                                 "超額年化": m["年化報酬"] - base["年化報酬"],
                                 "超額t值": t, "超額p值": p, **info})

            # 重疊分批版本：同樣的選股規則，只是把進場時點分散開
            if horizon >= 20:
                for top_n in (20, 50):
                    r, info = overlapping_portfolio(S, C, horizon, top_n,
                                                    COSTS["現股_3折"], n_tranche=3)
                    if r.empty:
                        continue
                    m = perf(r, f"重疊分批 前{top_n}名／{sname}（現股_3折,3批）")
                    exc = (r - bench_ret.reindex(r.index)).dropna()
                    t, p = nw_tstat(exc)
                    exc_series[m["策略"]] = exc
                    rows.append({**m, "持有期": horizon, "類別": "多頭組合",
                                 "排序": sname,
                                 "超額年化": m["年化報酬"] - base["年化報酬"],
                                 "超額t值": t, "超額p值": p, **info})

            # 多空組合：看優勢是否對稱。實務有平盤下不得放空等限制，僅供研究比較。
            r, info = cross_sectional_portfolio(S, C, horizon, 20, SHORT_COST, short_n=20)
            if not r.empty:
                m = perf(r, f"橫斷面多空 前後各20名／{sname}（融券3折）")
                exc = (r - bench_ret.reindex(r.index)).dropna()
                t, p = nw_tstat(exc)
                exc_series[m["策略"]] = exc
                rows.append({**m, "持有期": horizon, "類別": "多空組合",
                             "排序": sname,
                             "超額年化": m["年化報酬"] - base["年化報酬"],
                             "超額t值": t, "超額p值": p, **info})

        for ck in ("現股_3折", "ETF_3折", "台指期"):
            r, info = index_timing(P, twii, horizon, COSTS[ck])
            if r.empty:
                continue
            m = perf(r, f"大盤擇時（{ck}）")
            exc = (r - bench_ret.reindex(r.index)).dropna()
            t, p = nw_tstat(exc)
            exc_series[m["策略"]] = exc
            rows.append({**m, "持有期": horizon, "類別": "大盤擇時",
                         "超額年化": m["年化報酬"] - base["年化報酬"],
                         "超額t值": t, "超額p值": p, **info})

        if exc_series:
            pd.DataFrame(exc_series).to_csv(
                OUT_DIR / f"excess_returns_h{horizon}.csv", encoding="utf-8-sig")

        df = pd.DataFrame([x for x in rows if x.get("持有期") == horizon])
        show = [c for c in ["策略", "年化報酬", "夏普", "最大回撤",
                            "超額年化", "超額t值", "超額p值"] if c in df.columns]
        v = df[show].copy()
        for c in ("年化報酬", "最大回撤", "超額年化"):
            if c in v:
                v[c] = v[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
        for c in ("夏普", "超額t值", "超額p值"):
            if c in v:
                v[c] = v[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        print(v.to_string(index=False), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n完整結果：{OUT_DIR / 'portfolio_summary.csv'}")

    if "超額年化" in out.columns:
        cand = out.dropna(subset=["超額年化"]).sort_values("超額年化", ascending=False)
        if not cand.empty:
            print(f"\n{'=' * 76}\n超額報酬排名前 12"
                  f"（超額 t 值 > 2 才算統計上站得住）\n{'=' * 76}")
            c = cand.head(12)[["持有期", "策略", "年化報酬", "超額年化",
                               "夏普", "超額t值", "超額p值"]].copy()
            for col in ("年化報酬", "超額年化"):
                c[col] = c[col].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
            for col in ("夏普", "超額t值", "超額p值"):
                c[col] = c[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            print(c.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
