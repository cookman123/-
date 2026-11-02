#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
個股預測：選一套策略，給出方向、預測區間、建議進出場點，以及歷史相似走勢。

前端「個股預測」頁（frontend/p8_forecast.py）與 local_ask.py 共用。全部只用視角日期
（asof）當天收盤為止已知的資料：
  模型預測   predictions 表的 date ≤ asof（收盤後才算得出來）
  相似走勢   只拿「之後 H 天也在 asof 以前走完」的歷史片段，答案都已揭曉
  跌深事件   只拿 exit < asof 的事件（和 rag_rule.py 的檢索庫同一個原則）

策略
----
  h5 / h20 / h1  價量模型（nextday_horizon.py，含下市股的稽核版）。它是排序模型，
                 方向看當天全市場排名百分位：前 30% 偏多、後 30% 偏空
  analog         過去 L 天的累積報酬曲線，找歷史上最像的 k 段（歐氏距離），看它們之後 H 天怎麼走
  dip            跌深反彈規則（5 日跌 8%）＋候選 RAG 排除規則
  highvol        高波動＋120 日均線過濾（組合層級的規則，不預測單一股票漲跌）

進出場點：一律收盤後判斷、下一個交易日開盤成交，和所有回測同一個假設。
建議出場日一律是各策略回測時的固定持有期間。停損與目標價仍然畫出來當參考
（停損＝出場日 80% 區間下緣、目標＝出場日中線與 50% 區間上緣一半取大），但不是建議：
forecast_check.py 測過（2026-11-02），在模型的買進名單上照這兩條線出場，每筆平均比固定持有
少 0.74%（5 日，t −6.56）與 1.81%（20 日，t −5.18）。相似走勢改在「中位數最高那天」出場也比
固定 20 天差（−0.24%，t −2.17），所以出場日不用它。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent
DB = ROOT / "db" / "local.db"
UP_Q, DN_Q = 0.70, 0.30
# forecast_check.py 的實測（模型前 10 名、2014 年起約 3 萬筆）：照停損＋目標價出場減固定持有
STOP_TEST = "實測照停損＋目標價出場，每筆比固定持有少 0.74%（5 日，t −6.56）、1.81%（20 日，t −5.18）；勝率較高但大賺的被提早賣掉"          # 模型排名百分位的偏多／偏空界線
Z50, Z80 = 0.6745, 1.2816        # 常態近似的 50%／80% 區間
MAX_JUMP = 0.20                  # 單日 |log 報酬| 超過就當成資料跳動，不拿來比對
DIP_DROP = -0.08
HV_POOL, HV_N, HV_BUF, HV_HOLD = 250, 30, 40, 42

STRATEGIES = {
    "h5": dict(name="5 日模型", h=5, kind="model", evidence=(
        "含下市股、2026-11-02 重跑：買排名前 30 檔、緩衝 5 名，對沖大盤後年化 +10.1%（t 2.37）；"
        "前 10 檔只剩 +2.7%（t 0.93）。12 組一起檢定 RC p 0.051、SPA p 0.036，兩個都要 < 0.05，差一點沒通過。"
        "IC +0.050：排序有一點點用，單一股票的漲跌大多還是猜不準。")),
    "h20": dict(name="20 日模型", h=20, kind="model", evidence=(
        "含下市股重跑：IC +0.044；前 10 檔對沖後 +12.4%（t 2.19）、前 30 檔 +8.6%（t 2.16），"
        "但和 5 日模型同一組多重比較，一起沒有通過。")),
    "h1": dict(name="隔日模型", h=1, kind="model", evidence=(
        "隔日排序能力是真的（IC +0.069），但扣成本、對沖大盤後 4 組全部是負的（−1.5% 到 −25.7%），"
        "研究結論是不可交易。這裡只拿來看方向。")),
    "analog": dict(name="歷史相似走勢", h=20, kind="analog", evidence=(
        "2026-11-02 全市場檢驗（forecast_check.py，2014 年起 145 個測試日、3.6 萬筆）：只找這檔自己的歷史時，"
        "排序有一點點用（IC +0.022，t 3.66），但方向準確率 49.9%，和「永遠猜漲」的 49.1% 差不多，"
        "前 20% 減後 20% 每 20 天 +0.37%（t 1.90）不顯著。找全市場的版本完全沒用（抽樣 959 筆，IC −0.002）。"
        "只能當參考，不能單獨拿來交易。")),
    "dip": dict(name="跌深反彈規則", h=None, kind="dip", evidence=(
        "5 日內跌 8%、等止跌後進場。全樣本平均淨報酬是正的但很薄；"
        "RAG 排除規則還是候選（排除減保留 −0.78%、t −1.51，不顯著）。")),
    "highvol": dict(name="高波動＋均線過濾", h=HV_HOLD, kind="highvol", evidence=(
        "20 年年化 +16.8%、對大盤超額 t 1.46（不顯著），沖掉 beta 後也不顯著。"
        "這是 30 檔一起抱的組合規則，賺的是波動度溢酬，不是預測單一股票會漲。")),
}


# ---------------------------------------------------------------- 資料

def connect(db: str | Path = DB) -> sqlite3.Connection:
    return sqlite3.connect(db, check_same_thread=False)


def load_px(con, code: str) -> pd.DataFrame:
    return pd.read_sql("SELECT date, open, high, low, close, volume FROM prices WHERE code = ? ORDER BY date",
                       con, params=[code], parse_dates=["date"]).set_index("date")


def load_panel(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全市場收盤與成交量（日期 × 代號），相似走勢的全市場比對與高波動排名用。"""
    df = pd.read_sql("SELECT date, code, close, volume FROM prices", con, parse_dates=["date"])
    return (df.pivot(index="date", columns="code", values="close").sort_index(),
            df.pivot(index="date", columns="code", values="volume").sort_index())


def future_dates(index: pd.DatetimeIndex, asof: pd.Timestamp, n: int) -> list:
    """asof 之後 n 個交易日；資料裡有的用實際日期，不夠的用平日補。"""
    known = list(index[index > asof][:n])
    if len(known) < n:
        start = known[-1] if known else asof
        known += list(pd.bdate_range(start + pd.Timedelta(days=1), periods=n - len(known)))
    return [asof] + known


# ---------------------------------------------------------------- 共用

def _cone(n: int, mid_end: float, sigma: float) -> pd.DataFrame:
    """常態近似的錐形區間（log 報酬）；t = 0 是 asof 收盤。"""
    t = np.arange(n + 1)
    mid, s = mid_end * t / max(n, 1), sigma * np.sqrt(t)
    return pd.DataFrame({"mid": mid, "lo50": mid - Z50 * s, "hi50": mid + Z50 * s,
                         "lo80": mid - Z80 * s, "hi80": mid + Z80 * s}, index=t)


def _sigma(close: pd.Series, n: int = 60) -> float:
    r = np.log(close).diff().iloc[-n:]
    return float(r.std()) if r.notna().sum() > 10 else 0.03


def _plan(res: dict, enter: bool, exit_t: int, note: str = "") -> None:
    """建議進出場點：t = 1 開盤進場、exit_t 開盤出場（固定持有）；停損與目標價只是參考線，見檔頭的檢驗。"""
    p, c, d = res["path"], res["close"], res["dates"]
    exit_t = int(min(max(exit_t, 2), len(p) - 1))
    if enter:
        res["plan"] = dict(enter=True, entry_t=1, entry_date=d[1], exit_t=exit_t, exit_date=d[exit_t],
                           pre_exit=d[exit_t - 1],
                           entry_est=c, target=c * np.exp(max(p["mid"].iloc[exit_t], p["hi50"].iloc[exit_t] * 0.5)),
                           stop=c * np.exp(p["lo80"].iloc[exit_t]), note=note)
    else:
        res["plan"] = dict(enter=False, exit_date=d[1], note=note)


EXIT_MODES = {"fixed": "固定持有（建議）", "both": "停損＋目標價", "stop": "只停損", "target": "只目標價"}
# forecast_check.py 的欄名
EXIT_COLS = {"fixed": "固定", "both": "停損＋目標", "stop": "只停損", "target": "只目標"}


def simulate_exit(px: pd.DataFrame, entry_date, exit_date, stop: float | None, target: float | None):
    """照出場方式模擬：entry_date 開盤進場，持有期間開盤跳空越過停損／目標就用開盤價，盤中碰到用該價位
    （同一天兩者都碰到算停損），都沒碰到就 exit_date 開盤出場。和 forecast_check.py 的規則相同。
    回傳（出場日, 出場價, 原因）；價格還沒走到能判斷的那天就回傳 None。"""
    d = px.loc[(px.index >= pd.Timestamp(entry_date)) & (px.index <= pd.Timestamp(exit_date))]
    if d.empty or d.index[0] != pd.Timestamp(entry_date):
        return None
    for k, (dt, r) in enumerate(d.iterrows()):
        if dt == pd.Timestamp(exit_date):
            return dt, float(r["open"]), "到期"
        if k > 0 and stop is not None and r["open"] <= stop:
            return dt, float(r["open"]), "停損（開盤跳空）"
        if k > 0 and target is not None and r["open"] >= target:
            return dt, float(r["open"]), "目標（開盤跳空）"
        if stop is not None and r["low"] <= stop:
            return dt, float(stop), "停損"
        if target is not None and r["high"] >= target:
            return dt, float(target), "目標"
    return None


def exit_levels(plan: dict, mode: str) -> tuple:
    """出場方式對應的（停損, 目標）；不用的那條是 None。"""
    return (plan["stop"] if mode in ("both", "stop") else None,
            plan["target"] if mode in ("both", "target") else None)


def _base(key: str, code: str, hist: pd.DataFrame, full_index, n: int) -> dict:
    a = hist.index[-1]
    return dict(strategy=key, name=STRATEGIES[key]["name"], code=code, asof=a, close=float(hist["close"].iloc[-1]),
                h=n, dates=future_dates(full_index, a, n), direction="中性", tone="neutral",
                facts=[], warn=[], calib=None, plan=None, evidence=STRATEGIES[key]["evidence"])


# ---------------------------------------------------------------- 模型

_DAYS: dict = {}


def complete_days(con, h: int) -> pd.Series:
    """每天有預測的檔數；當天爬蟲只跑了一部分時只有幾十檔，排名不能用，不到 100 檔的日子不算。
    掃一次 predictions 表要將近一秒，逐日跑策略時會呼叫幾百次，所以快取起來。"""
    if h not in _DAYS:
        n = pd.read_sql("SELECT date, COUNT(*) n FROM predictions WHERE h = ? GROUP BY date", con,
                        params=[h]).set_index("date")["n"]
        _DAYS[h] = n[n >= 100]
    return _DAYS[h]


def model_plan(hist: pd.DataFrame, full_index, pred: float, h: int) -> dict:
    """只算建議進出場點（不查歷史紀錄），給逐日自動下單用；和 _model 偏多時的計畫完全相同。"""
    res = dict(close=float(hist["close"].iloc[-1]), dates=future_dates(full_index, hist.index[-1], h + 1),
               path=_cone(h + 1, np.log1p(pred), _sigma(hist["close"])))
    _plan(res, True, h + 1)
    return res["plan"]


def _model(con, key, code, hist, full_index, calib: bool = True) -> dict:
    h = STRATEGIES[key]["h"]
    res = _base(key, code, hist, full_index, h + 1)
    asof = res["asof"].strftime("%Y-%m-%d")
    days = complete_days(con, h)
    rows = con.execute("SELECT date, pred, pctile FROM predictions WHERE code = ? AND h = ? AND date <= ? "
                       "ORDER BY date DESC LIMIT 10", [code, h, asof]).fetchall()
    r = next((x for x in rows if x[0] in days.index), None)
    sig = _sigma(hist["close"])
    if not r:
        res.update(direction="無預測", tone="muted", path=_cone(h + 1, 0.0, sig))
        res["warn"].append("這檔在視角日期以前沒有模型預測（模型只算成交金額前 250 名的母體）。")
        return res
    d, pred, pc = r
    n_day = int(days[d])
    stale = int((hist.index > pd.Timestamp(d)).sum())
    rank = int(round((1 - pc) * n_day)) + 1
    res["direction"], res["tone"] = (("偏多", "bull") if pc >= UP_Q else ("偏空", "bear") if pc <= DN_Q
                                     else ("中性", "neutral"))
    # 標籤是 t+1 開盤到 t+1+h 開盤；路徑從 asof 收盤起算，第 1 天先放波動、不放報酬
    path = _cone(h + 1, np.log1p(pred), sig)
    res["path"] = path
    res["pred"], res["pctile"], res["rank"], res["n_day"], res["pred_date"] = pred, pc, rank, n_day, d
    res["facts"].append(f"{STRATEGIES[key]['name']}（{d} 收盤後）：預測之後 {h} 個交易日報酬 {pred:+.2%}，"
                        f"當天 {n_day} 檔中排第 {rank}（前 {1 - pc:.0%}），判定{res['direction']}。")
    if stale > 0:
        res["warn"].append(f"模型預測只到 {d}，比視角日期晚了 {stale} 個交易日沒有更新"
                           "（要重跑特徵庫與 nextday_horizon.py --tag _surv）。")
    # 這檔自己的歷史：被判偏多／偏空之後，實際 h 日報酬（逐日自動下單時不算，太慢）
    hp = pd.DataFrame(columns=["pred", "pctile"]) if not calib else pd.read_sql("SELECT date, pred, pctile FROM predictions WHERE code = ? AND h = ? AND date <= ?",
                     con, params=[code, h, asof], parse_dates=["date"]).set_index("date")
    o = hist["open"]
    lab = (o.shift(-(1 + h)) / o.shift(-1) - 1).reindex(hp.index)
    ok = lab.notna()
    if calib and ok.sum() >= 30:
        hp, lab = hp[ok], lab[ok]
        up, dn = hp["pctile"] >= UP_Q, hp["pctile"] <= DN_Q
        res["calib"] = dict(
            n=int(ok.sum()), base=float(lab.mean()), base_win=float((lab > 0).mean()),
            up_n=int(up.sum()), up_mean=float(lab[up].mean()) if up.any() else np.nan,
            up_win=float((lab[up] > 0).mean()) if up.any() else np.nan,
            dn_n=int(dn.sum()), dn_mean=float(lab[dn].mean()) if dn.any() else np.nan,
            dn_win=float((lab[dn] < 0).mean()) if dn.any() else np.nan,
            ic=float(hp["pred"].corr(lab, method="spearman")))
        c = res["calib"]
        res["facts"].append(
            f"這檔過去 {c['n']} 天的紀錄：判偏多 {c['up_n']} 次，之後 {h} 日平均 {c['up_mean']:+.2%}、"
            f"上漲比例 {c['up_win']:.0%}；判偏空 {c['dn_n']} 次，平均 {c['dn_mean']:+.2%}；"
            f"不分訊號平均 {c['base']:+.2%}、上漲比例 {c['base_win']:.0%}。")
    if res["direction"] == "偏多":
        _plan(res, True, h + 1, f"模型的標籤就是「明天開盤買、{h} 天後開盤賣」，出場日照這個。")
    elif res["direction"] == "偏空":
        _plan(res, False, 0, "模型偏空：不建議新買；已持有的話，下一個交易日開盤出場。")
    return res


# ---------------------------------------------------------------- 歷史相似走勢

def _windows(lp: np.ndarray, L: int, H: int, stride: int = 1):
    """每段 = 錨點 + L 天比對 + H 天之後；回傳比對段 x、之後 f（都相對錨點／比對段最後一天）與段尾索引。"""
    if len(lp) < L + H + 1:
        return None
    W = sliding_window_view(lp, L + H + 1)[::stride]
    ok = np.isfinite(W).all(1) & (np.abs(np.diff(W, axis=1)).max(1) <= MAX_JUMP)
    W = W[ok]
    end = (np.arange(len(lp) - L - H)[::stride])[ok] + L
    return W[:, 1:L + 1] - W[:, [0]], W[:, L + 1:] - W[:, [L]], end


def _greedy(order: np.ndarray, end: np.ndarray, k: int, gap: int) -> list:
    """依距離由近到遠挑，同一檔的兩段至少隔 gap 天，避免挑到同一段走勢的重複版本。"""
    out = []
    for i in order:
        if all(abs(end[i] - end[j]) >= gap for j in out):
            out.append(i)
            if len(out) == k:
                break
    return out


def analog_search(closes: pd.DataFrame, code: str, asof: pd.Timestamp, L: int = 20, H: int = 20,
                  k: int = 10, pool: str = "self") -> dict | None:
    """closes：日期 × 代號的收盤。pool="self" 只找這檔自己，"market" 找全市場。"""
    s = closes[code].loc[:asof].dropna()
    if len(s) < L + 1:
        return None
    lp = np.log(s.to_numpy())
    q = lp[-L:] - lp[-L - 1]
    cand = []
    for c in ([code] if pool == "self" else closes.columns):
        ser = s if c == code else closes[c].loc[:asof].dropna()
        w = _windows(np.log(ser.to_numpy()), L, H, 1 if pool == "self" else 2)
        if w is None or not len(w[0]):
            continue
        x, f, end = w
        d = np.sqrt(((x - q) ** 2).sum(1))
        pick = _greedy(np.argsort(d)[:200], end, k if pool == "self" else 3, L)
        idx = ser.index
        cand += [dict(code=c, start=idx[end[i] - L + 1], end=idx[end[i]], dist=float(d[i]),
                      fwd=float(np.expm1(f[i, -1])), x=x[i], f=f[i]) for i in pick]
    if not cand:
        return None
    m = pd.DataFrame(cand).sort_values("dist").head(k).reset_index(drop=True)
    F = np.column_stack([np.zeros(len(m)), np.vstack(m["f"])])
    return dict(matches=m, query=q, F=F, L=L, H=H, pool=pool)


def analog_selftest(close: pd.Series, L: int = 20, H: int = 20, k: int = 10, step: int = 10,
                    min_hist: int = 500) -> dict | None:
    """這檔自己的 walk-forward：每 step 天用當時已揭曉的片段預測之後 H 天，和實際比對。"""
    lp = np.log(close.dropna().to_numpy())
    w = _windows(lp, L, H)
    if w is None:
        return None
    x, f, end = w
    preds, reals = [], []
    for e0 in range(max(min_hist, L + H + 1), len(lp) - H, step):
        known = end + H <= e0
        if known.sum() < k * 5:
            continue
        q = lp[e0 - L + 1:e0 + 1] - lp[e0 - L]
        xi, fi, ei = x[known], f[known], end[known]
        d = np.sqrt(((xi - q) ** 2).sum(1))
        pick = _greedy(np.argsort(d)[:200], ei, k, L)
        preds.append(np.median(fi[pick, -1]))
        reals.append(lp[e0 + H] - lp[e0])
    if len(preds) < 20:
        return None
    p, r = np.array(preds), np.array(reals)
    up, dn = p > 0.01, p < -0.01
    return dict(n=len(p), hit=float((np.sign(p) == np.sign(r)).mean()), base_win=float((r > 0).mean()),
                corr=float(pd.Series(p).corr(pd.Series(r), method="spearman")),
                up_n=int(up.sum()), up_mean=float(np.expm1(r[up]).mean()) if up.any() else np.nan,
                dn_n=int(dn.sum()), dn_mean=float(np.expm1(r[dn]).mean()) if dn.any() else np.nan,
                base=float(np.expm1(r).mean()))


def _analog(key, code, hist, full_index, closes, L=20, H=20, k=10, pool="self", selftest=True) -> dict:
    res = _base(key, code, hist, full_index, H)
    if closes is None or code not in closes:
        closes = pd.DataFrame({code: hist["close"]})
    a = analog_search(closes, code, res["asof"], L, H, k, pool)
    if a is None:
        res.update(direction="資料不足", tone="muted", path=_cone(H, 0.0, _sigma(hist["close"])))
        res["warn"].append(f"歷史不夠長，找不到可比對的 {L}+{H} 天片段。")
        return res
    F = a["F"]
    qs = np.nanquantile(F, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
    res["path"] = pd.DataFrame({"lo80": qs[0], "lo50": qs[1], "mid": qs[2], "hi50": qs[3], "hi80": qs[4]})
    res["analog"] = a
    med, win = float(np.expm1(qs[2, -1])), float((F[:, -1] > 0).mean())
    res["direction"], res["tone"] = (("偏多", "bull") if med > 0.01 else ("偏空", "bear") if med < -0.01
                                     else ("中性", "neutral"))
    where = "這檔自己的歷史" if pool == "self" else "全市場"
    res["facts"].append(f"在{where}找到最像近 {L} 天走勢的 {len(F)} 段，之後 {H} 天的中位數 {med:+.2%}、"
                        f"上漲比例 {win:.0%}，判定{res['direction']}。")
    if selftest:
        t = analog_selftest(hist["close"], L, H, k)
        res["calib"] = t
        if t:
            res["facts"].append(
                f"同樣方法在這檔自己的歷史回測 {t['n']} 次：方向猜對 {t['hit']:.0%}"
                f"（同期上漲比例 {t['base_win']:.0%}），排序相關 {t['corr']:+.2f}；"
                f"判偏多時之後平均 {t['up_mean']:+.2%}、偏空時 {t['dn_mean']:+.2%}、不分訊號 {t['base']:+.2%}。")
    if res["direction"] == "偏多":
        _plan(res, True, H, f"固定持有到第 {H} 個交易日開盤出場；改在中位數最高那天出場測過，平均更差。")
    elif res["direction"] == "偏空":
        _plan(res, False, 0, "相似片段之後多半下跌：不建議新買；已持有的話，下一個交易日開盤出場。")
    return res


# ---------------------------------------------------------------- 跌深反彈

def _dip(con, key, code, hist, full_index) -> dict:
    asof = hist.index[-1].strftime("%Y-%m-%d")
    ev = pd.read_sql("SELECT code, signal, entry, exit, hold, net FROM events WHERE exit < ?", con, params=[asof])
    H = int(round(ev["hold"].median())) if len(ev) else 15
    res = _base(key, code, hist, full_index, H + 1)
    c = hist["close"]
    r5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 5 else np.nan
    hit = r5 <= DIP_DROP
    if len(ev) >= 50:
        lr = np.log1p(ev["net"].clip(lower=-0.95))
        qs = lr.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy()
        t = np.arange(H + 2)
        sc = np.sqrt(np.clip(t - 1, 0, None) / H)          # t = 1 才進場
        mid = lr.mean() * np.clip(t - 1, 0, None) / H
        res["path"] = pd.DataFrame({"mid": mid, "lo80": mid + (qs[0] - lr.mean()) * sc,
                                    "lo50": mid + (qs[1] - lr.mean()) * sc, "hi50": mid + (qs[3] - lr.mean()) * sc,
                                    "hi80": mid + (qs[4] - lr.mean()) * sc})
        wait = (pd.to_datetime(ev["entry"]) - pd.to_datetime(ev["signal"])).dt.days.median()
        own = ev[ev["code"] == code]
        res["facts"].append(
            f"視角日期以前已出場的跌深事件 {len(ev):,} 筆：平均淨報酬 {ev['net'].mean():+.2%}、"
            f"賺錢比例 {(ev['net'] > 0).mean():.0%}、持有中位數 {H} 天、訊號到進場中位數 {wait:.0f} 個日曆天"
            + (f"；這檔自己 {len(own)} 筆、平均 {own['net'].mean():+.2%}。" if len(own) else "；這檔自己沒有紀錄。"))
    else:
        res["path"] = _cone(H + 1, 0.0, _sigma(c))
        res["warn"].append("視角日期以前揭曉的事件太少，區間改用波動度估計。")
    res["facts"].insert(0, f"{asof} 近 5 日 {r5:+.1%}，跌深條件（5 日跌 8%）{'已' if hit else '未'}達標。")
    rd = con.execute("SELECT entry, score, threshold, excluded FROM rag_decisions WHERE code = ? AND entry <= ? "
                     "ORDER BY entry DESC LIMIT 1", [code, asof]).fetchone()
    recent = rd and (pd.Timestamp(asof) - pd.Timestamp(rd[0])).days <= 20
    if recent:
        res["facts"].append(f"最近一次事件 {rd[0]} 進場：RAG 規則分數 {rd[1]:+.2%}、"
                            f"門檻 {'無' if rd[2] is None else f'{rd[2]:+.2%}'}，規則判定{'排除' if rd[3] else '保留'}。")
    if recent and rd[3]:
        res["direction"], res["tone"] = "規則排除", "muted"
        _plan(res, False, 0, "RAG 候選規則判定排除：歷史上最像的 20 個事件平均不好，這次不進場。")
    elif hit or recent:
        res["direction"], res["tone"] = "規則進場", "bull"
        _plan(res, True, H + 1, "實際規則要等止跌才進場；這裡用明天開盤近似，出場照事件持有中位數。")
    else:
        res["direction"], res["tone"] = "無訊號", "muted"
    return res


# ---------------------------------------------------------------- 高波動＋均線

def _highvol(con, key, code, hist, full_index, closes, volumes) -> dict:
    res = _base(key, code, hist, full_index, HV_HOLD + 1)
    res["path"] = _cone(HV_HOLD + 1, 0.0, _sigma(hist["close"]))
    asof = res["asof"]
    r = con.execute("SELECT date, in_market FROM strategy_daily WHERE date <= ? ORDER BY date DESC LIMIT 1",
                    [asof.strftime("%Y-%m-%d")]).fetchone()
    on = bool(r and r[1])
    res["facts"].append(f"大盤趨勢過濾（{r[0] if r else '—'}）：{'在 120 日均線之上，策略持有中' if on else '跌破 120 日均線，策略空手'}。")
    rank = None
    if closes is not None and volumes is not None and code in closes:
        # 和 strategy_spec.py 一樣：換倉日用前一日為止的資料、簡單報酬的 60 日標準差
        C, V = closes.loc[:asof].iloc[-62:-1], volumes.loc[:asof].iloc[-62:-1]
        tv = (C * V).iloc[-60:].mean()
        pool = tv[C.iloc[-60:].notna().sum() >= 50].nlargest(HV_POOL).index
        vol = C[pool].pct_change().iloc[-60:].std().sort_values(ascending=False)
        if code in vol.index:
            rank = int(vol.index.get_loc(code)) + 1
            res["facts"].append(f"若 {asof:%Y-%m-%d} 換倉：在成交金額前 {HV_POOL} 名裡，60 日波動排第 {rank}"
                                f"（買前 {HV_N}、持股掉出前 {HV_BUF} 才賣）。")
        else:
            res["facts"].append(f"不在成交金額前 {HV_POOL} 名的選股範圍。")
    res["rank"] = rank
    if on and rank and rank <= HV_N:
        res["direction"], res["tone"] = "策略會買", "bull"
        _plan(res, True, HV_HOLD + 1, f"抱到下一次換倉（約 {HV_HOLD} 個交易日）；大盤跌破 120 日均線就全部出場。"
                                      "區間只用波動度估，中線放 0：這個策略不預測單一股票的漲跌。")
    elif on and rank and rank <= HV_BUF:
        res["direction"], res["tone"] = "緩衝帶", "neutral"
        res["facts"].append("在緩衝帶：已持有可以續抱，但不會新買。")
    else:
        res["direction"], res["tone"] = ("不在清單", "muted") if on else ("策略空手", "bear")
        if not on:
            _plan(res, False, 0, "大盤跌破 120 日均線：策略全部出場、空手等待。")
    return res


# ---------------------------------------------------------------- 入口

def forecast(con, code: str, asof: str | pd.Timestamp, strategy: str = "h5", px: pd.DataFrame | None = None,
             closes: pd.DataFrame | None = None, volumes: pd.DataFrame | None = None, calib: bool = True,
             **opt) -> dict:
    """回傳 dict：direction、tone、path（t × mid/lo50/hi50/lo80/hi80 的 log 報酬）、dates、plan、facts、
    warn、calib、evidence；相似走勢另有 analog。"""
    px = load_px(con, code) if px is None else px
    hist = px.loc[:pd.Timestamp(asof)]
    if len(hist) < 30:
        raise ValueError(f"{code} 在 {asof} 以前的價格不到 30 天")
    kind = STRATEGIES[strategy]["kind"]
    if kind == "model":
        return _model(con, strategy, code, hist, px.index, calib)
    if kind == "analog":
        return _analog(strategy, code, hist, px.index, closes, **({"selftest": False} if not calib else {}) | opt)
    if kind == "dip":
        return _dip(con, strategy, code, hist, px.index)
    return _highvol(con, strategy, code, hist, px.index, closes, volumes)


def summary_lines(con, code: str, asof: str) -> list:
    """給 local_ask.py：每種預測一行可引用的文字（高波動與全市場相似走勢太慢，不放）。"""
    px = load_px(con, code)
    if len(px.loc[:pd.Timestamp(asof)]) < 30:
        return []
    out = []
    for key in ("h5", "h20", "dip", "analog"):
        try:
            r = forecast(con, code, asof, key, px=px, **({"selftest": True} if key == "analog" else {}))
        except Exception:
            continue
        line = "".join(r["facts"])
        if r["plan"] and r["plan"]["enter"]:
            p = r["plan"]
            line += (f"建議 {p['entry_date']:%Y-%m-%d} 開盤進場、{p['exit_date']:%Y-%m-%d} 開盤出場"
                     f"（固定持有；停損參考 {p['stop']:.2f}、目標參考 {p['target']:.2f}，"
                     "但實測照這兩條線出場比固定持有差）。")
        out.append(f"【{r['name']}】{line}研究結論：{r['evidence']}")
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    code, asof = (sys.argv[1:3] + ["2330", "2026-10-30"])[:2] if len(sys.argv) < 3 else sys.argv[1:3]
    con = connect()
    for line in summary_lines(con, code, asof):
        print(line, "\n")
