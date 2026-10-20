#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
近十年台股大波動日 × 當天／前一天新聞。

問的是兩件事
------------
1. 大波動日（t）「當日盤前」的新聞，和一般日子比起來有沒有不一樣（量、情緒、主題）？
2. 「前一天」（t-1）的新聞，能不能事前預示 t 會出現大波動？

大波動日的定義
--------------
大盤 TWII 收盤對收盤報酬 r_t，除以前 250 個交易日報酬的標準差（只用 t-1 以前，
不含當天）得到 z_t。|z_t| > 2 叫大波動日，|z_t| > 3 叫極端日，|z_t| < 1 是對照組
（一般日）。用滾動標準化而不是固定門檻，是因為 2026 年指數水位和波動都比
2017 年高很多，固定 3% 會把大波動日全部擠到最近兩年。另外也輸出 |r| 前 50 名
當作「絕對幅度」的對照定義。

時間對齊（避免前視）
--------------------
所有文本一律換成台北時間，沿用 davis_tw_train 的規則：
    13:30 以前發布 → 歸當天　／　13:30 以後（或只有日期、時分為 00:00）→ 下一個交易日
兩個時間窗（前後相接、不重疊）：
    當日　= t-1 13:30 之後 ~ t 09:00 之前（t 的盤前，開盤前已知）
    前一天 = t-2 13:30 之後 ~ t-1 13:30（t-1 的盤前 + 盤中，t-1 收盤時已全部公開）
t 日 09:00 以後（盤中、盤後）發布的新聞一律不算進「當日」，避免拿事後報導解釋波動。
只有「前一天」是在 r_t 的起點（t-1 收盤）之前就已知的，所以預測檢定只用前一天；
「當日」盤前新聞開盤前已知，但收盤對收盤的 r_t 在那時已經起跑，只能算描述／對照。

新聞來源先剔除機器產生的制式訊息（【公告】、Factset 速報、排行榜），見 BOILER_PAT。

新聞量的「異常程度」用嚴格往回看的基準：log(1+則數) 減去前 60 個「同類型」
交易日（一般日 vs 週一／長假後）的中位數。同一個月後面的日子不會被拿來當基準。

PTT 的推／噓數、留言數會在發文後持續累積（可能晚好幾天），屬於事後資訊，
所以完全不使用，PTT 只看發文數和標題。

資料來源與覆蓋
--------------
    intl   鉅亨國際新聞（外匯／美股／國際總經／國際股市）  2014-01 ~ 2026-10
    cnyes  鉅亨個股新聞（data/stocks/*/news，跨股去重）      2019-01 ~ 2026-10
           （2025 之後來自 news_backfill_2025.py 的補抓檔 cnyes_news_*_2025fill.csv）
    yahoo  Yahoo 股市個股新聞（同上，跨股去重）              2025-10 ~ 2026-09
    ptt    PTT Stock 板全板（NEW_pttcc(19-24)/pttcc_clean） 2019-03 ~ 2024-12
2016-10 ~ 2018-12 只有國際新聞；2025 之後沒有 PTT。

輸出：results/volatility_news/
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_reversal import NEGATIVE_WORDS  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "volatility_news"
PTT_DIR = ROOT / "NEW_pttcc(19-24)" / "pttcc_clean"

START, END = "2016-10-01", "2026-10-13"
SIGMA_WIN = 250
BIG_Z, EXTREME_Z, NORMAL_Z = 2.0, 3.0, 1.0
CLOSE_H, CLOSE_M, OPEN_H = 13, 30, 9
MIN_DOCS = 3                      # 情緒比例至少要有幾則才算

SOURCES = ["intl", "cnyes", "yahoo", "ptt"]
SRC_NAME = {"intl": "國際新聞", "cnyes": "鉅亨個股新聞", "yahoo": "Yahoo個股新聞", "ptt": "PTT股板"}

# 在講台股大盤本身漲跌的「盤勢報導」：t 日盤中這類新聞是波動的結果，不是原因
RECAP_PAT = re.compile(r"(?:台股|大盤|加權指數|台指期?|集中市場|櫃買|上市櫃|台北股市)"
                       r".*(?:盤|收|點|漲|跌|挫|揚|翻|殺|攻)|"
                       r"(?:盤前|盤中|盤後|收盤|開盤)(?:要聞|速報|─|-|:|：)")

# 機器產生的制式訊息：Yahoo 約 7 成是【公告】（董事會決議、月營收）、2024 年國際新聞
# 約 13% 是「鉅亨速報 - Factset」目標價快訊；2024 年起還有「盤中速報 - XX(美股)大跌5%」這種
# 自動價格快訊（2024 約 3 千則、2026 約 6.5 千則，之前完全沒有），鉅亨個股另有「限量免費索取」廣告。
# 這些不是新聞事件，而且只出現在某些年份，留著會讓則數失真
# （例如 3 月股利公告季被當成「新聞爆量」），所以新聞來源一律先剔除；PTT 不套用。
BOILER_PAT = re.compile(r"速報|免費索取|Factset|FactSet|【公告】|排行|前\s?\d+\s?名|董事會決議|代子公司|"
                        r"代重要子公司|公告本公司|股東常會|零股交易|成交股數|個股成交|自結")

# 方向性詞表：專案原有的負面詞表 NEGATIVE_WORDS（個股事件為主）之外，
# 補上大盤層級常見的漲跌描述。標題命中任一即算該方向。
MKT_NEG = ["跌", "挫", "崩", "殺", "走低", "收黑", "翻黑", "拋售", "恐慌", "避險",
           "擔憂", "憂心", "疑慮", "衰退", "降息預期降溫", "鷹派", "緊縮", "危機", "倒閉", "制裁"]
MKT_POS = ["漲", "揚", "飆", "創高", "新高", "反彈", "走高", "收紅", "翻紅", "樂觀", "看好",
           "強勁", "優於預期", "回升", "利多", "寬鬆", "鴿派", "降息", "紓困", "刺激"]
NEG_PAT = re.compile("|".join(map(re.escape, sorted(set(MKT_NEG + NEGATIVE_WORDS), key=len, reverse=True))))
POS_PAT = re.compile("|".join(map(re.escape, sorted(MKT_POS, key=len, reverse=True))))
PROJ_NEG_PAT = re.compile("|".join(map(re.escape, NEGATIVE_WORDS)))

# 主題詞典：用來判斷大波動日有沒有「明確的新聞事件」，以及哪類主題在大波動日特別多
THEMES = {
    "疫情": r"疫情|肺炎|新冠|COVID|確診|封城|Omicron|Delta變種|病毒|三級警戒|清零",
    "Fed利率": r"Fed|聯準會|升息|降息|鮑爾|FOMC|縮表|利率決策|利率會議",
    "關稅貿易": r"關稅|貿易戰|貿易協議|貿易談判|對等|301|實體清單|出口管制|禁令|華為",
    "地緣政治": r"台海|共軍|軍演|解放軍|俄烏|烏克蘭|俄羅斯|普丁|戰爭|開戰|以色列|伊朗|哈瑪斯|中東|飛彈|北韓|地緣|空襲|攻擊",
    "經濟數據": r"CPI|通膨|PCE|非農|就業|失業率|PMI|GDP|零售銷售|物價",
    "半導體AI": r"台積電|半導體|輝達|NVIDIA|Nvidia|(?<![A-Za-z])AI(?![A-Za-z])|晶片|記憶體|費半|蘋果|DeepSeek",
    "匯率日圓": r"日圓|日銀|日本央行|匯率|新台幣|台幣|美元指數|套利交易|人民幣|貶值|升值",
    "債市殖利率": r"殖利率|公債|美債|債市|債券",
    "金融風險": r"銀行業|銀行危機|倒閉|恆大|違約|SVB|矽谷銀行|瑞信|擠兌|破產|流動性",
    "油價原物料": r"油價|原油|OPEC|布蘭特|西德州|黃金|金價|銅價",
    "政治選舉": r"選舉|大選|公投|川普|拜登|賀錦麗|關門|彈劾|罷免|國會",
    "台股政策": r"國安基金|證所稅|金管會|印花稅|當沖|融資|斷頭|期交所|證交所",
    "天災意外": r"地震|颱風|停電|乾旱|缺水|大火|爆炸",
}
THEME_PATS = {k: re.compile(v, re.IGNORECASE) for k, v in THEMES.items()}

STOPWORDS = set("""的 了 在 是 與 及 和 或 為 將 對 等 也 都 就 而 但 又 於 被 讓 從 向 到 上 下 中 後 前 今 日 年 月 週
 再 仍 已 即 逾 近 約 達 個 這 那 其 之 以 由 至 並 及其 今日 今天 昨日 明日 本週 下週 上週 美元 億元 萬元 新台幣元
 台股 美股 盤前 盤中 盤後 收盤 開盤 指數 股市 市場 點 漲 跌 新聞 快訊 速報 要聞 早盤 午盤 尾盤 一覽 消息
 Re Fw 閒聊 新聞 標的 請益 心得 情報 公告 問題 討論 分享""".split())


# ------------------------------------------------------------ 價格與大波動日

def _read_px(name: str) -> pd.DataFrame:
    px = pd.read_csv(DATA / "market" / f"{name}.csv", encoding="utf-8-sig")
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    flat = (px["volume"] <= 0) & (px["open"] == px["close"]) & (px["high"] == px["low"])
    return px.loc[~flat].set_index("date")


def market() -> pd.DataFrame:
    px = _read_px("TWII")
    m = pd.DataFrame(index=px.index)
    m["close"] = px["close"]
    m["r"] = px["close"].pct_change()
    m["gap"] = px["open"] / px["close"].shift(1) - 1
    m["intraday"] = px["close"] / px["open"] - 1
    m["sigma"] = m["r"].rolling(SIGMA_WIN, min_periods=120).std().shift(1)
    m["z"] = m["r"] / m["sigma"]
    m["absz"] = m["z"].abs()
    m["absz_lag1"] = m["absz"].shift(1)
    m["absz_lag5max"] = m["absz"].shift(1).rolling(5).max()
    m["absz_ma20"] = m["absz"].shift(1).rolling(20).mean()
    m["rv20_ratio"] = m["r"].shift(1).rolling(20).std() / m["sigma"]     # 近月波動 / 近一年波動
    cal = m.index.to_series()
    m["gap_days"] = cal.diff().dt.days
    m["after_break"] = (m["gap_days"] > 1).astype(int)          # 週一、長假後

    # 美股前一晚（台北 t 日清晨收盤的那一場）：外部資訊，盤前已知
    for nm in ("SOX", "GSPC"):
        us = _read_px(nm)[["close"]].copy()
        us["ret"] = us["close"].pct_change()
        us["sig"] = us["ret"].rolling(SIGMA_WIN, min_periods=120).std().shift(1)
        us["uz"] = us["ret"] / us["sig"]
        us = us.reset_index()[["date", "ret", "uz"]].rename(columns={"date": "us_date"})
        left = pd.DataFrame({"date": m.index})
        j = pd.merge_asof(left, us, left_on="date", right_on="us_date",
                          allow_exact_matches=False, direction="backward")
        # 取日期嚴格早於 t 的最近一場美股（台北時間 t 日清晨收盤）
        m[f"{nm.lower()}_ret"] = j["ret"].to_numpy()
        m[f"{nm.lower()}_z"] = j["uz"].to_numpy()
    m["sox_z_abs"] = m["sox_z"].abs()

    m = m[(m.index >= pd.Timestamp(START)) & (m.index <= pd.Timestamp(END))].copy()
    m["big"] = m["absz"] > BIG_Z
    m["extreme"] = m["absz"] > EXTREME_Z
    m["normal"] = m["absz"] < NORMAL_Z
    m["big_next"] = m["big"].shift(-1)
    m["r_next"] = m["r"].shift(-1)
    m["top50_abs"] = m["r"].abs().rank(ascending=False) <= 50
    return m


# ------------------------------------------------------------ 文本

def _to_taipei_naive(s: pd.Series, naive_is_taipei: bool) -> pd.Series:
    if naive_is_taipei:
        ts = pd.to_datetime(s, errors="coerce", format="mixed")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
        return ts
    ts = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    return ts.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)


def load_intl() -> pd.DataFrame:
    # intl_news_crawler 用 datetime.fromtimestamp 寫檔 → 無時區字串、本機時間＝台北時間。
    # 不能丟給 _read_text_csv（它把無時區當 UTC，會整批晚 8 小時）。
    parts = []
    for f in sorted((DATA / "intl_news").glob("*_20*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", usecols=["date", "title", "category", "news_id"])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = _to_taipei_naive(d["date"], naive_is_taipei=True)
    d = d.dropna(subset=["ts"]).drop_duplicates("news_id")
    return pd.DataFrame({"ts": d["ts"], "title": d["title"].fillna("").astype(str),
                         "src": "intl", "sub": d["category"]})


def load_stock_news() -> pd.DataFrame:
    # 同一則鉅亨新聞會被掛在好幾檔股票底下，跨股以 (來源, 標題) 去重，變成全市場新聞池
    parts = []
    for f in (DATA / "stocks").glob("*/news/*.csv"):
        src = "cnyes" if f.name.startswith("cnyes") else "yahoo"
        try:
            d = pd.read_csv(f, encoding="utf-8-sig", usecols=lambda c: c in ("date", "title"))
        except Exception:
            continue
        if "date" not in d or "title" not in d:
            continue
        d["src"] = src
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = _to_taipei_naive(d["date"], naive_is_taipei=False)
    d["title"] = d["title"].fillna("").astype(str).str.strip()
    d = d.dropna(subset=["ts"])
    d = d[d["title"] != ""].sort_values("ts").drop_duplicates(["src", "title"])
    return pd.DataFrame({"ts": d["ts"], "title": d["title"], "src": d["src"], "sub": ""})


def load_ptt_board() -> pd.DataFrame:
    """全板文章（不做個股歸屬），只留時間、標題、推噓數。逐行讀 2.7GB JSONL，結果快取。"""
    cache = OUT / "_cache_ptt_board.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    rows = []
    for f in sorted(PTT_DIR.glob("stock_*_clean.jsonl")):
        print(f"  讀 {f.name} …", flush=True)
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows.append((r.get("date") or r.get("date_ts"), r.get("title") or "",
                             r.get("n_push", 0), r.get("n_boo", 0), r.get("n_comments", 0)))
    d = pd.DataFrame(rows, columns=["date", "title", "n_push", "n_boo", "n_comments"])
    d["ts"] = _to_taipei_naive(d["date"], naive_is_taipei=True)   # PTT 原始時間就是台北時間
    for c in ("n_push", "n_boo", "n_comments"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d = d.dropna(subset=["ts"])
    out = pd.DataFrame({"ts": d["ts"], "title": d["title"].astype(str).str.strip(), "src": "ptt",
                        "sub": "", "n_push": d["n_push"], "n_boo": d["n_boo"],
                        "n_comments": d["n_comments"]})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)
    return out


def assign_day(docs: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """13:30 後（或時分全零）順延；再拆成盤前 / 盤中。"""
    ts = docs["ts"]
    after = (ts.dt.hour > CLOSE_H) | ((ts.dt.hour == CLOSE_H) & (ts.dt.minute >= CLOSE_M))
    midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    eff = ts.dt.normalize() + pd.to_timedelta((after | midnight).astype(int), unit="D")
    idx = cal.searchsorted(eff.to_numpy(), side="left")
    ok = idx < len(cal)
    docs = docs.loc[ok].copy()
    docs["day"] = cal[idx[ok]]
    # 發布當下 < 歸屬日 09:00 → 開盤前已知
    docs["pre_open"] = docs["ts"] < docs["day"] + pd.Timedelta(hours=OPEN_H)
    docs["recap"] = docs["title"].str.contains(RECAP_PAT, na=False)
    docs["neg"] = docs["title"].str.contains(NEG_PAT, na=False)
    docs["pos"] = docs["title"].str.contains(POS_PAT, na=False)
    docs["neg_proj"] = docs["title"].str.contains(PROJ_NEG_PAT, na=False)
    for k, p in THEME_PATS.items():
        docs[f"th_{k}"] = docs["title"].str.contains(p, na=False)
    return docs


def coverage_months(docs: pd.DataFrame) -> pd.Series:
    """每個來源每月則數；低於該來源月中位數 25% 的月份視為沒覆蓋。"""
    mc = docs.groupby(docs["day"].dt.to_period("M")).size()
    return mc[mc >= 0.25 * mc.median()]


# ------------------------------------------------------------ 每日特徵

def _backward_abn(logn: pd.Series, after_break: pd.Series) -> pd.Series:
    """log(1+n) 減去前 60 個同類型交易日的中位數（一般日／週一長假後分開），嚴格往回看。"""
    out = pd.Series(np.nan, index=logn.index)
    for flag, win in ((0, 60), (1, 20)):
        sub = logn[after_break == flag]
        base = sub.rolling(win, min_periods=max(10, win // 3)).median().shift(1)
        out.loc[sub.index] = sub - base
    return out


def daily_features(docs: pd.DataFrame, m: pd.DataFrame, src: str, covered: pd.Index) -> pd.DataFrame:
    days = m.index[m.index.to_period("M").isin(covered)]
    g = docs.groupby("day")
    f = pd.DataFrame(index=days)
    f["n"] = g.size().reindex(days).fillna(0)
    f["n_pre"] = docs[docs["pre_open"]].groupby("day").size().reindex(days).fillna(0)
    f["n_intra"] = f["n"] - f["n_pre"]
    nr = docs[~docs["recap"]]
    f["n_nonrecap"] = nr.groupby("day").size().reindex(days).fillna(0)
    f["n_recap"] = f["n"] - f["n_nonrecap"]
    f["n_pre_nonrecap"] = nr[nr["pre_open"]].groupby("day").size().reindex(days).fillna(0)

    ab = m.loc[days, "after_break"]
    for c in ("n", "n_pre", "n_nonrecap", "n_pre_nonrecap"):
        f[f"abn_{c}"] = _backward_abn(np.log1p(f[c]), ab)

    def share(col, base_df=docs, nmin=MIN_DOCS):
        cnt = base_df.groupby("day")[col].sum().reindex(days)
        tot = base_df.groupby("day").size().reindex(days)
        return (cnt / tot).where(tot >= nmin)

    f["neg_share"] = share("neg")
    f["pos_share"] = share("pos")
    f["tone"] = f["pos_share"] - f["neg_share"]
    f["neg_proj_share"] = share("neg_proj")
    f["tone_nonrecap"] = share("pos", nr) - share("neg", nr)
    f["recap_share"] = 1 - f["n_nonrecap"] / f["n"].replace(0, np.nan)
    # 情緒偏離：減去前 60 日平均，嚴格往回看
    f["tone_dev"] = f["tone"] - f["tone"].rolling(60, min_periods=20).mean().shift(1)
    f["neg_dev"] = f["neg_share"] - f["neg_share"].rolling(60, min_periods=20).mean().shift(1)
    for k in THEMES:
        f[f"th_{k}"] = share(f"th_{k}", nr, 1)
        # 則數（非盤勢），給「明確事件」旗標用
        f[f"thc_{k}"] = nr.groupby("day")[f"th_{k}"].sum().reindex(days).fillna(0)
    f.columns = [f"{src}_{c}" for c in f.columns]
    return f


# ------------------------------------------------------------ 統計工具

def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """OLS + 以群集（月份）為單位的穩健標準誤。"""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = pd.DataFrame(X * u[:, None]).groupby(groups).sum().to_numpy()
    meat = ug.T @ ug
    G, N, K = len(np.unique(groups)), len(y), X.shape[1]
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((N - 1) / max(N - K, 1))
    return b, np.sqrt(np.diag(V)), G


def compare(panel: pd.DataFrame, col: str, a_mask: pd.Series, b_mask: pd.Series) -> dict | None:
    """a 組 vs b 組：平均差、月份群集 t、Mann-Whitney p。"""
    x = panel[col]
    sel = (a_mask | b_mask) & x.notna()
    if sel.sum() < 30 or (a_mask & sel).sum() < 5:
        return None
    y = x[sel].to_numpy(float)
    d = a_mask[sel].astype(float).to_numpy()
    X = np.column_stack([np.ones_like(d), d])
    grp = panel.index[sel].to_period("M").astype(str).to_numpy()
    b, se, G = ols_cluster(y, X, grp)
    t = b[1] / se[1] if se[1] > 0 else np.nan
    xa, xb = y[d == 1], y[d == 0]
    try:
        mw = stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
    except ValueError:
        mw = np.nan
    return {"n_a": len(xa), "n_b": len(xb), "mean_a": xa.mean(), "mean_b": xb.mean(),
            "diff": b[1], "t_cluster": t, "p_cluster": 2 * stats.t.sf(abs(t), G - 1) if np.isfinite(t) else np.nan,
            "p_mannwhitney": mw, "months": G}


def logit_fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    mdl = LogisticRegression(C=np.inf, max_iter=2000)
    mdl.fit(X, y)
    p = np.clip(mdl.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return mdl, ll


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def predict_test(panel: pd.DataFrame, target: str, base_cols: list, news_cols: list, label: str) -> dict | None:
    """
    target ~ 基準（落後波動）vs 基準 + 新聞。
    in-sample 概似比檢定 + 逐年留一（leave-one-year-out）樣本外 AUC。
    """
    cols = base_cols + news_cols
    d = panel[[target] + cols].dropna()
    if len(d) < 200 or d[target].sum() < 10:
        return None
    y = d[target].astype(int).to_numpy()
    mu, sd = d[cols].mean(), d[cols].std().replace(0, 1)
    Z = ((d[cols] - mu) / sd).to_numpy()
    nb = len(base_cols)
    m0, ll0 = logit_fit(Z[:, :nb], y)
    m1, ll1 = logit_fit(Z, y)
    lr = 2 * (ll1 - ll0)
    p_lr = stats.chi2.sf(lr, len(news_cols))
    yrs = d.index.year.to_numpy()
    s0, s1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for yr in np.unique(yrs):
        tr, te = yrs != yr, yrs == yr
        if y[tr].sum() < 5:
            continue
        a, _ = logit_fit(Z[tr][:, :nb], y[tr])
        b, _ = logit_fit(Z[tr], y[tr])
        s0[te] = a.predict_proba(Z[te][:, :nb])[:, 1]
        s1[te] = b.predict_proba(Z[te])[:, 1]
    ok = ~np.isnan(s0)
    coefs = dict(zip(cols, m1.coef_[0]))
    # 新聞單獨的 AUC（第一個新聞變數），看有沒有任何排序能力
    solo = auc(y, d[news_cols[0]].to_numpy())
    return {"test": label, "target": target, "n_days": len(y), "n_events": int(y.sum()),
            "years": f"{d.index.min():%Y-%m}~{d.index.max():%Y-%m}",
            "LR_chi2": lr, "df": len(news_cols), "p_LR": p_lr,
            "AUC_in_base": auc(y, m0.predict_proba(Z[:, :nb])[:, 1]),
            "AUC_in_news": auc(y, m1.predict_proba(Z)[:, 1]),
            "AUC_oos_base": auc(y[ok], s0[ok]), "AUC_oos_news": auc(y[ok], s1[ok]),
            "AUC_news_alone": solo,
            **{f"coef_{k}": v for k, v in coefs.items()}}


# ------------------------------------------------------------ 關鍵詞

_JIEBA = None


def tokens(title: str) -> list:
    global _JIEBA
    if _JIEBA is None:
        import jieba
        jieba.setLogLevel(60)
        _JIEBA = jieba
    out = []
    for w in _JIEBA.lcut(title, HMM=False):
        w = w.strip()
        if len(w) < 2 or w in STOPWORDS or re.fullmatch(r"[\d\.\-/%:：,，%]+", w):
            continue
        if re.fullmatch(r"[A-Za-z]", w):
            continue
        out.append(w)
    return list(set(out))


def keyword_lift(day_titles: list, bg_df: Counter, bg_n: int, k: int = 8) -> list:
    """當天標題的詞頻 vs 該來源背景文件頻率，取 lift 最高、至少出現 2 則的詞。"""
    c = Counter(w for t in day_titles for w in tokens(t))
    n = max(len(day_titles), 1)
    scored = []
    for w, cnt in c.items():
        if cnt < 2:
            continue
        lift = ((cnt + 0.5) / n) / ((bg_df.get(w, 0) + 0.5) / bg_n)
        scored.append((cnt * math.log(lift), w, cnt))
    scored.sort(reverse=True)
    return [f"{w}({cnt})" for s, w, cnt in scored[:k] if s > 0]


# ------------------------------------------------------------ 主流程

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = market()
    full_cal = _read_px("TWII").index
    print(f"交易日 {len(m)}（{m.index.min():%Y-%m-%d} ~ {m.index.max():%Y-%m-%d}）"
          f"　大波動 {int(m['big'].sum())}　極端 {int(m['extreme'].sum())}　一般 {int(m['normal'].sum())}")

    print("讀新聞 …", flush=True)
    stock = load_stock_news()
    raw = {"intl": load_intl(), "cnyes": stock[stock["src"] == "cnyes"],
           "yahoo": stock[stock["src"] == "yahoo"], "ptt": load_ptt_board()}

    docs, feats_pre, feats_day, cov_rows = {}, [], [], []
    for s in SOURCES:
        r0 = raw[s]
        if s != "ptt":
            boiler = r0["title"].str.contains(BOILER_PAT, na=False)
            print(f"  {s:6s} 剔除制式訊息 {int(boiler.sum())} 則（{boiler.mean():.1%}）")
            r0 = r0[~boiler]
        d = assign_day(r0, full_cal)
        d = d[(d["day"] >= m.index.min()) & (d["day"] <= m.index.max())]
        cov = coverage_months(d)
        d = d[d["day"].dt.to_period("M").isin(cov.index)]
        docs[s] = d
        # 當日 = 歸到 t 且在 t 日 09:00 前發布；前一天 = 歸到 t-1 的全部（t-1 盤前 + 盤中）
        feats_pre.append(daily_features(d[d["pre_open"]], m, s, cov.index))
        feats_day.append(daily_features(d, m, s, cov.index))
        days_cov = m.index[m.index.to_period("M").isin(cov.index)]
        cov_rows.append({"source": s, "名稱": SRC_NAME[s], "docs": len(d),
                         "first_month": str(cov.index.min()), "last_month": str(cov.index.max()),
                         "covered_months": len(cov), "trading_days": len(days_cov),
                         "big_days_covered": int(m.loc[days_cov, "big"].sum()),
                         "docs_per_day_median": float(d.groupby("day").size().median()),
                         "pre_open_share": float(d["pre_open"].mean()),
                         "recap_share": float(d["recap"].mean())})
        print(f"  {s:6s} {len(d):>8} 則　{cov.index.min()} ~ {cov.index.max()}　"
              f"覆蓋 {len(cov)} 個月　大波動日 {cov_rows[-1]['big_days_covered']}")
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(OUT / "coverage.csv", index=False, encoding="utf-8-sig")
    # 每月則數（方便畫圖看覆蓋斷層）
    monthly = pd.DataFrame({s: docs[s].groupby(docs[s]["day"].dt.to_period("M")).size() for s in SOURCES})
    monthly.index = monthly.index.astype(str)
    monthly.to_csv(OUT / "coverage_monthly.csv", encoding="utf-8-sig")

    # 欄位 {src}_xxx = 當日（t 盤前）；L1_{src}_xxx = 前一天（t-1 整個歸屬窗）
    panel = m.join(pd.concat(feats_pre, axis=1), how="left")
    lag = pd.concat(feats_day, axis=1).reindex(m.index).shift(1)
    lag.columns = [f"L1_{c}" for c in lag.columns]
    panel = panel.join(lag)
    panel.to_csv(OUT / "daily_panel.csv", encoding="utf-8-sig", float_format="%.6g")

    # ---------------- 1) 大波動日 vs 一般日
    big, normal = panel["big"], panel["normal"]
    up, down = big & (panel["r"] > 0), big & (panel["r"] < 0)
    tests = []
    show = ["abn_n", "abn_n_nonrecap", "recap_share",
            "neg_share", "pos_share", "tone", "tone_nonrecap", "neg_proj_share", "tone_dev"]
    for s in SOURCES:
        for c in show:
            for when, pre in (("當日盤前", ""), ("前一天", "L1_")):
                col = f"{pre}{s}_{c}"
                if col not in panel:
                    continue
                for grp, a, b in (("大波動 vs 一般", big, normal),
                                  ("極端 vs 一般", panel["extreme"], normal),
                                  ("大跌 vs 大漲", down, up)):
                    r = compare(panel, col, a, b)
                    if r:
                        tests.append({"source": s, "feature": c, "when": when, "comparison": grp, **r})
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "stats_big_vs_normal.csv", index=False, encoding="utf-8-sig", float_format="%.5g")

    # ---------------- 2) 主題：哪類主題在大波動日特別多
    th_rows = []
    for s in SOURCES:
        for k in THEMES:
            for when, pre in (("當日盤前", ""), ("前一天", "L1_")):
                col = f"{pre}{s}_th_{k}"
                r = compare(panel, col, big, normal)
                if r:
                    th_rows.append({"source": s, "theme": k, "when": when,
                                    "lift": r["mean_a"] / r["mean_b"] if r["mean_b"] > 0 else np.nan, **r})
    th = pd.DataFrame(th_rows)
    th.to_csv(OUT / "theme_enrichment.csv", index=False, encoding="utf-8-sig", float_format="%.5g")

    # ---------------- 3) t-1 新聞能不能預示 t 的大波動
    # 基準模型只用價格：前一天、前 5 天最大、前 20 天平均的 |z|，近月 / 近一年波動比，以及 σ。
    # 波動有群聚性，新聞量又跟著波動走，基準不夠強的話新聞會「看起來」有預測力。
    base = ["absz_lag1", "absz_lag5max", "absz_ma20", "rv20_ratio", "sigma"]
    preds, lift_rows = [], []
    for s in SOURCES:
        L = lambda c: f"L1_{s}_{c}"  # noqa: E731
        # 前 20 日（到 t-2 為止）的負面比例：用來分辨「t-1 的新消息」和「最近整體氣氛偏空」
        panel[f"{s}_neg_ma20_lag2"] = panel[f"{s}_neg_share"].shift(2).rolling(20, min_periods=10).mean()
        specs = [("t-1 新聞量", [L("abn_n")]),
                 ("t-1 非盤勢新聞量", [L("abn_n_nonrecap")]),
                 ("t-1 負面比例", [L("neg_share")]),
                 ("前20日負面比例（到t-2）", [f"{s}_neg_ma20_lag2"]),
                 ("t-1 新聞量+情緒", [L("abn_n"), L("neg_share"), L("tone_dev")])]
        # 經濟意義：t-1 負面比例分五等分，各組隔天出現大波動的機率
        q = panel[[L("neg_share"), "big"]].dropna()
        if len(q) > 200:
            q["quintile"] = pd.qcut(q[L("neg_share")], 5, labels=False, duplicates="drop") + 1
            for k, g in q.groupby("quintile"):
                lift_rows.append({"source": s, "t-1 負面比例分組": int(k), "days": len(g),
                                  "neg_share_mean": g[L("neg_share")].mean(), "P(big_t)": g["big"].mean()})
        for label, cols in specs:
            if not all(c in panel for c in cols):
                continue
            for tgt in ("big", "extreme"):
                r = predict_test(panel, tgt, base, cols, f"{s}｜{label}")
                if r:
                    preds.append(r)
        # 對照：當日盤前新聞（開盤前已知，但收盤對收盤的 r_t 在 t-1 收盤時已經起跑）
        r = predict_test(panel, "big", base + ["sox_z_abs"], [f"{s}_abn_n"],
                         f"{s}｜當日盤前新聞量（控制前一晚費半）")
        if r:
            preds.append(r)
    preds = pd.DataFrame(preds)
    pd.DataFrame(lift_rows).to_csv(OUT / "predict_t1_neg_quintiles.csv", index=False, encoding="utf-8-sig",
                                   float_format="%.4g")
    preds.to_csv(OUT / "predict_t1.csv", index=False, encoding="utf-8-sig", float_format="%.5g")

    # t-1 情緒能不能預示「方向」：只看大波動日，t-1 tone 和 r_t 的相關
    dir_rows = []
    for s in SOURCES:
        for c in ("tone", "tone_dev", "neg_share"):
            col = f"L1_{s}_{c}"
            sub = panel.loc[big & panel[col].notna(), [col, "r"]]
            if len(sub) >= 15:
                rho, p = stats.spearmanr(sub[col], sub["r"])
                dir_rows.append({"source": s, "feature": f"t-1 {c}", "n_big_days": len(sub),
                                 "spearman_vs_r_t": rho, "p": p})
    pd.DataFrame(dir_rows).to_csv(OUT / "direction_t1.csv", index=False, encoding="utf-8-sig",
                                  float_format="%.5g")

    # ---------------- 3b) 「明確新聞事件」旗標：同一把尺套在每一天，才能和一般日比
    # 文本範圍一律是「前一天 + 當日盤前」，也就是 t 開盤前全部已知的東西
    # 主題集中：非盤勢文本中某主題 ≥ 3 則、比例 ≥ 一般日平均的 3 倍
    # 新聞爆量：任一來源、任一窗的非盤勢新聞量異常 > 0.5（約平常的 1.65 倍）
    theme_base = {s: {k: panel.loc[normal, f"{s}_th_{k}"].mean() for k in THEMES} for s in SOURCES}
    ev = pd.DataFrame(index=panel.index)
    hit = {}
    for s in SOURCES:
        cov_s = panel[f"{s}_n"].notna() & panel[f"L1_{s}_n"].notna()
        n_all = panel[f"{s}_n_nonrecap"] + panel[f"L1_{s}_n_nonrecap"]
        for k in THEMES:
            c_all = panel[f"{s}_thc_{k}"] + panel[f"L1_{s}_thc_{k}"]
            hit[(s, k)] = cov_s & (c_all >= 3) & (c_all / n_all.replace(0, np.nan) >= 3 * theme_base[s][k])
        ev[f"{s}_cov"] = cov_s
    ev["covered"] = ev[[f"{s}_cov" for s in SOURCES]].any(axis=1)
    ev["theme"] = pd.concat(hit, axis=1).any(axis=1)
    # 爆量不看 Yahoo：它是逐檔爬的個股頁，量跟著爬取範圍跳，2025 後一般日也有 16% 會「爆量」
    bs = [s for s in SOURCES if s != "yahoo"]
    ev["burst_t"] = panel[[f"{s}_abn_n_nonrecap" for s in bs]].max(axis=1) > 0.5
    ev["burst_t1"] = panel[[f"L1_{s}_abn_n_nonrecap" for s in bs]].max(axis=1) > 0.5
    ev["burst"] = ev["burst_t"] | ev["burst_t1"]
    ev["event"] = ev["covered"] & (ev["theme"] | ev["burst"])
    ev["us_shock"] = panel["sox_z"].abs() > 2
    ev["event_or_us"] = ev["event"] | (ev["covered"] & ev["us_shock"])
    # 嚴格版：只看「爆量」（主題旗標在一般日也常亮，鑑別力低）
    ev["burst_or_us"] = ev["covered"] & (ev["burst"] | ev["us_shock"])
    ev.join(panel[["r", "z", "big", "normal"]]).to_csv(OUT / "event_flags_daily.csv", encoding="utf-8-sig")
    ev_rows = []
    cov_mask = ev["covered"]
    for era, msk in (("全部有覆蓋", cov_mask),
                     ("2016-10~2018（只有國際新聞）", cov_mask & (panel.index < "2019-01-01")),
                     ("2019~2024（國際+個股+PTT）", cov_mask & (panel.index >= "2019-01-01") & (panel.index < "2025-01-01")),
                     ("2025~（國際+個股，無PTT）", cov_mask & (panel.index >= "2025-01-01"))):
        for c in ("theme", "burst_t", "burst_t1", "burst", "event", "us_shock", "event_or_us", "burst_or_us"):
            a, b = big & msk, normal & msk
            if a.sum() < 5:
                continue
            tab = [[int(ev.loc[a, c].sum()), int((~ev.loc[a, c]).sum())],
                   [int(ev.loc[b, c].sum()), int((~ev.loc[b, c]).sum())]]
            odds, pf = stats.fisher_exact(tab)
            tmp = panel.assign(_f=ev[c].astype(float))
            r = compare(tmp, "_f", a, b)
            ev_rows.append({"era": era, "flag": c, "big_days": int(a.sum()), "rate_big": ev.loc[a, c].mean(),
                            "normal_days": int(b.sum()), "rate_normal": ev.loc[b, c].mean(),
                            "odds_ratio": odds, "p_fisher": pf,
                            "t_cluster": r["t_cluster"] if r else np.nan,
                            "p_cluster": r["p_cluster"] if r else np.nan})
    ev_tab = pd.DataFrame(ev_rows)
    ev_tab.to_csv(OUT / "event_flag_rates.csv", index=False, encoding="utf-8-sig", float_format="%.4g")

    # ---------------- 4) 逐日清單 + 關鍵詞 + 標題
    print("斷詞算背景詞頻 …", flush=True)
    bg = {}
    for s in SOURCES:
        nr = docs[s][~docs[s]["recap"]]
        # 背景只取每月 400 則的抽樣，避免國際新聞 2014-16 量太大拖慢
        smp = nr.groupby(nr["day"].dt.to_period("M"), group_keys=False).apply(
            lambda g: g.sample(min(len(g), 400), random_state=0))
        dfc = Counter(w for t in smp["title"] for w in tokens(t))
        bg[s] = (dfc, len(smp))

    rows, md = [], []
    bigdays = panel[panel["big"] | panel["top50_abs"]].copy()
    for day, row in bigdays.iterrows():
        prev = panel.index[panel.index.get_loc(day) - 1] if panel.index.get_loc(day) > 0 else None
        rec = {"date": day.date(), "r_pct": round(row["r"] * 100, 2), "z": round(row["z"], 2),
               "gap_pct": round(row["gap"] * 100, 2), "intraday_pct": round(row["intraday"] * 100, 2),
               "big_z2": bool(row["big"]), "extreme_z3": bool(row["extreme"]), "top50_abs": bool(row["top50_abs"]),
               "sox_overnight_pct": round(row["sox_ret"] * 100, 2) if pd.notna(row["sox_ret"]) else np.nan,
               "sox_overnight_z": round(row["sox_z"], 2) if pd.notna(row["sox_z"]) else np.nan,
               "sources_covered": ",".join(s for s in SOURCES if pd.notna(row.get(f"{s}_n")))}
        themes_hit, titles_all, kw_all = Counter(), [], []
        for s in SOURCES:
            if pd.isna(row.get(f"{s}_n")):
                continue
            rec[f"{s}_n_t"] = int(row[f"{s}_n"])                     # 當日盤前
            rec[f"{s}_n_t_nonrecap"] = int(row[f"{s}_n_nonrecap"])
            rec[f"{s}_abn_t"] = round(row[f"{s}_abn_n"], 2) if pd.notna(row[f"{s}_abn_n"]) else np.nan
            rec[f"{s}_n_t1"] = row.get(f"L1_{s}_n")
            rec[f"{s}_abn_t1"] = round(row[f"L1_{s}_abn_n"], 2) if pd.notna(row.get(f"L1_{s}_abn_n")) else np.nan
            rec[f"{s}_tone_t"] = round(row[f"{s}_tone"], 3) if pd.notna(row[f"{s}_tone"]) else np.nan
            rec[f"{s}_tone_t1"] = round(row[f"L1_{s}_tone"], 3) if pd.notna(row.get(f"L1_{s}_tone")) else np.nan
            dd = docs[s]
            dd = dd[(((dd["day"] == day) & dd["pre_open"]) | (dd["day"] == prev)) & ~dd["recap"]]
            for k in THEMES:
                if hit[(s, k)].get(day, False):
                    themes_hit[k] += int(row[f"{s}_thc_{k}"] + row[f"L1_{s}_thc_{k}"])
            kw = keyword_lift(dd["title"].tolist(), *bg[s])
            rec[f"{s}_keywords"] = " ".join(kw)
            kw_all += kw
            # 代表性標題：命中最多前幾名關鍵詞的標題
            kws = [k.split("(")[0] for k in kw[:5]]
            if kws and len(dd):
                sc = dd["title"].apply(lambda t: sum(k in t for k in kws))
                pick = dd.assign(_s=sc, _when=np.where(dd["day"] == prev, "前一天", "當日盤前"))
                pick = pick[pick["_s"] > 0].sort_values(["_s", "ts"], ascending=[False, True]).head(3)
                titles_all += [f"[{SRC_NAME[s]}｜{w}｜{ts:%m-%d %H:%M}] {t}"
                               for t, w, ts in zip(pick["title"], pick["_when"], pick["ts"])]
        abn_any = max([row.get(f"{s}_abn_n_nonrecap", np.nan) for s in SOURCES
                       if pd.notna(row.get(f"{s}_abn_n_nonrecap", np.nan))], default=np.nan)
        rec["themes"] = "、".join(f"{k}({v})" for k, v in themes_hit.most_common(4))
        rec["news_abn_max"] = round(abn_any, 2) if pd.notna(abn_any) else np.nan
        has_cov = bool(rec["sources_covered"])
        rec["news_burst"] = bool(ev.at[day, "burst"])
        rec["us_shock"] = bool(ev.at[day, "us_shock"])
        rec["event"] = bool(ev.at[day, "event"])
        rec["burst_or_us"] = bool(ev.at[day, "burst_or_us"])
        rec["headlines"] = " ‖ ".join(titles_all[:6])
        rows.append(rec)

        flag = "（無新聞資料）" if not has_cov else ""
        md.append(f"### {day:%Y-%m-%d}　大盤 {row['r'] * 100:+.2f}%（z={row['z']:+.1f}；跳空 "
                  f"{row['gap'] * 100:+.2f}%、盤中 {row['intraday'] * 100:+.2f}%）{flag}")
        if pd.notna(row["sox_ret"]):
            md.append(f"- 前一晚費半 {row['sox_ret'] * 100:+.2f}%（z={row['sox_z']:+.1f}）")
        if has_cov:
            for s in SOURCES:
                if f"{s}_n_t" in rec:
                    t1 = rec[f"{s}_n_t1"]
                    md.append(f"- {SRC_NAME[s]}：前一天 {int(t1) if pd.notna(t1) else '—'} 則"
                              f"（異常 {rec[f'{s}_abn_t1']}）／當日盤前 {rec[f'{s}_n_t']} 則"
                              f"（非盤勢 {rec[f'{s}_n_t_nonrecap']}、異常 {rec[f'{s}_abn_t']}）；"
                              f"關鍵詞：{rec.get(f'{s}_keywords') or '—'}")
            md.append(f"- 主題：{rec['themes'] or '無明顯集中'}；新聞爆量 {'是' if rec['news_burst'] else '否'}；"
                      f"前一晚費半大動 {'是' if rec['us_shock'] else '否'}")
            for h in titles_all[:6]:
                md.append(f"  - {h}")
        md.append("")

    lst = pd.DataFrame(rows)
    lst.to_csv(OUT / "big_days.csv", index=False, encoding="utf-8-sig")
    head = ["# 大波動日逐日新聞摘要（自動產生）", "",
            f"定義：TWII 日報酬 / 前 {SIGMA_WIN} 日標準差，|z| > {BIG_Z}；另含 |r| 前 50 名。",
            "時間窗：當日（盤前）= t-1 13:30 ~ t 09:00；前一天 = t-2 13:30 ~ t-1 13:30（t-1 盤前 + 盤中）。"
            "t 日 09:00 以後的新聞完全不算。",
            "「異常」= log(1+則數) 減前 60 個同類交易日中位數（0.69 ≈ 平常的 2 倍）。"
            "關鍵詞與標題只取非盤勢報導，括號是則數。", ""]
    (OUT / "big_days_news.md").write_text("\n".join(head + md), encoding="utf-8")

    # ---------------- 主控台摘要
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n覆蓋：\n", cov_df.to_string(index=False))
    key = tests[(tests["comparison"] == "大波動 vs 一般") &
                tests["feature"].isin(["abn_n", "abn_n_nonrecap", "tone", "neg_share", "tone_nonrecap"])]
    print("\n大波動 vs 一般：\n", key[["source", "feature", "when", "n_a", "n_b", "mean_a", "mean_b",
                                   "t_cluster", "p_cluster", "p_mannwhitney"]].round(4).to_string(index=False))
    dn = tests[(tests["comparison"] == "大跌 vs 大漲") & tests["feature"].isin(["tone", "tone_nonrecap", "neg_share"])]
    print("\n大跌 vs 大漲：\n", dn[["source", "feature", "when", "n_a", "n_b", "mean_a", "mean_b",
                               "t_cluster", "p_cluster"]].round(4).to_string(index=False))
    print("\n預測：\n", preds[["test", "target", "n_days", "n_events", "LR_chi2", "p_LR", "AUC_oos_base",
                             "AUC_oos_news", "AUC_news_alone"]].round(4).to_string(index=False))
    tsig = th[th["p_cluster"] < 0.1]
    print("\n主題（p<0.1）：\n", tsig[["source", "theme", "when", "mean_a", "mean_b", "lift",
                                         "t_cluster", "p_cluster"]].round(4).to_string(index=False))
    print(f"\n大波動清單 {len(lst)} 天；有新聞覆蓋 {int((lst['sources_covered'] != '').sum())}；"
          f"新聞爆量或前一晚費半大動 {int(lst['burst_or_us'].sum())}；前一晚費半 |z|>2 {int(lst['us_shock'].sum())}")
    print("\n明確事件旗標（大波動 vs 一般）：\n", ev_tab.round(4).to_string(index=False))
    print(f"輸出 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
