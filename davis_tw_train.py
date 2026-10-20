#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股 DAViS 訓練流程（單檔、依提案步驟線性走）
================================================================================
本檔把《台股DAViS_研究提案》§4 METHOD 的訓練流程，按照簡報上的順序一步一步寫下來，
每一步都會在畫面上印出做了什麼、資料變成什麼樣子。目的是「看得懂、講得出來」，
適合放進論文的方法章節或口試說明。

與 davis_tw_all.py 的分工
------------------------
    davis_tw_all.py   完整實驗框架：消融矩陣 E0–E3／E1a-c、跨股票彙總、批次報表
    davis_tw_train.py 本檔：單一條訓練流程，線性、可逐段閱讀；預設跑「個股＋28 個產業類股」的週波段
兩者的模型與評估邏輯一致，差別只在組織方式。

訓練流程（對應簡報的模組）
--------------------------
    STEP 1  蒐集層 C′：讀入四個資料軸
    STEP 2  價量技術面特徵
    STEP 3  籌碼面特徵（RQ2 新增的資料軸）
    STEP 4  文本特徵：斷詞 → TF-IDF → PCA → Ward 分群
    STEP 5  切分：walk-forward（訓練 → 驗證 → 測試，嚴格依時序）
    STEP 6  分析層 A′：8 個基礎模型 → Huber 堆疊集成
    STEP 7  評估層：天真基準對照 → DM 檢定 → 含成本回測

資料來源（用你已經蒐集好的那些）
--------------------------------
    股價技術面  data/stocks/{代號}/price/yahoo_{代號}_{年}.csv   Yahoo 還原股價
    籌碼面      data/chips/{代號}.csv                            證交所 T86 三大法人買賣超
    財經新聞    data/stocks/{代號}/news/cnyes_news_{代號}.csv     鉅亨網（2020–2024 完整）
                data/stocks/{代號}/news/yahoo_news_{代號}.csv     Yahoo 股市（近期補充）
    論壇        data/forum/cmoney_{代號}.csv                     股市爆料同學會

安裝
----
    pip install pandas numpy scikit-learn scipy
    pip install jieba          # 中文斷詞；沒裝會退回字元 bigram，文本特徵品質會很差
    pip install ckip-transformers torch   # 選用；繁中斷詞品質最好但要下載模型

執行（常用指令，複製就能跑）
----------------------------
    # 不帶參數＝個股 2330 ＋ 全部 28 個產業類股，週波段（約 15 分鐘）
    python davis_tw_train.py

    # 只跑全部 28 個產業類股
    python davis_tw_train.py --level sector --features price chip --quiet

    # 指定多檔個股（指令要寫成一整行）
    python davis_tw_train.py --stocks 2330 2317 1101 2603 --features price chip --quiet

    # 切回原本的隔日沖
    python davis_tw_train.py --stock 2330 --horizon 1

    # 單一檔股票、印出完整七個步驟（適合看流程、寫論文）
    python davis_tw_train.py --level stock --stock 2330 --features price chip text

    # 指定幾個產業
    python davis_tw_train.py --level sector --sectors 半導體 金融 航運

參數速查
--------
    --level     both＝個股＋產業類股（預設）｜stock＝只跑個股｜sector＝只跑產業類股
    --stock / --stocks    個股代號（預設 2330）；--sectors 指定產業（預設全部 28 個）
    --features  price chip text 任選（預設三個都用）
    --horizon   預測期間（交易日）：5=週波段（預設）、1=隔日沖
    --start / --end       研究期間，預設 2020-01-01 ~ 2024-12-31
    --quiet     只印最後的彙總表
    --out       輸出目錄，預設 results/train_{時間戳}

判讀提醒
--------
    DA 要跟 always_long 比（不是跟 naive 比），MAPE 要跟 naive 比，
    而且差異必須通過 DM 檢定（p < 0.05）才算數。
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from tw_market_index import benchmark_stats, load_twii

# ============================================================================
# 設定
# ============================================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

START_DATE = "2019-01-01"      # 研究期間：6 年（2019–2024），約 1,460 個交易日
#                               籌碼面 T86 自 2015 起、鉅亨新聞自 2019 起，兩者都涵蓋得到
END_DATE = "2024-12-31"
WARMUP_DAYS = 180              # 技術指標暖機期，只算特徵不進訓練

TRAIN_DAYS, VAL_DAYS, TEST_DAYS, STEP_DAYS = 600, 120, 120, 120

# 預測期間（交易日），預設兩種都跑，結果並排比較：
#     1 = 隔日沖（預測隔日報酬，訊號為真就持有一天）
#     5 = 週波段（預測未來一週累積報酬，進場後至少持有一週）
# 兩種並列的理由：台股來回成本 0.585%，隔日沖幾乎每天都要吃一次，
# 波段則是一次成本攤在一週的報酬上，兩者的成本結構差很多，值得直接對照。
# 成本拖累 = 252/h × 0.585%：h=1 每年 −147%、h=5 −29%、h=20 −7.4%、h=60 −2.5%。
# 實測模型選股每 5 日只比全市場等權多約 0.24 個百分點，根本蓋不住 h=1/5 的成本，
# 所以把月波段（20 日）與季波段（60 日）一起納入，才問得出「拉長週期能不能成立」。
HORIZONS = [1, 5, 20, 60]

FEE_RATE = 0.001425            # 手續費（買賣各一次）
TAX_RATE = 0.003               # 證交稅（賣出）
TRADING_DAYS = 252

MARKET_CLOSE = (13, 30)        # 收盤後發布的文件順延到下一交易日

BASE_MODELS = ("ridge", "lasso", "elasticnet", "svr",
               "knn", "random_forest", "extra_trees", "gbrt")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("davis_train")


def banner(step: str, title: str) -> None:
    log.info("\n" + "=" * 76)
    log.info(f"{step}  {title}")
    log.info("=" * 76)


# ============================================================================
# STEP 1　蒐集層 C′：讀入四個資料軸
# ============================================================================


def load_price(code: str) -> pd.DataFrame | None:
    """讀逐年的 Yahoo 還原股價並串成一張表。"""
    files = sorted((DATA / "stocks" / code / "price").glob("yahoo_*.csv"))
    files = [f for f in files if "merged" not in f.stem]
    if not files:
        return None

    px = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = (px.dropna(subset=["date", "close"])
            .sort_values("date").drop_duplicates("date", keep="last").set_index("date"))

    # Yahoo 在停牌／補值日會給 volume=0 且 OHLC 全等的假列，逐日報酬率會被灌 0
    flat = (px["volume"] <= 0) & (px["open"] == px["close"]) & (px["high"] == px["low"])
    px = px.loc[~flat, ["open", "high", "low", "close", "volume"]]
    return _apply_price_fixes(code, px)


_PRICE_FIXES = None


def _apply_price_fixes(code: str, px: pd.DataFrame) -> pd.DataFrame:
    """
    套用 data/price_fixes.csv（tool/make_price_fixes.py 產生）。

    Yahoo 有少數減資、現金增資沒做分割調整，也有單日錯價（例如 1752 在 2025-01-10
    收盤是前一天的 101 倍）。這些超過漲跌幅限制的假跳動會灌進波動度訊號和報酬，
    所以把跳動日之前的價格整段乘上係數，讓那一天的報酬變成 0（2026-10-15 加入）。
    """
    global _PRICE_FIXES
    if _PRICE_FIXES is None:
        f = DATA / "price_fixes.csv"
        _PRICE_FIXES = (pd.read_csv(f, dtype={"code": str}, parse_dates=["date"])
                        if f.exists() else pd.DataFrame(columns=["code", "date", "factor"]))
    fx = _PRICE_FIXES[_PRICE_FIXES["code"] == code]
    if fx.empty:
        return px
    px = px.copy()
    for _, r in fx.iterrows():
        before = px.index < r["date"]
        px.loc[before, ["open", "high", "low", "close"]] *= r["factor"]
        px.loc[before, "volume"] /= r["factor"]
    return px


def load_chips(code: str) -> pd.DataFrame | None:
    """讀證交所 T86 轉出的三大法人買賣超（單位：股）。"""
    path = DATA / "chips" / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").set_index("date")


def _read_text_csv(path: Path, source: str) -> pd.DataFrame:
    """讀單一文本檔；沒有可用時間戳的列一律丟棄（對齊不到交易日就不能用）。"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if "date" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({"ts": pd.to_datetime(df["date"], errors="coerce", utc=True)})
    out["ts"] = out["ts"].dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    out["title"] = df.get("title", "").fillna("").astype(str)
    body = next((c for c in ("text", "summary", "content") if c in df.columns), None)
    out["text"] = df[body].fillna("").astype(str) if body else ""
    out["source"] = source
    # PTT 另有推／噓／留言數，是新聞沒有的情緒訊號，原樣帶下去給 step4 彙總
    for col in ("n_push", "n_boo", "n_comments"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["ts"])
    return out[(out["title"].str.strip() != "") | (out["text"].str.strip() != "")]


def load_news(code: str) -> pd.DataFrame:
    """鉅亨網（歷史主幹）+ Yahoo 股市（近期補充）。"""
    d = DATA / "stocks" / code / "news"
    parts = []
    for pattern, src in ((f"cnyes_news_{code}.csv", "cnyes"), (f"yahoo_news_{code}.csv", "yahoo")):
        p = d / pattern
        if p.exists():
            parts.append(_read_text_csv(p, src))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_forum(code: str) -> pd.DataFrame:
    """股市爆料同學會。"""
    p = DATA / "forum" / f"cmoney_{code}.csv"
    return _read_text_csv(p, "cmoney") if p.exists() else pd.DataFrame()


def load_ptt(code: str) -> pd.DataFrame:
    """PTT Stock 板（由 ptt_to_stock.py 依代號／公司名歸屬後產生）。"""
    p = DATA / "ptt" / f"{code}.csv"
    return _read_text_csv(p, "ptt") if p.exists() else pd.DataFrame()


def step1_load(code: str, start: str, end: str, quiet: bool = False) -> dict:
    banner("STEP 1", f"蒐集層 C′：讀入 {code} 的四個資料軸")

    px = load_price(code)
    if px is None or px.empty:
        raise SystemExit(f"找不到 {code} 的股價資料（data/stocks/{code}/price/）")

    # 多載入半年暖機期：技術指標最長回看 60 日，否則研究期初會整段被 NaN 吃掉
    eval_start = pd.Timestamp(start)
    px = px[(px.index >= eval_start - pd.Timedelta(days=WARMUP_DAYS)) & (px.index <= pd.Timestamp(end))]

    chips = load_chips(code)
    news = load_news(code)
    forum = load_forum(code)
    ptt = load_ptt(code)

    in_scope = px[px.index >= eval_start]
    if not quiet:
        log.info(f"  股價　  {len(in_scope):>6} 個交易日　{in_scope.index.min():%Y-%m-%d} ~ "
                 f"{in_scope.index.max():%Y-%m-%d}（另含 {len(px) - len(in_scope)} 日暖機期）")
        if chips is not None:
            c = chips[(chips.index >= eval_start) & (chips.index <= pd.Timestamp(end))]
            log.info(f"  籌碼面　{len(c):>6} 個交易日　欄位：{', '.join(chips.columns[:4])} …")
        else:
            log.info("  籌碼面　  無（data/chips/ 缺這一檔）")
        for name, df in (("新聞　", news), ("論壇　", forum), ("PTT　", ptt)):
            inr = (df[(df["ts"] >= eval_start) & (df["ts"] <= pd.Timestamp(end))]
                   if len(df) else df)
            if len(inr):
                by_src = "、".join(f"{k} {v}" for k, v in inr["source"].value_counts().items())
                log.info(f"  {name}  {len(inr):>6} 則　{inr['ts'].min():%Y-%m-%d} ~ "
                         f"{inr['ts'].max():%Y-%m-%d}（{by_src}）")
            elif len(df):
                # 有檔案但全部落在研究期間之外（同學會只有近期快照就會這樣）
                log.info(f"  {name}    研究期間內 0 則（檔案共 {len(df)} 則，"
                         f"最早 {df['ts'].min():%Y-%m-%d}）")
            else:
                log.info(f"  {name}    無")

    return {"code": code, "px": px, "chips": chips, "news": news, "forum": forum,
            "ptt": ptt, "eval_start": eval_start}


# ============================================================================
# 產業類股：把同產業的個股合成一個「類股指數」再照同一條流程訓練
# ============================================================================

INDUSTRY_MAP = DATA / "industry_stock_map.json"


def load_industry_map() -> dict:
    """讀 data/industry_stock_map.json → {產業: [股票代號, ...]}。"""
    if not INDUSTRY_MAP.exists():
        raise SystemExit(f"找不到產業分類清單 {INDUSTRY_MAP}（請先跑 tw_stock_crawler.py）")
    rows = json.loads(INDUSTRY_MAP.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        out.setdefault(r["category"], []).append(r["code"])
    return out


def all_stock_codes() -> list:
    """股票清單裡的全部代號（data/industry_stock_map.json）。"""
    if not INDUSTRY_MAP.exists():
        log.warning(f"找不到 {INDUSTRY_MAP}，退回只跑 2330")
        return ["2330"]
    rows = json.loads(INDUSTRY_MAP.read_text(encoding="utf-8"))
    return [r["code"] for r in rows]


def stock_to_sector() -> dict:
    """反查表：{股票代號: 產業}。"""
    rows = json.loads(INDUSTRY_MAP.read_text(encoding="utf-8")) if INDUSTRY_MAP.exists() else []
    return {r["code"]: r["category"] for r in rows}


MIN_SECTOR_MEMBERS = 3        # 一天至少要有幾檔成分股有報價，才算得出類股報酬


def build_sector_index(codes: list, start: str, end: str) -> tuple:
    """
    把同產業個股合成等權類股指數。

    用「等權平均日報酬」而不是平均價格：平均價格會被高價股主導，
    成分股價位差異大時合出來的序列也沒有經濟意義。
    日內欄位（開高低）先各自除以前一日收盤變成相對值再平均，
    這樣 STEP 2 的日內結構特徵仍然成立；成交量直接加總。

    日期用聯集而非交集：每天只用「當天有報價的成分股」計算，
    成分股不足 MIN_SECTOR_MEMBERS 檔的日子才剔除。
    取交集的話，產業裡只要有一檔是近期才上市，整段歷史就會被砍掉。
    """
    frames = {}
    for code in codes:
        px = load_price(code)
        if px is not None and len(px) >= 100:
            frames[code] = px
    if not frames:
        return None, []

    lo = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    hi = pd.Timestamp(end)
    idx = None
    for px in frames.values():                       # 聯集
        idx = px.index if idx is None else idx.union(px.index)
    idx = idx[(idx >= lo) & (idx <= hi)].sort_values()
    if len(idx) < 200:
        return None, list(frames)

    rets, o_rel, h_rel, l_rel, vols = [], [], [], [], []
    for px in frames.values():
        p_ = px.reindex(idx)
        prev = p_["close"].shift(1)
        rets.append(p_["close"] / prev - 1)
        o_rel.append(p_["open"] / prev)
        h_rel.append(p_["high"] / prev)
        l_rel.append(p_["low"] / prev)
        vols.append(p_["volume"])

    R = pd.concat(rets, axis=1)
    avail = R.notna().sum(axis=1)                    # 當天有報價的成分股數
    keep = avail >= min(MIN_SECTOR_MEMBERS, len(frames))
    idx = idx[keep]
    if len(idx) < 200:
        return None, list(frames)

    R = R.loc[idx]
    ret = R.mean(axis=1, skipna=True)
    close = 100.0 * (1 + ret.fillna(0)).cumprod()    # 指數基期 100
    prev_close = close.shift(1)
    index = pd.DataFrame({
        "open": pd.concat(o_rel, axis=1).loc[idx].mean(axis=1, skipna=True) * prev_close,
        "high": pd.concat(h_rel, axis=1).loc[idx].mean(axis=1, skipna=True) * prev_close,
        "low": pd.concat(l_rel, axis=1).loc[idx].mean(axis=1, skipna=True) * prev_close,
        "close": close,
        "volume": pd.concat(vols, axis=1).loc[idx].sum(axis=1, skipna=True),
    }, index=idx)
    index.attrs["avg_members"] = float(avail.loc[idx].mean())
    return index.dropna(), list(frames)


def build_sector_chips(codes: list, index: pd.DataFrame) -> pd.DataFrame | None:
    """產業籌碼面：各成分股的買賣超股數直接加總（單位一致，可加總）。"""
    frames = [c for c in (load_chips(x) for x in codes) if c is not None and not c.empty]
    if not frames:
        return None
    cols = set(frames[0].columns)
    for f in frames[1:]:
        cols &= set(f.columns)
    cols = [c for c in frames[0].columns if c in cols]
    if not cols:
        return None
    total = None
    for f in frames:
        part = f[cols].reindex(index.index)
        total = part if total is None else total.add(part, fill_value=0)
    return total


def build_sector_text(codes: list, loader) -> pd.DataFrame:
    """產業文本：把成分股的文件合併後去重（同一則新聞常同時標記多檔個股）。"""
    parts = [d for d in (loader(x) for x in codes) if d is not None and len(d)]
    if not parts:
        return pd.DataFrame()
    allp = pd.concat(parts, ignore_index=True)
    return allp.drop_duplicates(subset=["ts", "title"]).sort_values("ts").reset_index(drop=True)


def step1_load_sector(sector: str, codes: list, start: str, end: str,
                      quiet: bool = False) -> dict:
    banner("STEP 1", f"蒐集層 C′：合成「{sector}」類股指數（{len(codes)} 檔成分股）")

    index, used = build_sector_index(codes, start, end)
    if index is None or index.empty:
        raise SystemExit(f"{sector}：成分股報價不足，無法合成指數")

    chips = build_sector_chips(used, index)
    news = build_sector_text(used, load_news)
    forum = build_sector_text(used, load_forum)
    ptt = build_sector_text(used, load_ptt)
    eval_start = pd.Timestamp(start)
    in_scope = index[index.index >= eval_start]

    if not quiet:
        log.info(f"  成分股　{len(used)} 檔：{'、'.join(used)}")
        avg_n = index.attrs.get("avg_members", float(len(used)))
        log.info(f"  指數　　{len(in_scope)} 個交易日　{in_scope.index.min():%Y-%m-%d} ~ "
                 f"{in_scope.index.max():%Y-%m-%d}（等權日報酬合成，基期 100，"
                 f"每日平均 {avg_n:.1f} 檔成分股）")
        log.info(f"  籌碼面　{'成分股買賣超加總' if chips is not None else '無'}")
        for name, df in (("新聞", news), ("論壇", forum), ("PTT", ptt)):
            inr = df[(df["ts"] >= eval_start) & (df["ts"] <= pd.Timestamp(end))] if len(df) else df
            log.info(f"  {name}　　{len(inr)} 則（已跨成分股去重）")

    return {"code": sector, "px": index, "chips": chips, "news": news, "forum": forum,
            "ptt": ptt, "eval_start": eval_start, "members": used}


# ============================================================================
# STEP 2　價量技術面特徵
# ============================================================================


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def step2_price_features(px: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    banner("STEP 2", "價量技術面特徵（前綴 px_）")
    c, h, l, o, v = px["close"], px["high"], px["low"], px["open"], px["volume"]
    f = pd.DataFrame(index=px.index)

    ret1 = c.pct_change()
    f["px_ret1"] = ret1
    for k in (2, 3, 5, 10, 20):
        f[f"px_ret{k}"] = c.pct_change(k)
    for k in (5, 10, 20):
        f[f"px_vol{k}"] = ret1.rolling(k).std()
    for lag in (1, 2, 3):
        f[f"px_ret1_lag{lag}"] = ret1.shift(lag)

    for k in (3, 5, 10, 20, 60):                    # 均線乖離用比值，避免非定態
        f[f"px_ma{k}_dev"] = c / c.rolling(k).mean() - 1
    f["px_ma5_ma20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1

    for k in (5, 20, 60):                           # 區間位置
        hi, lo = h.rolling(k).max(), l.rolling(k).min()
        f[f"px_pos{k}"] = (c - lo) / (hi - lo).replace(0, np.nan)

    f["px_hl_range"] = (h - l) / c.shift(1)         # 日內結構
    f["px_oc_gap"] = o / c.shift(1) - 1
    f["px_close_loc"] = (c - l) / (h - l).replace(0, np.nan)

    logv = np.log1p(v)                              # 量能
    f["px_logvol"] = logv
    for k in (5, 20):
        f[f"px_vol_ratio{k}"] = v / v.rolling(k).mean().replace(0, np.nan)
    f["px_dvol_ma5"] = logv - logv.rolling(5).mean()

    # ---- 台股實務常用、原本缺的幾項 ----
    # KD（隨機指標）：台灣散戶最常用的超買超賣指標，低檔黃金交叉是常見進場依據
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = ((c - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    k_line = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d_line = k_line.ewm(alpha=1 / 3, adjust=False).mean()
    f["px_k"] = k_line / 100
    f["px_d"] = d_line / 100
    f["px_kd_diff"] = (k_line - d_line) / 100       # >0 為 K 在 D 之上（黃金交叉後）

    # ATR：波動幅度的絕對水準，用來把跌幅門檻標準化（跌 8% 對不同波動度的股票意義不同）
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    f["px_atr14"] = atr14 / c
    f["px_drop_in_atr"] = (c / c.shift(5) - 1) / (atr14 / c).replace(0, np.nan)

    # OBV：量價配合。下跌時 OBV 沒同步走弱，代表賣壓可能已經衰竭
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    f["px_obv_dev"] = (obv - obv.rolling(20).mean()) / v.rolling(20).mean().replace(0, np.nan)

    # 均線多空排列：短中長期均線的相對位置，判斷趨勢結構
    ma5, ma20_, ma60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    f["px_ma_align"] = (np.sign(ma5 - ma20_) + np.sign(ma20_ - ma60)) / 2
    f["px_ma60_dev"] = c / ma60 - 1

    f["px_rsi14"] = _rsi(c) / 100                   # 動能
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["px_macd"] = macd / c
    f["px_macd_sig"] = (macd - macd.ewm(span=9).mean()) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["px_bb_z"] = (c - ma20) / sd20.replace(0, np.nan)

    f["px_limit_up"] = (ret1 >= 0.095).astype(float)   # 台股 ±10% 漲跌幅限制
    f["px_limit_dn"] = (ret1 <= -0.095).astype(float)
    f["px_dow"] = px.index.dayofweek.astype(float)
    f["px_month"] = px.index.month.astype(float)

    f = f.replace([np.inf, -np.inf], np.nan)
    if not quiet:
        log.info(f"  產生 {f.shape[1]} 個特徵：報酬率／波動／均線乖離／區間位置／"
                 f"量能／RSI／MACD／布林／漲跌停旗標／日曆效應")
        log.info("  全部只用 t 日（含）以前的資料，預測目標是 t+1，故無前視偏誤")
    return f


# ============================================================================
# STEP 3　籌碼面特徵（RQ2 新增的資料軸）
# ============================================================================


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    """
    20 日 z 分數。標準差為 0（例如投信連續 20 日完全沒進出）時，偏離度定義為 0，
    不是遺漏值 —— 否則整列會在 dropna 被剔除，冷門類股／小型股會因此損失大量樣本。
    暖機期（不足 window 日）仍維持 NaN，因為那是真的還算不出來。
    """
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    z = (x - mu) / sd.replace(0, np.nan)
    return z.mask(sd.eq(0), 0.0)


def _sign_streak(x: pd.Series) -> pd.Series:
    """連續同向天數（正為連買、負為連賣）。"""
    out, run, prev = np.zeros(len(x)), 0.0, 0.0
    for i, s in enumerate(np.sign(x.fillna(0)).to_numpy()):
        run = run + s if s != 0 and s == prev else s
        prev, out[i] = s, run
    return pd.Series(out, index=x.index)


def step3_chip_features(chips: pd.DataFrame | None, px: pd.DataFrame,
                        quiet: bool = False) -> pd.DataFrame:
    banner("STEP 3", "籌碼面特徵（前綴 chip_）—— 原論文情境不存在的資料軸")
    if chips is None or chips.empty:
        log.info("  沒有籌碼面資料，本步驟略過")
        return pd.DataFrame(index=px.index)

    df = chips.reindex(px.index)
    vol = px["volume"].replace(0, np.nan)
    f = pd.DataFrame(index=px.index)

    cols = [c for c in ("foreign_investor", "investment_trust", "dealer",
                        "dealer_self", "dealer_hedge") if c in df.columns]
    for col in cols:
        x = df[col]
        f[f"chip_{col}_norm"] = x / vol                       # 買賣超佔當日成交量比重
        f[f"chip_{col}_ma5"] = (x / vol).rolling(5).mean()
        f[f"chip_{col}_ma20"] = (x / vol).rolling(20).mean()
        f[f"chip_{col}_z20"] = _rolling_z(x, 20)
        f[f"chip_{col}_streak"] = _sign_streak(x)             # 連買／連賣天數

    if cols:
        total = df["inst_total"] if "inst_total" in df.columns else df[
            [c for c in cols if c in ("foreign_investor", "investment_trust", "dealer")]].sum(axis=1)
        f["chip_inst_total_norm"] = total / vol
        f["chip_inst_total_ma5"] = (total / vol).rolling(5).mean()
        if {"foreign_investor", "investment_trust"} <= set(cols):
            # 外資與投信是否同向：籌碼共識訊號
            f["chip_fi_it_agree"] = np.sign(df["foreign_investor"]) * np.sign(df["investment_trust"])

    f = f.replace([np.inf, -np.inf], np.nan)
    if not quiet:
        log.info(f"  來源欄位：{', '.join(cols)}")
        log.info(f"  產生 {f.shape[1]} 個特徵：買賣超佔量比／5 日與 20 日均值／20 日 z 分數／"
                 f"連買連賣天數／三大法人合計／外資投信同向")
        log.info("  買賣超是盤後揭露，用 t 日數值預測 t+1 沒有前視問題")
    return f


# ============================================================================
# STEP 4　文本特徵：斷詞 → TF-IDF → PCA → Ward 分群
# ============================================================================


class Tokenizer:
    """CKIP 斷詞；未安裝時退回 jieba，再不行退回字元 bigram。"""

    _PUNCT = re.compile(r"[^一-鿿A-Za-z0-9]+")

    def __init__(self):
        self.kind, self._ws, self._jieba = "ckip", None, None
        try:
            from ckip_transformers.nlp import CkipWordSegmenter
            self._ws = CkipWordSegmenter(model="bert-base")
        except Exception:
            self.kind = "jieba"
        if self.kind == "jieba":
            try:
                import jieba
                self._jieba = jieba
            except Exception:
                self.kind = "bigram"

    def __call__(self, texts: list) -> list:
        cleaned = [self._PUNCT.sub(" ", t) for t in texts]
        if self.kind == "ckip":
            return [" ".join(w for w in ws if w.strip())
                    for ws in self._ws(cleaned, use_delim=True, batch_size=64)]
        if self.kind == "jieba":
            return [" ".join(w for w in self._jieba.cut(t) if w.strip()) for t in cleaned]
        return [" ".join(s[i:i + 2] for i in range(len(s) - 1))
                for s in (t.replace(" ", "") for t in cleaned)]


class TextFeatures:
    """
    沿用 DAViS-C 的設計：TF-IDF → PCA（稀疏矩陣用 TruncatedSVD）→ Ward 階層分群。

    防洩漏的關鍵：這三樣東西只能用「該折訓練窗內的文件」擬合，
    驗證與測試窗的文件只做 transform，所以本類別必須在每一折內重新建立。
    """

    def __init__(self, prefix: str, tokenizer: Tokenizer,
                 n_pca: int = 30, n_clusters: int = 10, lookback: int = 5):
        self.prefix, self.tok = prefix, tokenizer
        self.n_pca, self.n_clusters, self.lookback = n_pca, n_clusters, lookback
        self.fitted = False

    @staticmethod
    def _to_trading_day(ts: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
        """收盤後（或無時分）發布的文件順延到下一個交易日，避免用到尚未公開的資訊。"""
        after = (ts.dt.hour > MARKET_CLOSE[0]) | (
            (ts.dt.hour == MARKET_CLOSE[0]) & (ts.dt.minute >= MARKET_CLOSE[1]))
        midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0)
        eff = ts.dt.normalize() + pd.to_timedelta((after | midnight).astype(int), unit="D")
        idx = calendar.searchsorted(eff.to_numpy(), side="left")
        beyond = idx >= len(calendar)
        out = pd.Series(calendar[np.clip(idx, 0, len(calendar) - 1)], index=ts.index)
        out[beyond] = pd.NaT
        return out

    @staticmethod
    def ensure_tokens(docs: pd.DataFrame, tokenizer) -> pd.DataFrame:
        """
        先把整批文件斷好詞存進 tokens 欄，之後每一折、每一種預測期間直接取用。

        不快取的話，同一批文件會被重複斷詞：6 折 × 4 種預測期間 × (fit + transform)
        ＝ 48 次。以 2330 的 1.5 萬則新聞計，光斷詞就要 16 分鐘，而實際只需 20 秒。
        """
        if docs is None or docs.empty or "tokens" in docs.columns:
            return docs
        texts = (docs["title"].fillna("") + " " + docs["text"].fillna("")).str.strip().tolist()
        docs = docs.copy()
        docs["tokens"] = tokenizer(texts)
        return docs

    def _tokens(self, docs: pd.DataFrame) -> list:
        if "tokens" in docs.columns:
            return docs["tokens"].tolist()
        return self.tok((docs["title"].fillna("") + " " + docs["text"].fillna("")).str.strip().tolist())

    def fit(self, docs: pd.DataFrame) -> "TextFeatures":
        texts = (docs["title"] + " " + docs["text"]).str.strip().tolist()
        if len(texts) < max(20, self.n_clusters * 2):
            return self
        toks = self._tokens(docs)
        self.tfidf = TfidfVectorizer(max_features=20000, min_df=min(3, max(1, len(texts) // 50)),
                                     max_df=0.8, token_pattern=r"(?u)\S+")
        X = self.tfidf.fit_transform(toks)
        if X.shape[1] < 3:
            return self
        n_comp = int(min(self.n_pca, X.shape[1] - 1, max(2, X.shape[0] - 1)))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        Z = self.svd.fit_transform(X)
        n_clu = int(min(self.n_clusters, max(2, Z.shape[0] // 5)))
        # Ward 是 O(n^2)，共用語料有十萬筆等級的文件，必須抽樣後再用最近中心法外推。
        # 分群只是要定出「主題中心」，抽樣不影響中心位置的估計。
        if Z.shape[0] > WARD_MAX_SAMPLES:
            rng = np.random.default_rng(42)
            sel = rng.choice(Z.shape[0], WARD_MAX_SAMPLES, replace=False)
            Zc = Z[sel]
        else:
            Zc = Z
        labels = AgglomerativeClustering(n_clusters=n_clu, linkage="ward").fit_predict(Zc)
        self.centroids = NearestCentroid().fit(Zc, labels)
        self.n_clu, self.n_comp, self.fitted = n_clu, n_comp, True
        return self

    def transform(self, docs: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.fitted or docs.empty:
            return pd.DataFrame(index=calendar)
        d = docs.copy()
        d["day"] = self._to_trading_day(d["ts"], calendar)
        d = d.dropna(subset=["day"])
        if d.empty:
            return pd.DataFrame(index=calendar)

        toks = self._tokens(d)
        Z = self.svd.transform(self.tfidf.transform(toks))
        per_doc = pd.DataFrame(Z, columns=[f"{self.prefix}pc{i}" for i in range(Z.shape[1])])
        per_doc["day"] = d["day"].to_numpy()
        per_doc["cluster"] = self.centroids.predict(Z)

        cols = [c for c in per_doc.columns if c.startswith(self.prefix)]
        daily = per_doc.groupby("day")[cols].mean()
        counts = (per_doc.groupby(["day", "cluster"]).size().unstack(fill_value=0)
                         .reindex(columns=range(self.n_clu), fill_value=0))
        counts.columns = [f"{self.prefix}clu{c}_n" for c in counts.columns]
        daily = daily.join(counts, how="outer")
        daily[f"{self.prefix}n_docs"] = per_doc.groupby("day").size()

        daily = daily.reindex(calendar)
        n_cols = [c for c in daily.columns if c.endswith("_n") or c.endswith("n_docs")]
        daily[n_cols] = daily[n_cols].fillna(0.0)          # 當天沒文就是 0 篇
        cont = [c for c in daily.columns if c not in n_cols]
        daily[cont] = daily[cont].ffill(limit=self.lookback).fillna(0.0)

        rolled = daily.rolling(self.lookback, min_periods=1).mean().add_suffix(f"_ma{self.lookback}")
        return pd.concat([daily, rolled], axis=1)


TEXT_SOURCES = ("news", "forum", "ptt")


def _all_codes() -> list:
    d = DATA / "stocks"
    return sorted(x.name for x in d.iterdir()
                  if x.is_dir() and any((x / "price").glob("*.csv"))) if d.exists() else []


def _load_corpus(source: str, tokenizer) -> pd.DataFrame:
    """把全市場某一來源的文件讀成一份語料並斷詞（整個流程只做一次）。"""
    if source in _CORPUS:
        return _CORPUS[source]
    loader = {"news": load_news, "forum": load_forum, "ptt": load_ptt}[source]
    parts = []
    for c in _all_codes():
        d = loader(c)
        if d is not None and len(d):
            parts.append(d)
    if not parts:
        _CORPUS[source] = pd.DataFrame()
        return _CORPUS[source]
    allp = (pd.concat(parts, ignore_index=True)
              .drop_duplicates(subset=["ts", "title"])
              .sort_values("ts").reset_index(drop=True))
    log.info(f"  共用語料［{source}］：{len(allp):,} 則（跨 {len(parts)} 檔、已去重），斷詞中…")
    allp = TextFeatures.ensure_tokens(allp, tokenizer)
    _CORPUS[source] = allp
    return allp


def get_pooled_model(source: str, prefix: str, train_end: pd.Timestamp, tokenizer):
    """
    取得（或擬合）某來源在 train_end 之前的共用主題模型。

    train_end 一律往下取整到當月 1 日再當快取鍵：一來把快取鍵數量壓到每月一個，
    二來用到的文件必定早於 train_end，不會有洩漏。
    """
    key = (source, pd.Timestamp(train_end).to_period("M").start_time)
    if key in _POOLED:
        return _POOLED[key]
    corpus = _load_corpus(source, tokenizer)
    if corpus.empty:
        _POOLED[key] = None
        return None
    train_docs = corpus[corpus["ts"] < key[1]]
    tf = TextFeatures(prefix, tokenizer, n_pca=POOLED_PCA,
                      n_clusters=POOLED_CLUSTERS).fit(train_docs)
    log.info(f"  共用主題模型［{source} @ {key[1]:%Y-%m}］：以 {len(train_docs):,} 則擬合"
             f"（{'成功' if tf.fitted else '樣本不足'}）")
    _POOLED[key] = tf if tf.fitted else None
    return _POOLED[key]

# 由 --text-sources 設定；None 代表全用。做單一資料軸消融時（例如只留 ptt）才會改。
TEXT_SOURCE_FILTER = None

# 由 --no-tfidf 關閉。關掉後不做 TF-IDF／SVD／Ward，只保留 PTT 的推噓反應特徵，
# 用來分辨「文本內容」與「推噓情緒」哪一半（如果有的話）帶來效果。
TEXT_USE_TFIDF = True

# 由 --pooled-text 開啟。把主題模型改成「全市場共用」而非逐檔擬合。
#
# 為什麼要這樣改：每檔股票的新聞中位數只有 131 則（六年），訓練窗內約 53 則，
# 卻要在上面擬合 TF-IDF + SVD + Ward 再吐出約 91 欄特徵——383 檔裡有 176 檔
# 訓練窗內不到 50 則。主題模型根本沒有樣本可估，產出的是噪音。
# 改成用「全部股票訓練窗內的文件」擬合一次，再對各檔的文件做 transform，
# 主題向量就有十萬筆等級的樣本支撐，個股特徵的意義變成
# 「今天這檔的新聞落在哪些（估得準的）主題上」。
POOLED_TEXT = False
POOLED_PCA = 10          # 逐檔版是 30；共用版可以估得準，但輸出維度要壓低
POOLED_CLUSTERS = 5      # 逐檔版是 10
WARD_MAX_SAMPLES = 5000

# 由 --select-k 開啟：每折只用訓練窗挑出關聯最強的 K 個特徵。
# 針對的是「特徵多而弱」的情況——600 列訓練樣本配上百個欄位，
# KNN／SVR／樹模型會被無關維度拖垮。挑選只看訓練窗，不碰驗證與測試窗。
SELECT_K = 0

_CORPUS: dict = {}       # source -> 全市場文件（已斷詞）
_POOLED: dict = {}       # (source, 月初) -> 已擬合的 TextFeatures


def step4_text_features(bundle: dict, train_end: pd.Timestamp, tokenizer: Tokenizer,
                        quiet: bool = False, sources=None) -> pd.DataFrame:
    """
    以 train_end 之前的文件擬合，再轉換全期間（每一折都要重新呼叫）。

    sources 可指定只用部分文本來源（例如只用 ptt），用來做單一資料軸的消融；
    None 代表全用。
    """
    calendar = bundle["px"].index
    frames = []
    use = sources if sources is not None else TEXT_SOURCE_FILTER
    for name, prefix in (("news", "txt_news_"), ("forum", "txt_forum_"),
                         ("ptt", "txt_ptt_")):
        if use is not None and name not in use:
            continue
        docs = bundle.get(name)
        if docs is None or docs.empty:
            continue
        if TEXT_USE_TFIDF:
            if POOLED_TEXT:
                tf = get_pooled_model(name, prefix, train_end, tokenizer)
            else:
                tf = TextFeatures(prefix, tokenizer)
                tf.fit(docs[docs["ts"] <= train_end])
            if tf is not None:
                out = tf.transform(docs, calendar)
                if not out.empty:
                    frames.append(out)
        # PTT 獨有的推／噓／留言數：直接彙總，不需擬合，因此沒有洩漏疑慮
        if name == "ptt":
            meta = ptt_reaction_features(docs, calendar)
            if not meta.empty:
                frames.append(meta)
            if not quiet:
                log.info(f"  {name:5s}：訓練窗 {len(train_docs)} 篇 → "
                         f"{tf.n_comp} 維主成分 + {tf.n_clu} 群，共 {out.shape[1]} 個特徵")
        elif not quiet:
            log.info(f"  {name:5s}：訓練窗文件不足，無法擬合文本模型")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=calendar)


# ============================================================================
# STEP 5　切分：walk-forward（訓練 → 驗證 → 測試，嚴格依時序）
# ============================================================================


@dataclass
class Fold:
    idx: int
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def describe(self) -> str:
        f = lambda i: f"{i[0]:%Y-%m-%d}~{i[-1]:%Y-%m-%d}"
        return (f"fold{self.idx}｜訓練 {f(self.train)}｜驗證 {f(self.val)}｜測試 {f(self.test)}")


def build_targets(px: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    t 列的 y = 未來 horizon 個交易日的累積報酬率（§3 RQ3：目標從價位改成報酬率）。

    horizon=1 是隔日沖；horizon=5 是週波段。注意 horizon>1 時相鄰樣本的目標會重疊
    （t 與 t+1 的預測期間有 h-1 天共用），所以下游的顯著性檢定要用 Newey–West
    修正自相關，回測也要改成「進場後持有 h 天、期間不重複進場」。
    """
    c = px["close"]
    fwd = c.shift(-horizon)
    ret = fwd / c - 1
    return pd.DataFrame({"close_t": c, "close_fwd": fwd, "ret_next": ret, "y": ret}, index=px.index)


def assemble(feature_frames: list, px: pd.DataFrame, eval_start: pd.Timestamp,
             horizon: int = 1) -> pd.DataFrame:
    """合併特徵與目標，丟掉暖機期與任何含缺值的列（不插補，避免製造假訊號）。"""
    feats = pd.concat([f for f in feature_frames if not f.empty], axis=1)
    df = pd.concat([feats, build_targets(px, horizon)], axis=1)
    valid = [c for c in feats.columns if df[c].notna().sum() > 0.5 * len(df)]
    df = df[valid + ["y", "close_t", "close_fwd", "ret_next"]].dropna()
    df = df[df.index >= eval_start]
    df.attrs["features"] = valid
    return df


def ptt_reaction_features(docs: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    PTT 的推文反應彙總（前綴 txt_ptt_）——新聞與同學會沒有這個訊號。

    推(push)＝認同、噓(boo)＝反對，推噓比是現成的情緒指標，不必靠情緒詞典，
    也不受斷詞品質影響。全部只是「當日已公開資料」的加總，不涉及任何擬合，
    因此沒有洩漏問題；時間對齊沿用同一條規則（13:30 後發文順延到下一個交易日）。
    """
    need = {"n_push", "n_boo"}
    if docs is None or docs.empty or not need.issubset(docs.columns):
        return pd.DataFrame(index=calendar)

    d = docs.copy()
    d["day"] = TextFeatures._to_trading_day(d["ts"], calendar)
    d = d.dropna(subset=["day"])
    if d.empty:
        return pd.DataFrame(index=calendar)

    g = d.groupby("day")
    out = pd.DataFrame(index=calendar)
    out["txt_ptt_push"] = g["n_push"].sum()
    out["txt_ptt_boo"] = g["n_boo"].sum()
    out["txt_ptt_posts"] = g.size()
    if "n_comments" in d.columns:
        out["txt_ptt_cmts"] = g["n_comments"].sum()
    out = out.fillna(0.0)

    tot = out["txt_ptt_push"] + out["txt_ptt_boo"]
    out["txt_ptt_ratio"] = ((out["txt_ptt_push"] - out["txt_ptt_boo"])
                            / tot.where(tot > 0)).fillna(0.0)
    # 熱度要看「相對於自己平常」，用回看視窗標準化；shift(1) 確保只用過去
    for col in ("txt_ptt_posts", "txt_ptt_push"):
        ma = out[col].rolling(20, min_periods=5).mean().shift(1)
        out[f"{col}_z20"] = ((out[col] - ma) / ma.replace(0, np.nan)).fillna(0.0)
    for col in ("txt_ptt_ratio", "txt_ptt_posts"):
        out[f"{col}_ma5"] = out[col].rolling(5, min_periods=1).mean()
    return out


def select_features(tr: pd.DataFrame, cols: list, k: int) -> list:
    """
    只用訓練窗，依與目標的 Spearman 絕對相關挑前 k 個特徵。

    用等級相關而非皮爾森：報酬率有厚尾，少數極端值會主導線性相關。
    整個計算只碰 tr，驗證與測試窗完全沒參與，所以不構成洩漏。
    """
    y = tr["y"].to_numpy(float)
    if len(y) < 30:
        return cols
    ry = pd.Series(y).rank().to_numpy()
    ry = ry - ry.mean()
    den_y = np.sqrt((ry ** 2).sum())
    scores = {}
    for c in cols:
        x = tr[c].to_numpy(float)
        if not np.isfinite(x).all() or np.nanstd(x) == 0:
            continue
        rx = pd.Series(x).rank().to_numpy()
        rx = rx - rx.mean()
        den = np.sqrt((rx ** 2).sum()) * den_y
        scores[c] = abs(float((rx * ry).sum()) / den) if den > 0 else 0.0
    if not scores:
        return cols
    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


def step5_split(index: pd.DatetimeIndex, quiet: bool = False,
                horizon: int = 1) -> list:
    banner("STEP 5", "切分：walk-forward（訓練 → 驗證 → 測試，嚴格依時序，含 purging）")
    folds, i, start = [], 0, 0
    # purging（Lopez de Prado）：t 列的標籤是「未來 horizon 天的報酬」，
    # 所以訓練窗最後 horizon 列的答案是用驗證期的價格算出來的，驗證窗最後
    # horizon 列的答案是用測試期的價格算出來的。後者尤其嚴重——Huber 元模型
    # 就是用驗證窗的預測與標籤擬合的，等於讓元模型看到測試期的結果。
    # h=60、驗證窗 120 天時，有一半的元模型訓練標籤會偷看測試期。
    # 修正方式：每個窗的尾端各切掉 horizon 列，讓標籤不跨越窗的邊界。
    # 把原始窗各放大 purge 天再切掉尾端，讓「實際可用」的樣本數固定為
    # TRAIN_DAYS／VAL_DAYS／TEST_DAYS，不隨 horizon 改變。否則 h 越長驗證窗
    # 越小（h=60 時只剩一半），跨週期的比較就會被元模型的樣本數差異汙染。
    purge = max(0, int(horizon))
    span = (TRAIN_DAYS + purge) + (VAL_DAYS + purge) + TEST_DAYS
    while start + span <= len(index):
        tr_end = start + TRAIN_DAYS + purge      # 訓練窗原始結束點
        va_end = tr_end + VAL_DAYS + purge       # 驗證窗原始結束點
        te_end = min(va_end + TEST_DAYS, len(index))
        train = index[0:tr_end - purge]          # 擴張窗（維持原設計），切掉尾端 purge 列
        val = index[tr_end:va_end - purge]
        test = index[va_end:te_end]
        if len(train) > 50 and len(val) > 10 and len(test) > 0:
            folds.append(Fold(i, train, val, test))
        i, start = i + 1, start + STEP_DAYS

    if not quiet:
        if purge > 1:
            log.info(f"  purging：h={purge}，訓練／驗證窗各放大 {purge} 天後切掉尾端 "
                     f"{purge} 列，實際可用樣本維持 {TRAIN_DAYS}／{VAL_DAYS}／{TEST_DAYS}")
        log.info(f"  可用 {len(index)} 個交易日，視窗 {TRAIN_DAYS}／{VAL_DAYS}／{TEST_DAYS}，"
                 f"每次前進 {STEP_DAYS} 日 → {len(folds)} 折")
        for f in folds:
            log.info(f"    {f.describe()}")
        log.info("  測試窗永遠在最後，調參絕不會看到測試期之後的資料（修正原論文的切分順序倒置）")
    return folds


# ============================================================================
# STEP 6　分析層 A′：8 個基礎模型 → Huber 堆疊集成
# ============================================================================


def build_base_models(seed: int = 42) -> dict:
    scaled = lambda est: Pipeline([("scaler", StandardScaler()), ("est", est)])
    return {
        "ridge": scaled(Ridge(alpha=1.0)),
        "lasso": scaled(Lasso(alpha=1e-4, max_iter=20000, random_state=seed)),
        "elasticnet": scaled(ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=20000, random_state=seed)),
        "svr": scaled(SVR(kernel="rbf", C=1.0, epsilon=0.005, gamma="scale")),
        "knn": scaled(KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                               max_features="sqrt", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                           max_features="sqrt", n_jobs=-1, random_state=seed),
        "gbrt": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                          subsample=0.8, random_state=seed),
    }


class StackedEnsemble:
    """
    三段式堆疊（沿用原論文的 Huber meta-regressor）：
        1. 8 個基礎模型用「訓練窗」擬合
        2. 在「驗證窗」產生預測矩陣，Huber meta 用這個矩陣擬合
        3. 「測試窗」只做預測

    刻意不用交叉驗證的 out-of-fold：在時序資料上那會讓 meta 看到未來樣本。
    """

    def __init__(self):
        self.base = build_base_models()
        self.names = []

    def fit(self, X_tr, y_tr, X_va, y_va):
        Xtr, Xva = np.asarray(X_tr, float), np.asarray(X_va, float)
        for name, model in self.base.items():
            try:
                model.fit(Xtr, np.asarray(y_tr, float))
                self.names.append(name)
            except Exception as e:
                log.warning(f"    基礎模型 {name} 擬合失敗：{e}")
        if not self.names:
            raise RuntimeError("所有基礎模型都擬合失敗")
        P = self._base_matrix(Xva)
        self.meta = Pipeline([("scaler", StandardScaler()),
                              ("est", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000))])
        self.meta.fit(P, np.asarray(y_va, float))
        return self

    def _base_matrix(self, X):
        return np.column_stack([self.base[n].predict(X) for n in self.names])

    def predict(self, X):
        return self.meta.predict(self._base_matrix(np.asarray(X, float)))

    def predict_base(self, X) -> pd.DataFrame:
        return pd.DataFrame(self._base_matrix(np.asarray(X, float)), columns=self.names)

    def weights(self) -> pd.Series:
        """Huber meta 的係數 = 各基礎模型的貢獻權重（原框架的可解釋性賣點）。"""
        coef = getattr(self.meta.named_steps["est"], "coef_", None)
        if coef is None:
            return pd.Series(dtype=float)
        return pd.Series(np.ravel(coef), index=self.names).sort_values(key=np.abs, ascending=False)


# ============================================================================
# STEP 7　評估層：天真基準 → DM 檢定 → 含成本回測
# ============================================================================


def directional_accuracy(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(yt) > 1e-12          # 實際報酬率為 0 的日子不列入分母
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask]))) if mask.any() else float("nan")


def price_mape(close_t, ret_true, ret_pred) -> float:
    """把報酬率預測還原成價位再算 MAPE，才能跟原論文的 0.93% 對照。"""
    p_true = np.asarray(close_t, float) * (1 + np.asarray(ret_true, float))
    p_pred = np.asarray(close_t, float) * (1 + np.asarray(ret_pred, float))
    mask = np.abs(p_true) > 1e-12
    return float(np.mean(np.abs((p_true[mask] - p_pred[mask]) / p_true[mask])) * 100)


def diebold_mariano(y_true, pred1, pred2, loss: str = "mse", h: int = 1):
    """
    H0：兩組預測的期望損失相同；stat < 0 表示 pred1 較佳。
    含 Harvey–Leybourne–Newbold 小樣本修正。用它取代成對 t 檢定，
    因為後者忽略預測誤差的序列相關（原論文顯著性主張站不住腳的原因）。
    """
    yt, p1, p2 = (np.asarray(x, float) for x in (y_true, pred1, pred2))
    ok = np.isfinite(yt) & np.isfinite(p1) & np.isfinite(p2)
    yt, p1, p2 = yt[ok], p1[ok], p2[ok]
    if loss == "mse":
        d = (yt - p1) ** 2 - (yt - p2) ** 2
    else:                                   # 方向損失（1 - 命中）
        keep = np.abs(yt) > 1e-12
        yt, p1, p2 = yt[keep], p1[keep], p2[keep]
        d = (np.sign(yt) != np.sign(p1)).astype(float) - (np.sign(yt) != np.sign(p2)).astype(float)

    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    d_bar = float(np.mean(d))
    # Newey–West 長期變異數：h>1（波段、目標重疊）時一定要做，否則標準誤會低估
    var = float(np.mean((d - d_bar) ** 2))
    for k in range(1, h):
        var += 2 * float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
    if var <= 0:
        return float("nan"), float("nan")
    stat = d_bar / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def _verdict(stat: float, p: float) -> str:
    """DM 檢定的判讀：stat < 0 表示模型較佳，> 0 表示基準較佳。"""
    if not np.isfinite(stat) or not np.isfinite(p) or p >= 0.05:
        return "未達顯著（兩者無法區分）"
    return "模型顯著較佳" if stat < 0 else "基準顯著較佳（模型較差）"


ROUND_TRIP_COST = 2 * FEE_RATE + TAX_RATE      # 一次完整進出的成本率


def backtest(test: pd.DataFrame, pred: np.ndarray, horizon: int = 1,
             threshold: float = 0.0) -> dict:
    """
    long_flat 策略（現股可行，不放空）：

        預測報酬 > 門檻 → 進場，並至少持有 horizon 個交易日
        最短持有期滿後，訊號仍為真就續抱（不會多付一次來回成本），轉為假才出場

    horizon=1 時就是原本的隔日沖行為。成本只在真正進出場時計：
    進場付手續費 0.1425%，出場付手續費 0.1425% + 證交稅 0.3%。
    """
    sig = np.asarray(pred, float)
    # 逐日損益要用「隔日報酬」，不能用 h 日累積報酬（否則權益曲線會重複計算）
    daily = test["close_t"].pct_change().shift(-1).fillna(0.0).to_numpy(float)
    n = len(sig)

    pos = np.zeros(n)
    cost = np.zeros(n)
    in_pos, hold_until, trades, entry_equity = False, -1, [], None
    equity = 1.0

    for i in range(n):
        if not in_pos and sig[i] > threshold:              # 進場
            in_pos, hold_until = True, i + horizon
            cost[i] += FEE_RATE
            entry_equity = equity
        elif in_pos and i >= hold_until and sig[i] <= threshold:   # 期滿且訊號轉弱 → 出場
            in_pos = False
            cost[i] += FEE_RATE + TAX_RATE
            trades.append(equity * (1 - FEE_RATE - TAX_RATE) / entry_equity - 1)

        pos[i] = 1.0 if in_pos else 0.0
        equity *= (1 + pos[i] * daily[i] - cost[i])

    if in_pos:                                             # 最後一天強制平倉
        cost[-1] += FEE_RATE + TAX_RATE
        equity *= (1 - FEE_RATE - TAX_RATE)
        trades.append(equity / entry_equity - 1)

    net = pos * daily - cost
    curve = np.cumprod(1 + net)
    peak = np.maximum.accumulate(curve)
    tr = np.array(trades) if trades else np.array([])
    sharpe = (float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(TRADING_DAYS))
              if len(net) > 1 and np.std(net, ddof=1) > 0 else float("nan"))
    return {
        "n_days": n,
        "n_trades": len(tr),
        "exposure": float(np.mean(pos)),
        "avg_hold_days": float(np.sum(pos) / len(tr)) if len(tr) else float("nan"),
        "win_rate": float(np.mean(tr > 0)) if len(tr) else float("nan"),
        "avg_trade": float(np.mean(tr)) if len(tr) else float("nan"),
        "total_return": float(curve[-1] - 1),
        "sharpe": sharpe,
        "max_drawdown": float(np.min(curve / peak - 1)),
        "cost_drag": float(np.sum(cost)),
    }


def buy_and_hold(test: pd.DataFrame, horizon: int = 1) -> dict:
    """買進持有：測試期第一天買、最後一天賣，成本收一次來回。"""
    c = test["close_t"].to_numpy(float)
    if len(c) < 2:
        return {"total_return": float("nan"), "sharpe": float("nan")}
    total = c[-1] / c[0] - 1 - ROUND_TRIP_COST
    daily = np.diff(c) / c[:-1]
    sharpe = (float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS))
              if np.std(daily, ddof=1) > 0 else float("nan"))
    return {"total_return": float(total), "sharpe": sharpe}


def step7_evaluate(pred_df: pd.DataFrame, close_full: pd.Series, horizon: int = 1,
                   quiet: bool = False, bench: pd.Series | None = None) -> dict:
    banner("STEP 7", f"評估層：天真基準對照 → DM 檢定 → 含成本回測"
                     f"（預測期間 {horizon} 日）")

    yt, yp = pred_df["ret_next"].to_numpy(float), pred_df["pred"].to_numpy(float)
    naive = np.zeros(len(pred_df))                        # 預測「持平」：報酬率恆為 0
    always_long = np.full(len(pred_df), 1e-8)             # 恆預測上漲：DA 的正確對照組
    ma5 = (close_full.rolling(5).mean().reindex(pred_df.index) / pred_df["close_t"] - 1).to_numpy()

    res = {
        "n": len(pred_df),
        "horizon": horizon,
        "DA": directional_accuracy(yt, yp),
        "DA_always_long": directional_accuracy(yt, always_long),
        "DA_ma5": directional_accuracy(yt, ma5),
        "MAPE": price_mape(pred_df["close_t"], yt, yp),
        "MAPE_naive": price_mape(pred_df["close_t"], yt, naive),
        "MAPE_ma5": price_mape(pred_df["close_t"], yt, ma5),
    }
    # h>1 時相鄰樣本的預測期間重疊，DM 必須用 Newey–West 修正
    res["DM_mse_stat"], res["DM_mse_p"] = diebold_mariano(yt, yp, naive, "mse", h=horizon)
    res["DM_dir_stat"], res["DM_dir_p"] = diebold_mariano(yt, yp, always_long, "sign", h=horizon)

    # 穩健性：只取每隔 horizon 天一筆的不重疊子樣本再算一次方向準確率
    if horizon > 1:
        sub = slice(None, None, horizon)
        res["DA_nonoverlap"] = directional_accuracy(yt[sub], yp[sub])
        res["DA_nonoverlap_long"] = directional_accuracy(yt[sub], always_long[sub])
        res["n_nonoverlap"] = len(yt[sub])

    bt = backtest(pred_df, yp, horizon)
    bh = buy_and_hold(pred_df, horizon)
    res.update({f"bt_{k}": v for k, v in bt.items()})
    res["bt_sharpe_buyhold"] = bh["sharpe"]
    res["bt_return_buyhold"] = bh["total_return"]
    # 大盤對照：投資人真正的替代方案是買指數，不是抱這一檔
    res.update(benchmark_stats(bench, pred_df.index, TRADING_DAYS))
    res["excess_vs_bench"] = res["bt_total_return"] - res["bench_return"]

    if not quiet:
        log.info(f"  測試樣本　{res['n']} 日（{pred_df.index.min():%Y-%m-%d} ~ "
                 f"{pred_df.index.max():%Y-%m-%d}）"
                 + (f"，其中不重疊子樣本 {res['n_nonoverlap']} 筆" if horizon > 1 else ""))
        log.info("")
        log.info("  ── 方向準確率（主指標）──")
        log.info(f"    模型 DA          {res['DA']:.4f}")
        log.info(f"    always_long      {res['DA_always_long']:.4f}　← 正確的對照組（樣本期上漲日比例）")
        log.info(f"    5 日均線         {res['DA_ma5']:.4f}")
        if horizon > 1:
            log.info(f"    不重疊子樣本     模型 {res['DA_nonoverlap']:.4f}　"
                     f"vs always_long {res['DA_nonoverlap_long']:.4f}"
                     f"（{res['n_nonoverlap']} 筆，排除目標重疊的影響）")
        log.info(f"    DM 方向檢定      stat={res['DM_dir_stat']:.3f}　p={res['DM_dir_p']:.4f}"
                 f"　{_verdict(res['DM_dir_stat'], res['DM_dir_p'])}")
        log.info("")
        log.info("  ── 價位誤差（輔指標）──")
        log.info(f"    模型 MAPE        {res['MAPE']:.4f}%")
        log.info(f"    naive MAPE       {res['MAPE_naive']:.4f}%　← 沒有這一列，MAPE 無從判讀")
        log.info(f"    DM 誤差檢定      stat={res['DM_mse_stat']:.3f}　p={res['DM_mse_p']:.4f}"
                 f"　{_verdict(res['DM_mse_stat'], res['DM_mse_p'])}")
        log.info("")
        log.info("  ── 含成本回測 ──")
        log.info(f"    模型　　　  總報酬 {bt['total_return']:+.2%}　夏普 {bt['sharpe']:.3f}　"
                 f"最大回撤 {bt['max_drawdown']:.2%}")
        log.info(f"    　　　　　  交易 {bt['n_trades']} 次　勝率 {bt['win_rate']:.2%}　"
                 f"平均每筆 {bt['avg_trade']:+.3%}　平均持有 {bt['avg_hold_days']:.1f} 日　"
                 f"持倉比重 {bt['exposure']:.1%}")
        log.info(f"    買進持有　  總報酬 {bh['total_return']:+.2%}　夏普 {bh['sharpe']:.3f}")
        log.info(f"    大盤加權　  總報酬 {res['bench_return']:+.2%}　夏普 {res['bench_sharpe']:.3f}"
                 f"　← 超額報酬 {res['excess_vs_bench']:+.2%}")
        log.info(f"    成本拖累　  累計 {bt['cost_drag']:.2%}"
                 f"（單次來回 {ROUND_TRIP_COST:.4%}，實際進出 {bt['n_trades']} 趟）")
    return res


# ============================================================================
# 主流程：把七個步驟串起來
# ============================================================================


def train_one_stock(code: str, feature_groups: list, start: str, end: str,
                    horizon: int = 1, quiet: bool = False,
                    bundle: dict | None = None) -> dict | None:
    """個股或類股指數都走這一條流程；差別只在 bundle 怎麼來的。"""
    if bundle is None:
        bundle = step1_load(code, start, end, quiet)
    px, eval_start = bundle["px"], bundle["eval_start"]

    frames = []
    if "price" in feature_groups:
        frames.append(step2_price_features(px, quiet))
    if "chip" in feature_groups:
        frames.append(step3_chip_features(bundle["chips"], px, quiet))

    _srcs = TEXT_SOURCE_FILTER or TEXT_SOURCES
    # 注意不能寫成 bundle.get(k) or []：DataFrame 的真值判斷會直接丟 ValueError
    need_text = "text" in feature_groups and any(
        len(bundle[k]) for k in _srcs if bundle.get(k) is not None)
    tokenizer = Tokenizer() if (need_text and TEXT_USE_TFIDF) else None
    if need_text:
        # 斷詞只做一次，後面所有折與預測期間共用（見 TextFeatures.ensure_tokens）
        if TEXT_USE_TFIDF:
            for key in _srcs:
                if bundle.get(key) is not None and len(bundle[key]):
                    bundle[key] = TextFeatures.ensure_tokens(bundle[key], tokenizer)
    if need_text and not quiet:
        banner("STEP 4", f"文本特徵：斷詞（{tokenizer.kind}）→ TF-IDF → PCA → Ward 分群")
        log.info("  注意：這三樣只用該折訓練窗的文件擬合，所以每一折都會重算一次")

    # 先用全期間算一次，只為了決定有效日期範圍與折的切法
    probe_text = (step4_text_features(bundle, px.index[-1], tokenizer, quiet=True)
                  if need_text else pd.DataFrame(index=px.index))
    panel = assemble(frames + [probe_text], px, eval_start, horizon)
    if panel.empty:
        log.warning(f"  [{code}] 合併後沒有可用樣本")
        return None

    folds = step5_split(panel.index, quiet, horizon=horizon)
    if not folds:
        log.warning(f"  [{code}] 樣本不足，切不出任何一折")
        return None

    banner("STEP 6", f"分析層 A′：8 個基礎模型 → Huber 堆疊集成"
                      f"（預測目標：未來 {horizon} 日累積報酬率）")
    preds, weight_rows = [], []
    for fold in folds:
        if need_text:
            # 文本模型折內重擬合（防洩漏的關鍵一步）
            text_fold = step4_text_features(bundle, fold.train[-1], tokenizer, quiet=True)
            fold_panel = assemble(frames + [text_fold], px, eval_start, horizon)
        else:
            fold_panel = panel

        cols = fold_panel.attrs["features"]
        tr = fold_panel.loc[fold_panel.index.intersection(fold.train)]
        va = fold_panel.loc[fold_panel.index.intersection(fold.val)]
        te = fold_panel.loc[fold_panel.index.intersection(fold.test)]
        if len(tr) < 100 or len(va) < 20 or len(te) < 10:
            log.warning(f"  fold{fold.idx} 樣本不足（{len(tr)}/{len(va)}/{len(te)}），跳過")
            continue

        if SELECT_K and len(cols) > SELECT_K:
            cols = select_features(tr, cols, SELECT_K)

        model = StackedEnsemble().fit(tr[cols], tr["y"], va[cols], va["y"])
        out = te[["ret_next", "close_t", "close_fwd"]].copy()
        out["pred"] = model.predict(te[cols])
        out["fold"] = fold.idx
        preds.append(out)

        w = model.weights()
        weight_rows.append(pd.DataFrame({"fold": fold.idx, "model": w.index, "weight": w.to_numpy()}))
        if not quiet:
            top = "、".join(f"{n} {v:+.3f}" for n, v in w.head(3).items())
            log.info(f"  fold{fold.idx}：特徵 {len(cols)} 個｜訓練 {len(tr)}／驗證 {len(va)}／測試 {len(te)} 日"
                     f"｜權重前三 {top}")

    if not preds:
        return None

    pred_df = pd.concat(preds)
    result = step7_evaluate(pred_df, px["close"], horizon, quiet, bench=load_twii())
    result["code"] = code
    result["features"] = ",".join(feature_groups)
    return {"summary": result, "predictions": pred_df,
            "weights": pd.concat(weight_rows) if weight_rows else pd.DataFrame(),
            "folds": [f.describe() for f in folds]}


PROGRESS_FILE = "progress.jsonl"


def read_progress(out_dir: Path) -> list:
    """讀出已完成的結果列。用 JSON Lines 而非 CSV 追加，是因為不同預測期間的
    欄位不一樣（h>1 會多出不重疊子樣本的欄位），CSV 追加會因欄數不符而壞掉。"""
    path = out_dir / PROGRESS_FILE
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue                      # 中斷時可能留下半行，跳過即可
    return rows


def load_done(out_dir: Path) -> set:
    """已完成的「標的 × 預測期間」組合，用於斷點續跑。"""
    return {(str(r.get("code")), int(r.get("horizon", 0))) for r in read_progress(out_dir)}


def append_summary(out_dir: Path, row: dict) -> None:
    """每跑完一個組合就立刻追加一行，中途斷掉也不會前功盡棄。"""
    clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
    with open(out_dir / PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分鐘"
    return f"{seconds / 3600:.1f} 小時"


def _fmt(v, kind="f4"):
    """報告用的數字格式；缺值一律印成 -，不要讓 nan 污染表格。"""
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    if kind == "pct":
        return f"{v:+.2%}"
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.4f}"


def _rows_table(df, cols, headers) -> list:
    """把 DataFrame 轉成 Markdown 表格的字串列。"""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c, kind in cols:
            cells.append(_fmt(r.get(c), kind) if kind else str(r.get(c, "")))
        out.append("| " + " | ".join(cells) + " |")
    return out


def write_report(summary: pd.DataFrame, out_dir: Path, args, horizons: list) -> Path:
    """
    把彙總結果寫成一份可讀的 report.md：產業類股在前、個股在後。

    主控台只留最精簡的數字，細節看這份檔案 —— 幾百個標的的逐步驟輸出
    在終端機裡根本讀不完。
    """
    L = []
    L.append("# 台股 DAViS 訓練結果報告")
    L.append("")
    L.append(f"- 產生時間：{datetime.now():%Y-%m-%d %H:%M}")
    L.append(f"- 研究期間：{args.start} ~ {args.end}")
    L.append(f"- 特徵組：{'＋'.join(args.features)}")
    L.append(f"- 預測期間：{'、'.join('隔日沖 (h=1)' if h == 1 else f'持有 {h} 日 (h={h})' for h in horizons)}")
    L.append(f"- Walk-forward：訓練 {TRAIN_DAYS}／驗證 {VAL_DAYS}／測試 {TEST_DAYS} 日，"
             f"每次前進 {STEP_DAYS} 日")
    L.append(f"- 交易成本：手續費 {FEE_RATE:.4%}×2 ＋ 證交稅 {TAX_RATE:.2%}"
             f"（單次來回 {ROUND_TRIP_COST:.4%}）")
    L.append("")
    L.append("> **判讀規則**：DA 要跟 `always_long`（樣本期上漲比例）比，不是跟 naive 比；"
             "差異必須 DM 檢定 p < 0.05 才算數；報酬要跟大盤超額比，"
             "跑幾十個標的時還要留意多重檢定（28 個標的在 α=0.05 下本來就有約 1.4 個偽陽性）。")
    L.append("")

    cols = [("code", None), ("n", "int"), ("DA", "f4"), ("DA_always_long", "f4"),
            ("DM_dir_p", "f4"), ("bt_n_trades", "int"), ("bt_total_return", "pct"),
            ("bt_sharpe", "f3"), ("bt_return_buyhold", "pct"),
            ("bench_return", "pct"), ("excess_vs_bench", "pct")]
    headers = ["標的", "測試日", "DA", "always_long", "DM p", "交易次數",
               "模型報酬", "夏普", "買進持有", "大盤", "超額"]

    # ---------- 一、產業類股 ----------
    sec = summary[summary["level"] == "sector"] if "level" in summary.columns else summary.iloc[0:0]
    L.append(f"## 一、產業類股（{sec['code'].nunique() if len(sec) else 0} 個）")
    L.append("")
    if len(sec):
        for h in horizons:
            part = sec[sec["horizon"] == h].sort_values("excess_vs_bench", ascending=False)
            if part.empty:
                continue
            L.append(f"### {'隔日沖' if h == 1 else f'週波段（持有 {h} 日）'}")
            L.append("")
            L += _rows_table(part, cols, headers)
            L.append("")
            beat = (part["DA"] > part["DA_always_long"]).mean()
            sig = ((part["DM_dir_p"] < 0.05) & (part["DA"] > part["DA_always_long"])).sum()
            bad = ((part["DM_dir_p"] < 0.05) & (part["DA"] < part["DA_always_long"])).sum()
            L.append(f"小結：DA 贏過 always_long 的有 {beat:.0%}；"
                     f"顯著較佳 {sig} 個、顯著較差 {bad} 個；"
                     f"超額報酬為正的有 {(part['excess_vs_bench'] > 0).sum()}／{len(part)} 個。")
            L.append("")
    else:
        L.append("（本次未跑產業類股層級，加 `--level sector` 或 `--level both`）")
        L.append("")

    # ---------- 二、個股 ----------
    stk = summary[summary["level"] == "stock"] if "level" in summary.columns else summary
    L.append(f"## 二、個股（{stk['code'].nunique() if len(stk) else 0} 檔）")
    L.append("")
    if len(stk) and "sector" in stk.columns and stk["sector"].nunique() > 1:
        L.append("### 依產業彙總")
        L.append("")
        for h in horizons:
            part = stk[stk["horizon"] == h]
            if part.empty:
                continue
            g = (part.groupby("sector")
                 .agg(n_codes=("code", "nunique"), mean_da=("DA", "mean"),
                      mean_long=("DA_always_long", "mean"),
                      mean_excess=("excess_vs_bench", "mean"))
                 .reset_index())
            g["beat_rate"] = (part.assign(w=part["DA"] > part["DA_always_long"])
                              .groupby("sector")["w"].mean().to_numpy())
            g = g.sort_values("mean_excess", ascending=False)
            L.append(f"**{'隔日沖' if h == 1 else f'週波段（持有 {h} 日）'}**")
            L.append("")
            L += _rows_table(g, [("sector", None), ("n_codes", "int"), ("mean_da", "f4"),
                                 ("mean_long", "f4"), ("beat_rate", "f4"), ("mean_excess", "pct")],
                             ["產業", "檔數", "平均 DA", "always_long", "贏過基準比例", "平均超額"])
            L.append("")

    if len(stk):
        L.append("### 個股明細")
        L.append("")
        for h in horizons:
            part = stk[stk["horizon"] == h].sort_values("excess_vs_bench", ascending=False)
            if part.empty:
                continue
            L.append(f"**{'隔日沖' if h == 1 else f'週波段（持有 {h} 日）'}**"
                     f"（依超額報酬排序，共 {len(part)} 檔）")
            L.append("")
            L += _rows_table(part, cols, headers)
            L.append("")

    # ---------- 三、整體判讀 ----------
    L.append("## 三、整體判讀")
    L.append("")
    for h in horizons:
        part = summary[summary["horizon"] == h]
        if part.empty:
            continue
        beat = (part["DA"] > part["DA_always_long"]).mean()
        sig_good = ((part["DM_dir_p"] < 0.05) & (part["DA"] > part["DA_always_long"])).sum()
        sig_bad = ((part["DM_dir_p"] < 0.05) & (part["DA"] < part["DA_always_long"])).sum()
        pos = (part["excess_vs_bench"] > 0).sum()
        L.append(f"- **{'隔日沖' if h == 1 else f'持有 {h} 日'}**（{len(part)} 個標的）："
                 f"平均 DA {part['DA'].mean():.4f} vs always_long {part['DA_always_long'].mean():.4f}；"
                 f"DA 贏過基準 {beat:.0%}；方向顯著較佳 {sig_good} 個、顯著較差 {sig_bad} 個；"
                 f"超額報酬為正 {pos}／{len(part)} 個；"
                 f"平均夏普 {part['bt_sharpe'].mean():.3f} vs 買進持有 {part['bt_sharpe_buyhold'].mean():.3f}")
    L.append("")
    L.append(f"完整數值：`summary.csv`；逐日預測：`predictions_*.csv`；"
             f"堆疊權重：`meta_weights_*.csv`。")

    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="台股 DAViS 訓練流程（依提案步驟線性執行）")
    ap.add_argument("--level", default="both", choices=["stock", "sector", "both"],
                    help="分析層級：both=個股＋產業類股（預設）、stock=只跑個股、sector=只跑產業類股")
    ap.add_argument("--stock", default=None,
                    help="要訓練的單一股票代號；不指定就跑股票清單裡的全部標的")
    ap.add_argument("--stocks", nargs="*", default=None, help="一次跑多檔")
    ap.add_argument("--sectors", nargs="*", default=None,
                    help="要跑的產業（預設全部 28 個），例如 --sectors 半導體 金融 航運")
    ap.add_argument("--features", nargs="+", default=["price", "chip", "text"],
                    choices=["price", "chip", "text"], help="要使用的特徵組")
    ap.add_argument("--pooled-text", action="store_true",
                    help="主題模型改用全市場共用語料擬合（而非逐檔），修正小樣本問題")
    ap.add_argument("--select-k", type=int, default=0,
                    help="每折只保留訓練窗關聯最強的 K 個特徵（0=不挑選）")
    ap.add_argument("--no-tfidf", action="store_true",
                    help="不做 TF-IDF／分群，只保留 PTT 推噓反應特徵")
    ap.add_argument("--text-sources", nargs="+", default=None,
                    choices=list(TEXT_SOURCES),
                    help="只用部分文本來源（做單一資料軸消融時用），預設三個全用")
    ap.add_argument("--horizon", type=int, nargs="+", default=HORIZONS,
                    help="預測期間（交易日），可給多個一起跑。預設 1 5"
                         "（1=隔日沖、5=週波段）")
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    ap.add_argument("--quiet", action="store_true", help="只印最後結果")
    ap.add_argument("--out", default=None, help="輸出目錄（預設 results/train_{時間}）")
    args = ap.parse_args(argv)
    global TEXT_SOURCE_FILTER, TEXT_USE_TFIDF, POOLED_TEXT, SELECT_K
    if args.pooled_text:
        POOLED_TEXT = True
        log.info("  主題模型：全市場共用語料（逐檔樣本太少，見 POOLED_TEXT 註解）")
    if args.select_k:
        SELECT_K = args.select_k
        log.info(f"  折內特徵挑選：每折只留訓練窗關聯最強的 {args.select_k} 個特徵")
    if args.no_tfidf:
        TEXT_USE_TFIDF = False
        log.info("  已關閉 TF-IDF／分群，文本側只保留 PTT 推噓反應特徵")
    if args.text_sources:
        TEXT_SOURCE_FILTER = tuple(args.text_sources)
        log.info(f"  文本來源限定為：{'、'.join(args.text_sources)}")

    codes = args.stocks or ([args.stock] if args.stock else all_stock_codes())
    out_dir = Path(args.out) if args.out else RESULTS / f"train_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = sorted(set(args.horizon))
    hz_label = "、".join("隔日沖" if h == 1 else f"持有 {h} 日" for h in horizons)
    log.info(f"研究期間 {args.start} ~ {args.end}｜特徵組 {'＋'.join(args.features)}｜"
             f"預測期間 {hz_label}｜層級 {args.level}")

    # 要跑哪些「標的」：個股用代號，類股用產業名（並帶著成分股清單）
    jobs = []
    if args.level in ("stock", "both"):
        sec_of = stock_to_sector()
        jobs += [{"kind": "stock", "name": c, "sector": sec_of.get(c, ""), "bundle": None}
                 for c in codes]
    if args.level in ("sector", "both"):
        imap = load_industry_map()
        wanted = args.sectors or list(imap)
        for sec in wanted:
            if sec not in imap:
                log.warning(f"沒有這個產業：{sec}（可用：{'、'.join(list(imap)[:6])} …）")
                continue
            jobs.append({"kind": "sector", "name": sec, "sector": sec, "members": imap[sec]})

    log.info(f"標的：{'、'.join(j['name'] for j in jobs[:10])}"
             f"{f' …共 {len(jobs)} 個' if len(jobs) > 10 else ''}")

    total_runs = len(jobs) * len(horizons)
    done_set = load_done(out_dir)
    # 每次訓練的粗估秒數：純價量／籌碼很快，文本要重跑 TF-IDF + 分群所以貴很多
    per_run = 30 if "text" in args.features else 8
    if any(j["kind"] == "sector" for j in jobs):
        per_run *= 1.6                     # 類股要載入十幾檔成分股、文本也更多
    remain = max(total_runs - len(done_set), 0)
    log.info(f"共 {len(jobs)} 個標的 × {len(horizons)} 種預測期間 = {total_runs} 次訓練"
             f"{f'（已完成 {len(done_set)} 次，續跑剩 {remain} 次）' if done_set else ''}"
             f"｜預估還要 {fmt_eta(remain * per_run)}")
    if done_set:
        log.info(f"斷點續跑：沿用 {out_dir}／{PROGRESS_FILE} 的既有結果")

    if len(jobs) > 3 and not args.quiet:
        log.info(f"標的數 {len(jobs)} 個，自動切換為精簡輸出（逐步驟細節只在單一標的時顯示）")
        args.quiet = True

    rows, all_preds, done, t0 = [], [], 0, datetime.now()
    for job in jobs:
        name = job["name"]
        if all((name, h) in done_set for h in horizons):
            done += len(horizons)
            continue                       # 這個標的的所有週期都跑過了
        # 資料只讀一次，多個預測期間共用（產業類股要載入十幾檔成分股，重讀很浪費）
        try:
            bundle = (step1_load_sector(name, job["members"], args.start, args.end, args.quiet)
                      if job["kind"] == "sector"
                      else step1_load(name, args.start, args.end, args.quiet))
        except SystemExit as e:
            log.warning(str(e))
            continue
        except Exception as e:
            log.warning(f"[{name}] 讀取資料失敗：{type(e).__name__}: {e}")
            continue

        for hz in horizons:
            done += 1
            if (name, hz) in done_set:
                continue
            try:
                got = train_one_stock(name, args.features, args.start, args.end,
                                      hz, args.quiet, bundle=bundle)
            except Exception as e:
                log.warning(f"[{name}] horizon={hz} 訓練失敗：{type(e).__name__}: {e}")
                continue
            if not got:
                continue
            summary = got["summary"]
            summary["level"] = job["kind"]
            summary["sector"] = job["sector"]
            if job["kind"] == "sector":
                summary["n_members"] = len(bundle.get("members", []))
            rows.append(summary)
            append_summary(out_dir, summary)          # 立刻落地，斷了也不會白跑
            all_preds.append(got["predictions"].assign(code=name, horizon=hz))
            safe = re.sub(r"[^\w\u4e00-\u9fff]", "_", name)
            got["predictions"].to_csv(out_dir / f"predictions_{safe}_h{hz}.csv",
                                      encoding="utf-8-sig")
            if not got["weights"].empty:
                got["weights"].to_csv(out_dir / f"meta_weights_{safe}_h{hz}.csv",
                                      index=False, encoding="utf-8-sig")
            if args.quiet and done % 10 == 0:
                spent = (datetime.now() - t0).total_seconds()
                rate = spent / max(len(rows), 1)
                left = (total_runs - done) * rate
                log.info(f"  進度 {done}/{total_runs}　已花 {fmt_eta(spent)}　"
                         f"預估剩餘 {fmt_eta(left)}")

    if not rows:
        log.error("沒有任何股票訓練成功")
        return 1

    # 彙總以落地的進度檔為準，這樣續跑時也會把先前的結果一起算進來
    progress = read_progress(out_dir)
    summary = pd.DataFrame(progress) if progress else pd.DataFrame(rows)
    summary = summary.drop_duplicates(subset=["code", "horizon"], keep="last")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "config.json").write_text(json.dumps({
        "period": [args.start, args.end], "features": args.features, "stocks": codes,
        "horizon": args.horizon,
        "walk_forward": {"train": TRAIN_DAYS, "val": VAL_DAYS,
                         "test": TEST_DAYS, "step": STEP_DAYS},
        "cost": {"fee": FEE_RATE, "tax": TAX_RATE},
        "base_models": list(BASE_MODELS), "meta": "HuberRegressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    banner("完成", "跨股票彙總")
    cols = ["level", "code", "sector", "horizon", "n", "DA", "DA_always_long", "DM_dir_p",
            "bt_n_trades", "bt_win_rate", "bt_total_return", "bt_sharpe",
            "bt_return_buyhold", "bt_sharpe_buyhold", "bench_return", "excess_vs_bench"]
    cols = [c for c in cols if c in summary.columns]
    log.info(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # 預測期間對照：隔日沖 vs 週波段，哪個比較有搞頭
    if summary["horizon"].nunique() > 1:
        log.info("\n── 預測期間對照 ──")
        hz = (summary.groupby(["level", "horizon"])
              .agg(n_targets=("code", "nunique"), mean_da=("DA", "mean"),
                   mean_always_long=("DA_always_long", "mean"),
                   mean_sharpe=("bt_sharpe", "mean"),
                   sharpe_buyhold=("bt_sharpe_buyhold", "mean"),
                   mean_excess=("excess_vs_bench", "mean")))
        hz["beat_rate"] = (summary.assign(w=summary["DA"] > summary["DA_always_long"])
                           .groupby(["level", "horizon"])["w"].mean())
        hz.columns = ["標的數", "平均DA", "對照always_long", "平均夏普",
                      "買進持有夏普", "平均超額", "贏過基準比例"]
        log.info(hz.to_string(float_format=lambda x: f"{x:.4f}"))

    # 個股結果按產業分組：呼應原論文「來源效益依產業而異」的發現
    stock_rows = summary[summary["level"] == "stock"] if "level" in summary.columns else summary
    if len(stock_rows) > 3 and "sector" in stock_rows.columns and stock_rows["sector"].nunique() > 1:
        log.info("\n── 個股結果依產業分組 ──")
        grp = (stock_rows.groupby("sector")
               .agg(n_codes=("code", "nunique"), mean_da=("DA", "mean"),
                    mean_always_long=("DA_always_long", "mean"),
                    mean_sharpe=("bt_sharpe", "mean")))
        grp["beat_rate"] = (stock_rows.assign(w=stock_rows["DA"] > stock_rows["DA_always_long"])
                            .groupby("sector")["w"].mean())
        grp.columns = ["檔數", "平均DA", "對照always_long", "平均夏普", "贏過基準比例"]
        log.info(grp.sort_values("平均DA", ascending=False)
                 .to_string(float_format=lambda x: f"{x:.4f}"))

    report = write_report(summary, out_dir, args, horizons)
    log.info(f"\n報告：{report}")
    log.info(f"輸出目錄：{out_dir}")
    log.info("判讀提醒：DA 要跟 always_long 比、MAPE 要跟 naive 比，"
             "且差異必須通過 DM 檢定（p < 0.05）才算數。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # 進度已經逐筆寫進 summary.csv，用同一個 --out 重跑就會接續
        log.info("\n已中斷。進度保留在 summary.csv，用相同的 --out 重跑即可接續。")
        raise SystemExit(130)
