#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股 DAViS 延伸研究：訓練與評估管線（單檔版）
================================================================================
本檔由 davis_tw/ 套件合併而成，功能完全相同，只是不需要套件目錄。
把這支 .py 放在專案根目錄（與 data/ 同層）即可執行。

依《台股DAViS_研究提案》的 §4 METHOD／§4 EXPERIMENT 實作：保留 DAViS 三模組骨架，
只替換提案指定的三處 —— 資料軸（RQ2 籌碼面）、預測目標（RQ3 報酬率／方向）、
評估協定（斷點 1–4 的修正）。

安裝
----
    pip install pandas numpy scikit-learn scipy
    # 選用：xgboost lightgbm（更強的基礎模型）
    # 選用：ckip-transformers torch 或 jieba（文本特徵 E1／E3 需要）

執行
----
    python davis_tw_all.py prepare-chips                       # T86 → data/chips/*.csv（跑一次）
    python davis_tw_all.py run --experiments E0 E2 --max-stocks 20
    python davis_tw_all.py run --stocks 2330 1101 --experiments E0 E2 --train-days 600 \
        --val-days 120 --test-days 120 --step-days 120

判讀規則（避免重蹈原論文的斷點）
--------------------------------
  * MAPE 必須與 MAPE_naive 並排看：台股 ±10% 漲跌幅限制讓價位 MAPE 天生偏低。
  * DA 的對照組是 always_long（樣本期上漲日比例），不是 naive。
  * 任何「較佳」的宣稱都要看 dm_tests.csv，平均 DA 高 0.5 個百分點不算證據。
  * 消融比較已對齊各實驗的共同日期區間，否則差異分不清是特徵還是期間造成的。

防洩漏的三個設計點
------------------
  1. 每折固定訓練 → 驗證 → 測試，測試窗永遠最後。
  2. TF-IDF/PCA/Ward 折內用該折訓練窗重擬合，驗證／測試窗只做 transform。
  3. 堆疊不用交叉驗證：base 用訓練窗、meta 用驗證窗，避免 meta 看到未來樣本。
"""
from __future__ import annotations

import argparse
import json
import logging
import numpy as np
import pandas as pd
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tw_market_index import benchmark_stats, load_twii

# 本檔由多個模組合併而成，原始碼以 C.（設定）、M.（指標）、bl.（基準）前綴互相引用。
# 合併後這些前綴一律指向本模組自身，程式碼因而無需改寫，也方便與套件版對照。
C = M = bl = data_io = sys.modules[__name__]

logger = logging.getLogger("davis_tw")


##############################################################################
# 設定（原 davis_tw/config.py）
##############################################################################

"""全域設定：路徑、實驗矩陣、交易成本、評估協定參數。

對應簡報 §4 METHOD／§4 EXPERIMENT。所有「與 RQ 無關的環節」都集中在這裡，
改動實驗時只動 EXPERIMENTS 與 WalkForwardConfig，確保實驗差異可歸因。"""

# ============================== 路徑 ==============================

PROJECT_ROOT = Path(__file__).resolve().parent   # 單檔版：本腳本與 data/ 同層
DATA_DIR = PROJECT_ROOT / "data"
STOCKS_DIR = DATA_DIR / "stocks"          # data/stocks/{code}/price|news
CHIPS_DIR = DATA_DIR / "chips"            # data/chips/{code}.csv （FinMind 籌碼面，待補）
PTT_DIR = DATA_DIR / "ptt"                # data/ptt/{code}.csv   （PTT Stock 板，待補）
FORUM_DIR = DATA_DIR / "forum"            # data/forum/{code}.csv （CMoney 同學會，待補）
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"

# ============================== 特徵組（消融矩陣用） ==============================

# 特徵欄位以前綴分群，dataset 依實驗設定開關整群。
GROUP_PREFIX = {
    "price": "px_",
    "news": "txt_news_",
    "ptt": "txt_ptt_",
    "forum": "txt_forum_",
    "chip": "chip_",
}

TEXT_GROUPS = ("news", "ptt", "forum")

# 簡報 §4 EXPERIMENT 的消融矩陣
EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "E0":  ("price",),                                   # 僅價量（基線）
    "E1":  ("price", "news", "ptt", "forum"),            # 價量＋文本（原論文 w-c 的台股複製）
    "E1a": ("price", "news"),                            # 文本再拆：新聞
    "E1b": ("price", "ptt"),                             # 文本再拆：PTT
    "E1c": ("price", "forum"),                           # 文本再拆：論壇
    "E2":  ("price", "chip"),                            # 價量＋籌碼面
    "E3":  ("price", "news", "ptt", "forum", "chip"),    # 全特徵
}

# ============================== 預測目標（§3 RQ3：目標重設） ==============================

# "return"    : 隔日報酬率 r_{t+1} = close_{t+1}/close_t - 1（迴歸，主要設定）
# "log_return": 隔日對數報酬率
# "price"     : 隔日收盤價（僅為與原論文對照用，受漲跌幅限制影響，見 §5 RISKS）
TARGET = "return"

# 預測期間（交易日），消融矩陣預設兩種都跑：
#     1 = 隔日沖（預測隔日報酬）
#     5 = 週波段（預測未來一週累積報酬，進場後至少持有一週）
# 兩者的成本結構差很多：台股來回成本 0.585%，隔日沖幾乎天天要吃一次，
# 波段是一次成本攤在一週的報酬上，所以「籌碼面有沒有增益」要分開問。
# 注意 h>1 時相鄰樣本的目標會重疊，DM 檢定必須用 Newey–West 修正（已處理）。
HORIZONS = [1, 5]

# ============================== Walk-forward 評估協定（修正斷點 1） ==============================


@dataclass(frozen=True)
class WalkForwardConfig:
    """訓練→驗證→測試嚴格依時序前進；驗證集永遠夾在訓練與測試之間。"""

    # 視窗長度配合 2020–2024 的研究期間（約 1,215 個交易日）：
    # 600+120+120 = 840、每次前進 120 日 → 4 折、共 480 個測試日。
    # 期間若縮短（--start 延後），需同步把視窗調小，否則會切不出足夠折數。
    train_days: int = 600     # 約 2.4 年交易日
    val_days: int = 120       # 約半年（供 meta-regressor 與門檻校準）
    test_days: int = 120      # 約半年
    step_days: int = 120      # 每次前進一個測試窗，測試集互不重疊
    expanding: bool = True    # True: 訓練窗擴張；False: 固定長度滾動
    min_folds: int = 2        # 少於此折數視為資料不足


WALK_FORWARD = WalkForwardConfig()

# ============================== 交易成本與回測（修正斷點 4） ==============================


@dataclass(frozen=True)
class CostConfig:
    """台股現股當日買進、隔日賣出的成本結構。"""

    fee_rate: float = 0.001425       # 券商手續費（買賣各一次）
    fee_discount: float = 1.0        # 折讓，例如 6 折填 0.6
    tax_rate: float = 0.003          # 證交稅（賣出時課徵）
    min_fee: float = 0.0             # 低消（單股回測通常設 0，避免與部位大小耦合）
    slippage: float = 0.0            # 滑價（以報酬率計），保守可設 0.0005

    def round_trip_cost(self) -> float:
        """一次完整進出（買+賣）的成本率，直接自報酬率扣除。"""
        fee = self.fee_rate * self.fee_discount
        return 2 * fee + self.tax_rate + 2 * self.slippage


COST = CostConfig()

TRADING_DAYS_PER_YEAR = 252

# ============================== 文本前處理（DAViS-C 沿用） ==============================


@dataclass(frozen=True)
class TextConfig:
    """TF-IDF → PCA → Ward 分群，沿用原論文設計；斷詞改用 CKIP。"""

    tokenizer: str = "ckip"          # "ckip" | "jieba"（CKIP 未安裝時自動退回）
    max_features: int = 20000
    min_df: int = 3
    max_df: float = 0.8
    n_pca: int = 30                  # PCA 降維後維度
    n_clusters: int = 10             # Ward 分群數 → 每日各群文件數／情緒均值為特徵
    lookback_days: int = 5           # 每日彙總的文本回看視窗
    use_sentiment: bool = True       # 是否加入情緒分數（詞典法，見 features_text）


TEXT = TextConfig()

# ============================== 模型（DAViS-A 沿用 8 基礎模型 + Huber 堆疊） ==============================


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    n_jobs: int = -1
    # 基礎模型清單；名稱對應 models.build_base_models()
    base_models: tuple[str, ...] = (
        "ridge", "lasso", "elasticnet", "svr",
        "knn", "random_forest", "extra_trees", "gbrt",
    )
    # 若已安裝 xgboost / lightgbm，可換成含 "xgboost" 的清單
    meta_model: str = "huber"        # Huber 堆疊集成（原論文設計）
    refit_base_on_val: bool = False  # 保持 base 與 meta 校準一致，預設不重訓


MODEL = ModelConfig()

# ============================== 研究範圍 ==============================


@dataclass(frozen=True)
class ScopeConfig:
    """簡報 §4：每產業取市值最高且文本量足者，20–25 檔；近 4–5 年日頻。"""

    # 研究期間：2019-01-01 ~ 2024-12-31，共 6 年、約 1,460 個交易日。
    # 涵蓋多空循環（2019 多頭、2020 疫情急跌與反彈、2022 修正、2023–24 AI 行情），
    # 且完整落在證交所 T86 籌碼面的可得範圍內，E0–E3 期間一致、可直接比較。
    start_date: str = "2019-01-01"
    end_date: str | None = "2024-12-31"
    max_stocks: int = 25
    min_rows: int = 500              # 單股可用列數下限（研究期間內），不足則剔除
    # 挑股規則。False = 全市場依成交金額中位數取前 max_stocks 檔；
    # True  = 每個產業各取成交金額最高的 per_industry_top 檔（＝提案 §4 的
    #         「每產業取市值最高且文本量足」），28 個產業都有代表，
    #         產業層級的比較才不會被大型電子股洗掉。
    per_industry: bool = False
    per_industry_top: int = 1
    # 暖機期：技術指標最長回看 60 日，若從 start_date 才開始載入，
    # 2022 年初會因 NaN 被整段丟掉。多載入半年只用來算特徵，不進入評估。
    warmup_days: int = 180
    stocks: tuple[str, ...] = field(default_factory=tuple)  # 指定清單則覆蓋自動挑選


SCOPE = ScopeConfig()


##############################################################################
# 資料讀取層（原 davis_tw/data_io.py）
##############################################################################

"""資料讀取層（對應 DAViS-C′ 的輸出端）。

把爬蟲落地的檔案讀成統一格式：
    price : DataFrame[date, open, high, low, close, volume]（date 為索引，已排序去重）
    news / ptt / forum : DataFrame[date, title, text, source]
    chip  : DataFrame[date, <籌碼面數值欄位...>]

重要：文本與籌碼面若缺檔或缺日期欄，一律回傳 None 並記錄原因，
由 dataset 層決定該實驗是否可執行 —— 不用假資料填補，避免結論失真。"""

_DATE_CANDIDATES = ("date", "time", "datetime", "publish_time", "created_at", "日期")


# ============================== 股票清單 ==============================


def list_available_stocks(stocks_dir: Path | None = None) -> list[str]:
    """列出 data/stocks 下所有有價格檔的股票代號。"""
    stocks_dir = stocks_dir or C.STOCKS_DIR
    if not stocks_dir.exists():
        return []
    codes = []
    for d in sorted(stocks_dir.iterdir()):
        if d.is_dir() and any((d / "price").glob("*.csv")):
            codes.append(d.name)
    return codes


# ============================== 價格 ==============================


def load_price(code: str, stocks_dir: Path | None = None) -> pd.DataFrame | None:
    """讀取單一股票的逐年價格檔並串成一張表。"""
    stocks_dir = stocks_dir or C.STOCKS_DIR
    files = sorted((stocks_dir / code / "price").glob("*.csv"))
    if not files:
        logger.warning("[%s] 找不到價格檔", code)
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 讀取失敗 %s：%s", code, f.name, exc)
            continue
        if "date" not in df.columns:
            continue
        frames.append(df)

    if not frames:
        return None

    px = pd.concat(frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"])

    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        if col not in px.columns:
            logger.warning("[%s] 價格缺欄位 %s", code, col)
            return None
        px[col] = pd.to_numeric(px[col], errors="coerce")

    px = (px.dropna(subset=["close"])
            .sort_values("date")
            .drop_duplicates(subset="date", keep="last")
            .set_index("date"))

    # Yahoo 在停牌／補值日會給出 volume=0 且 OHLC 全等的假列，逐日報酬率會被灌 0，
    # 直接剔除以免污染方向準確率（DA）的分母。
    flat = (px["volume"] <= 0) & (px["open"] == px["close"]) & (px["high"] == px["low"])
    if flat.any():
        logger.info("[%s] 剔除 %d 筆零成交量補值列", code, int(flat.sum()))
        px = px.loc[~flat]

    return px[numeric]


# ============================== 文本 ==============================


def _find_date_column(df: pd.DataFrame) -> str | None:
    for cand in _DATE_CANDIDATES:
        if cand in df.columns and df[cand].notna().any() and df[cand].astype(str).str.strip().ne("").any():
            return cand
    return None


def _load_text_csvs(files: list[Path], code: str, kind: str) -> pd.DataFrame | None:
    """共用的文本讀取：需要可解析的日期欄，否則視為不可用。"""
    if not files:
        logger.info("[%s] 無 %s 文本檔", code, kind)
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] %s 讀取失敗 %s：%s", code, kind, f.name, exc)
            continue
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    date_col = _find_date_column(raw)
    if date_col is None:
        logger.warning(
            "[%s] %s 共 %d 筆但沒有可用的日期欄（欄位=%s）——"
            "無法做時序對齊，該來源本次略過",
            code, kind, len(raw), list(raw.columns),
        )
        return None

    out = pd.DataFrame()
    # 爬蟲輸出的是帶時區的 ISO 字串（+08:00）。統一轉成台北時間後拿掉時區資訊，
    # 否則後面跟 tz-naive 的交易日、訓練窗結束日比較會直接 TypeError。
    ts = pd.to_datetime(raw[date_col], errors="coerce", utc=True)
    try:
        ts = ts.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out["date"] = ts
    out["title"] = raw.get("title", "").fillna("").astype(str)
    body_col = next((c for c in ("content", "text", "body", "summary") if c in raw.columns), None)
    out["text"] = raw[body_col].fillna("").astype(str) if body_col else ""
    out["source"] = raw.get("source", "").fillna("").astype(str) if "source" in raw.columns else kind

    out = out.dropna(subset=["date"])
    out = out[(out["title"].str.strip() != "") | (out["text"].str.strip() != "")]
    if out.empty:
        return None
    # 只保留日期（盤中時間另做處理：見 features_text 的 T+1 對齊）
    out["ts"] = out["date"]
    out["date"] = out["date"].dt.normalize()
    return out.sort_values("ts").reset_index(drop=True)


def load_news(code: str, stocks_dir: Path | None = None) -> pd.DataFrame | None:
    stocks_dir = stocks_dir or C.STOCKS_DIR
    return _load_text_csvs(sorted((stocks_dir / code / "news").glob("*.csv")), code, "news")


def load_ptt(code: str, ptt_dir: Path | None = None) -> pd.DataFrame | None:
    ptt_dir = ptt_dir or C.PTT_DIR
    return _load_text_csvs(sorted(ptt_dir.glob(f"*{code}*.csv")), code, "ptt")


def load_forum(code: str, forum_dir: Path | None = None) -> pd.DataFrame | None:
    forum_dir = forum_dir or C.FORUM_DIR
    return _load_text_csvs(sorted(forum_dir.glob(f"*{code}*.csv")), code, "forum")


# ============================== 籌碼面（RQ2 核心資料軸） ==============================

# FinMind 常見欄位 → 統一命名。缺哪個就跳過哪個，不硬湊。
CHIP_COLUMN_ALIASES = {
    "foreign_investor": ("foreign_investor", "Foreign_Investor", "外資買賣超"),
    "investment_trust": ("investment_trust", "Investment_Trust", "投信買賣超"),
    "dealer": ("dealer", "Dealer", "自營商買賣超"),
    "dealer_self": ("dealer_self", "自營商買賣超(自行買賣)"),
    "dealer_hedge": ("dealer_hedge", "自營商買賣超(避險)"),
    "inst_total": ("inst_total", "三大法人買賣超"),
    "margin_purchase": ("MarginPurchaseTodayBalance", "margin_balance", "融資餘額"),
    "short_sale": ("ShortSaleTodayBalance", "short_balance", "融券餘額"),
    "securities_lending": ("SecuritiesLendingVolume", "lending_volume", "借券賣出"),
    "day_trade_volume": ("DayTrading_Volume", "day_trade_volume", "當沖量"),
}


def load_chips(code: str, chips_dir: Path | None = None) -> pd.DataFrame | None:
    """讀取單股籌碼面日資料（三大法人／融資券／借券／當沖）。"""
    chips_dir = chips_dir or C.CHIPS_DIR
    files = sorted(chips_dir.glob(f"*{code}*.csv"))
    if not files:
        logger.info("[%s] 無籌碼面資料（data/chips 尚未建立）", code)
        return None

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 籌碼面讀取失敗 %s：%s", code, f.name, exc)
    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    date_col = _find_date_column(raw)
    if date_col is None:
        logger.warning("[%s] 籌碼面缺日期欄，略過", code)
        return None

    out = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce")})
    kept = []
    for std_name, aliases in CHIP_COLUMN_ALIASES.items():
        col = next((a for a in aliases if a in raw.columns), None)
        if col is None:
            continue
        out[std_name] = pd.to_numeric(raw[col], errors="coerce")
        kept.append(std_name)

    if not kept:
        logger.warning("[%s] 籌碼面檔案無可辨識欄位（欄位=%s）", code, list(raw.columns))
        return None

    out = (out.dropna(subset=["date"])
              .sort_values("date")
              .drop_duplicates(subset="date", keep="last")
              .set_index("date"))
    logger.info("[%s] 籌碼面欄位：%s", code, kept)
    return out[kept]


# ============================== 產業分類 ==============================


def load_industry_map(path: Path | None = None) -> dict:
    """讀 data/industry_stock_map.json → {產業: [股票代號, ...]}。"""
    path = path or (C.DATA_DIR / "industry_stock_map.json")
    if not path.exists():
        logger.warning("找不到產業分類清單 %s", path)
        return {}
    out: dict[str, list[str]] = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        out.setdefault(r["category"], []).append(r["code"])
    return out


def stock_to_sector(path: Path | None = None) -> dict:
    """反查表：{股票代號: 產業}。"""
    path = path or (C.DATA_DIR / "industry_stock_map.json")
    if not path.exists():
        return {}
    return {r["code"]: r["category"] for r in json.loads(path.read_text(encoding="utf-8"))}


##############################################################################
# 特徵：價量技術面（原 davis_tw/features_price.py）
##############################################################################

"""價量技術面特徵（特徵組 price，前綴 px_）。

所有特徵在時點 t 只使用 <= t 的資訊；目標為 t+1，故不存在前視偏誤。"""

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_price_features(px: pd.DataFrame) -> pd.DataFrame:
    """由 OHLCV 建立技術面特徵，全部以 px_ 為前綴。"""
    c, h, l, o, v = (px["close"], px["high"], px["low"], px["open"], px["volume"])
    f = pd.DataFrame(index=px.index)

    # --- 報酬率與波動 ---
    ret1 = c.pct_change()
    f["px_ret1"] = ret1
    for k in (2, 3, 5, 10, 20):
        f[f"px_ret{k}"] = c.pct_change(k)
    for k in (5, 10, 20):
        f[f"px_vol{k}"] = ret1.rolling(k).std()
    f["px_ret1_lag1"] = ret1.shift(1)
    f["px_ret1_lag2"] = ret1.shift(2)
    f["px_ret1_lag3"] = ret1.shift(3)

    # --- 均線乖離（用比值而非絕對價位，避免非定態） ---
    for k in (3, 5, 10, 20, 60):
        ma = c.rolling(k).mean()
        f[f"px_ma{k}_dev"] = c / ma - 1
    f["px_ma5_ma20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1

    # --- 區間位置 ---
    for k in (5, 20, 60):
        hi, lo = h.rolling(k).max(), l.rolling(k).min()
        f[f"px_pos{k}"] = (c - lo) / (hi - lo).replace(0, np.nan)

    # --- 日內結構 ---
    f["px_hl_range"] = (h - l) / c.shift(1)
    f["px_oc_gap"] = o / c.shift(1) - 1
    f["px_close_loc"] = (c - l) / (h - l).replace(0, np.nan)

    # --- 量能 ---
    logv = np.log1p(v)
    f["px_logvol"] = logv
    for k in (5, 20):
        f[f"px_vol_ratio{k}"] = v / v.rolling(k).mean().replace(0, np.nan)
    f["px_dvol_ma5"] = logv - logv.rolling(5).mean()

    # --- 動能指標 ---
    f["px_rsi14"] = _rsi(c, 14) / 100
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["px_macd"] = macd / c
    f["px_macd_sig"] = (macd - macd.ewm(span=9).mean()) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["px_bb_z"] = (c - ma20) / sd20.replace(0, np.nan)

    # --- 漲跌幅限制相關（§5 RISKS：台股 ±10%） ---
    f["px_limit_up"] = (ret1 >= 0.095).astype(float)
    f["px_limit_dn"] = (ret1 <= -0.095).astype(float)

    # --- 日曆效應 ---
    f["px_dow"] = px.index.dayofweek.astype(float)
    f["px_month"] = px.index.month.astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


##############################################################################
# 特徵：籌碼面（原 davis_tw/features_chip.py）
##############################################################################

"""籌碼面特徵（特徵組 chip，前綴 chip_）—— 本研究相對原論文的新增資料軸（RQ2）。

原始欄位是「當日」公開數值（三大法人買賣超、融資券餘額、借券、當沖），
盤後才揭露，因此在預測 t+1 時可使用 t 日數值，無前視問題。"""

def build_chip_features(chips: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """把籌碼面原始日資料轉成量綱穩定的特徵（以成交量／流通量為分母做標準化）。"""
    df = chips.reindex(px.index)  # 對齊交易日；缺日不前填，交由下游 dropna 處理
    vol = px["volume"].replace(0, np.nan)
    f = pd.DataFrame(index=px.index)

    inst_cols = [c for c in ("foreign_investor", "investment_trust", "dealer",
                             "dealer_self", "dealer_hedge") if c in df.columns]
    for col in inst_cols:
        x = df[col]
        f[f"chip_{col}_norm"] = x / vol                          # 買賣超佔當日成交量比重
        f[f"chip_{col}_ma5"] = (x / vol).rolling(5).mean()
        f[f"chip_{col}_ma20"] = (x / vol).rolling(20).mean()
        f[f"chip_{col}_z20"] = _rolling_z(x, 20)
        f[f"chip_{col}_streak"] = _sign_streak(x)                # 連買／連賣天數

    if inst_cols:
        # 有官方「三大法人買賣超」欄就直接用，避免自營商自行買賣／避險重複相加
        if "inst_total" in df.columns:
            total = df["inst_total"]
        else:
            total = df[[c for c in inst_cols if c in ("foreign_investor", "investment_trust", "dealer")]].sum(axis=1)
        f["chip_inst_total_norm"] = total / vol
        f["chip_inst_total_ma5"] = (total / vol).rolling(5).mean()
        if {"foreign_investor", "investment_trust"} <= set(inst_cols):
            # 外資與投信同向與否：籌碼共識訊號
            f["chip_fi_it_agree"] = np.sign(df["foreign_investor"]) * np.sign(df["investment_trust"])

    if "margin_purchase" in df.columns:
        m = df["margin_purchase"]
        f["chip_margin_chg"] = m.pct_change().replace([np.inf, -np.inf], np.nan)
        f["chip_margin_chg5"] = m.pct_change(5).replace([np.inf, -np.inf], np.nan)
        f["chip_margin_z20"] = _rolling_z(m, 20)

    if "short_sale" in df.columns:
        s = df["short_sale"]
        f["chip_short_chg"] = s.pct_change().replace([np.inf, -np.inf], np.nan)
        f["chip_short_z20"] = _rolling_z(s, 20)
        if "margin_purchase" in df.columns:
            # 券資比：融券餘額 / 融資餘額，軋空壓力的常用代理變數
            f["chip_short_margin_ratio"] = s / df["margin_purchase"].replace(0, np.nan)

    if "securities_lending" in df.columns:
        f["chip_lending_norm"] = df["securities_lending"] / vol
        f["chip_lending_ma5"] = (df["securities_lending"] / vol).rolling(5).mean()

    if "day_trade_volume" in df.columns:
        f["chip_daytrade_ratio"] = df["day_trade_volume"] / vol
        f["chip_daytrade_ma5"] = (df["day_trade_volume"] / vol).rolling(5).mean()

    return f.replace([np.inf, -np.inf], np.nan)


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
    sign = np.sign(x.fillna(0))
    out = np.zeros(len(sign))
    run = 0.0
    prev = 0.0
    for i, s in enumerate(sign.to_numpy()):
        run = run + s if s != 0 and s == prev else s
        prev = s
        out[i] = run
    return pd.Series(out, index=x.index)


##############################################################################
# 特徵：文本（原 davis_tw/features_text.py）
##############################################################################

"""文本特徵（特徵組 news / ptt / forum，前綴 txt_{來源}_）。

沿用 DAViS-C 的設計：斷詞 → TF-IDF → PCA 降維 → Ward 階層分群，
每個交易日彙總成「各群文件數 + 主成分均值 + 情緒分數 + 發文量」。
差異僅在斷詞器改用 CKIP Transformers（繁中，附 18 類 NER）。

防洩漏設計（對應斷點 1 的修正）：
    TF-IDF 詞彙、PCA、Ward 分群中心一律只用「訓練窗內的文件」擬合，
    驗證／測試窗的文件只做 transform。故本類別必須在每個 walk-forward
    折內重新 fit，不可全樣本 fit 一次。

時間對齊：
    收盤 13:30 之後（或非交易日）發布的文件，歸屬到下一個交易日，
    避免用尚未公開的資訊預測當日。"""

MARKET_CLOSE_HOUR = 13
MARKET_CLOSE_MINUTE = 30

_PUNCT = re.compile(r"[^一-鿿A-Za-z0-9]+")


# ============================== 斷詞 ==============================


class Tokenizer:
    """CKIP 斷詞；未安裝或載入失敗時退回 jieba，再不行退回字元 bigram。"""

    def __init__(self, kind: str = "ckip"):
        self.kind = kind
        self._ws = None
        self._jieba = None
        if kind == "ckip":
            try:
                from ckip_transformers.nlp import CkipWordSegmenter  # type: ignore

                self._ws = CkipWordSegmenter(model="bert-base")
                logger.info("使用 CKIP Transformers 斷詞")
            except Exception as exc:  # noqa: BLE001
                logger.warning("CKIP 不可用（%s），退回 jieba", exc)
                self.kind = "jieba"
        if self.kind == "jieba":
            try:
                import jieba  # type: ignore

                self._jieba = jieba
            except Exception:  # noqa: BLE001
                logger.warning("jieba 亦不可用，退回字元 bigram")
                self.kind = "bigram"

    def __call__(self, texts: list[str]) -> list[str]:
        """回傳以空白分隔的詞串，供 TfidfVectorizer 直接吃。"""
        cleaned = [_PUNCT.sub(" ", t) for t in texts]
        if self.kind == "ckip":
            batches = self._ws(cleaned, use_delim=True, batch_size=64)
            return [" ".join(w for w in ws if w.strip()) for ws in batches]
        if self.kind == "jieba":
            return [" ".join(w for w in self._jieba.cut(t) if w.strip()) for t in cleaned]
        return [" ".join(_char_bigrams(t)) for t in cleaned]


def _char_bigrams(text: str) -> list[str]:
    s = text.replace(" ", "")
    return [s[i:i + 2] for i in range(len(s) - 1)]


# ============================== 情緒（詞典法） ==============================


@dataclass
class SentimentLexicon:
    """NTUSD 之類的正負詞表；放在 data/lexicon/positive.txt、negative.txt。"""

    positive: set
    negative: set

    @classmethod
    def load(cls, lexicon_dir=None):
        lexicon_dir = lexicon_dir or (C.DATA_DIR / "lexicon")
        pos_f, neg_f = lexicon_dir / "positive.txt", lexicon_dir / "negative.txt"
        if not (pos_f.exists() and neg_f.exists()):
            logger.info("未找到情緒詞典（%s），略過情緒特徵", lexicon_dir)
            return None
        pos = {ln.strip() for ln in pos_f.read_text(encoding="utf-8").splitlines() if ln.strip()}
        neg = {ln.strip() for ln in neg_f.read_text(encoding="utf-8").splitlines() if ln.strip()}
        return cls(pos, neg)

    def score(self, tokenized: str) -> float:
        """(正-負)/(正+負)，無命中則 0。"""
        toks = tokenized.split()
        p = sum(1 for t in toks if t in self.positive)
        n = sum(1 for t in toks if t in self.negative)
        return (p - n) / (p + n) if (p + n) else 0.0


# ============================== 主體 ==============================


class TextFeatureExtractor:
    """單一文本來源的特徵抽取器；每個 walk-forward 折內獨立 fit。"""

    def __init__(self, source, cfg=None, tokenizer=None, lexicon=None):
        self.source = source
        self.cfg = cfg or C.TEXT
        self.prefix = f"txt_{source}_"
        self.tokenizer = tokenizer or Tokenizer(self.cfg.tokenizer)
        self.lexicon = lexicon if lexicon is not None else (
            SentimentLexicon.load() if self.cfg.use_sentiment else None
        )
        self.tfidf = None
        self.svd = None
        self.centroids = None
        self.n_clusters_ = 0
        self.n_comp_ = 0
        self._fitted = False

    # ---------- 內部 ----------

    @staticmethod
    def _assign_trading_day(ts, calendar):
        """把文件時戳對應到「可用於預測」的交易日：收盤後發布者順延至下一交易日。"""
        after_close = (ts.dt.hour > MARKET_CLOSE_HOUR) | (
            (ts.dt.hour == MARKET_CLOSE_HOUR) & (ts.dt.minute >= MARKET_CLOSE_MINUTE)
        )
        # 無時分資訊（00:00）時保守視為收盤後
        no_time = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
        eff = ts.dt.normalize() + pd.to_timedelta((after_close | no_time).astype(int), unit="D")
        idx = calendar.searchsorted(eff.to_numpy(), side="left")
        beyond = idx >= len(calendar)
        idx = np.clip(idx, 0, len(calendar) - 1)
        out = pd.Series(calendar[idx], index=ts.index)
        out[beyond] = pd.NaT  # 超出行事曆（尚無對應交易日）的文件丟棄
        return out

    def _docs_text(self, docs):
        return (docs["title"].fillna("") + " " + docs["text"].fillna("")).str.strip().tolist()

    @staticmethod
    def ensure_tokens(docs, tokenizer):
        """
        先把整批文件斷好詞存進 tokens 欄，之後每折、每個實驗、每種預測期間直接取用。

        不快取的話同一批文件會被重複斷詞：6 折 × 4 實驗 × 4 種預測期間 ×（fit+transform），
        以 2330 的 1.5 萬則新聞計，光斷詞就要好幾小時，而實際只需 20 秒。
        """
        if docs is None or docs.empty or "tokens" in docs.columns:
            return docs
        texts = (docs["title"].fillna("") + " " + docs["text"].fillna("")).str.strip().tolist()
        docs = docs.copy()
        docs["tokens"] = tokenizer(texts)
        return docs

    def _tokens(self, docs):
        if "tokens" in docs.columns:
            return docs["tokens"].tolist()
        return self.tokenizer(self._docs_text(docs))

    # ---------- 對外 ----------

    def fit(self, train_docs):
        """只用訓練窗文件擬合 TF-IDF / PCA / Ward。"""
        texts = self._docs_text(train_docs) if train_docs is not None else []
        if len(texts) < max(20, self.cfg.n_clusters * 2):
            logger.warning("[%s] 訓練窗文件僅 %d 篇，不足以擬合文本模型", self.source, len(texts))
            self._fitted = False
            return self

        tokenized = self._tokens(train_docs)
        self.tfidf = TfidfVectorizer(
            max_features=self.cfg.max_features,
            min_df=min(self.cfg.min_df, max(1, len(texts) // 50)),
            max_df=self.cfg.max_df,
            token_pattern=r"(?u)\S+",
        )
        X = self.tfidf.fit_transform(tokenized)
        if X.shape[1] < 3:
            logger.warning("[%s] TF-IDF 詞彙過少（%d），略過文本特徵", self.source, X.shape[1])
            self._fitted = False
            return self

        # 稀疏矩陣用 TruncatedSVD（未置中的 PCA，扮演原論文 PCA 的角色）
        n_comp = int(min(self.cfg.n_pca, X.shape[1] - 1, max(2, X.shape[0] - 1)))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        Z = self.svd.fit_transform(X)

        # Ward 階層分群：對訓練文件分群後，用最近中心法把新文件指派到既有群
        n_clu = int(min(self.cfg.n_clusters, max(2, Z.shape[0] // 5)))
        labels = AgglomerativeClustering(n_clusters=n_clu, linkage="ward").fit_predict(Z)
        self.centroids = NearestCentroid().fit(Z, labels)
        self.n_clusters_ = n_clu
        self.n_comp_ = n_comp
        self._fitted = True
        logger.info("[%s] 文本模型擬合完成：%d 篇、%d 維、%d 群",
                    self.source, len(texts), n_comp, n_clu)
        return self

    def transform(self, docs, calendar):
        """把（任意期間的）文件轉成以交易日為索引的特徵表。"""
        empty = pd.DataFrame(index=calendar)
        if not self._fitted or docs is None or docs.empty:
            return empty

        docs = docs.copy()
        docs["trading_day"] = self._assign_trading_day(docs["ts"], calendar)
        docs = docs.dropna(subset=["trading_day"])
        if docs.empty:
            return empty

        tokenized = self._tokens(docs)
        X = self.tfidf.transform(tokenized)
        Z = self.svd.transform(X)
        labels = self.centroids.predict(Z)

        per_doc = pd.DataFrame(Z, columns=[f"{self.prefix}pc{i}" for i in range(Z.shape[1])])
        per_doc["trading_day"] = docs["trading_day"].to_numpy()
        per_doc["cluster"] = labels
        if self.lexicon is not None:
            per_doc[f"{self.prefix}sent"] = [self.lexicon.score(t) for t in tokenized]

        agg_cols = [c for c in per_doc.columns if c.startswith(self.prefix)]
        daily = per_doc.groupby("trading_day")[agg_cols].mean()

        # 各群當日文件數（DAViS 以群別代表主題，數量代表該主題熱度）
        counts = (per_doc.groupby(["trading_day", "cluster"]).size()
                          .unstack(fill_value=0)
                          .reindex(columns=range(self.n_clusters_), fill_value=0))
        counts.columns = [f"{self.prefix}clu{c}_n" for c in counts.columns]
        daily = daily.join(counts, how="outer")
        daily[f"{self.prefix}n_docs"] = per_doc.groupby("trading_day").size()

        daily = daily.reindex(calendar)
        # 文件數缺日 = 當日無文，補 0；連續型（主成分／情緒）以回看視窗前填
        n_cols = [c for c in daily.columns if c.endswith("_n") or c.endswith("n_docs")]
        daily[n_cols] = daily[n_cols].fillna(0.0)
        cont_cols = [c for c in daily.columns if c not in n_cols]
        daily[cont_cols] = daily[cont_cols].ffill(limit=self.cfg.lookback_days).fillna(0.0)

        # 回看視窗的滾動彙總（原論文以近期文本影響當日預測）
        k = self.cfg.lookback_days
        rolled = daily.rolling(k, min_periods=1).mean().add_suffix(f"_ma{k}")
        doc_sum = daily[n_cols].rolling(k, min_periods=1).sum().add_suffix(f"_sum{k}")
        return pd.concat([daily, rolled, doc_sum], axis=1)

    def fit_transform_split(self, docs, calendar, train_end):
        """便利方法：以 train_end 以前的文件 fit，再 transform 全期間。"""
        if docs is None or docs.empty:
            return pd.DataFrame(index=calendar)
        train_docs = docs[docs["ts"] <= train_end]
        return self.fit(train_docs).transform(docs, calendar)


##############################################################################
# 資料集組裝（原 davis_tw/dataset.py）
##############################################################################

"""資料集組裝：把價量／文本／籌碼三條資料軸併成單股的日頻面板，並產生預測目標。

關鍵設計
--------
1. 價量與籌碼面特徵都是「因果轉換」（只用 <= t 的資料），可一次算好；
   文本特徵含 TF-IDF/PCA/Ward 這類需要擬合的步驟，必須在每個 walk-forward
   折內用該折的訓練窗重新擬合，因此拆成 `build_text_features(train_end)`。
2. 特徵欄位以群前綴命名，消融矩陣（E0–E3、E1a-c）只是選欄位，不改模型。
3. 目標預設為隔日報酬率（§3 RQ3）；同時保留 close_t 以便還原價位算 MAPE。"""

class StockData:
    """單一股票的全部原始資料與特徵組裝邏輯。"""

    def __init__(self, code: str, px: pd.DataFrame,
                 docs: dict[str, pd.DataFrame | None] | None = None,
                 chips: pd.DataFrame | None = None,
                 eval_start: pd.Timestamp | None = None):
        self.code = code
        self.px = px
        self.docs = docs or {}
        self.chips = chips
        # 評估起始日：px 會多含一段暖機期供技術指標計算，但該段不列入訓練與評估
        self.eval_start = eval_start
        self.members: list[str] = []          # 產業類股才有：成分股清單
        self._static: pd.DataFrame | None = None

    # ---------- 載入 ----------

    @classmethod
    def load(cls, code: str, shared_tokenizer=None) -> "StockData | None":
        px = data_io.load_price(code)
        if px is None or len(px) < 100:
            logger.warning("[%s] 價格資料不足，略過", code)
            return None
        docs = {
            "news": data_io.load_news(code),
            "ptt": data_io.load_ptt(code),
            "forum": data_io.load_forum(code),
        }
        chips = data_io.load_chips(code)
        return cls(code, px, docs, chips)

    # ---------- 可用資料軸 ----------

    def available_groups(self) -> set[str]:
        """回報這檔股票實際拿得到哪些特徵組，供實驗排程判斷可否執行。"""
        groups = {"price"}
        for src in C.TEXT_GROUPS:
            d = self.docs.get(src)
            if d is not None and not d.empty:
                groups.add(src)
        if self.chips is not None and not self.chips.empty:
            groups.add("chip")
        return groups

    # ---------- 特徵 ----------

    def build_static_features(self) -> pd.DataFrame:
        """價量 + 籌碼面特徵（不需擬合，全期間一次算完）。"""
        if self._static is not None:
            return self._static
        parts = [build_price_features(self.px)]
        if self.chips is not None and not self.chips.empty:
            parts.append(build_chip_features(self.chips, self.px))
        self._static = pd.concat(parts, axis=1)
        return self._static

    def build_text_features(self, train_end: pd.Timestamp,
                            sources=C.TEXT_GROUPS,
                            tokenizer: Tokenizer | None = None,
                            lexicon: SentimentLexicon | None = None) -> pd.DataFrame:
        """以 train_end 之前的文件擬合文本模型，再轉換全期間（每折呼叫一次）。"""
        calendar = self.px.index
        frames = []
        for src in sources:
            d = self.docs.get(src)
            if d is None or d.empty:
                continue
            # 斷詞只做一次，之後所有折／實驗／預測期間共用
            if tokenizer is not None and "tokens" not in d.columns:
                d = TextFeatureExtractor.ensure_tokens(d, tokenizer)
                self.docs[src] = d
            ext = TextFeatureExtractor(src, tokenizer=tokenizer, lexicon=lexicon)
            frames.append(ext.fit_transform_split(d, calendar, train_end))
        if not frames:
            return pd.DataFrame(index=calendar)
        return pd.concat(frames, axis=1)

    # ---------- 目標 ----------

    def build_targets(self, target: str = C.TARGET, horizon: int = 1) -> pd.DataFrame:
        """
        t 列的 y 代表 t+horizon 的實現值；最後 horizon 列必為 NaN 並於下游剔除。

        horizon=1 是隔日沖，horizon=5 是週波段。h>1 時相鄰樣本的目標會重疊
        （t 與 t+1 的預測期間共用 h-1 天），所以顯著性檢定要做 Newey–West 修正、
        回測也要改成「進場後至少持有 h 天」，否則會高估顯著性與交易成本。
        """
        c = self.px["close"]
        nxt = c.shift(-horizon)
        out = pd.DataFrame(index=self.px.index)
        out["close_t"] = c
        out["close_next"] = nxt
        out["ret_next"] = nxt / c - 1
        if target == "return":
            out["y"] = out["ret_next"]
        elif target == "log_return":
            out["y"] = np.log(nxt / c)
        elif target == "price":
            out["y"] = nxt
        else:
            raise ValueError(f"未知的預測目標：{target}")
        out["y_dir"] = np.sign(out["ret_next"])
        return out


# ============================== 面板組裝 ==============================


def select_columns(features: pd.DataFrame, groups) -> list[str]:
    """依特徵組前綴選欄位（消融矩陣的唯一實作點）。"""
    prefixes = tuple(C.GROUP_PREFIX[g] for g in groups)
    return [c for c in features.columns if c.startswith(prefixes)]


def assemble(stock: StockData, text_features: pd.DataFrame | None,
             groups, target: str = C.TARGET, horizon: int = 1) -> pd.DataFrame:
    """
    合併特徵與目標，回傳含 y / close_t / ret_next 與所選特徵欄的表。
    只丟棄「特徵或目標為 NaN」的列，不做任何插補（避免製造假訊號）。
    """
    static = stock.build_static_features()
    feats = static if text_features is None or text_features.empty else \
        pd.concat([static, text_features], axis=1)

    cols = select_columns(feats, groups)
    if not cols:
        raise ValueError(f"[{stock.code}] 特徵組 {groups} 沒有對應欄位")

    tgt = stock.build_targets(target, horizon)
    df = pd.concat([feats[cols], tgt], axis=1)

    # 全欄為 NaN 或方差為 0 的特徵直接剔除（例如某籌碼欄位整段缺）
    valid = [c for c in cols if df[c].notna().sum() > 0.5 * len(df)]
    dropped = sorted(set(cols) - set(valid))
    if dropped:
        logger.info("[%s] 剔除缺值過多的特徵 %d 欄", stock.code, len(dropped))

    df = df[valid + ["y", "y_dir", "close_t", "close_next", "ret_next"]]
    df = df.dropna()
    if stock.eval_start is not None:
        df = df[df.index >= stock.eval_start]   # 丟掉暖機期，正式研究期間才進模型
    df.attrs["feature_columns"] = valid
    df.attrs["code"] = stock.code
    return df


def load_universe(codes=None, scope: C.ScopeConfig = C.SCOPE) -> list[StockData]:
    """
    載入研究範圍內的股票。未指定清單時，依「資料完整度」排序取前 max_stocks 檔
    （簡報 §4 的挑選原則是市值＋文本量，取得市值資料後可在此換成正式規則）。
    """
    codes = list(codes or scope.stocks or data_io.list_available_stocks())
    loaded: list[StockData] = []
    for code in codes:
        sd = StockData.load(code)
        if sd is None:
            continue
        px = sd.px
        start = pd.Timestamp(scope.start_date) if scope.start_date else None
        if start is not None:
            # 多載入 warmup_days 天供技術指標暖機，assemble 時再截掉
            px = px[px.index >= start - pd.Timedelta(days=scope.warmup_days)]
            sd.eval_start = start
        if scope.end_date:
            px = px[px.index <= pd.Timestamp(scope.end_date)]
        in_scope = px if start is None else px[px.index >= start]
        if len(in_scope) < scope.min_rows:
            logger.info("[%s] 期間內僅 %d 個交易日（< %d），略過", code, len(in_scope), scope.min_rows)
            continue
        sd.px = px
        loaded.append(sd)

    turnover = lambda s: float((s.px["close"] * s.px["volume"]).median())

    if not scope.stocks and scope.per_industry:
        # 提案 §4 的挑選原則：每產業取市值（這裡以成交金額中位數代理）最高的幾檔。
        # 全市場 top-N 會被大型電子股洗掉 —— 實測 20 檔只涵蓋 12/28 個產業，
        # 產業層級的比較就做不出來，所以分層抽樣才是對的做法。
        sec_of = data_io.stock_to_sector()
        by_sec: dict[str, list] = {}
        for sd in loaded:
            by_sec.setdefault(sec_of.get(sd.code, "未分類"), []).append(sd)
        picked = []
        for sec, group in by_sec.items():
            group.sort(key=turnover, reverse=True)
            picked.extend(group[:scope.per_industry_top])
        loaded = picked
        logger.info("挑股規則：每產業取成交金額最高 %d 檔 → %d 個產業、%d 檔",
                    scope.per_industry_top, len(by_sec), len(loaded))
    elif not scope.stocks and len(loaded) > scope.max_stocks:
        # 全市場依成交金額中位數（代理市值）排序，取前 max_stocks 檔
        loaded.sort(key=turnover, reverse=True)
        loaded = loaded[:scope.max_stocks]
    logger.info("研究範圍：%d 檔（%s ...）", len(loaded), ", ".join(s.code for s in loaded[:8]))
    return loaded


# ============================== 產業類股 ==============================

MIN_SECTOR_MEMBERS = 3       # 一天至少要有幾檔成分股有報價，才算得出類股報酬


def build_sector_index(members: list, scope: C.ScopeConfig) -> pd.DataFrame | None:
    """
    把同產業個股合成等權類股指數。

    用等權「日報酬」而非平均價格：平均價格會被高價股主導。日內欄位（開高低）
    先各自除以前一日收盤變成相對值再平均，成交量直接加總。
    日期取聯集並要求當天至少 MIN_SECTOR_MEMBERS 檔有報價 —— 取交集的話，
    產業裡只要有一檔是近期才上市，整段歷史就會被砍掉。
    """
    frames = {}
    for code in members:
        px = data_io.load_price(code)
        if px is not None and len(px) >= 100:
            frames[code] = px
    if not frames:
        return None

    start = pd.Timestamp(scope.start_date) if scope.start_date else None
    idx = None
    for px in frames.values():
        idx = px.index if idx is None else idx.union(px.index)
    if start is not None:
        idx = idx[idx >= start - pd.Timedelta(days=scope.warmup_days)]
    if scope.end_date:
        idx = idx[idx <= pd.Timestamp(scope.end_date)]
    idx = idx.sort_values()
    if len(idx) < 200:
        return None

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
    avail = R.notna().sum(axis=1)
    idx = idx[avail >= min(MIN_SECTOR_MEMBERS, len(frames))]
    if len(idx) < 200:
        return None

    ret = R.loc[idx].mean(axis=1, skipna=True)
    close = 100.0 * (1 + ret.fillna(0)).cumprod()          # 指數基期 100
    prev_close = close.shift(1)
    index = pd.DataFrame({
        "open": pd.concat(o_rel, axis=1).loc[idx].mean(axis=1, skipna=True) * prev_close,
        "high": pd.concat(h_rel, axis=1).loc[idx].mean(axis=1, skipna=True) * prev_close,
        "low": pd.concat(l_rel, axis=1).loc[idx].mean(axis=1, skipna=True) * prev_close,
        "close": close,
        "volume": pd.concat(vols, axis=1).loc[idx].sum(axis=1, skipna=True),
    }, index=idx).dropna()
    index.attrs["members"] = list(frames)
    return index


def build_sector_chips(members: list, index: pd.DataFrame) -> pd.DataFrame | None:
    """產業籌碼面：成分股買賣超股數直接加總（單位一致，可加總）。"""
    frames = [c for c in (data_io.load_chips(m) for m in members) if c is not None and not c.empty]
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


def build_sector_docs(members: list, loader) -> pd.DataFrame | None:
    """產業文本：成分股文件合併後去重（同一則新聞常同時標記多檔個股）。"""
    parts = [d for d in (loader(m) for m in members) if d is not None and len(d)]
    if not parts:
        return None
    allp = pd.concat(parts, ignore_index=True)
    return allp.drop_duplicates(subset=["ts", "title"]).sort_values("ts").reset_index(drop=True)


def load_sector(sector: str, members: list, scope: C.ScopeConfig = C.SCOPE) -> StockData | None:
    """把一個產業包成 StockData，後續流程與個股完全相同。"""
    index = build_sector_index(members, scope)
    if index is None:
        logger.warning("[%s] 成分股報價不足，無法合成類股指數", sector)
        return None
    used = index.attrs.get("members", members)
    docs = {
        "news": build_sector_docs(used, data_io.load_news),
        "ptt": build_sector_docs(used, data_io.load_ptt),
        "forum": build_sector_docs(used, data_io.load_forum),
    }
    sd = StockData(sector, index, docs, build_sector_chips(used, index))
    if scope.start_date:
        sd.eval_start = pd.Timestamp(scope.start_date)
    sd.members = used
    logger.info("[%s] 類股指數：%d 檔成分股、%d 個交易日", sector, len(used), len(index))
    return sd


def load_sector_universe(sectors=None, scope: C.ScopeConfig = C.SCOPE) -> list[StockData]:
    """載入要分析的產業類股。"""
    imap = data_io.load_industry_map()
    if not imap:
        logger.error("沒有產業分類清單，無法做產業層級分析")
        return []
    wanted = list(sectors) if sectors else list(imap)
    out = []
    for sec in wanted:
        if sec not in imap:
            logger.warning("沒有這個產業：%s", sec)
            continue
        sd = load_sector(sec, imap[sec], scope)
        if sd is not None:
            out.append(sd)
    logger.info("產業範圍：%d 個（%s ...）", len(out), "、".join(s.code for s in out[:6]))
    return out


##############################################################################
# 模型：8 基礎模型 + Huber 堆疊（原 davis_tw/models.py）
##############################################################################

"""模型層（DAViS-A′）：8 個基礎迴歸模型 + Huber 堆疊集成。

沿用原論文架構，只改預測目標（價位 → 報酬率）。堆疊方式刻意採
「訓練窗 fit base → 驗證窗產生 meta 訓練資料 → 測試窗評估」的時序三段式，
而非交叉驗證的 out-of-fold，理由是後者在時序資料上會讓 meta 看到未來樣本
（原論文斷點 1 的同類問題）。

Huber meta-regressor 的係數即各基礎模型的貢獻權重，
對應簡報 §1「可解釋性是框架的實務賣點」。"""

# ============================== 基礎模型 ==============================


def _make_model(name: str, seed: int, n_jobs: int):
    """單一基礎模型；需要標準化的模型自帶 StandardScaler（在折內 fit，無洩漏）。"""
    scaled = lambda est: Pipeline([("scaler", StandardScaler()), ("est", est)])

    if name == "linear":
        return scaled(LinearRegression())
    if name == "ridge":
        return scaled(Ridge(alpha=1.0, random_state=seed))
    if name == "lasso":
        return scaled(Lasso(alpha=1e-4, max_iter=20000, random_state=seed))
    if name == "elasticnet":
        return scaled(ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=20000, random_state=seed))
    if name == "svr":
        return scaled(SVR(kernel="rbf", C=1.0, epsilon=0.005, gamma="scale"))
    if name == "knn":
        return scaled(KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=n_jobs))
    if name == "mlp":
        return scaled(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600,
                                   early_stopping=True, random_state=seed))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     max_features="sqrt", n_jobs=n_jobs, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                   max_features="sqrt", n_jobs=n_jobs, random_state=seed)
    if name == "gbrt":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                         max_depth=3, subsample=0.8, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor  # type: ignore

        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            n_jobs=n_jobs, random_state=seed, tree_method="hist")
    if name == "lightgbm":
        from lightgbm import LGBMRegressor  # type: ignore

        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.8,
                             n_jobs=n_jobs, random_state=seed, verbose=-1)
    raise ValueError(f"未知的基礎模型：{name}")


def build_base_models(cfg: C.ModelConfig = C.MODEL) -> dict:
    """依設定建立基礎模型字典；缺套件的模型會被跳過並記錄。"""
    models = {}
    for name in cfg.base_models:
        try:
            models[name] = _make_model(name, cfg.random_state, cfg.n_jobs)
        except ImportError as exc:
            logger.warning("跳過基礎模型 %s（未安裝：%s）", name, exc)
    if not models:
        raise RuntimeError("沒有任何可用的基礎模型")
    return models


# ============================== 堆疊集成 ==============================


class DavisStackedEnsemble:
    """
    三段式堆疊：
        fit(X_tr, y_tr, X_va, y_va)
            1. 各 base model 以訓練窗擬合
            2. 於驗證窗產生預測矩陣 P_va
            3. Huber meta 以 P_va → y_va 擬合
        predict(X_te) = meta(base_preds(X_te))
    """

    def __init__(self, cfg: C.ModelConfig = C.MODEL):
        self.cfg = cfg
        self.base_models_ = build_base_models(cfg)
        self.meta_ = None
        self.feature_names_: list[str] = []
        self.base_names_: list[str] = []

    # ---------- 內部 ----------

    def _meta_estimator(self):
        if self.cfg.meta_model == "huber":
            # 報酬率量級小（1e-2），epsilon 取 1.35 為 Huber 標準值，
            # 交由前置的標準化（下方 _scale_preds）處理量綱。
            return HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
        if self.cfg.meta_model == "ridge":
            return Ridge(alpha=1.0)
        if self.cfg.meta_model == "linear":
            return LinearRegression()
        raise ValueError(f"未知的 meta 模型：{self.cfg.meta_model}")

    def _base_matrix(self, X) -> np.ndarray:
        return np.column_stack([self.base_models_[n].predict(X) for n in self.base_names_])

    # ---------- 對外 ----------

    def fit(self, X_tr, y_tr, X_va, y_va) -> "DavisStackedEnsemble":
        self.feature_names_ = list(X_tr.columns) if isinstance(X_tr, pd.DataFrame) else []
        Xtr = np.asarray(X_tr, dtype=float)
        Xva = np.asarray(X_va, dtype=float)

        fitted = []
        for name, model in self.base_models_.items():
            try:
                model.fit(Xtr, np.asarray(y_tr, dtype=float))
                fitted.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("基礎模型 %s 擬合失敗：%s", name, exc)
        if not fitted:
            raise RuntimeError("所有基礎模型皆擬合失敗")
        self.base_names_ = fitted

        P_va = self._base_matrix(Xva)
        self.meta_ = Pipeline([("scaler", StandardScaler()), ("est", self._meta_estimator())])
        self.meta_.fit(P_va, np.asarray(y_va, dtype=float))

        if self.cfg.refit_base_on_val:
            # 需要時可用 train+val 重訓 base（預測分佈會與 meta 校準時略有偏移，預設關閉）
            X_all = np.vstack([Xtr, Xva])
            y_all = np.concatenate([np.asarray(y_tr, float), np.asarray(y_va, float)])
            for name in self.base_names_:
                self.base_models_[name].fit(X_all, y_all)
        return self

    def predict(self, X) -> np.ndarray:
        return self.meta_.predict(self._base_matrix(np.asarray(X, dtype=float)))

    def predict_base(self, X) -> pd.DataFrame:
        """各基礎模型的個別預測，用於「集成 vs 單一模型」的 DM 檢定（斷點 3）。"""
        P = self._base_matrix(np.asarray(X, dtype=float))
        return pd.DataFrame(P, columns=self.base_names_)

    def meta_weights(self) -> pd.Series:
        """Huber meta 的標準化係數 = 各基礎模型的貢獻權重（可解釋性產出）。"""
        est = self.meta_.named_steps["est"]
        coef = getattr(est, "coef_", None)
        if coef is None:
            return pd.Series(dtype=float)
        return pd.Series(np.ravel(coef), index=self.base_names_).sort_values(key=np.abs, ascending=False)

    def feature_importance(self) -> pd.Series:
        """樹模型的平均特徵重要度（僅供描述性解讀，非因果宣稱）。"""
        if not self.feature_names_:
            return pd.Series(dtype=float)
        acc, n = None, 0
        for name in self.base_names_:
            imp = getattr(self.base_models_[name], "feature_importances_", None)
            if imp is None or len(imp) != len(self.feature_names_):
                continue
            acc = imp.astype(float) if acc is None else acc + imp
            n += 1
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(acc / n, index=self.feature_names_).sort_values(ascending=False)


##############################################################################
# 評估指標與 DM 檢定（原 davis_tw/metrics.py）
##############################################################################

"""評估指標與統計檢定（對應斷點 2、3 的修正）。

主指標：DA（方向準確率）；輔指標：MAPE / RMSE。
顯著性一律用 Diebold–Mariano 檢定（含 Harvey–Leybourne–Newbold 小樣本修正），
不用成對 t 檢定 —— 後者忽略預測誤差的序列相關，是原論文斷點 3 的來源。"""

# ============================== 點預測指標 ==============================


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def directional_accuracy(y_true, y_pred) -> float:
    """
    方向準確率：以「實際報酬率」與「預測報酬率」的正負號比對。
    實際報酬率恰為 0 的日子（漲跌停鎖死、無量）不列入分母，避免灌水。
    """
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(yt) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask])))


def price_mape_from_returns(close_t, ret_true, ret_pred) -> float:
    """把報酬率預測還原成價位後計算 MAPE，供與原論文的 0.93% 對照。"""
    close_t = np.asarray(close_t, float)
    p_true = close_t * (1 + np.asarray(ret_true, float))
    p_pred = close_t * (1 + np.asarray(ret_pred, float))
    return mape(p_true, p_pred)


def evaluate_predictions(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """df 需含 ret_next（實際）、close_t 與 pred_col（預測報酬率）。"""
    yt, yp = df["ret_next"].to_numpy(float), df[pred_col].to_numpy(float)
    return {
        "n": int(len(df)),
        "DA": directional_accuracy(yt, yp),
        "MAPE_price": price_mape_from_returns(df["close_t"], yt, yp),
        "RMSE_ret": rmse(yt, yp),
        "MAE_ret": mae(yt, yp),
        "corr": float(np.corrcoef(yt, yp)[0, 1]) if len(df) > 2 and np.std(yp) > 0 else float("nan"),
    }


# ============================== Diebold–Mariano ==============================


@dataclass
class DMResult:
    stat: float
    p_value: float
    n: int
    loss: str
    better: str  # "model1" | "model2" | "tie"

    def as_dict(self) -> dict:
        return {"DM_stat": self.stat, "DM_p": self.p_value,
                "DM_n": self.n, "DM_loss": self.loss, "DM_better": self.better}


def diebold_mariano(y_true, pred1, pred2, h: int = 1, loss: str = "mse",
                    hln_correction: bool = True) -> DMResult:
    """
    H0：兩組預測的期望損失相同。
    stat < 0 表示 pred1 損失較低（較佳）。

    loss:
        "mse"   均方誤差（點預測比較）
        "mae"   絕對誤差
        "sign"  方向損失（1 - 命中），用於以 DA 為主指標時的顯著性檢定
    """
    yt = np.asarray(y_true, float)
    p1, p2 = np.asarray(pred1, float), np.asarray(pred2, float)
    mask = np.isfinite(yt) & np.isfinite(p1) & np.isfinite(p2)
    yt, p1, p2 = yt[mask], p1[mask], p2[mask]

    if loss == "mse":
        l1, l2 = (yt - p1) ** 2, (yt - p2) ** 2
    elif loss == "mae":
        l1, l2 = np.abs(yt - p1), np.abs(yt - p2)
    elif loss == "sign":
        keep = np.abs(yt) > 1e-12
        yt, p1, p2 = yt[keep], p1[keep], p2[keep]
        l1 = (np.sign(yt) != np.sign(p1)).astype(float)
        l2 = (np.sign(yt) != np.sign(p2)).astype(float)
    else:
        raise ValueError(f"未知的損失函數：{loss}")

    d = l1 - l2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return DMResult(float("nan"), float("nan"), n, loss, "tie")

    d_bar = float(np.mean(d))
    # Newey–West 長期變異數（自相關至 h-1 階）
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for k in range(1, h):
        gk = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var += 2 * gk
    if var <= 0:
        return DMResult(float("nan"), float("nan"), n, loss, "tie")

    stat = d_bar / np.sqrt(var / n)
    if hln_correction:
        # Harvey, Leybourne & Newbold (1997) 小樣本修正
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= adj
        p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))

    better = "tie"
    if p < 0.05:
        better = "model1" if stat < 0 else "model2"
    return DMResult(float(stat), float(p), n, loss, better)


# ============================== 交易績效指標 ==============================


def sharpe_ratio(returns, risk_free: float = 0.0,
                 periods: int = C.TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    excess = r - risk_free / periods
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods))


def sortino_ratio(returns, periods: int = C.TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(downside, ddof=1) * np.sqrt(periods))


def max_drawdown(returns) -> float:
    """回傳最大回撤（負值，例如 -0.23 表示 -23%）。"""
    r = np.asarray(returns, float)
    r = np.nan_to_num(r)
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else float("nan")


def annualized_return(returns, periods: int = C.TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    total = float(np.prod(1 + r))
    return float(total ** (periods / len(r)) - 1)


##############################################################################
# 天真基準（原 davis_tw/baselines.py）
##############################################################################

"""天真基準（對應斷點 2 的修正）。

原論文沒有列出 naive baseline，導致 MAPE 0.93% 無從判讀。
本模組提供三組必列基準，所有實驗的報表都會強制帶上：

    naive       : p̂_{t+1} = p_t，等價於「預測隔日報酬率為 0」
    always_long : 恆predict 上漲 —— DA 的正確對照組（= 樣本期的上漲日比例）
    ma3/ma5 : 以近 k 日均價作為隔日預測，換算成報酬率
    drift   : 以訓練窗平均報酬率作為常數預測（含漂移的 naive）
    buy_hold: 交易層基準，永遠持有

注意：naive 的預測報酬率恆為 0，其方向準確率在本專案的定義下為 0
（sign(0) != sign(y)），這是刻意的 —— 它凸顯「零預測」不提供方向資訊，
價位 MAPE 卻可以很低，正是斷點 2 想揭露的現象。因此：
    比價位誤差（MAPE/MSE）→ 對照 naive
    比方向準確率（DA）    → 對照 always_long
拿 naive 當 DA 的對照組會得到「模型永遠顯著較佳」的空洞結論。"""

def naive_return(df: pd.DataFrame) -> np.ndarray:
    """p̂_{t+1} = p_t → 報酬率預測恆為 0。"""
    return np.zeros(len(df))


def ma_return(df: pd.DataFrame, close_full: pd.Series, window: int) -> np.ndarray:
    """p̂_{t+1} = MA_k(p_t) → r̂ = MA_k(p_t)/p_t - 1。用完整價格序列算 MA 再對齊。"""
    ma = close_full.rolling(window).mean().reindex(df.index)
    return (ma / df["close_t"] - 1).to_numpy(float)


def drift_return(df: pd.DataFrame, train_mean_ret: float) -> np.ndarray:
    """以訓練窗的平均報酬率作為常數預測（避免用到測試窗資訊）。"""
    return np.full(len(df), float(train_mean_ret))


def build_baselines(test_df: pd.DataFrame, close_full: pd.Series,
                    train_mean_ret: float) -> dict[str, np.ndarray]:
    """回傳 {基準名稱: 預測報酬率陣列}，索引與 test_df 對齊。"""
    return {
        "naive": naive_return(test_df),
        "always_long": np.full(len(test_df), 1e-8),
        "ma3": ma_return(test_df, close_full, 3),
        "ma5": ma_return(test_df, close_full, 5),
        "drift": drift_return(test_df, train_mean_ret),
    }


##############################################################################
# 含成本回測（原 davis_tw/backtest.py）
##############################################################################

"""含成本交易回測（對應斷點 4 的修正：誤差小 ≠ 可獲利）。

策略設定（刻意簡單，讓「訊號價值」而非「策略工程」成為被檢驗的對象）：
    long_flat  : 預測報酬 > threshold 則隔日持有，否則空手（台股現股可行）
    long_short : 預測方向做多／做空（需信用交易，僅供對照）

成本：手續費 0.1425%（買賣各一次，可設折讓）＋ 證交稅 0.3%（賣出），
      僅在「部位變動」的當日計費，不做每日虛擬進出以免高估成本。"""

@dataclass
class BacktestResult:
    equity: pd.Series
    net_returns: pd.Series
    positions: pd.Series
    stats: dict

    def to_dict(self) -> dict:
        return dict(self.stats)


def _position_from_signal(pred: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    if mode == "long_flat":
        return (pred > threshold).astype(float)
    if mode == "long_short":
        return np.sign(pred) * (np.abs(pred) > threshold)
    if mode == "buy_hold":
        return np.ones(len(pred))
    raise ValueError(f"未知的策略模式：{mode}")


def backtest(test_df: pd.DataFrame, pred: np.ndarray, mode: str = "long_flat",
             threshold: float = 0.0, cost: C.CostConfig = C.COST,
             horizon: int = 1) -> BacktestResult:
    """
    long_flat：訊號 > 門檻就進場，並至少持有 horizon 個交易日；最短持有期滿後
    訊號仍為真就續抱（不會多付一次來回成本），轉為假才出場。horizon=1 等同逐日再平衡。

    逐日損益用 close_t 自己算出的「隔日報酬」，不是 h 日累積報酬 ——
    後者會讓權益曲線重複計算同一段漲跌。
    """
    idx = test_df.index
    sig = np.asarray(pred, float)
    n = len(sig)
    fee = cost.fee_rate * cost.fee_discount
    daily = test_df["close_t"].pct_change().shift(-1).fillna(0.0).to_numpy(float)
    want_all = (np.ones(n) if mode == "buy_hold"
                else _position_from_signal(sig, mode, threshold))

    pos = np.zeros(n)
    entry_cost = np.zeros(n)
    exit_cost = np.zeros(n)
    in_pos, hold_until, n_trades = False, -1, 0

    for i in range(n):
        want = want_all[i]
        if not in_pos and want != 0:
            in_pos, hold_until = True, i + horizon
            entry_cost[i] += fee + cost.slippage
            n_trades += 1
        elif in_pos and i >= hold_until and want == 0:
            in_pos = False
            exit_cost[i] += fee + cost.tax_rate + cost.slippage
        pos[i] = want if in_pos else 0.0

    if in_pos:                                   # 最後一日強制平倉
        exit_cost[-1] += fee + cost.tax_rate + cost.slippage

    gross = pos * daily
    net = gross - entry_cost - exit_cost
    equity = pd.Series(np.cumprod(1 + net), index=idx, name="equity")

    stats = {
        "strategy": mode,
        "horizon": horizon,
        "n_days": int(n),
        "n_trades": n_trades,
        "exposure": float(np.mean(np.abs(pos))),
        "avg_hold_days": float(np.sum(np.abs(pos)) / n_trades) if n_trades else float("nan"),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else float("nan"),
        "ann_return": annualized_return(net),
        "sharpe": sharpe_ratio(net),
        "sortino": sortino_ratio(net),
        "max_drawdown": max_drawdown(net),
        "gross_ann_return": annualized_return(gross),
        "cost_drag": float(np.sum(entry_cost + exit_cost)),
        "win_rate": float(np.mean(net[pos != 0] > 0)) if np.any(pos != 0) else float("nan"),
    }
    return BacktestResult(equity, pd.Series(net, index=idx, name="net_ret"),
                          pd.Series(pos, index=idx, name="position"), stats)


def backtest_suite(test_df: pd.DataFrame, pred: np.ndarray,
                   baselines: dict[str, np.ndarray] | None = None,
                   threshold: float = 0.0, horizon: int = 1) -> pd.DataFrame:
    """模型策略 + 買進持有 + 各基準訊號策略的績效對照表。"""
    rows = []
    rows.append({"name": "model", **backtest(test_df, pred, "long_flat", threshold,
                                         horizon=horizon).to_dict()})
    rows.append({"name": "buy_hold", **backtest(test_df, np.ones(len(test_df)), "buy_hold").to_dict()})
    for bname, bpred in (baselines or {}).items():
        if bname == "always_long":
            continue  # 與 buy_hold 等價，不重複列
        if np.allclose(np.nan_to_num(bpred), 0):
            continue  # naive 恆為 0，策略上等同空手，無須列
        rows.append({"name": bname, **backtest(test_df, bpred, "long_flat", threshold,
                                        horizon=horizon).to_dict()})
    return pd.DataFrame(rows)


##############################################################################
# Walk-forward 評估（原 davis_tw/walkforward.py）
##############################################################################

"""Walk-forward 滾動評估（對應斷點 1 的修正）。

切分順序嚴格依時序：訓練 → 驗證 → 測試，且測試窗永遠在最後。
原論文把測試期（2017/06–10）放在驗證期（2017/10–2018/02）之前，
調參時等於看過測試期之後的資料；本模組在結構上排除這種可能。

每一折內部的流程：
    1. 以「訓練窗結束日」為界，重新擬合文本模型（TF-IDF/PCA/Ward）
    2. 基礎模型以訓練窗擬合
    3. Huber meta 以驗證窗擬合
    4. 於測試窗預測，並同步產生基準預測（naive/MA/drift）
折與折之間不共用任何擬合物件。"""

# ============================== 切分 ==============================


@dataclass
class Fold:
    idx: int
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def describe(self) -> str:
        f = lambda i: f"{i[0].date()}~{i[-1].date()}"
        return f"fold{self.idx}: train {f(self.train)} | val {f(self.val)} | test {f(self.test)}"


def make_folds(index: pd.DatetimeIndex, cfg: C.WalkForwardConfig = C.WALK_FORWARD) -> list[Fold]:
    """以交易日數（非日曆日）切窗，確保每折樣本數一致。"""
    n = len(index)
    span = cfg.train_days + cfg.val_days + cfg.test_days
    if n < span:
        logger.warning("樣本 %d 日 < 單折所需 %d 日，無法切折", n, span)
        return []

    folds, i, start = [], 0, 0
    while start + span <= n:
        tr_end = start + cfg.train_days
        va_end = tr_end + cfg.val_days
        te_end = min(va_end + cfg.test_days, n)
        train_start = 0 if cfg.expanding else start
        folds.append(Fold(i, index[train_start:tr_end], index[tr_end:va_end], index[va_end:te_end]))
        i += 1
        start += cfg.step_days
    return folds


# ============================== 單股單實驗的執行 ==============================


@dataclass
class RunResult:
    code: str
    experiment: str
    predictions: pd.DataFrame                 # 逐日：pred / ret_next / close_t / fold
    baseline_predictions: pd.DataFrame        # 逐日：各基準的預測報酬率
    base_model_predictions: pd.DataFrame      # 逐日：8 個基礎模型各自的預測
    meta_weights: pd.DataFrame                # 每折的 Huber 權重
    feature_importance: pd.Series = field(default_factory=pd.Series)
    n_features: int = 0
    folds: list = field(default_factory=list)


def run_stock_experiment(stock: StockData, experiment: str, groups=None,
                         wf: C.WalkForwardConfig = C.WALK_FORWARD,
                         model_cfg: C.ModelConfig = C.MODEL,
                         target: str = C.TARGET,
                         restrict_index: pd.DatetimeIndex | None = None,
                         tokenizer=None, horizon: int = 1) -> RunResult | None:
    """對單一股票跑完一個實驗設定（E0/E1/.../E3）的所有 walk-forward 折。"""
    groups = groups or C.EXPERIMENTS[experiment]
    needs_text = any(g in C.TEXT_GROUPS for g in groups)

    # 先用「全期間」組一次，取得可用日期索引與特徵欄位（文本欄位稍後每折重算）
    text_all = None
    if needs_text:
        # 這裡只是為了取得欄位與日期範圍，實際模型用的是折內重擬合的版本
        text_all = stock.build_text_features(stock.px.index[-1], sources=[g for g in groups if g in C.TEXT_GROUPS],
                                             tokenizer=tokenizer)
    try:
        panel = assemble(stock, text_all, groups, target, horizon)
    except ValueError as exc:
        logger.warning("[%s/%s] %s", stock.code, experiment, exc)
        return None

    if restrict_index is not None:
        panel = panel.loc[panel.index.intersection(restrict_index)]

    folds = make_folds(panel.index, wf)
    if len(folds) < wf.min_folds:
        logger.warning("[%s/%s] 可用折數 %d < %d，略過", stock.code, experiment, len(folds), wf.min_folds)
        return None

    preds, base_preds, base_rows, weight_rows, importances = [], [], [], [], []
    close_full = stock.px["close"]

    for fold in folds:
        # --- 文本模型每折重新擬合（防洩漏的關鍵） ---
        if needs_text:
            text_fold = stock.build_text_features(
                fold.train[-1], sources=[g for g in groups if g in C.TEXT_GROUPS], tokenizer=tokenizer)
            fold_panel = assemble(stock, text_fold, groups, target, horizon)
            if restrict_index is not None:
                fold_panel = fold_panel.loc[fold_panel.index.intersection(restrict_index)]
        else:
            fold_panel = panel

        cols = fold_panel.attrs["feature_columns"]
        tr = fold_panel.loc[fold_panel.index.intersection(fold.train)]
        va = fold_panel.loc[fold_panel.index.intersection(fold.val)]
        te = fold_panel.loc[fold_panel.index.intersection(fold.test)]
        if len(tr) < 100 or len(va) < 20 or len(te) < 10:
            logger.info("[%s/%s] %s 樣本不足（%d/%d/%d），跳過此折",
                        stock.code, experiment, fold.idx, len(tr), len(va), len(te))
            continue

        model = DavisStackedEnsemble(model_cfg)
        try:
            model.fit(tr[cols], tr["y"], va[cols], va["y"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s/%s] fold%d 擬合失敗：%s", stock.code, experiment, fold.idx, exc)
            continue

        yhat = model.predict(te[cols])
        out = te[["ret_next", "close_t", "close_next", "y", "y_dir"]].copy()
        out["pred"] = yhat
        out["fold"] = fold.idx
        out["code"] = stock.code
        preds.append(out)

        # 基準：drift 用訓練窗均值，不碰測試窗
        bp = pd.DataFrame(bl.build_baselines(te, close_full, float(tr["ret_next"].mean())), index=te.index)
        bp["fold"] = fold.idx
        base_preds.append(bp)

        # 各基礎模型的個別預測（供「集成 vs 單一模型」DM 檢定）
        bm = model.predict_base(te[cols])
        bm.index = te.index
        bm["fold"] = fold.idx
        base_rows.append(bm)

        w = model.meta_weights()
        weight_rows.append(pd.DataFrame({"fold": fold.idx, "model": w.index, "weight": w.to_numpy()}))
        imp = model.feature_importance()
        if not imp.empty:
            importances.append(imp)

    if not preds:
        return None

    fi = (pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
          if importances else pd.Series(dtype=float))
    return RunResult(
        code=stock.code,
        experiment=experiment,
        predictions=pd.concat(preds),
        baseline_predictions=pd.concat(base_preds),
        base_model_predictions=pd.concat(base_rows),
        meta_weights=pd.concat(weight_rows) if weight_rows else pd.DataFrame(),
        feature_importance=fi,
        n_features=len(panel.attrs["feature_columns"]),
        folds=[f.describe() for f in folds],
    )


# ============================== 實驗可比性：共同日期索引 ==============================


def common_evaluation_index(stock: StockData, experiments, target: str = C.TARGET,
                            tokenizer=None, horizon: int = 1) -> pd.DatetimeIndex | None:
    """
    取各實驗特徵皆非缺值的日期交集。

    必要性：籌碼面 T86 只到 2024/12，文本回溯亦不完整；若不對齊，
    E0 會比 E2/E3 多跑好幾年，實驗間的 DA/MAPE 差異就分不清是
    「特徵有效」還是「期間不同」。對齊後所有實驗共用同一段測試期。
    """
    idx = None
    for exp in experiments:
        groups = C.EXPERIMENTS[exp]
        text = None
        if any(g in C.TEXT_GROUPS for g in groups):
            text = stock.build_text_features(stock.px.index[-1],
                                             sources=[g for g in groups if g in C.TEXT_GROUPS],
                                             tokenizer=tokenizer)
        try:
            panel = assemble(stock, text, groups, target, horizon)
        except ValueError:
            return None
        idx = panel.index if idx is None else idx.intersection(panel.index)
        if len(idx) == 0:
            return None
    return idx


##############################################################################
# T86 籌碼面轉檔（原 davis_tw/prepare_chips.py）
##############################################################################

"""把證交所 T86（三大法人買賣超日報）的逐日 JSON 轉成單股籌碼面 CSV。

輸入：data/twse/twse/t86/t86_YYYYMMDD.json
        每檔是「某一交易日、全市場所有個股」的一張表
輸出：data/chips/{code}.csv
        欄位 date, foreign_investor, investment_trust, dealer,
             dealer_self, dealer_hedge, inst_total（單位：股）

用法：
    python -m davis_tw.prepare_chips              # 全部股票
    python -m davis_tw.prepare_chips 2330 1101    # 只轉指定股票

備註：
  - 欄位以 JSON 內的 fields 名稱動態對應，不用固定索引，
    避免證交所改版欄位順序造成錯位。
  - 非交易日的檔案 stat 不是 OK，直接略過。
  - 買賣超為「當日盤後」公開資訊，用於預測 t+1 無前視問題。"""

T86_DIR = C.DATA_DIR / "twse" / "twse" / "t86"

# 證交所欄位名稱 → 專案統一欄名
FIELD_MAP = {
    "外資買賣超股數": "foreign_investor",
    "外陸資買賣超股數(不含外資自營商)": "foreign_investor",
    "投信買賣超股數": "investment_trust",
    "自營商買賣超股數": "dealer",
    "自營商買賣超股數(自行買賣)": "dealer_self",
    "自營商買賣超股數(避險)": "dealer_hedge",
    "三大法人買賣超股數": "inst_total",
}


def _to_number(x: str) -> float:
    s = str(x).replace(",", "").replace("--", "").strip()
    if s in ("", "-", "X"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_t86_dir(t86_dir: Path = T86_DIR, codes: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """掃過所有日檔，累積成 {股票代號: DataFrame}。"""
    files = sorted(t86_dir.glob("t86_*.json"))
    if not files:
        raise FileNotFoundError(f"找不到 T86 檔案：{t86_dir}")

    rows: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("讀取失敗 %s：%s", f.name, exc)
            continue
        if payload.get("stat") != "OK":
            skipped += 1
            continue

        date = pd.to_datetime(payload.get("date") or f.stem.split("_")[-1], format="%Y%m%d")
        fields = payload.get("fields") or []
        # 欄位索引：同名欄位取第一個（例如新舊版外資欄名並存時）
        idx_map = {}
        for i, name in enumerate(fields):
            std = FIELD_MAP.get(name.strip())
            if std and std not in idx_map:
                idx_map[std] = i
        if not idx_map:
            logger.warning("%s 無可辨識欄位：%s", f.name, fields)
            continue

        for row in payload.get("data", []):
            code = str(row[0]).strip()
            if codes is not None and code not in codes:
                continue
            rec = {"date": date}
            for std, i in idx_map.items():
                rec[std] = _to_number(row[i]) if i < len(row) else float("nan")
            rows[code].append(rec)

    logger.info("T86 解析完成：%d 個交易日檔、%d 檔股票（略過非交易日 %d 檔）",
                len(files) - skipped, len(rows), skipped)
    return {code: pd.DataFrame(recs).sort_values("date") for code, recs in rows.items()}


def write_chip_csvs(out_dir: Path = C.CHIPS_DIR, codes=None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_set = set(codes) if codes else None
    tables = parse_t86_dir(codes=codes_set)

    written = []
    for code, df in tables.items():
        if len(df) < 100:      # 資料過短的（下市／新上市）不輸出
            continue
        path = out_dir / f"{code}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    logger.info("已輸出 %d 檔籌碼面 CSV → %s", len(written), out_dir)
    return written


def prepare_chips_main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])
    paths = write_chip_csvs(codes=argv or None)
    if paths:
        sample = pd.read_csv(paths[0], encoding="utf-8-sig")
        print(f"\n範例 {paths[0].name}：{len(sample)} 列 "
              f"{sample['date'].min()} ~ {sample['date'].max()}")
        print(sample.head(3).to_string(index=False))
    return 0


##############################################################################
# 實驗主程式（原 davis_tw/run_experiment.py）
##############################################################################

"""實驗主程式：消融矩陣 × 修正後的評估協定（簡報 §4 EXPERIMENT）。

用法
----
    # 先把證交所 T86 轉成籌碼面 CSV（只需做一次）
    python -m davis_tw.prepare_chips

    # 跑基線與籌碼面實驗，10 檔股票
    python -m davis_tw.run_experiment --experiments E0 E2 --max-stocks 10

    # 指定股票與全矩陣
    python -m davis_tw.run_experiment --stocks 2330 2317 1101 --experiments E0 E1 E2 E3

輸出（results/{run_id}/）
    predictions.csv     逐日預測（含實際報酬率、所屬折）
    per_stock.csv       單股 × 實驗的指標
    summary.csv         實驗層級彙總（跨股票平均與勝率）
    dm_tests.csv        Diebold–Mariano 檢定結果
    backtest.csv        含成本回測績效
    meta_weights.csv    Huber 堆疊權重（可解釋性產出）
    run_config.json     本次執行的完整設定，供複現"""

# ============================== 評估 ==============================


def experiment_runnable(exp: str, avail: set) -> bool:
    """
    這個實驗的資料齊不齊。

    文本來源（新聞／PTT／論壇）採「至少有一個就能跑」，而不是全部到齊 ——
    E1 原本定義成 price+news+ptt+forum，但 PTT 從未蒐集，結果整批 E1／E3
    在所有標的上都被靜默跳過，消融矩陣等於只比了 E0 vs E2。
    少一個來源只是少幾欄特徵（select_columns 會自動略過），不該讓整個實驗消失；
    實際用到哪些來源會記在結果的 text_sources 欄，判讀時要一併看。
    """
    need = set(C.EXPERIMENTS[exp])
    text_need = need & set(C.TEXT_GROUPS)
    if not (need - set(C.TEXT_GROUPS)) <= avail:      # 價量／籌碼這類必要軸
        return False
    if text_need and not (text_need & avail):          # 文本至少要有一個來源
        return False
    return True


def evaluate_run(res, threshold: float = 0.0, horizon: int = 1,
                 bench=None) -> dict:
    """單股單實驗的指標 + 對天真基準的 DM 檢定 + 含成本回測。"""
    pred = res.predictions
    row = {"code": res.code, "experiment": res.experiment, "horizon": horizon,
           "n_features": res.n_features}
    row.update(M.evaluate_predictions(pred))

    bp = res.baseline_predictions.reindex(pred.index)
    for name in ("naive", "always_long", "ma3", "ma5", "drift"):
        if name not in bp.columns:
            continue
        row[f"DA_{name}"] = M.directional_accuracy(pred["ret_next"], bp[name])
        row[f"MAPE_{name}"] = M.price_mape_from_returns(pred["close_t"], pred["ret_next"], bp[name])

    # 主張「模型優於天真基準」必須通過 DM 檢定（斷點 2 + 3）
    # 誤差比 naive、方向比 always_long —— 各自對上正確的對照組
    dm_mse = M.diebold_mariano(pred["ret_next"], pred["pred"], bp["naive"],
                               loss="mse", h=horizon)
    dm_sign = M.diebold_mariano(pred["ret_next"], pred["pred"], bp["always_long"],
                                loss="sign", h=horizon)
    row["DM_vs_naive_mse_p"] = dm_mse.p_value
    row["DM_vs_naive_mse_stat"] = dm_mse.stat
    row["DM_dir_vs_long_p"] = dm_sign.p_value
    row["DM_dir_vs_long_stat"] = dm_sign.stat

    bt = backtest_suite(pred, pred["pred"].to_numpy(), threshold=threshold, horizon=horizon)
    model_bt = bt[bt["name"] == "model"].iloc[0]
    bh_bt = bt[bt["name"] == "buy_hold"].iloc[0]
    row["sharpe"] = model_bt["sharpe"]
    row["total_return"] = model_bt["total_return"]
    row["ann_return"] = model_bt["ann_return"]
    row["max_drawdown"] = model_bt["max_drawdown"]
    row["n_trades"] = model_bt["n_trades"]
    row["sharpe_buyhold"] = bh_bt["sharpe"]
    row["ann_return_buyhold"] = bh_bt["ann_return"]
    # 大盤對照：投資人真正的替代方案是買指數，不是抱這一檔
    row.update(benchmark_stats(bench, pred.index, C.TRADING_DAYS_PER_YEAR))
    row["excess_vs_bench"] = row["total_return"] - row["bench_return"]
    return row


def dm_between_experiments(results: dict, code: str, exp1: str, exp2: str,
                           horizon: int = 1) -> list[dict]:
    """同一檔股票、兩個實驗之間的 DM 檢定（消融比較的統計依據）。"""
    r1, r2 = results.get((code, exp1, horizon)), results.get((code, exp2, horizon))
    if r1 is None or r2 is None:
        return []
    p1, p2 = r1.predictions, r2.predictions
    idx = p1.index.intersection(p2.index)
    if len(idx) < 30:
        return []
    out = []
    for loss in ("mse", "sign"):
        dm = M.diebold_mariano(p1.loc[idx, "ret_next"], p1.loc[idx, "pred"],
                               p2.loc[idx, "pred"], loss=loss, h=horizon)
        out.append({"code": code, "horizon": horizon,
                    "comparison": f"{exp1} vs {exp2}", **dm.as_dict()})
    return out


def dm_ensemble_vs_base(res, horizon: int = 1) -> list[dict]:
    """集成 vs 各單一基礎模型（原論文斷點 3 的直接回應）。"""
    pred = res.predictions
    bm = res.base_model_predictions.reindex(pred.index)
    rows = []
    for col in bm.columns:
        if col == "fold":
            continue
        dm = M.diebold_mariano(pred["ret_next"], pred["pred"], bm[col], loss="sign", h=horizon)
        rows.append({"code": res.code, "horizon": horizon,
                     "comparison": f"{res.experiment}:ensemble vs {col}", **dm.as_dict()})
    return rows


# ============================== 主流程 ==============================


def run_experiment_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="台股 DAViS 延伸研究：消融實驗")
    ap.add_argument("--experiments", nargs="+", default=["E0"],
                    choices=list(C.EXPERIMENTS), help="要跑的實驗代號")
    ap.add_argument("--level", default="stock", choices=["stock", "sector", "both"],
                    help="分析層級：stock=個股（預設）、sector=產業類股指數、both=兩者都跑")
    ap.add_argument("--stocks", nargs="*", default=None, help="指定股票代號（預設自動挑選）")
    ap.add_argument("--sectors", nargs="*", default=None,
                    help="指定產業（預設全部 28 個），例如 --sectors 半導體 金融 航運")
    ap.add_argument("--max-stocks", type=int, default=C.SCOPE.max_stocks)
    ap.add_argument("--per-industry", action="store_true",
                    help="改用提案 §4 的挑股規則：每產業取成交金額最高的幾檔"
                         "（預設是全市場前 max-stocks 檔，產業涵蓋會偏電子）")
    ap.add_argument("--per-industry-top", type=int, default=1,
                    help="搭配 --per-industry：每產業取幾檔（預設 1）")
    ap.add_argument("--start", default=C.SCOPE.start_date)
    ap.add_argument("--end", default=C.SCOPE.end_date)
    ap.add_argument("--horizon", type=int, nargs="+", default=C.HORIZONS,
                    help="預測期間（交易日），可給多個一起跑。預設 1 5"
                         "（1=隔日沖、5=週波段）")
    ap.add_argument("--target", default=C.TARGET, choices=["return", "log_return", "price"])
    ap.add_argument("--threshold", type=float, default=0.0, help="回測進場門檻（預測報酬率）")
    ap.add_argument("--train-days", type=int, default=C.WALK_FORWARD.train_days)
    ap.add_argument("--val-days", type=int, default=C.WALK_FORWARD.val_days)
    ap.add_argument("--test-days", type=int, default=C.WALK_FORWARD.test_days)
    ap.add_argument("--step-days", type=int, default=C.WALK_FORWARD.step_days)
    ap.add_argument("--rolling", action="store_true", help="改用固定長度滾動窗（預設為擴張窗）")
    ap.add_argument("--no-align", action="store_true",
                    help="不對齊各實驗的日期交集（不建議：實驗間將不可比）")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")

    wf = C.WalkForwardConfig(train_days=args.train_days, val_days=args.val_days,
                             test_days=args.test_days, step_days=args.step_days,
                             expanding=not args.rolling)
    scope = C.ScopeConfig(start_date=args.start, end_date=args.end,
                          max_stocks=args.max_stocks,
                          stocks=tuple(args.stocks) if args.stocks else (),
                          per_industry=args.per_industry,
                          per_industry_top=args.per_industry_top)

    horizons = sorted(set(args.horizon))
    bench = load_twii()            # 大盤加權指數，所有績效的最終對照組
    logger.info("預測期間：%s", "、".join("隔日沖" if h == 1 else f"持有 {h} 日" for h in horizons))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = C.RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 依層級決定要分析的標的：個股、產業類股指數，或兩者
    stocks = []
    if args.level in ("stock", "both"):
        stocks += load_universe(scope=scope)
    if args.level in ("sector", "both"):
        stocks += load_sector_universe(args.sectors, scope=scope)
    if not stocks:
        logger.error("研究範圍內沒有可用標的；請確認 data/stocks 是否已建立")
        return 1

    sec_of = {} if args.level == "sector" else stock_to_sector()

    # 文本斷詞器共用一個實例（CKIP 模型載入成本高）
    tokenizer = Tokenizer(C.TEXT.tokenizer) if any(
        g in C.TEXT_GROUPS for e in args.experiments for g in C.EXPERIMENTS[e]) else None

    results, eval_rows, dm_rows, bt_rows, weight_rows = {}, [], [], [], []
    skipped = []

    for sd in stocks:
        avail = sd.available_groups()
        runnable = [e for e in args.experiments if experiment_runnable(e, avail)]
        missing = [e for e in args.experiments if e not in runnable]
        if missing:
            need = sorted({g for e in missing for g in C.EXPERIMENTS[e]} - avail)
            logger.info("[%s] 可用資料軸：%s", sd.code, "、".join(sorted(avail)))
            logger.warning("[%s] 略過 %s（缺資料軸：%s）", sd.code, ",".join(missing), ",".join(need))
            skipped.extend({"code": sd.code, "experiment": e, "reason": f"缺 {need}"} for e in missing)
        if not runnable:
            continue

        for hz in horizons:
            # 對齊：所有可跑實驗共用同一段有效日期，確保消融比較的差異只來自特徵
            restrict = (None if args.no_align else
                        common_evaluation_index(sd, runnable, args.target, tokenizer, hz))
            if restrict is not None and len(restrict) < wf.train_days + wf.val_days + wf.test_days:
                logger.warning("[%s] h=%d 對齊後僅 %d 個交易日，不足一折，略過",
                               sd.code, hz, len(restrict))
                continue

            for exp in runnable:
                logger.info("[%s] h=%d 執行 %s ...", sd.code, hz, exp)
                res = run_stock_experiment(sd, exp, wf=wf, target=args.target,
                                           restrict_index=restrict, tokenizer=tokenizer,
                                           horizon=hz)
                if res is None:
                    continue
                results[(sd.code, exp, hz)] = res
                row = evaluate_run(res, args.threshold, hz, bench=bench)
                used_text = sorted(set(C.EXPERIMENTS[exp]) & set(C.TEXT_GROUPS) & avail)
                row["text_sources"] = ",".join(used_text) if used_text else ""
                row["level"] = "sector" if sd.members else "stock"
                row["sector"] = sd.code if sd.members else sec_of.get(sd.code, "")
                if sd.members:
                    row["n_members"] = len(sd.members)
                eval_rows.append(row)
                dm_rows.extend(dm_ensemble_vs_base(res, hz))

                bt = backtest_suite(res.predictions, res.predictions["pred"].to_numpy(),
                                    threshold=args.threshold, horizon=hz)
                bt.insert(0, "experiment", exp)
                bt.insert(0, "code", sd.code)
                bt_rows.append(bt)

                if not res.meta_weights.empty:
                    w = res.meta_weights.copy()
                    w.insert(0, "horizon", hz)
                    w.insert(0, "experiment", exp)
                    w.insert(0, "code", sd.code)
                    weight_rows.append(w)

    if not eval_rows:
        logger.error("沒有任何實驗成功完成")
        return 1

    # ---------- 消融比較的 DM 檢定 ----------
    codes = sorted({c for c, _, _ in results})
    pairs = [("E1", "E0"), ("E2", "E0"), ("E3", "E0"), ("E3", "E1"), ("E3", "E2"),
             ("E1a", "E0"), ("E1b", "E0"), ("E1c", "E0")]
    for code in codes:
        for hz in horizons:
            for a, b in pairs:
                dm_rows.extend(dm_between_experiments(results, code, a, b, hz))

    # ---------- 落地 ----------
    per_stock = pd.DataFrame(eval_rows)
    per_stock.to_csv(out_dir / "per_stock.csv", index=False, encoding="utf-8-sig")

    all_preds = pd.concat([r.predictions.assign(experiment=e, horizon=h)
                           for (c, e, h), r in results.items()])
    all_preds.to_csv(out_dir / "predictions.csv", encoding="utf-8-sig")

    agg = (per_stock.groupby(["experiment", "horizon"])
           .agg(n_stocks=("code", "nunique"),
                DA=("DA", "mean"),
                DA_std=("DA", "std"),
                DA_always_long=("DA_always_long", "mean"),
                DA_ma5=("DA_ma5", "mean"),
                MAPE_price=("MAPE_price", "mean"),
                MAPE_naive=("MAPE_naive", "mean"),
                RMSE_ret=("RMSE_ret", "mean"),
                sharpe=("sharpe", "mean"),
                sharpe_buyhold=("sharpe_buyhold", "mean"),
                ann_return=("ann_return", "mean"),
                max_drawdown=("max_drawdown", "mean"),
                n_trades=("n_trades", "mean"),
                bench_return=("bench_return", "mean"),
                excess_vs_bench=("excess_vs_bench", "mean"))
           .reset_index())
    # 跨股票勝率：DA > 0.5 的比例，以及方向 DM 顯著優於 naive 的比例
    agg["pct_DA_above_50"] = (per_stock.assign(w=per_stock["DA"] > 0.5)
                              .groupby(["experiment", "horizon"])["w"].mean().to_numpy())
    # 方向顯著優於 always_long 的股票比例（H1/H2 的主要判準）
    agg["pct_dir_beat_long_sig"] = (per_stock.assign(
        w=(per_stock["DM_dir_vs_long_p"] < 0.05) & (per_stock["DM_dir_vs_long_stat"] < 0))
        .groupby(["experiment", "horizon"])["w"].mean().to_numpy())
    # 誤差顯著低於 naive 的股票比例
    agg["pct_mse_beat_naive_sig"] = (per_stock.assign(
        w=(per_stock["DM_vs_naive_mse_p"] < 0.05) & (per_stock["DM_vs_naive_mse_stat"] < 0))
        .groupby(["experiment", "horizon"])["w"].mean().to_numpy())
    agg.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    # 個股結果依產業分組：呼應原論文「來源效益依產業而異」的發現
    if "level" in per_stock.columns:
        sub = per_stock[(per_stock["level"] == "stock") & (per_stock["sector"] != "")]
        if len(sub) > 3 and sub["sector"].nunique() > 1:
            by_sec = (sub.groupby(["sector", "experiment"])
                      .agg(n=("code", "nunique"), DA=("DA", "mean"),
                           DA_long=("DA_always_long", "mean"), sharpe=("sharpe", "mean"))
                      .reset_index())
            by_sec["beat_long"] = by_sec["DA"] > by_sec["DA_long"]
            by_sec.to_csv(out_dir / "by_sector.csv", index=False, encoding="utf-8-sig")

    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out_dir / "dm_tests.csv", index=False, encoding="utf-8-sig")
    if bt_rows:
        pd.concat(bt_rows).to_csv(out_dir / "backtest.csv", index=False, encoding="utf-8-sig")
    if weight_rows:
        pd.concat(weight_rows).to_csv(out_dir / "meta_weights.csv", index=False, encoding="utf-8-sig")
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False, encoding="utf-8-sig")

    (out_dir / "run_config.json").write_text(json.dumps({
        "run_id": run_id,
        "experiments": args.experiments,
        "target": args.target,
        "scope": asdict(scope),
        "walk_forward": asdict(wf),
        "model": asdict(C.MODEL),
        "text": asdict(C.TEXT),
        "cost": asdict(C.COST),
        "threshold": args.threshold,
        "aligned": not args.no_align,
        "stocks": [s.code for s in stocks],
        "horizons": horizons,
        "folds_example": next(iter(results.values())).folds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 主控台摘要 ----------
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + "=" * 78)
    print(f"實驗彙總（run_id={run_id}，{per_stock['code'].nunique()} 檔股票）")
    print("=" * 78)
    cols = ["experiment", "horizon", "n_stocks", "DA", "DA_always_long", "pct_DA_above_50",
            "pct_dir_beat_long_sig", "MAPE_price", "MAPE_naive",
            "pct_mse_beat_naive_sig", "sharpe", "sharpe_buyhold", "excess_vs_bench"]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 兩個層級都跑時，分開再列一次，否則平均會把個股與類股混在一起
    if "level" in per_stock.columns and per_stock["level"].nunique() > 1:
        print("\n依層級分開：")
        lv = (per_stock.groupby(["level", "experiment"])
              .agg(標的數=("code", "nunique"), DA=("DA", "mean"),
                   DA_always_long=("DA_always_long", "mean"),
                   sharpe=("sharpe", "mean"), sharpe_buyhold=("sharpe_buyhold", "mean"))
              .reset_index())
        print(lv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n輸出目錄：{out_dir}")
    print("提醒：MAPE 需與 MAPE_naive 對照才可判讀；DA 的優勢需由 dm_tests.csv 支持。")
    return 0

################################################################################
# 命令列進入點
################################################################################


def main(argv=None) -> int:
    """把兩個子命令（prepare-chips / run）分派到對應的主程式。"""
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("prepare-chips", "prepare_chips"):
        return prepare_chips_main(rest)
    if cmd == "run":
        return run_experiment_main(rest)
    print(f"未知的子命令：{cmd}\n可用：prepare-chips | run（加 --help 看說明）")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
