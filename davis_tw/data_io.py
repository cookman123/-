# -*- coding: utf-8 -*-
"""
資料讀取層（對應 DAViS-C′ 的輸出端）。

把爬蟲落地的檔案讀成統一格式：
    price : DataFrame[date, open, high, low, close, volume]（date 為索引，已排序去重）
    news / ptt / forum : DataFrame[date, title, text, source]
    chip  : DataFrame[date, <籌碼面數值欄位...>]

重要：文本與籌碼面若缺檔或缺日期欄，一律回傳 None 並記錄原因，
由 dataset 層決定該實驗是否可執行 —— 不用假資料填補，避免結論失真。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import config as C

logger = logging.getLogger(__name__)

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
