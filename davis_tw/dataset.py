# -*- coding: utf-8 -*-
"""
資料集組裝：把價量／文本／籌碼三條資料軸併成單股的日頻面板，並產生預測目標。

關鍵設計
--------
1. 價量與籌碼面特徵都是「因果轉換」（只用 <= t 的資料），可一次算好；
   文本特徵含 TF-IDF/PCA/Ward 這類需要擬合的步驟，必須在每個 walk-forward
   折內用該折的訓練窗重新擬合，因此拆成 `build_text_features(train_end)`。
2. 特徵欄位以群前綴命名，消融矩陣（E0–E3、E1a-c）只是選欄位，不改模型。
3. 目標預設為隔日報酬率（§3 RQ3）；同時保留 close_t 以便還原價位算 MAPE。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C
from . import data_io
from .features_chip import build_chip_features
from .features_price import build_price_features
from .features_text import TextFeatureExtractor, Tokenizer, SentimentLexicon

logger = logging.getLogger(__name__)


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
