# -*- coding: utf-8 -*-
"""
文本特徵（特徵組 news / ptt / forum，前綴 txt_{來源}_）。

沿用 DAViS-C 的設計：斷詞 → TF-IDF → PCA 降維 → Ward 階層分群，
每個交易日彙總成「各群文件數 + 主成分均值 + 情緒分數 + 發文量」。
差異僅在斷詞器改用 CKIP Transformers（繁中，附 18 類 NER）。

防洩漏設計（對應斷點 1 的修正）：
    TF-IDF 詞彙、PCA、Ward 分群中心一律只用「訓練窗內的文件」擬合，
    驗證／測試窗的文件只做 transform。故本類別必須在每個 walk-forward
    折內重新 fit，不可全樣本 fit 一次。

時間對齊：
    收盤 13:30 之後（或非交易日）發布的文件，歸屬到下一個交易日，
    避免用尚未公開的資訊預測當日。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid

from . import config as C

logger = logging.getLogger(__name__)

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
