# -*- coding: utf-8 -*-
"""
全域設定：路徑、實驗矩陣、交易成本、評估協定參數。

對應簡報 §4 METHOD／§4 EXPERIMENT。所有「與 RQ 無關的環節」都集中在這裡，
改動實驗時只動 EXPERIMENTS 與 WalkForwardConfig，確保實驗差異可歸因。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ============================== 路徑 ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
