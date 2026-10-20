# davis_tw — 台股 DAViS 延伸研究：訓練與評估管線

依《台股DAViS_研究提案》的 §4 METHOD／§4 EXPERIMENT 實作。保留 DAViS 三模組骨架，
只替換提案指定的三處：**資料軸（RQ2 籌碼面）**、**預測目標（RQ3 報酬率／方向）**、
**評估協定（斷點 1–4 的修正）**。凡與 RQ 無關的環節一律沿用原設計，讓實驗差異可歸因。

## 模組對照

| 提案中的設計 | 實作位置 |
|---|---|
| 蒐集層 C′：價量／籌碼／文本 | `data_io.py`、`prepare_chips.py` |
| CKIP 斷詞 → TF-IDF → PCA/Ward | `features_text.py` |
| 籌碼面特徵（三大法人等） | `features_chip.py` |
| 價量技術面特徵 | `features_price.py` |
| 分析層 A′：8 基礎模型 + Huber 堆疊 | `models.py` |
| 目標改為隔日報酬率／方向 | `dataset.py::build_targets`、`config.TARGET` |
| 特徵組可開關（消融矩陣 E0–E3、E1a-c） | `config.EXPERIMENTS`、`dataset.select_columns` |
| **斷點 1** walk-forward 依時序滾動 | `walkforward.py` |
| **斷點 2** naive／MA 基準強制對照 | `baselines.py` |
| **斷點 3** DM 檢定取代成對 t 檢定 | `metrics.diebold_mariano` |
| **斷點 4** 含成本交易回測 | `backtest.py` |
| 實驗排程與報表 | `run_experiment.py` |

## 安裝

```bash
pip install -r davis_tw/requirements.txt
```

## 執行

```bash
python -m davis_tw.prepare_chips
```

```bash
python -m davis_tw.run_experiment --experiments E0 E2 --max-stocks 20
```

輸出在 `results/{run_id}/`：`summary.csv`（實驗層彙總）、`per_stock.csv`、`predictions.csv`、
`dm_tests.csv`、`backtest.csv`、`meta_weights.csv`、`run_config.json`（完整設定，供複現）。

主要參數：

| 參數 | 說明 |
|---|---|
| `--experiments E0 E1 E1a E1b E1c E2 E3` | 消融矩陣的實驗代號 |
| `--stocks 2330 2317 ...` | 指定標的；不給則依成交金額中位數自動挑 `--max-stocks` 檔 |
| `--start / --end` | 研究期間，預設 2020-01-01 ~ 2024-12-31 |
| `--train-days / --val-days / --test-days / --step-days` | walk-forward 視窗（交易日數） |
| `--rolling` | 改用固定長度滾動窗（預設為擴張窗） |
| `--threshold` | 回測進場門檻（預測報酬率） |
| `--target return \| log_return \| price` | 預測目標；`price` 僅為與原論文對照用 |

## 防洩漏的三個設計點

1. **切分順序**：每折固定為訓練 → 驗證 → 測試，測試窗永遠最後（`walkforward.make_folds`）。
2. **文本模型折內重擬合**：TF-IDF 詞彙、PCA、Ward 分群中心只用該折訓練窗的文件擬合，
   驗證／測試窗的文件只做 transform（`features_text.TextFeatureExtractor`）。
3. **堆疊不用交叉驗證**：base 用訓練窗、meta 用驗證窗，避免 out-of-fold 在時序資料上讓
   meta 看到未來樣本（`models.DavisStackedEnsemble`）。

另外，文件依發布時間對齊到「可用於預測」的交易日：收盤 13:30 後發布者順延至下一交易日。

## 判讀規則（避免重蹈原論文的斷點）

- **MAPE 必須與 `MAPE_naive` 並排看**。台股 ±10% 漲跌幅限制讓價位 MAPE 天生偏低，
  單看 1% 以下的數字沒有意義。
- **DA 的對照組是 `always_long`（樣本期上漲日比例），不是 naive**。
  naive 的預測報酬率恆為 0，在本專案的定義下 DA = 0，拿它當方向對照會得到空洞的顯著結果。
- **任何「較佳」的宣稱都要看 `dm_tests.csv`**。`summary.csv` 的 `pct_dir_beat_long_sig`、
  `pct_mse_beat_naive_sig` 才是 H1／H2 的判準，平均 DA 高 0.5 個百分點不算證據。
- **消融比較已對齊期間**。各實驗共用同一段有效日期（`common_evaluation_index`），
  否則 E0 會比 E2/E3 多跑好幾年，差異就分不清是特徵有效還是期間不同。加 `--no-align` 可關閉，但不建議。

## 目前的資料狀態

| 資料軸 | 狀態 | 可跑的實驗 |
|---|---|---|
| 價量 | `data/stocks/{code}/price/`，267 檔、2011–2026（Yahoo 還原股價） | E0 ✅ |
| 籌碼面 | `data/twse/twse/t86/` → `prepare_chips` → `data/chips/`，1,344 檔、2015-01-05～2024-12-31（三大法人買賣超） | E2 ✅ |
| 新聞（鉅亨） | `tw_text_crawler.py cnyes` → `data/stocks/{code}/news/cnyes_news_{code}.csv`，含精確發布時間，**可回溯 2020–2024** | E1／E1a ✅（需先跑爬蟲） |
| 新聞（Yahoo） | `tw_text_crawler.py yahoo-news`，已修好時間戳並支援「載入更多」：每檔可取到伺服器庫存上限（1101 共 111 則約回溯一年；2330／2317 各 200 則但僅數天～兩週） | 僅供近期補充 |
| 論壇（同學會） | `tw_text_crawler.py cmoney` → `data/forum/cmoney_{code}.csv`，含 createTime、看多看空標記、表情反應數；只有近期快照 | E1c ⚠️（歷史不足） |
| PTT | 尚未蒐集（`data/ptt/`） | E1b ❌ |

### 要補的三件事

1. **文本歷史只有鉅亨網補得上**。Yahoo 的庫存上限約 200 則、同學會只有近期快照，
   兩者都回溯不到 2020 —— 大型股尤其明顯（2330 的 200 則只涵蓋 5 天）。
   要跑 2020–2024 的 E1／E1a，先執行：
   `python tw_text_crawler.py cnyes --start 2020-01-01 --end 2024-12-31 --max-stocks 25`
   （約 2.5 小時，逐週寫檔並記 checkpoint）。
2. **E1b／E1c 的歷史缺口**。PTT Stock 板有公開的歷史存檔可補；同學會若要納入，
   實務上只能從現在開始每日累積，或把該項消融限制在近期子區間並在論文中說明。
3. **籌碼面目前只有三大法人**。融資融券（TWSE `MI_MARGN`）、借券、當沖標的尚未蒐集；
   `features_chip.py` 已預留 `margin_purchase`／`short_sale`／`securities_lending`／
   `day_trade_volume` 的處理邏輯，資料補進 `data/chips/{code}.csv` 就會自動納入。

### 研究期間與視窗

預設 **2020-01-01 ~ 2024-12-31**，5 年、約 1,215 個交易日，涵蓋 2020 疫情急跌與反彈、
2022 修正、2023–24 AI 行情，且完整落在證交所 T86 籌碼面的可得範圍內，
E0–E3 期間一致、可直接比較。

另外會多載入半年（`ScopeConfig.warmup_days`）作為技術指標的暖機期，
該段只用來算特徵，不進入訓練與評估 —— 訓練實際從 2020-01-02 起算。

walk-forward 預設視窗 600／120／120（訓練／驗證／測試），每次前進 120 個交易日，
可切出 **4 折、共 480 個測試日**：

```
fold0: train 2020-01-02~2022-06-28 | val 2022-06-29~2022-12-15 | test 2022-12-16~2023-06-26
fold1: train 2020-01-02~2022-12-15 | val 2022-12-16~2023-06-26 | test 2023-06-27~2023-12-15
fold2: train 2020-01-02~2023-06-26 | val 2023-06-27~2023-12-15 | test 2023-12-18~2024-06-19
fold3: train 2020-01-02~2023-12-15 | val 2023-12-18~2024-06-19 | test 2024-06-20~2024-12-16
```

價格資料本身有到 2026-09，若只跑 `E0`／`E1`（不需要籌碼面）可用 `--end` 放寬到最新；
但期間縮短時記得同步調小視窗，否則會切不出足夠折數。

## 可調整的設計選擇

- `config.ModelConfig.base_models`：預設 8 個 sklearn 模型。裝了 xgboost／lightgbm 之後
  換上即可（原論文的 8 個模型清單請對照論文表格再定案）。
- `config.TextConfig`：斷詞器、TF-IDF 維度、PCA 維度、Ward 群數、文本回看視窗。
- `config.CostConfig`：手續費折讓、滑價。預設為無折讓、無滑價的保守設定。
- `backtest.backtest`：`long_flat`（現股可行）與 `long_short`（需信用交易）兩種策略。
