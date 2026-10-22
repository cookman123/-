# sandbox — 模擬交易沙盒

給一筆虛擬資金，讓策略在台股歷史日資料上下單。沙盒負責撮合、扣成本、檢查風控、
記帳、出績效報表；策略只需要回傳目標權重。設計與取捨見
[`docs/SANDBOX_PLAN.md`](../docs/SANDBOX_PLAN.md)。

沙盒不是本 repo 研究的一部分：不修改既有腳本、`davis_tw/`、`data/`，也不推翻
[`results/研究結論總整理.md`](../results/研究結論總整理.md) 的結論。目前只附驗證用的
假策略，沒有接任何實際訊號。

## 模組

| 模組 | 內容 |
|---|---|
| `data.py` | 讀 `data/` 的價格檔並還原成原始價；公司行動表；防偷看未來的 `DataView`；資料稽核 |
| `broker.py` | 升降單位、漲跌停價、成本、把目標權重換成股數並撮合 |
| `portfolio.py` | 現金、持股、帳冊 |
| `risk.py` | 權重格式、個股上限、單日虧損停損 |
| `engine.py` | 主迴圈：策略 → 公司行動 → 撮合 → 評價 |
| `strategies.py` | 驗證用策略：`DoNothing`、`BuyAndHold`、`Scripted`、`EqualWeight`、`RandomPick` |
| `report.py` | 績效指標、相對基準的 beta／alpha、報表輸出 |
| `run.py` | 命令列入口 |

## 安裝

只用到 `requirements.txt` 裡的 pandas、numpy。跑測試另需 pytest（開發用，不在
`requirements.txt` 的必要項目裡）：

```bash
pip install pandas numpy pytest
```

## 資料

沙盒讀的是本 repo 爬蟲建好的 `data/`（約 13 GB，不在 repo 裡，建法見根目錄 README）：

| 用途 | 來源 |
|---|---|
| 存活股 | `data/stocks/<代號>/price/yahoo_*.csv`，套 `data/price_fixes.csv` |
| 下市股 | `data/delisted/price/<代號>.csv` |
| ETF | `data/market/<代號>.csv`（名單見 `data.ETFS`） |
| 除權息、減資 | `data/corp_actions/exright.csv`、`reduction.csv` |
| 交易日曆 | `data/market/TWII.csv` 的日期 |

Yahoo 的價格是分割調整價（配股、減資往回調整）。撮合需要真實能成交的價格，所以
`RepoData.load` 會用公司行動的分割比把價格還原回去（`adjust_delisted.py` 的反向）。
建好資料後先跑一次稽核，確認還原後的價格與證交所公布的前日收盤一致：

```bash
python -m sandbox.data
```

## 執行

```bash
python -m sandbox.run --strategy random --start 2011-01-01 --end 2024-12-31
```

```bash
python -m sandbox.run --help
```

輸出在 `results/sandbox/<名稱>/`：`report.md`、`config.json`、`summary.csv`（會上傳），
`nav.csv`、`positions.csv`、`ledger.csv`（只留本機）。每次執行都會同時跑一份買進持有
0050（同一套成本與規則）當基準。

主要參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--strategy` | `random` | `do_nothing`、`buy_and_hold`、`equal_weight`、`random` |
| `--fill` | `next_open` | t 日收盤決策後在 t+1 開盤成交；`next_close` 改為收盤 |
| `--odd-lot` | 關 | 以 1 股為單位；預設整股 1000 股 |
| `--fee-discount`／`--fee-min` | 0.6／20 | 手續費折數、每筆下限（元） |
| `--slippage` | 0.002 | 買價上調、賣價下調 |
| `--zero-cost` | 關 | 手續費、稅、滑價全部設為 0 |
| `--max-stock-weight` | 0.10 | 個股權重上限（ETF 不受限） |
| `--max-daily-loss`／`--halt-days` | 0.05／5 | 單日淨值跌幅超過門檻後，幾個交易日只准減倉 |
| `--n`／`--every`／`--seed` | 10／20／0 | `random`：每幾個交易日挑幾檔 |

## 寫自己的策略

```python
from sandbox.data import Panel, RepoData
from sandbox.engine import run
from sandbox.report import metrics


class Momentum:
    def generate(self, date, view):
        cal = view.calendar()
        if len(cal) > 1 and cal[-2].month == date.month:   # 只在每月第一個交易日換倉
            return None
        close = view.field("close").iloc[-60:]
        ret = (close.iloc[-1] / close.iloc[0] - 1).dropna()
        top = ret[ret.index.isin(view.universe())].nlargest(10).index
        return {c: 0.1 for c in top}


panel = Panel.build(RepoData(), "2015-01-01", "2024-12-31")
print(metrics(run(Momentum(), panel)))
```

- `generate` 在 t 日收盤後被呼叫，`view` 只看得到 t 日以前的資料；查詢之後的日期會丟
  `LookaheadError`。
- 回傳 `{代號: 目標權重}`，沒列出的持股目標為 0；回傳 `None` 表示不調整。
  `{}` 是全部賣光，跟 `None` 不同。

## 交易規則摘要

- 整股 1000 股，目標股數向下取整；現金不足時依比例縮單，不允許負現金。
- 手續費 0.1425% × 折數，元以下捨去並套下限；證交稅賣出時收，股票 0.3%、ETF 0.1%。
- 漲跌幅 2015-06-01 前 ±7%、之後 ±10%；開盤即漲停（跌停）鎖死時買進（賣出）拒單。
- 停牌沒有價格時拒單；持股以最近收盤價評價。下市股在最後交易日之後以最後收盤價轉成現金。
- 除權息、減資在撮合前處理：現金股利入帳、配股增加股數、減資換股，零股以參考價折現。
- 每一筆拒單、風控介入、公司行動都寫進帳冊並附原因。

## 測試

```bash
python -m pytest tests/sandbox
```

測試只用 `tests/sandbox/conftest.py` 在暫存資料夾產生的合成資料，不需要 `data/`。
與 `davis_tw_train.load_price` 的對照測試在本機有 `data/` 時才會跑。

## 目前的驗證狀態

| 項目 | 狀態 |
|---|---|
| 合成資料上的全部測試（會計恆等式、防偷看、漲跌停、公司行動、風控、指標） | 通過 |
| 真實資料的稽核（`python -m sandbox.data`） | **尚未執行** |
| 真實資料上買進持有 0050 的對帳、隨機策略回測 | **尚未執行** |
| 0050 含息報酬與外部公布數字的抽查 | **尚未執行** |

真實資料的驗證全部通過之前，沙盒產出的報酬不得被引用。
