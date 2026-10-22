# 台股 DAViS 延伸研究

畢業專題。把 DAViS（多資料軸、多模型堆疊的股價預測框架）延伸到台灣股市，
加入籌碼面與文本（新聞、PTT、國際新聞）資料，檢驗「換資料軸能不能提升預測、能不能變成賺錢的策略」。

完整結論在 [`results/研究結論總整理.md`](results/研究結論總整理.md)，
唯一存活效應的可執行規格在 [`results/策略規格_高波動加趨勢過濾.md`](results/策略規格_高波動加趨勢過濾.md)。

---

## 一句話結論

**以公開的價量、籌碼、文本資料，在台股找不到扣成本、沖掉市場風險後仍然存在的可交易 alpha。**
能賺錢的組合賺的都是市場風險溢酬；用台指期取得同樣的曝險更便宜。

這個結論的價值在於每一條路徑都有記錄下來的失敗原因，而且多數是方法論陷阱，不是資料不夠。

## 正面結論：真的存在的效應

- **橫斷面波動度溢酬**：每季在成交金額前 250 名裡買近 60 日波動最高的 30 檔。
  - 季頻版通過 White Reality Check 與 Hansen SPA（p 0.037／0.015）。
  - 加回現金股利後對母體超額 +10.24%（t 2.36）。
  - 低波動組合對稱地顯著為負。
  - 納入下市股後只少約 0.4～1.2 個百分點。
  - 但它是 beta 大於 1 的風險補償：beta 中性後 +7.84%（t 1.87）不顯著，對同波動的槓桿大盤只差 −1.43%（t 0.25）。
  - 2003 起的日頻長期版 RC p 0.053～0.061，在門檻邊緣。
- **跌深反彈**：2011–2024 每筆淨報酬 +0.34%（群集 t 2.41），大型股更強。
  但做成組合後輸給持有 0050（20 格年化差 −10.64%，t −2.34）。
- **費半隔夜訊號**：對台股現貨盤中報酬有預測力（強弱組差 +0.365%，群集 t 7.04），
  但台指期開盤就已經完全反映，現股要賺得付 0.39% 成本，比訊號本身大。
- **日頻預測力存在**：隔日橫斷面 IC 約 +0.07。但改成隔日開盤成交後全部為負；
  降換手的版本扣成本後轉正，沖掉 beta 後不顯著。

## 方法論貢獻（論文主體）

研究中發現的幾個會讓結果虛胖的錯誤，每個都有修正前後的數字：

1. **walk-forward 缺少 purging**：標籤是未來 h 日報酬，驗證窗尾端會看到測試期。
   修正前 h=60 超額 t 3.94、RC p 0.0002；修正後 t 1.25、RC p 0.34。
2. **選股清單用當期市值**：拿 2024 年的排名挑 2011 年的股票。
   2011 年前 120 名與 2024 年只重疊 58%；改成逐日成交金額排名後，超額由 +7.20% 變 −4.27%。
3. **成交不到的價格**：用收盤價算訊號又用同一個收盤價成交。
   多空年化 +162% 改成隔日開盤成交後變 −19%；模型選中的股票觸及漲跌停的比例是母體的 8.5 倍。
4. **資料缺陷**：Yahoo 存活股整段缺 2011 年 1～8 月；Yahoo 價格其實是分割調整價；
   有 376 筆假跳動（漏調整的減資、單日錯價），會優先灌進高波動、短週期的策略。
5. **回測的前視篩選**：換倉時只保留「整段持有期都有報價」的股票，等於事先知道誰會停牌下市。

另外採用群集標準誤（全市場同時下跌時逐筆 t 會從 3.87 灌到 15.20）、
White Reality Check／Hansen SPA 校正資料窺探、事先固定變體再一起檢定。

---

## 目錄結構

```
├─ davis_tw_train.py        DAViS 訓練流程（purging、43 項技術指標、文本特徵、8 模型＋Huber 堆疊）
├─ davis_tw/                DAViS 模組化版本與 E0–E3 消融
├─ feature_engineering.py   105 個特徵的特徵庫（fe_families.py 定義各族）
├─ universe_pit.py          時點正確的選股範圍（逐日成交金額排名）
├─ reality_check.py         White Reality Check／Hansen SPA
├─ portfolio_backtest.py    橫斷面組合回測（實務費率、重疊分批）
├─ quarterly_*.py           季頻因子檢定、驗證、alpha 分解
├─ factor_test.py、highvol_*.py、strategy_spec.py      波動度溢酬與趨勢過濾
├─ nextday_*.py、next_day_*.py                         隔日預測與可交易性診斷
├─ news_reversal.py、adaptive_strategy.py、run_long.py、reversal_two_investors.py   跌深反彈
├─ overnight_lead.py、taifex_crawler.py                海外隔夜訊號、期貨時段分解
├─ survivorship_fix.py、delisted_*.py、adjust_delisted.py   生存者偏誤（納入下市股）
├─ fill_2011_gap.py、corp_actions.py、dividend_check.py     資料修補、除權息、現金股利
├─ tw_*_crawler.py、intl_news_crawler.py、daily_update.py    資料爬蟲與每日更新
├─ sandbox/                 模擬交易沙盒：虛擬資金、台股撮合、成本、風控、績效報表（見 sandbox/README.md）
├─ tests/sandbox/           沙盒的測試（合成資料，不需 data/）
├─ tool/                    稽核與對照工具（價格跳動掃描、前視 A/B、2011 修正 A/B）
└─ results/                 結論文件、各分析的摘要表與執行紀錄
```

`results/` 只放結論文件、摘要 CSV 與小型紀錄；逐檔預測、交易明細、修正前備份等大檔沒有上傳。

## 資料

原始資料（約 13 GB）**沒有放進 repo**：檔案太大，而且新聞與 PTT 全文有版權疑慮。
repo 只附兩個小檔：`data/industry_stock_map.json`（股票與產業對照）和
`data/price_fixes.csv`（Yahoo 價格假跳動的修正表）。

資料來源與對應的爬蟲：

- 個股股價：Yahoo Finance（`tw_stock_crawler.py`、`refresh_prices.py`），
  2011 年 1～8 月缺口用證交所 MI_INDEX 與櫃買中心補（`fill_2011_gap.py`）
- 指數與 ETF：`tw_market_index.py`、`fetch_index_targets.py`
- 三大法人買賣超：證交所 T86 日報（`davis_tw/prepare_chips.py` 轉成單股 CSV）
- 本益比、股價淨值比、殖利率：證交所 BWIBBU（`tw_fundamentals_crawler.py`）
- 除權息與減資：證交所 TWT49U、TWTAUU（`corp_actions.py`）
- 下市股：證交所 OpenAPI 終止上市清單與 STOCK_DAY（`delisted_crawler.py`、`delisted_prices.py`）
- 個股新聞、PTT：`tw_text_crawler.py`、`ptt_to_stock.py`；國際新聞：`intl_news_crawler.py`
- 台指期分時段：`taifex_crawler.py`

各腳本的參數用 `python 腳本.py --help` 查看。

## 執行環境

Python 3.12，套件見 `requirements.txt`：

```bash
pip install -r requirements.txt
```

建好資料後，主要結果的重現順序大致是：
`universe_pit.py`（母體快取）→ `feature_engineering.py`（特徵庫）→
`quarterly_factors.py`／`quarterly_verify.py`／`quarterly_alpha.py`（季頻）→
`highvol_controls.py`／`highvol_diagnose.py`／`strategy_spec.py`（長期版與趨勢過濾）。
`tool/rerun_audit.sh` 是資料稽核後重跑所有存活策略的完整腳本。

模擬交易沙盒（`sandbox/`）用同一份 `data/`，執行與測試方式見
[`sandbox/README.md`](sandbox/README.md)。測試另需 pytest。

## 限制

- 未計入流動性衝擊；大盤基準為價格指數
- 基本面資料自 2015 年起；PTT 與個股新聞自 2019 年起；國際新聞自 2014 年起
- 主題詞表是人工編寫的關鍵詞比對，不是語意模型
- 股價來源 Yahoo 有漏調整的公司行動，已用修正表處理已知的 222 筆，但無法保證沒有遺漏

詳見結論文件第十二節。

## 免責聲明

本專案為學術研究，所有回測結果都不構成投資建議。
