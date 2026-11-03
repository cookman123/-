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
- **營收年增率（唯一不是 beta 的候選）**：補上公開資訊觀測站的月營收後，營收年增率最高的十分位減最低的十分位，
  扣成本年化約 +13%，對大盤 beta 0.41（只到 2024 年時 0.05）、alpha 年化 +8.9%（t 2.36）。
  控制價格動能、納入下市股、排除 2025–2026 後都還在。
  但和 105 個價量特徵一起做多重比較時 RC p 0.086、SPA p 0.022，屬邊緣；2011–2018 也不顯著。
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
├─ financials_crawler.py、fundamental_factors.py、fundamental_verify.py   財報與營收因子
├─ revenue_event_study.py  月營收公告事件研究（四種基準：母體、成交值×波動、產業、大盤）
├─ frontend/               本機 Streamlit 前端（app.py 進入點）
├─ restricted.py、fetch_restricted.py、restricted_check.py   處置／注意／全額交割股名單與候選池可交易性檢查
├─ numcheck.py、attest_local.py、llm_random_control.py      地端 LLM 的數字核對、決策存證、隨機同檔數對照
├─ tw_*_crawler.py、intl_news_crawler.py、daily_update.py    資料爬蟲與每日更新
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
- 月營收（2010 起）、季報損益表與資產負債表（2013 起）：公開資訊觀測站彙總表（`financials_crawler.py`）
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

## 本機前端

介面沿用 [jameschen108/tw-stock-quant](https://github.com/jameschen108/tw-stock-quant) 的 Streamlit 前端（同樣的版面、樣式、
資料狀態列與「執行」視窗），內容換成本研究的資料與結果：

- **總覽**：大盤是否在 120 日線之上、本月營收事件（創新高且年增≥30%、年增≤−20%）、高波動清單、研究重點
- **個股**：383 檔的價格、月營收（標出創歷史新高）、季財報、三大法人、營收事件後的超額報酬紀錄
- **策略**：高波動＋趨勢過濾的回測、beta 分解、5 日標籤策略的累積淨值、118／121 個特徵的季頻檢定
- **營收研究**：事件研究的四種基準比較、財報因子、五道驗證
- **產業**：照朋友後來加的主題頁做的強弱排行、產業指數走勢、成分股；他的主題成分大多不在本研究的 383 檔裡，改用產業分類（第一次開要讀全部行情，約 40 秒）
- **研究結論**：`results/研究結論總整理.md` 依章節閱讀

```bash
pip install -r frontend/requirements.txt
python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

瀏覽器開 `http://127.0.0.1:8501`。前端只讀 `data/` 與 `results/`，唯一的例外是「執行」視窗裡的三個固定指令
（`daily_update.py --no-news`、`strategy_spec.py`、`revenue_event_study.py`）。頁首會提醒股價、母體快取或清單是否過舊。
沒有朋友專案的 TEJ 授權資料、LLM 分析師與模擬持股，這三塊沒有搬過來。

## 量化端與地端 LLM 的分工（併入 tw-stock-quant 的做法）

原則：**規則能查表、能回測的放量化端；讀新聞、整理理由、在規則給的範圍內挑選才交給地端 LLM（gemma3:12b），
而且 LLM 的輸出要能被核對、被存證，並且要和規則與隨機挑選比過才採用。**

量化端
- **處置／注意／全額交割股**（`restricted.py`、`fetch_restricted.py`，取自朋友的專案）：交易所公告的硬條件，
  隔日開盤多半買不到。`llm_judge.candidates()` 先排除再給 LLM 挑；`strategy_spec.py` 的清單加上「交易限制」欄
  （只標示，不改規格）。`restricted_check.py` 量化回測硬算這些股票會多算多少
- **營收驚喜 SUR**（取自朋友的 `surprise.py`）：加進 `revenue_event_study.py` 的事件。每筆平均超額比
  「創新高且年增≥30%」小，但前後兩段都穩定（對成交值×波動配對全期 t 6.3）

地端 LLM
- **數字核對**（`numcheck.py`，取自朋友的評測數字比對）：`local_ask.py` 的回答裡，資料中找不到、或正負方向
  寫反的數字，會要求模型重寫一次，還是有就在前端標出來
- **決策存證**（`attest_local.py`，取自朋友的 `attest.py`）：`llm_decisions` 每一列算雜湊、串成鏈，
  `paper_daily.py` 每天跑完自動存證；`verify` 能指出哪一列被改過
- **隨機同檔數對照**（`llm_random_control.py`）：LLM 挑 k 檔，就從同一群候選隨機挑 k 檔重複 5000 次，
  看 LLM 的成績在隨機分布的哪裡（事後補充，採用與否仍照原本 t > 2 的規則）

結果都在前端「策略 → 地端 LLM 評估」。

## 限制

- 未計入流動性衝擊；大盤基準為價格指數
- 估值資料（本益比等）自 2015 年起、季報自 2013 年起、月營收自 2010 年起；PTT 與個股新聞自 2019 年起；國際新聞自 2014 年起
- 主題詞表是人工編寫的關鍵詞比對，不是語意模型
- 股價來源 Yahoo 有漏調整的公司行動，已用修正表處理已知的 222 筆，但無法保證沒有遺漏

詳見結論文件第十二節。

## 免責聲明

本專案為學術研究，所有回測結果都不構成投資建議。
