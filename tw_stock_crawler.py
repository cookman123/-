#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股資料蒐集腳本
====================================
股價技術面：
    - Yahoo 股市      (query1.finance.yahoo.com 歷史報價端點)

財經新聞：
    - 鉅亨網台股新聞分類 API (api.cnyes.com) —— 可回溯 2020 以前，帶精確發布時間
    - Yahoo股市個股新聞頁 (tw.stock.yahoo.com/quote/{代號}.TW/news) —— 近期快照

論壇：
    - 股市爆料同學會 (www.cmoney.tw/forum/stock/{代號}) —— 近期快照，需 Selenium

備註：股票分類清單仍會用證交所 ISIN 查詢系統 (isin.twse.com.tw) 取得
官方產業別對照表（用來依 Yahoo 分類挑股票），但「股價資料」已不再
抓證交所，全部改用 Yahoo。

執行方式（子任務可分次跑，各自有 checkpoint）：
    python tw_stock_crawler.py price                       # 股價，逐年
    python tw_stock_crawler.py cnyes --max-stocks 25       # 鉅亨網新聞（預設 2020–2024）
    python tw_stock_crawler.py yahoo-news --max-stocks 25  # Yahoo 近期新聞
    python tw_stock_crawler.py cmoney --max-stocks 25      # 同學會論壇
    python tw_stock_crawler.py all                         # 以上全部

抓取範圍：
  - 股價預設近 15 年（可調整 START_DATE）。
  - 新聞／論壇的回溯能力差很多，這點直接影響研究設計：
      鉅亨網   可回溯（預設 2020-01-01 ~ 2024-12-31，用 --start/--end 調整）
      Yahoo    只有近 1–2 個月、數十則
      同學會   只有近期（單檔約一週～數十則）
    所以 2020–2024 的文本資料軸請以鉅亨網為主幹，另外兩個來源定位為
    「近期補充」或「從現在開始每日累積」。

時間戳記（2026/09 重寫的重點）：
  舊版只解析 Yahoo 新聞頁的 DOM，而 DOM 只顯示相對時間（「1 小時前」
  「上個月」「2 個月前」），還原不到哪一天，所以輸出的 time 欄全是空的。
  沒有發布時間就無法把新聞對齊交易日，文本特徵在方法上不能用。
  現在三個來源都改從各自的結構化資料取時間（Yahoo 讀頁面內嵌 JSON 的
  pubtime、鉅亨讀 publishAt、同學會讀 __NUXT__ 的 createTime），
  統一輸出成 date 欄（ISO 8601、台北時間）。

避免被 ban 的機制：
  1. 單一 requests.Session，關閉多執行緒併發，全部循序執行
  2. 每次請求後強制隨機延遲（API 類較短、一般網頁爬取較保守）
  3. 429 / 5xx / 逾時 → tenacity 指數退避重試（次數見 MAX_RETRY_ATTEMPTS）
  4. 逐年 / 逐週分批下載，每個小任務完成就存檔 + 寫 checkpoint.json，
     腳本中斷後重跑會自動跳過已完成的部分（斷點續傳）
  5. 抓取前檢查該網域 robots.txt，若明確禁止的路徑就跳過並記錄警告

重要提醒：
  - 各站頁面結構可能隨時改版。三個解析器都會在「解析不到東西」時發出
    明確警告（而不是安靜地寫出空檔案），看到警告就去檢查對應的正則／JS。
  - 同學會需要 Selenium：`pip install selenium webdriver-manager`。
    它的 JSON API 需要 X-Version: 3.0 標頭，補上後仍回 401（要授權 token），
    未登入拿不到，所以改用瀏覽器讀伺服器端算好的 __NUXT__ 狀態。
  - cmoney 的 TLS 憑證在 Python 3.13 會驗證失敗（Missing Subject Key
    Identifier），連 robots.txt 都讀不到，所以那一段改用瀏覽器讀 robots。
  - 若該網站條款明確禁止自動化存取，請改用它們的官方合作 API 或
    向對方申請資料授權，不要硬爬。
"""

import os
import re
import argparse
import hashlib
import sys
import json
import time
import random
import logging
import urllib.robotparser as robotparser
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ============================== 使用者設定 ==============================

# 每個 Yahoo 產業分類先抓幾檔（先小規模測試整條 pipeline，之後可調大）
STOCKS_PER_CATEGORY = 10

# Yahoo奇摩股市／證交所官方產業分類 -> 對應「證交所 ISIN 公開清單」產業別欄位
# 使用「包含關係」比對（例如 "半導體" 會比對到官方欄位 "半導體業"），
# 這樣即使官方欄位用字略有出入（工業/業/纖維…等後綴不一致）也抓得到。
# 這份清單就是你截圖裡 Yahoo 股市顯示的分類；ETF/ETN/存託憑證/認購/
# 認售/指數類/市牛證/市熊證屬於「金融商品」不是個別公司股票，故不納入。
YAHOO_CATEGORIES = [
    "水泥", "食品", "塑膠", "紡織", "電機機械",
    "電器電纜", "化學", "生技", "玻璃", "造紙",
    "鋼鐵", "橡膠", "汽車", "半導體", "電腦週邊",
    "光電", "通訊網路", "電子零組件", "電子通路", "資訊服務",
    "其他電子", "營建", "航運", "觀光餐旅", "金融",
    "貿易百貨", "油電燃氣", "運動休閒", "居家生活", "數位雲端",
    "綠能環保",
]

TWSE_ISIN_LIST_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"


def fetch_official_industry_list() -> pd.DataFrame:
    """
    從證交所 ISIN 公開資訊查詢抓「全部上市證券」清單，取得官方權威的
    代號/名稱/產業別對照表。這是動態抓取＋現場解析，不是憑記憶硬編代號，
    可避免股票代號跟公司對不上、抓錯資料的風險。
    網頁本身是 Big5 編碼，這裡用 response.encoding='big5' 正確解碼。
    """
    logger.info("正在向證交所 ISIN 查詢系統抓取官方產業分類清單……")
    resp = session.get(TWSE_ISIN_LIST_URL, timeout=20)
    resp.encoding = "big5"
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="h4")
    if table is None:
        raise RuntimeError("找不到證交所 ISIN 清單表格，網站結構可能已變更")

    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) != 7:
            continue
        code_name = tds[0].get_text(strip=True)
        if "\u3000" not in code_name:  # 全形空白分隔代號與名稱
            continue
        code, name = code_name.split("\u3000", 1)
        # td 順序：[0]代號+名稱 [1]ISIN [2]上市日 [3]市場別 [4]產業別 [5]CFICode [6]備註
        market_type = tds[3].get_text(strip=True)
        industry = tds[4].get_text(strip=True)
        # 只保留 4 位數字代號的「股票」（排除權證、ETF等代號格式不同的商品）
        if not re.fullmatch(r"\d{4}", code):
            continue
        if market_type != "上市" or not industry:
            continue
        rows.append({"code": code, "name": name, "industry": industry})

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"])
    logger.info(f"官方清單抓取完成，共 {len(df)} 檔上市股票、{df['industry'].nunique()} 個產業別")
    return df


# ============================== 市值排名（挑股用） ==============================
# 提案 §4 的挑股原則是「每產業取市值最高且文本量足」。市值 = 收盤價 × 發行股數，
# 而證交所 OpenAPI 提供兩份可以組出市值的公開資料：
#   STOCK_DAY_ALL   全市場最新一日收盤價（一次請求拿到全部）
#   t187ap03_L      上市公司基本資料，含實收資本額
# 台股面額 10 元，發行股數 ≒ 實收資本額 / 10，故 市值 ≒ 收盤價 × 實收資本額 / 10。

TWSE_STOCK_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_COMPANY_INFO = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
PAR_VALUE = 10.0          # 台股面額

# Yahoo 對「該股票在這段期間還沒上市」一律回 400。連續這麼多年都沒資料，
# 就判定是近年才上市，跳過其餘年份，不必逐年試。
NO_DATA_YEARS_BEFORE_SKIP = 3


def fetch_market_cap() -> dict:
    """回傳 {股票代號: 市值（元）}；任一來源失敗就回空字典，由呼叫端決定如何退路。"""
    caps = {}
    try:
        prices = {}
        for row in http_get(TWSE_STOCK_DAY_ALL, as_json=True) or []:
            code = str(row.get("Code", "")).strip()
            try:
                prices[code] = float(str(row.get("ClosingPrice", "")).replace(",", ""))
            except (TypeError, ValueError):
                continue
        polite_sleep(*API_DELAY_RANGE)

        for row in http_get(TWSE_COMPANY_INFO, as_json=True) or []:
            code = str(row.get("公司代號", "")).strip()
            if code not in prices:
                continue
            try:
                capital = float(str(row.get("實收資本額", "")).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if capital > 0:
                caps[code] = prices[code] * capital / PAR_VALUE
        polite_sleep(*API_DELAY_RANGE)
    except Exception as e:
        logger.warning(f"取得市值資料失敗（{e}），挑股將退回代號排序")
        return {}

    logger.info(f"已取得 {len(caps)} 檔上市公司的市值")
    return caps


def build_stock_list_by_yahoo_category(stocks_per_category: int = STOCKS_PER_CATEGORY,
                                       rebuild: bool = False):
    """
    依 Yahoo 股市分類，從官方產業清單挑出每類前 N 檔，組成 STOCK_LIST。
    結果會快取到 data/industry_stock_map.json，重跑腳本不用重抓（也比較禮貌）。
    若抓取失敗（離線 / 網站改版），會退回內建的小型備援清單，並提示錯誤。
    """
    cache_path = DATA_DIR / "industry_stock_map.json"
    if cache_path.exists() and not rebuild:
        logger.info(f"使用已快取的產業分類清單：{cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if cache_path.exists() and rebuild:
        backup = cache_path.with_suffix(".json.bak")
        cache_path.replace(backup)
        logger.info(f"重建股票清單，舊清單已備份到 {backup}")

    try:
        df = fetch_official_industry_list()
    except Exception as e:
        logger.error(f"抓取官方產業分類清單失敗（{e}），改用內建備援清單")
        fallback = [
            {"code": "2330", "name": "台積電", "category": "半導體"},
            {"code": "2317", "name": "鴻海", "category": "其他電子"},
            {"code": "2881", "name": "富邦金", "category": "金融"},
            {"code": "2603", "name": "長榮", "category": "航運"},
        ]
        return fallback

    # 依市值排序挑股（提案 §4）。舊版用 matched.head() 等於「代號最小的前 N 檔」，
    # 金融類會只挑到彰銀、台中銀這類小型銀行，富邦金／國泰金／中信金全被排除，
    # 半導體也會漏掉聯發科 —— 樣本因此系統性偏向代號小的老公司與小型股。
    caps = fetch_market_cap()
    if not caps:
        logger.warning("沒有市值資料，本次挑股退回代號排序（樣本會偏向小型股，請留意）")

    stock_list = []
    for yahoo_cat in YAHOO_CATEGORIES:
        matched = df[df["industry"].str.contains(yahoo_cat, na=False)].copy()
        if matched.empty:
            logger.warning(f"[分類無對應] Yahoo分類「{yahoo_cat}」在官方產業別欄位中找不到符合的股票，跳過")
            continue
        if caps:
            matched["market_cap"] = matched["code"].map(caps).fillna(0.0)
            matched = matched.sort_values("market_cap", ascending=False)
        picked = matched.head(stocks_per_category)
        for _, row in picked.iterrows():
            item = {"code": row["code"], "name": row["name"], "category": yahoo_cat}
            if caps:
                item["market_cap"] = float(row.get("market_cap", 0.0))
            stock_list.append(item)
        top = "、".join(f"{r['name']}" for _, r in picked.head(3).iterrows())
        logger.info(f"[分類完成] {yahoo_cat}：取得 {len(picked)} 檔"
                    f"（市值前三：{top}）")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stock_list, f, ensure_ascii=False, indent=2)
    logger.info(f"已存檔分類結果：{cache_path}（共 {len(stock_list)} 檔）")
    return stock_list


# STOCK_LIST 在主流程開始時才動態建立（見 main()），因為它需要先建立好
# logger / DATA_DIR 等基礎設施。這裡先給一個空殼，避免其他函式在載入時報錯。
STOCK_LIST = []

END_DATE = date.today()
START_DATE = date(END_DATE.year - 15, END_DATE.month, END_DATE.day)

# API 類來源（證交所、Yahoo報價）：延遲可以短一點
API_DELAY_RANGE = (1.0, 2.0)
# 一般網頁爬取（Yahoo新聞頁）：延遲拉長，更保守
WEB_DELAY_RANGE = (3.0, 6.0)
# 每抓完一檔股票的冷卻時間
PER_STOCK_COOLDOWN_RANGE = (8, 15)

# 單次請求失敗後最多重試幾次（含第一次嘗試）。
# 目前先設成 1：失敗就記錄下來、直接跳到下一筆，不重試，
# 讓你先跑一輪看整體流程跟成功率。之後要打開重試機制，
# 把這裡改回 5（原本的預設值）即可，其他都不用動。
MAX_RETRY_ATTEMPTS = 1

DATA_DIR = Path("data")
STOCKS_DIR = DATA_DIR / "stocks"
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"
LOG_FILE = DATA_DIR / "crawler.log"

for d in (DATA_DIR, STOCKS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def stock_price_dir(code: str) -> Path:
    """每檔股票的股價資料夾：data/stocks/<code>/price/"""
    d = STOCKS_DIR / code / "price"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stock_news_dir(code: str) -> Path:
    """每檔股票的新聞資料夾：data/stocks/<code>/news/"""
    d = STOCKS_DIR / code / "news"
    d.mkdir(parents=True, exist_ok=True)
    return d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tw_stock_crawler")

session = requests.Session()
session.headers.update(
    {
        # 誠實標示自己是研究用途的爬蟲，附聯絡方式，這是禮貌也是慣例
        "User-Agent": "Mozilla/5.0 (academic-research-crawler; contact: your_email@example.com)",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9",
    }
)

_robots_cache = {}


class RateLimitError(Exception):
    pass


# ============================== 共用工具 ==============================

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt: dict) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def task_key(source: str, code: str, sub: str) -> str:
    return f"{source}:{code}:{sub}"


def polite_sleep(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


def robots_allowed(url: str) -> bool:
    """檢查目標網域 robots.txt 是否允許抓取此路徑，結果做快取避免重複下載 robots.txt"""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in _robots_cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(root, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            logger.warning(f"無法讀取 robots.txt：{root}，保守起見視為不允許")
            _robots_cache[root] = None
        else:
            _robots_cache[root] = rp

    rp = _robots_cache[root]
    if rp is None:
        return False
    return rp.can_fetch(session.headers["User-Agent"], url)


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception_type((RateLimitError, requests.exceptions.RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def http_get(url: str, params: dict = None, as_json: bool = True):
    resp = session.get(url, params=params, timeout=15)
    if resp.status_code == 429:
        logger.warning(f"429 Too Many Requests: {url}")
        raise RateLimitError("rate limited")
    if resp.status_code >= 500:
        raise requests.exceptions.RequestException(f"{resp.status_code} server error: {url}")
    resp.raise_for_status()
    return resp.json() if as_json else resp.text


def year_chunks(start: date, end: date):
    y = start.year
    while y <= end.year:
        chunk_start = max(date(y, 1, 1), start)
        chunk_end = min(date(y, 12, 31), end)
        yield y, chunk_start, chunk_end
        y += 1



# ============================== 2. Yahoo 股市（股價技術面，備援/交叉驗證）==============================
# 使用 Yahoo Finance 的歷史報價端點（query1.finance.yahoo.com），
# 台股代號需加 .TW（上市）或 .TWO（上櫃），一次可用 period1/period2
# (unix timestamp) 拉一個區間，故用「年」為單位分批，跟其他來源節奏一致。

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def crawl_yahoo_price(checkpoint: dict) -> None:
    logger.info("===== 開始抓取：Yahoo 股市（股價技術面）=====")
    for stock in STOCK_LIST:
        code = stock["code"]
        symbol = f"{code}.TW"  # 上櫃股票請改成 .TWO
        no_data_years = 0      # 連續幾年回 400（尚未上市）
        for year, y_start, y_end in year_chunks(START_DATE, END_DATE):
            key = task_key("yahoo_price", code, str(year))
            if checkpoint.get(key) == "done":
                logger.info(f"[跳過-已完成] {key}")
                polite_sleep(*API_DELAY_RANGE)
                continue

            logger.info(f"[抓取中] Yahoo股市 / {symbol} / {year}")
            try:
                url = YAHOO_CHART_URL.format(symbol=symbol)
                period1 = int(datetime(y_start.year, y_start.month, y_start.day).timestamp())
                period2 = int(datetime(y_end.year, y_end.month, y_end.day).timestamp()) + 86400
                data = http_get(
                    url,
                    params={
                        "period1": period1,
                        "period2": period2,
                        "interval": "1d",
                        "events": "history",
                    },
                )
                result = (data.get("chart") or {}).get("result")
                if not result:
                    err = (data.get("chart") or {}).get("error")
                    logger.info(f"[無資料] {key}（{err}）")
                else:
                    r0 = result[0]
                    timestamps = r0.get("timestamp", [])
                    quote = r0["indicators"]["quote"][0]
                    df = pd.DataFrame(
                        {
                            "date": [datetime.utcfromtimestamp(t).date() for t in timestamps],
                            "open": quote.get("open"),
                            "high": quote.get("high"),
                            "low": quote.get("low"),
                            "close": quote.get("close"),
                            "volume": quote.get("volume"),
                        }
                    )
                    out_path = stock_price_dir(code) / f"yahoo_{code}_{year}.csv"
                    df.to_csv(out_path, index=False, encoding="utf-8-sig")
                    logger.info(f"[已存檔] {out_path}（{len(df)} 筆）")

                checkpoint[key] = "done"
                save_checkpoint(checkpoint)
                no_data_years = 0
            except requests.exceptions.HTTPError as e:
                # Yahoo 對「該股票在這段期間還沒上市」一律回 400。這不是失敗，
                # 重跑一百次也不會有資料，所以記成已完成、不要再浪費請求，
                # 也不要用 ERROR 洗版（新股動輒有十幾個上市前的年份）。
                if getattr(e.response, "status_code", None) == 400:
                    no_data_years += 1
                    logger.info(f"[尚未上市／無此期間資料] {key}")
                    checkpoint[key] = "done"
                    save_checkpoint(checkpoint)
                    # 連續多年都沒資料 → 這檔是近年才上市，直接跳到下一檔
                    if no_data_years >= NO_DATA_YEARS_BEFORE_SKIP:
                        logger.info(f"[略過] {code} 連續 {no_data_years} 年無資料，"
                                    f"應為近年新上市，跳過其餘年份")
                        break
                else:
                    logger.error(f"[失敗，之後重跑會補抓] {key}: {e}")
            except Exception as e:
                logger.error(f"[失敗，之後重跑會補抓] {key}: {e}")

            polite_sleep(*API_DELAY_RANGE)

        polite_sleep(*PER_STOCK_COOLDOWN_RANGE)


# ============================== 3. 財經新聞與論壇（帶時間戳記）==============================
# 這一段已經拆到獨立檔案 tw_text_crawler.py（鉅亨網新聞／Yahoo 新聞／股市爆料同學會），
# 它會從本檔匯入共用的 Session、重試、延遲、robots 檢查與 checkpoint，程式碼只有一份。
# 本檔的 main() 執行文本任務時會延遲匯入該模組（放在函式內，避免循環匯入）。
#
# 文本蒐集的預設期間：與 davis_tw 的研究期間一致。放在這裡是因為兩支程式都要用，
# tw_text_crawler 會從本檔匯入。
TEXT_START_DATE = date(2019, 1, 1)
TEXT_END_DATE = date(2024, 12, 31)


# ============================== 合併輸出 ==============================

def merge_stock_prices() -> None:
    """把每檔股票逐年的股價 CSV 合併成一個檔案，存在該股票自己的 price/ 資料夾底下"""
    for stock in STOCK_LIST:
        code = stock["code"]
        pattern_dir = stock_price_dir(code)
        files = sorted(pattern_dir.glob(f"yahoo_{code}_*.csv"))
        files = [f for f in files if "merged" not in f.stem]
        if not files:
            continue
        dfs = [pd.read_csv(f) for f in files]
        merged = pd.concat(dfs, ignore_index=True)
        out_path = pattern_dir / f"yahoo_{code}_merged.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"[合併完成] {out_path}（共 {len(merged)} 筆）")


# ============================== 主流程 ==============================

def main(argv=None) -> int:
    """
    子任務可個別執行，方便分次跑（每個任務都有 checkpoint，中斷後重跑會接續）：

        python tw_stock_crawler.py price      # 股價（Yahoo 歷史報價，逐年）
        python tw_stock_crawler.py cnyes      # 鉅亨網新聞（可回溯 2020–2024，約 2.5 小時）
        python tw_stock_crawler.py yahoo-news # Yahoo 個股新聞近期快照
        python tw_stock_crawler.py cmoney     # 股市爆料同學會（需 Selenium）
        python tw_stock_crawler.py all        # 以上全部

    常用選項：
        --max-stocks 25            只抓清單前 N 檔（預設全部）
        --stocks 2330 1101         指定股票代號
        --start 2020-01-01 --end 2024-12-31   鉅亨網的回溯區間
        --loose                    鉅亨網放寬比對（收「標題提到公司名」但未標記代號的新聞）
        --show-browser             同學會不使用無頭瀏覽器
    """
    global STOCK_LIST

    ap = argparse.ArgumentParser(description="台股資料蒐集（股價／新聞／論壇）")
    ap.add_argument("task", nargs="?", default="all",
                    choices=["all", "price", "cnyes", "yahoo-news", "cmoney"],
                    help="要執行的子任務，預設 all")
    ap.add_argument("--stocks", nargs="*", default=None, help="指定股票代號")
    ap.add_argument("--max-stocks", type=int, default=None, help="只抓清單前 N 檔")
    ap.add_argument("--start", default=str(TEXT_START_DATE), help="鉅亨網回溯起日")
    ap.add_argument("--end", default=str(TEXT_END_DATE), help="鉅亨網回溯迄日")
    ap.add_argument("--loose", action="store_true", help="鉅亨網放寬個股比對")
    ap.add_argument("--show-browser", action="store_true", help="同學會顯示瀏覽器視窗")
    ap.add_argument("--rebuild-list", action="store_true",
                    help="重建股票清單（依市值重新挑選每產業前 N 檔），舊清單會備份")
    args = ap.parse_args(argv)

    STOCK_LIST = build_stock_list_by_yahoo_category(STOCKS_PER_CATEGORY,
                                                    rebuild=args.rebuild_list)
    if args.stocks:
        wanted = set(args.stocks)
        STOCK_LIST = [s for s in STOCK_LIST if s["code"] in wanted]
        missing = wanted - {s["code"] for s in STOCK_LIST}
        if missing:
            logger.warning(f"清單中沒有這些股票，已略過：{sorted(missing)}")
    elif args.max_stocks:
        STOCK_LIST = STOCK_LIST[:args.max_stocks]

    if not STOCK_LIST:
        logger.error("沒有可抓的股票，請確認 --stocks 或產業分類清單")
        return 1

    logger.info(f"任務：{args.task}｜股票 {len(STOCK_LIST)} 檔"
                f"（{', '.join(s['code'] for s in STOCK_LIST[:8])}{' ...' if len(STOCK_LIST) > 8 else ''}）")

    checkpoint = load_checkpoint()
    text_start = datetime.strptime(args.start, "%Y-%m-%d").date()
    text_end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.task in ("all", "price"):
        logger.info(f"股價抓取區間：{START_DATE} ~ {END_DATE}（約 15 年）")
        crawl_yahoo_price(checkpoint)
        merge_stock_prices()

    if args.task in ("all", "cnyes", "yahoo-news", "cmoney"):
        # 延遲匯入：tw_text_crawler 會從本檔匯入共用工具，放在函式內才不會循環匯入
        from tw_text_crawler import run_text_tasks

        run_text_tasks(STOCK_LIST, checkpoint,
                       task="all" if args.task == "all" else args.task,
                       start=text_start, end=text_end,
                       strict=not args.loose, headless=not args.show_browser)

    logger.info("抓取完成。接著可跑：python davis_tw_all.py run --experiments E0 E1 E2 E3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
