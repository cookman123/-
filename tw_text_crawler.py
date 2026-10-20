#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股文本資料蒐集（財經新聞與論壇）
====================================================================
本檔從 tw_stock_crawler.py 拆出來，專責「文本」資料軸；股價仍留在主爬蟲。
共用的基礎設施（Session、重試、延遲、robots 檢查、checkpoint、資料夾、
股票清單）一律從 tw_stock_crawler 匯入，程式碼只維護一份。

兩種用法等價，挑順手的：
    python tw_text_crawler.py cnyes --max-stocks 25        # 直接跑這支
    python tw_stock_crawler.py cnyes --max-stocks 25       # 主爬蟲會轉呼叫這裡

子任務：
    cnyes       鉅亨網台股新聞（可回溯，預設 2020-01-01 ~ 2024-12-31，約 2.5 小時）
    yahoo-news  Yahoo 個股新聞（初始頁 + 載入更多，每檔可到上百則）
    cmoney      股市爆料同學會（需 Selenium）
    all         以上全部

選項：--stocks / --max-stocks / --start / --end / --loose / --show-browser / --force

同學會任務的穩定性：每頁都有載入逾時、失敗就重開瀏覽器、每 40 檔主動輪替一次、
連續失敗 8 檔就中止並報錯；同一天已抓過的股票預設跳過（要重抓加 --force）。
"""

import argparse
import base64
import hashlib
import html
import json
import re
import time
import urllib.robotparser as robotparser
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# 共用基礎設施一律取自主爬蟲，不重複實作
from tw_stock_crawler import (
    API_DELAY_RANGE,
    DATA_DIR,
    PER_STOCK_COOLDOWN_RANGE,
    STOCKS_PER_CATEGORY,
    TEXT_END_DATE,
    TEXT_START_DATE,
    WEB_DELAY_RANGE,
    build_stock_list_by_yahoo_category,
    http_get,
    load_checkpoint,
    logger,
    polite_sleep,
    robots_allowed,
    save_checkpoint,
    session,
    stock_news_dir,
    task_key,
)


# ============================== 3. 財經新聞與論壇（帶時間戳記）==============================
# 這一段在 2026/09 重寫過。舊版只解析 Yahoo 新聞頁的 DOM，而 DOM 上只顯示
# 相對時間（「1 小時前」「上個月」「2 個月前」），還原不到「哪一天」，
# 所以輸出的 CSV 裡 time 欄全是空的。沒有發布時間就無法把新聞對齊到交易日，
# 文本特徵在方法上不能用（硬套會造成前視偏誤）。
#
# 三個來源，都輸出精確到秒的時間戳（欄位名稱一律是 date，ISO 8601、台北時間）：
#
#   cnyes  鉅亨網台股新聞分類 API —— **要回溯歷史就用這個**
#          https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?startAt=&endAt=
#          每筆有 publishAt（epoch 秒）與 stock 標記（該則新聞關聯到哪些個股），
#          本腳本就用這個標記把新聞分流給各檔股票，掃一次分類就能供應所有目標股，
#          比逐股查詢省下數十倍請求。
#          注意：鉅亨的關鍵字搜尋端點 search/news 只准查最近 1 年（更早會回 422
#          "start at 的時間必須大於 -1 year."），所以回溯一定要走分類端點。
#          分類端點單次查詢可翻頁數有上限（約 50 頁），故把期間切成一週一窗。
#
#   yahoo  Yahoo 股市個股新聞頁
#          https://tw.stock.yahoo.com/quote/{code}.TW/news
#          時間藏在頁面內嵌的 JSON（"pubtime":epoch 毫秒），伺服器端就渲染好了，
#          用 requests 就拿得到，不需要 Selenium。初始頁只有 20 則，其餘靠
#          「載入更多」的串流端點往回翻（見下方 YAHOO_STREAM_URL 的說明），
#          可取到伺服器持有的全部庫存。
#          實測：1101 共 111 則、回溯約一年；2330／2317 各 200 則，但新聞量大，
#          只涵蓋數天到兩週。也就是說「則數」有解，「回溯期間」仍受新聞密度限制。
#
#   cmoney 股市爆料同學會（論壇資料軸，近期快照）
#          https://www.cmoney.tw/forum/stock/{code}
#          時間在 window.__NUXT__ 的 createTime（epoch 毫秒），另有 bullOrBear
#          （看多／看空標記）與 emojiCount（可當推噓比一類的結構訊號）。
#          它的 JSON API 需要 X-Version: 3.0 標頭，補上後仍回 401（要授權 token），
#          未登入拿不到，所以改用 Selenium 讀伺服器端算好的 __NUXT__ 狀態。
#          另注意：cmoney 的 TLS 憑證在 Python 3.13 會驗證失敗
#          （Missing Subject Key Identifier），所以連 robots.txt 都改用瀏覽器讀。
#          限制：同樣是近期快照（單檔約一週～數十則）。
#
# ⚠️ 研究設計上的重點：Yahoo 的庫存上限約 200 則、同學會只有近期快照，
#    兩者都補不出 2020–2024 的歷史（大型股尤其明顯，200 則只涵蓋幾天）。
#    要跑那段期間的文本消融實驗，新聞請用 cnyes。

TAIPEI_TZ = timezone(timedelta(hours=8))


def _strip_html(text: str) -> str:
    """
    把 API 回傳的 HTML 內文轉成純文字（TF-IDF 前處理只需要文字）。

    先解碼、再移除標籤 —— 順序反了會出事：鉅亨回傳的是「實體編碼過」的標籤
    （&lt;p&gt;），若先移除 <...> 再解碼，編碼過的標籤會躲過移除、解碼後反而
    變成真正的 <p>，最後被斷詞成 p、br 之類的垃圾詞灌進 TF-IDF。
    （這個順序錯誤曾讓 145,755 則新聞全部帶著 HTML 標籤進入特徵。）
    最後再解碼一次，處理內容本身二次編碼的情況。
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _write_text_csv(rows: list, path: Path, dedup_key: str) -> int:
    """寫檔並與既有資料合併去重；回傳最終列數。"""
    df = pd.DataFrame(rows)
    if path.exists():
        try:
            old = pd.read_csv(path, encoding="utf-8-sig")
            if dedup_key in old.columns:
                df = pd.concat([old, df], ignore_index=True)
        except Exception as e:
            logger.warning(f"既有檔案讀取失敗（{e}），直接覆寫：{path.name}")
    if "date" not in df.columns:
        return 0
    # 舊版沒有時間戳的資料列（date 空白）在這裡一併清掉 —— 留著會在特徵階段
    # 被靜默丟棄，反而讓人以為文本資料量比實際多
    df = df[df["date"].notna() & (df["date"].astype(str).str.strip() != "")]
    df = df.drop_duplicates(subset=[dedup_key], keep="last").sort_values("date")
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()], errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


# ------------------------------ 3-1. 鉅亨網（歷史回溯） ------------------------------

CNYES_CATEGORY_URL = "https://api.cnyes.com/media/api/v1/newslist/category/{cat}"
CNYES_CATEGORIES = ("tw_stock",)   # 可再加 "tw_stock_news"、"headline"，代價是請求數倍增
CNYES_PER_PAGE = 30
CNYES_WINDOW_DAYS = 7              # 台股新聞約每日 3 頁，一週約 21 頁，安全落在翻頁上限內
CNYES_MAX_PAGES = 45               # API 約在第 50 頁後回 422，設 45 為安全上限



def _time_windows(start: date, end: date, days: int = CNYES_WINDOW_DAYS):
    """把研究期間切成可翻完頁的時間窗。"""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def _cnyes_page(cat: str, start: date, stop: date, page: int):
    """取一頁分類新聞；失敗（含 422 翻過頭）回 None，由呼叫端決定停止。"""
    params = {
        "limit": CNYES_PER_PAGE,
        "page": page,
        "startAt": int(datetime.combine(start, datetime.min.time()).timestamp()),
        "endAt": int(datetime.combine(stop, datetime.max.time()).timestamp()),
    }
    try:
        payload = http_get(CNYES_CATEGORY_URL.format(cat=cat), params=params, as_json=True)
    except Exception as e:
        logger.warning(f"[鉅亨] {cat} {start}~{stop} 第 {page} 頁取得失敗：{e}")
        return None
    polite_sleep(*API_DELAY_RANGE)
    return (payload or {}).get("items")


def crawl_cnyes_news(stock_list: list, checkpoint: dict,
                     start: date = TEXT_START_DATE, end: date = TEXT_END_DATE,
                     strict: bool = True, categories=CNYES_CATEGORIES) -> None:
    """
    掃過鉅亨網台股新聞分類，依 API 的 stock 標記分流到傳入的各檔股票。

    strict=True（預設）只收「stock 標記含該代號」的文章 —— 最乾淨的對應方式。
    strict=False 另收「標題或摘要提到公司名」的文章，recall 較高但可能誤收
    （例如「中鋼」會命中「中鋼構」的新聞）。
    """
    logger.info(f"===== 開始抓取：鉅亨網新聞（{start} ~ {end}）=====")
    if not robots_allowed(CNYES_CATEGORY_URL.format(cat="tw_stock")):
        logger.warning("[鉅亨] robots.txt 不允許，略過")
        return

    targets = {s["code"]: s["name"] for s in stock_list}
    # checkpoint 要綁定「這次的股票集合」：只記日期的話，之後換一批股票重跑會被
    # 誤判為已完成而整段跳過，新股票就永遠抓不到資料
    fingerprint = hashlib.md5(",".join(sorted(targets)).encode("utf-8")).hexdigest()[:8]
    for w_start, w_stop in _time_windows(start, end):
        key = task_key("cnyes_news", fingerprint, f"{w_start:%Y%m%d}")
        if checkpoint.get(key) == "done":
            logger.info(f"[鉅亨] {w_start}~{w_stop} 已完成，跳過")
            continue

        buckets = {c: {} for c in targets}
        scanned = 0
        for cat in categories:
            page = 1
            while page <= CNYES_MAX_PAGES:
                items = _cnyes_page(cat, w_start, w_stop, page)
                if not items:
                    break
                data = items.get("data") or []
                if not data:
                    break
                scanned += len(data)
                for x in data:
                    nid = x.get("newsId")
                    tags = x.get("stock") or []
                    title = re.sub(r"</?mark>", "", x.get("title") or "")
                    summary = x.get("summary") or ""
                    for code, name in targets.items():
                        tagged = code in tags
                        if not tagged and not (not strict and (name in title or name in summary)):
                            continue
                        buckets[code][nid] = {
                            "date": datetime.fromtimestamp(x["publishAt"], TAIPEI_TZ).isoformat(),
                            "title": title,
                            "text": _strip_html(x.get("content") or "") or summary,
                            "summary": summary,
                            "source": "cnyes",
                            "url": f"https://news.cnyes.com/news/id/{nid}",
                            "news_id": nid,
                            "stock_tags": ",".join(tags),
                            "tagged": int(tagged),
                        }
                if page >= (items.get("last_page") or 1):
                    break
                page += 1

        hit = 0
        for code, rows in buckets.items():
            if rows:
                _write_text_csv(list(rows.values()),
                                stock_news_dir(code) / f"cnyes_news_{code}.csv", "news_id")
                hit += len(rows)
        logger.info(f"[鉅亨] {w_start}~{w_stop} 掃過 {scanned} 則，命中目標個股 {hit} 則")

        # 每個時間窗結束就落地並記 checkpoint，中斷後重跑會自動接續
        checkpoint[key] = "done"
        save_checkpoint(checkpoint)

    for code, name in targets.items():
        path = stock_news_dir(code) / f"cnyes_news_{code}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            logger.info(f"[鉅亨] {code} {name}：共 {len(df)} 則"
                        f"（{str(df['date'].min())[:10]} ~ {str(df['date'].max())[:10]}）")
        else:
            logger.warning(f"[鉅亨] {code} {name}：該期間沒有抓到任何新聞")


# ------------------------------ 3-2. Yahoo 股市新聞（近期快照） ------------------------------

YAHOO_NEWS_URL = "https://tw.stock.yahoo.com/quote/{code}.TW/news"

# 頁面內嵌 JSON 的每筆記錄形如：
#   "pubtime":1789041671000,...,"url":"https://tw.stock.yahoo.com/news/....html",
#   ...,"title":"水泥雙雄8月營收不同調　台泥年增1成",...,"publisher":"中央社財經"
# 欄位順序不固定，所以先用 pubtime 定位，再在其後的片段裡各自找 url／title／publisher。
YAHOO_TS_RE = re.compile(r'"pubtime":"?(\d{13})')
YAHOO_URL_RE = re.compile(r'"(?:url|link)":"(https://tw\.stock\.yahoo\.com/news/[^"]+?\.html)"')
YAHOO_TITLE_RE = re.compile(r'"title":"([^"]{4,300}?)"')
YAHOO_PUB_RE = re.compile(r'"publisher":"([^"]{0,60}?)"')
YAHOO_UUID_RE = re.compile(r'"id":"([0-9a-f]{8}-[0-9a-f-]{27})"')
YAHOO_SCAN_WINDOW = 4000           # 一筆記錄的欄位散落範圍
JS_SLASH = chr(92) + "u002F"       # 原始碼把斜線寫成 JS 跳脫序列，先還原成 / 再比對

# --- 「載入更多」：把每檔從 20 則提高到上百則（2026/09/12 新增） ---
# 個股新聞頁初始只渲染 20 則，其餘靠捲動載入。捲動觸發的是這個端點：
#   POST /_td-stock/api/resource   resource=ApacStreamService, operation=read
# 用 requests 直接打就會成功（不需要瀏覽器、不需要 crumb），回應是乾淨的 JSON：
#   {"g0":{"data":{"stream_items":[{link,pubtime,id,title,publisher,summary}...],
#                  "stream_pagination":{...nextParams.after...},
#                  "stream_remaining":N,"stream_total":M}}}
# 兩個關鍵：
#   1. params.ui 一定要帶 "pubtime_maxage": 0，否則回應不含 pubtime（只有相對時間可用，等於白抓）
#   2. 翻頁靠回應裡的 nextParams.after（base64 cursor），不是 offset；
#      第一個 cursor 由我們用初始頁最後一則的 pubtime + uuid 自行組出
# 實測上限即伺服器回報的 stream_total：1101 為 111 則（可回溯約 10 個月）、
# 2330 為 200 則（但新聞量大，只涵蓋數天）。
YAHOO_STREAM_URL = ("https://tw.stock.yahoo.com/_td-stock/api/resource"
                    "?bkt=&crumb=undefined&device=desktop&ecma=modern"
                    "&feature=useCG%2CuseCGV2&intl=tw&lang=zh-Hant-TW&partner=none"
                    "&region=TW&site=finance&tz=Asia%2FTaipei&ver=1.4.914")
YAHOO_LIST_ALIAS = "finance-TW-zh-Hant-TW-ticker-all"
YAHOO_LIST_ID = "7d31ccc0-9394-11ea-bbbf-8277882e32ab"
YAHOO_BATCH = 20                   # 伺服器每批固定回 20 則
YAHOO_MAX_BATCHES = 15             # 防呆上限（20×15 = 300 則）


def _unescape_js(s: str) -> str:
    """把 JS 字串字面量的跳脫（u3000 這類）還原成正常文字。"""
    try:
        return json.loads('"' + s.rstrip(chr(92)) + '"')
    except Exception:
        return s


def parse_yahoo_news_page(html: str) -> list:
    """
    從頁面原始碼的內嵌 JSON 取出「時間 + 網址 + 標題 + 來源 + uuid」。

    不要改回去解析 DOM —— DOM 只顯示相對時間（「上個月」「2 個月前」），
    無法還原到日，正是舊版 time 欄位空白的原因。
    """
    rows = []
    for m in YAHOO_TS_RE.finditer(html):
        chunk = html[m.end(): m.end() + YAHOO_SCAN_WINDOW].replace(JS_SLASH, "/")
        u = YAHOO_URL_RE.search(chunk)
        if not u:                  # 廣告等非新聞記錄沒有文章網址
            continue
        t = YAHOO_TITLE_RE.search(chunk)
        p = YAHOO_PUB_RE.search(chunk)
        uid = YAHOO_UUID_RE.search(chunk)
        rows.append({
            "date": datetime.fromtimestamp(int(m.group(1)) / 1000, TAIPEI_TZ).isoformat(),
            "title": _unescape_js(t.group(1)) if t else "",
            "text": "",
            "source": _unescape_js(p.group(1)) if p else "yahoo",
            "url": _unescape_js(u.group(1)),
            "_ts": int(m.group(1)),
            "_uuid": uid.group(1) if uid else "",
        })
    dedup = {r["url"]: r for r in rows}     # 同一則在頁面 JSON 可能出現多次
    return sorted(dedup.values(), key=lambda r: r["date"])


def _yahoo_cursor(index: int, pubtime_ms: int, uuid: str) -> str:
    """組出串流分頁用的 base64 cursor。"""
    payload = {"index": index, "sort_fields": [pubtime_ms, uuid], "cursor_version": 1}
    return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _yahoo_stream_body(code: str, after: str) -> dict:
    syms = [f"{code}.TW", f"{code}.TWO"]
    return {
        "requests": {"g0": {
            "resource": "ApacStreamService",
            "operation": "read",
            "params": {
                "category": "CUSTOM:NEWS",
                # pubtime_maxage=0 才會回傳發布時間；summary 順便帶回摘要當內文
                "ui": {"view": "mega", "pubtime_maxage": 0, "summary": True},
                "useNCP": True, "useCG": True, "useCGV2": True,
                "cgParams": {"overrides": {
                    "listAlias": YAHOO_LIST_ALIAS,
                    "listId": YAHOO_LIST_ID,
                    "queryVariables": json.dumps({"tickerSymbol": syms}),
                }},
                "ncpParams": {
                    "query": {"id": "twfinance-partner-stream", "namespace": "abu",
                              "version": "v1", "s": syms},
                    "body": {"gqlVariables": {"lightyearList": {
                        "pagination": {"nextParams": {"after": after, "first": YAHOO_BATCH}}}}},
                },
                "batches": {"pagination": True, "size": YAHOO_BATCH, "timeout": 1300, "total": 200},
            }}},
        "context": {"feature": "useCG,useCGV2", "bkt": "", "device": "desktop", "intl": "tw",
                    "lang": "zh-Hant-TW", "partner": "none", "region": "TW", "site": "finance",
                    "tz": "Asia/Taipei", "ver": "1.4.914", "ecma": "modern"},
    }


def _yahoo_stream_page(code: str, after: str):
    """取一批串流新聞，回傳 (資料列, 下一個 cursor, 剩餘則數)。"""
    resp = session.post(
        YAHOO_STREAM_URL,
        data=json.dumps(_yahoo_stream_body(code, after)).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "X-Requested-With": "XMLHttpRequest",
                 "Origin": "https://tw.stock.yahoo.com",
                 "Referer": YAHOO_NEWS_URL.format(code=code)},
        timeout=25,
    )
    if resp.status_code != 200:
        logger.warning(f"[Yahoo新聞] {code} 串流回應 HTTP {resp.status_code}，停止翻頁")
        return [], None, 0
    try:
        data = resp.json()["g0"]["data"]
    except Exception as e:
        logger.warning(f"[Yahoo新聞] {code} 串流回應無法解析（{e}），停止翻頁")
        return [], None, 0

    rows = []
    for it in data.get("stream_items") or []:
        link, ts = it.get("link") or it.get("url"), it.get("pubtime")
        if not link or not ts:          # 廣告等非新聞項目沒有這兩個欄位
            continue
        rows.append({
            "date": datetime.fromtimestamp(int(ts) / 1000, TAIPEI_TZ).isoformat(),
            "title": it.get("title") or "",
            "text": it.get("summary") or "",
            "source": it.get("publisher") or "yahoo",
            "url": link,
            "_ts": int(ts),
            "_uuid": it.get("id") or "",
        })
    after_next = (data.get("stream_pagination", {}).get("gqlVariables", {})
                      .get("lightyearList", {}).get("pagination", {})
                      .get("nextParams", {}).get("after"))
    return rows, after_next, int(data.get("stream_remaining") or 0)


def fetch_yahoo_news(code: str, max_batches: int = YAHOO_MAX_BATCHES) -> list:
    """
    抓單檔的 Yahoo 新聞：初始頁 20 則 + 一路翻「載入更多」直到伺服器說沒有了。
    """
    html = http_get(YAHOO_NEWS_URL.format(code=code), as_json=False)
    polite_sleep(*WEB_DELAY_RANGE)
    initial = parse_yahoo_news_page(html) if html else []
    if not initial:
        return []

    items = {r["url"]: r for r in initial}
    last = min(initial, key=lambda r: r["_ts"])      # 最舊的一則才是往回翻的起點
    if not last["_uuid"]:
        logger.info(f"[Yahoo新聞] {code} 初始頁取不到文章 uuid，僅保留 {len(items)} 則")
        return list(items.values())

    after = _yahoo_cursor(len(initial), last["_ts"], last["_uuid"])
    for _ in range(max_batches):
        rows, after, remaining = _yahoo_stream_page(code, after)
        new = [r for r in rows if r["url"] not in items]
        for r in new:
            items[r["url"]] = r
        polite_sleep(*API_DELAY_RANGE)
        if not after or not new or remaining <= 0:
            break
    return sorted(items.values(), key=lambda r: r["date"])


def crawl_yahoo_news(stock_list: list, checkpoint: dict,
                     max_batches: int = YAHOO_MAX_BATCHES) -> None:
    """抓 Yahoo 個股新聞（初始頁 + 載入更多，requests 即可，不需 Selenium）。"""
    logger.info("===== 開始抓取：Yahoo 股市新聞 =====")
    for stock in stock_list:
        code, name = stock["code"], stock["name"]
        url = YAHOO_NEWS_URL.format(code=code)
        if not robots_allowed(url):
            logger.warning(f"[Yahoo新聞] robots.txt 不允許：{url}")
            continue

        try:
            rows = fetch_yahoo_news(code, max_batches)
        except Exception as e:
            logger.warning(f"[Yahoo新聞] {code} {name} 抓取失敗：{type(e).__name__}: {e}")
            continue

        if not rows:
            logger.warning(f"[Yahoo新聞] {code} {name} 解析不到內嵌 JSON —— "
                           f"版型可能已改，請重新檢查 YAHOO_TS_RE / YAHOO_URL_RE")
            continue

        # 內部欄位（翻頁用的時間戳與 uuid）不寫進 CSV
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        n = _write_text_csv(clean, stock_news_dir(code) / f"yahoo_news_{code}.csv", "url")
        logger.info(f"[Yahoo新聞] {code} {name}：本次 {len(clean)} 則"
                    f"（{clean[0]['date'][:10]} ~ {clean[-1]['date'][:10]}），檔案共 {n} 則")
        checkpoint[task_key("yahoo_news", code, "latest")] = datetime.now(TAIPEI_TZ).isoformat()
        save_checkpoint(checkpoint)
        polite_sleep(*PER_STOCK_COOLDOWN_RANGE)


# ------------------------------ 3-3. 股市爆料同學會（論壇） ------------------------------

CMONEY_FORUM_URL = "https://www.cmoney.tw/forum/stock/{code}"

# 從 window.__NUXT__ 撈文章陣列：物件同時具備 createTime 與 content 才算數
CMONEY_EXTRACT_JS = r"""
function findArticles(o, depth, seen){
  if(!o || typeof o !== 'object' || depth > 8 || seen.has(o)) return null;
  seen.add(o);
  if(Array.isArray(o) && o.length && o[0] && typeof o[0] === 'object'
     && 'createTime' in o[0] && 'content' in o[0]) return o;
  for(var k in o){ var r = findArticles(o[k], depth+1, seen); if(r) return r; }
  return null;
}
var arr = window.__NUXT__ ? findArticles(window.__NUXT__, 0, new Set()) : null;
if(!arr) return [];
return arr.map(function(a){
  var c = a.content || {};
  var tags = (c.commodityTags || []).map(function(t){ return t.key + ':' + t.bullOrBear; });
  var e = a.emojiCount || {};
  return {id: String(a.id), createTime: a.createTime, title: c.title || '',
          text: c.text || '', tags: tags.join(','), commentCount: a.commentCount || 0,
          like: e.like||0, dislike: e.dislike||0, laugh: e.laugh||0, money: e.money||0,
          shock: e.shock||0, cry: e.cry||0, think: e.think||0, angry: e.angry||0};
});
"""


# --- 逾時與重啟策略（2026/09/11 修正） ---
# 症狀：整晚每一檔都以 120 秒逾時失敗
#   HTTPConnectionPool(host='localhost', port=53325): Read timed out.
# 那個 localhost 連線是 Python 跟 ChromeDriver 之間的通道，代表瀏覽器本身沒回應。
# 原因是舊版用「同一個 driver 跑完整份清單」（數百檔、數小時），headless Chrome
# 累積大量 SPA 分頁後記憶體膨脹卡死；而迴圈只是 catch 住例外繼續跑，
# 於是每檔都空等 120 秒，整晚毫無產出。
# 對策：頁面層級逾時 + 連續失敗就重開瀏覽器 + 固定檔數輪替 + 連續失敗過多直接中止。
CMONEY_PAGE_LOAD_TIMEOUT = 45      # 單頁載入上限（秒）
CMONEY_SCRIPT_TIMEOUT = 30         # execute_script 上限（秒）
CMONEY_COMMAND_TIMEOUT = 60        # 與 ChromeDriver 通訊的上限，取代預設的 120 秒
CMONEY_DRIVER_RECYCLE = 40         # 每抓幾檔就重開一次瀏覽器，避免記憶體累積
CMONEY_MAX_CONSECUTIVE_FAILS = 8   # 連續失敗幾檔就判定環境有問題並中止


def _build_selenium_driver(headless: bool = True):
    """建立 Selenium driver（需要 pip install selenium webdriver-manager）。"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.remote.remote_connection import RemoteConnection
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as e:
        raise RuntimeError("同學會需要 Selenium：pip install selenium webdriver-manager") from e

    # 瀏覽器沒回應時，讓 Python 端 60 秒就放棄，而不是預設的 120 秒。
    # Selenium 4.48 建議改用 ClientConfig，但 webdriver.Chrome() 目前不吃這個參數
    # （只有 Remote 吃），本機 Chrome 仍得走 set_timeout，故把棄用警告壓下來。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            RemoteConnection.set_timeout(CMONEY_COMMAND_TIMEOUT)
    except Exception:
        pass

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1280,2000")
    # 長時間連續抓取的穩定性選項：關掉不需要的子系統，降低記憶體與當機機率
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--blink-settings=imagesEnabled=false")   # 不載圖，省頻寬與記憶體
    opts.add_argument(f"--user-agent={session.headers['User-Agent']}")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(CMONEY_PAGE_LOAD_TIMEOUT)
    driver.set_script_timeout(CMONEY_SCRIPT_TIMEOUT)
    return driver


def _quit_driver_quietly(driver) -> None:
    """關閉 driver；已經卡死的 driver 連 quit() 都可能拋例外，這裡吞掉。"""
    try:
        driver.quit()
    except Exception as e:
        logger.debug(f"[同學會] 關閉瀏覽器時發生例外（可忽略）：{e}")


def _robots_allowed_via_driver(driver, url: str) -> bool:
    """
    用瀏覽器讀 robots.txt 再判斷。

    cmoney 的 TLS 憑證在 Python 3.13 會驗證失敗（Missing Subject Key Identifier），
    robots_allowed() 讀不到就會保守地回 False，等於整段被跳過；
    改用已經開好的瀏覽器取 robots.txt，就能照常做這項檢查。
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    try:
        driver.get(root + "/robots.txt")
        text = driver.find_element("tag name", "body").text
        rp = robotparser.RobotFileParser()
        rp.parse(text.splitlines())
        return rp.can_fetch(session.headers["User-Agent"], url)
    except Exception as e:
        logger.warning(f"[同學會] 以瀏覽器讀 robots.txt 失敗（{e}），保守起見跳過")
        return False


def _fetch_cmoney_articles(driver, code: str, scroll_rounds: int) -> list:
    """載入單檔的同學會頁面並取出文章陣列（逾時或當機由呼叫端處理）。"""
    driver.get(CMONEY_FORUM_URL.format(code=code))
    time.sleep(4)
    for _ in range(scroll_rounds):       # 捲動讓前端把更多貼文塞進狀態
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    return driver.execute_script(CMONEY_EXTRACT_JS) or []


def crawl_cmoney_forum(stock_list: list, checkpoint: dict,
                       headless: bool = True, scroll_rounds: int = 6,
                       skip_done_today: bool = True) -> None:
    """
    用 Selenium 讀同學會頁面的 __NUXT__ 狀態，取得帶時間戳的貼文。

    長時間連續抓取的穩定性處理（見上方 CMONEY_* 常數的說明）：
      * 每頁都有載入與腳本逾時，卡住的頁面會很快放棄而不是空等
      * 任何一檔失敗就重開瀏覽器，避免一個壞掉的 session 拖垮後面全部
      * 每 CMONEY_DRIVER_RECYCLE 檔主動重開一次，控制記憶體
      * 連續失敗超過門檻就中止並明確報錯，不再整晚空轉
    """
    logger.info("===== 開始抓取：股市爆料同學會（論壇）=====")
    forum_dir = DATA_DIR / "forum"
    forum_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(TAIPEI_TZ).date().isoformat()
    driver = _build_selenium_driver(headless)
    consecutive_fails = 0
    processed = 0

    try:
        if not _robots_allowed_via_driver(driver, CMONEY_FORUM_URL.format(code="1101")):
            logger.warning("[同學會] robots.txt 不允許，略過")
            return

        for stock in stock_list:
            code, name = stock["code"], stock["name"]

            # 同學會是「當下快照」，同一天重跑同一檔沒有意義；中斷後重跑可直接接續
            done_at = checkpoint.get(task_key("cmoney_forum", code, "latest"), "")
            if skip_done_today and done_at.startswith(today):
                logger.info(f"[同學會] {code} {name} 今天已抓過，跳過")
                continue

            # 定期輪替瀏覽器，避免長時間累積導致無回應
            if processed and processed % CMONEY_DRIVER_RECYCLE == 0:
                logger.info(f"[同學會] 已處理 {processed} 檔，重開瀏覽器以釋放記憶體")
                _quit_driver_quietly(driver)
                driver = _build_selenium_driver(headless)

            try:
                raw = _fetch_cmoney_articles(driver, code, scroll_rounds)
                consecutive_fails = 0
            except Exception as e:
                consecutive_fails += 1
                logger.warning(f"[同學會] {code} {name} 抓取失敗"
                               f"（連續 {consecutive_fails} 次）：{type(e).__name__}: {e}")
                # 失敗多半代表瀏覽器已經沒回應，重開一個再繼續
                _quit_driver_quietly(driver)
                if consecutive_fails >= CMONEY_MAX_CONSECUTIVE_FAILS:
                    logger.error(f"[同學會] 連續 {consecutive_fails} 檔失敗，判定瀏覽器環境有問題，"
                                 f"中止本任務（已完成 {processed} 檔，重跑會自動接續）。"
                                 f"請確認 Chrome 版本、記憶體是否足夠，或加 --show-browser 觀察畫面")
                    return
                driver = _build_selenium_driver(headless)
                continue
            finally:
                processed += 1

            if not raw:
                logger.warning(f"[同學會] {code} {name} 讀不到 __NUXT__ 文章陣列 —— "
                               f"版型可能已改，請重新檢查 CMONEY_EXTRACT_JS")
                continue

            rows = []
            for a in raw:
                if not a.get("createTime"):
                    continue
                rows.append({
                    "date": datetime.fromtimestamp(a["createTime"] / 1000, TAIPEI_TZ).isoformat(),
                    "title": a.get("title") or "",
                    "text": a.get("text") or "",
                    "source": "cmoney_forum",
                    "url": f"https://www.cmoney.tw/forum/article/{a['id']}",
                    "article_id": a["id"],
                    "tags": a.get("tags") or "",       # 代號:看多(1)/看空(2)/中性(0)
                    "comment_count": a.get("commentCount", 0),
                    # 表情反應可當推噓比一類的結構訊號
                    "like": a.get("like", 0), "dislike": a.get("dislike", 0),
                    "laugh": a.get("laugh", 0), "money": a.get("money", 0),
                    "shock": a.get("shock", 0), "cry": a.get("cry", 0),
                    "think": a.get("think", 0), "angry": a.get("angry", 0),
                })

            if rows:
                n = _write_text_csv(rows, forum_dir / f"cmoney_{code}.csv", "article_id")
                span = f"{min(r['date'] for r in rows)[:10]} ~ {max(r['date'] for r in rows)[:10]}"
                logger.info(f"[同學會] {code} {name}：本次 {len(rows)} 則（{span}），檔案共 {n} 則")
                checkpoint[task_key("cmoney_forum", code, "latest")] = datetime.now(TAIPEI_TZ).isoformat()
                save_checkpoint(checkpoint)
            polite_sleep(*WEB_DELAY_RANGE)
    finally:
        _quit_driver_quietly(driver)
        logger.info(f"[同學會] 本次處理 {processed} 檔")


# ============================== 主流程 ==============================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="台股文本資料蒐集（新聞與論壇）")
    ap.add_argument("task", nargs="?", default="all",
                    choices=["all", "cnyes", "yahoo-news", "cmoney"],
                    help="要執行的子任務，預設 all")
    ap.add_argument("--stocks", nargs="*", default=None, help="指定股票代號")
    ap.add_argument("--max-stocks", type=int, default=None, help="只抓清單前 N 檔")
    ap.add_argument("--start", default=str(TEXT_START_DATE), help="鉅亨網回溯起日")
    ap.add_argument("--end", default=str(TEXT_END_DATE), help="鉅亨網回溯迄日")
    ap.add_argument("--loose", action="store_true", help="鉅亨網放寬個股比對")
    ap.add_argument("--show-browser", action="store_true", help="同學會顯示瀏覽器視窗")
    ap.add_argument("--force", action="store_true",
                    help="同學會：同一天已抓過的股票也重抓（預設會跳過，方便中斷後接續）")
    args = ap.parse_args(argv)

    stock_list = build_stock_list_by_yahoo_category(STOCKS_PER_CATEGORY)
    if args.stocks:
        wanted = set(args.stocks)
        stock_list = [s for s in stock_list if s["code"] in wanted]
        missing = wanted - {s["code"] for s in stock_list}
        if missing:
            logger.warning(f"清單中沒有這些股票，已略過：{sorted(missing)}")
    elif args.max_stocks:
        stock_list = stock_list[:args.max_stocks]

    if not stock_list:
        logger.error("沒有可抓的股票，請確認 --stocks 或產業分類清單")
        return 1

    logger.info(f"任務：{args.task}｜股票 {len(stock_list)} 檔"
                f"（{', '.join(s['code'] for s in stock_list[:8])}"
                f"{' ...' if len(stock_list) > 8 else ''}）")

    checkpoint = load_checkpoint()
    run_text_tasks(stock_list, checkpoint, task=args.task,
                   start=datetime.strptime(args.start, "%Y-%m-%d").date(),
                   end=datetime.strptime(args.end, "%Y-%m-%d").date(),
                   strict=not args.loose, headless=not args.show_browser,
                   skip_done_today=not args.force)

    logger.info("文本抓取完成。接著可跑：python davis_tw_all.py run --experiments E0 E1 E2 E3")
    return 0


def run_text_tasks(stock_list: list, checkpoint: dict, task: str = "all",
                   start: date = TEXT_START_DATE, end: date = TEXT_END_DATE,
                   strict: bool = True, headless: bool = True,
                   skip_done_today: bool = True) -> None:
    """三個文本任務的統一入口；tw_stock_crawler.main() 也是呼叫這個。"""
    if task in ("all", "cnyes"):
        crawl_cnyes_news(stock_list, checkpoint, start, end, strict=strict)
    if task in ("all", "yahoo-news"):
        crawl_yahoo_news(stock_list, checkpoint)
    if task in ("all", "cmoney"):
        crawl_cmoney_forum(stock_list, checkpoint, headless=headless,
                           skip_done_today=skip_done_today)


if __name__ == "__main__":
    raise SystemExit(main())
