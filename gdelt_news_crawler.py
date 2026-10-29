#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
英文國際新聞（GDELT DOC 2.0 API）：只抓大波動日前後，之後每天抓最新一天。

為什麼要英文新聞
----------------
現有的國際新聞全部來自鉅亨網的中文分類，多半是翻譯或改寫外電。英文原文用來做
穩健性檢查：換成另一種語言、另一批媒體，「新聞量與負面比例只反映市場緊張度、
不能事前預示方向」的結論還成不成立。

為什麼不全抓
------------
GDELT 每 5 秒最多一次請求。逐日全抓 2017 起約一萬多次請求、要跑一整天以上，
而研究關心的只有大波動日附近。所以：
    --bigdays  依 results/volatility_news/big_days.csv，每個大波動日 t 抓 UTC
               的 t−3 ～ t+1 五天（GDELT DOC API 只回溯到 2017-01-01）
    --daily    每天抓前兩個 UTC 日（已抓過的自動跳過），接在 daily_update 之後跑

時間對齊
--------
GDELT 的 seendate 是 UTC。存檔時另存台北時間。台北 t 日 09:00 開盤＝UTC t 日 01:00，
所以 UTC t−1 整天的英文新聞都在台股 t 日開盤前，可直接接上「當日盤前」窗。

兩組查詢（事先固定）
--------------------
    market：美股、Fed、通膨、關稅、油價
    asia  ：台灣、台積電、半導體、晶片、南海
每組每天取 GDELT 依相關度排序的前 250 則（API 上限）。

輸出：data/intl_news_en/gdelt_{年}.csv
      欄位 date（台北時間）, seendate_utc, title, url, domain, language,
      sourcecountry, query, fetched_at；以 url 去重
斷點：data/intl_news_en/checkpoint.json，記錄已完成的（查詢, UTC 日）
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "intl_news_en"
CKPT = OUT / "checkpoint.json"
BIGDAYS = ROOT / "results" / "volatility_news" / "big_days.csv"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "Mozilla/5.0 (student thesis research; rate-limited)"}
DELAY = 6.0                      # GDELT 要求每 5 秒最多一次
EARLIEST = pd.Timestamp("2017-01-01")
TPE = timezone(timedelta(hours=8))

QUERIES = {
    "market": '("stock market" OR "wall street" OR "federal reserve" OR inflation OR tariff '
              'OR "oil price") sourcelang:english',
    "asia": '(Taiwan OR TSMC OR semiconductor OR chipmaker OR "chip stocks" OR "South China Sea") '
            'sourcelang:english',
}


def load_ckpt() -> set:
    if CKPT.exists():
        return set(tuple(x) for x in json.loads(CKPT.read_text(encoding="utf-8")))
    return set()


def save_ckpt(done: set) -> None:
    CKPT.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")


def fetch_day(query: str, day: pd.Timestamp) -> list:
    """抓單一查詢、單一 UTC 日的文章清單。429 時退避重試。"""
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": "250",
              "sort": "hybridrel",
              "startdatetime": f"{day:%Y%m%d}000000",
              "enddatetime": f"{day + pd.Timedelta(days=1):%Y%m%d}000000"}
    for k in range(6):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=90)
        except requests.RequestException as e:
            print(f"    連線失敗 {e}，{30 * (k + 1)} 秒後重試", flush=True)
            time.sleep(30 * (k + 1))
            continue
        if r.status_code == 429:
            print(f"    429 被限速，{30 * (k + 1)} 秒後重試", flush=True)
            time.sleep(30 * (k + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        txt = r.text.strip()
        if not txt or not txt.startswith("{"):
            return []                 # 當天沒有符合的文章時 GDELT 回空白
        return json.loads(txt).get("articles", []) or []
    raise RuntimeError("連續被限速，稍後再跑（已完成的部分有存斷點）")


def to_rows(arts: list, qname: str) -> pd.DataFrame:
    now = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for a in arts:
        sd = a.get("seendate", "")
        try:
            utc = datetime.strptime(sd, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({"date": utc.astimezone(TPE).strftime("%Y-%m-%d %H:%M:%S"),
                     "seendate_utc": utc.strftime("%Y-%m-%d %H:%M:%S"),
                     "title": a.get("title", ""), "url": a.get("url", ""),
                     "domain": a.get("domain", ""), "language": a.get("language", ""),
                     "sourcecountry": a.get("sourcecountry", ""),
                     "query": qname, "fetched_at": now})
    return pd.DataFrame(rows)


def append(df: pd.DataFrame) -> int:
    """依台北時間的年份寫入；以 url 去重。回傳新增筆數。"""
    if df.empty:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    added = 0
    for y, g in df.groupby(df["date"].str[:4]):
        f = OUT / f"gdelt_{y}.csv"
        old = pd.read_csv(f, encoding="utf-8-sig") if f.exists() else pd.DataFrame(columns=g.columns)
        new = g[~g["url"].isin(set(old["url"]))]
        if len(new):
            pd.concat([old, new], ignore_index=True).sort_values("date") \
                .to_csv(f, index=False, encoding="utf-8-sig")
            added += len(new)
    return added


def run(days: list) -> None:
    done = load_ckpt()
    todo = [(q, d) for d in days for q in QUERIES if (q, f"{d:%Y-%m-%d}") not in done]
    print(f"UTC 日 {len(days)} 天 × {len(QUERIES)} 組查詢，待抓 {len(todo)} 次"
          f"（約 {len(todo) * DELAY / 60:.0f} 分鐘）", flush=True)
    total = 0
    for i, (q, d) in enumerate(todo, 1):
        arts = fetch_day(QUERIES[q], d)
        n = append(to_rows(arts, q))
        total += n
        done.add((q, f"{d:%Y-%m-%d}"))
        save_ckpt(done)
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}　{d:%Y-%m-%d} {q}：{len(arts)} 則，累計新增 {total}", flush=True)
        time.sleep(DELAY)
    print(f"完成，新增 {total} 則 → {OUT}")


def bigday_dates() -> list:
    b = pd.read_csv(BIGDAYS, encoding="utf-8-sig", parse_dates=["date"])
    b = b[b["big_z2"].astype(str).str.lower() == "true"] if "big_z2" in b else b
    days = sorted({d + pd.Timedelta(days=k) for d in b["date"] for k in range(-3, 2)})
    skipped = sum(1 for d in days if d < EARLIEST)
    days = [d for d in days if d >= EARLIEST and d <= pd.Timestamp.utcnow().tz_localize(None).normalize()]
    print(f"大波動日 {len(b)} 天；前後展開成 {len(days)} 個 UTC 日"
          f"（{skipped} 天早於 GDELT 起點 2017-01-01，略過）")
    return days


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="GDELT 英文新聞：大波動日前後＋每日更新")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bigdays", action="store_true", help="抓所有大波動日 t−3 ～ t+1")
    g.add_argument("--daily", action="store_true", help="抓前兩個 UTC 日（每日更新用）")
    a = ap.parse_args(argv)
    if a.bigdays:
        run(bigday_dates())
    else:
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        run([today - pd.Timedelta(days=2), today - pd.Timedelta(days=1)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
