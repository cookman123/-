#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTT Stock 板每日增量爬蟲。

專案原本的 PTT 資料是外部提供的整理檔（NEW_pttcc(19-24)/pttcc_clean/，2019～2024），
沒有爬蟲，所以 2025 年起沒有 PTT。這支從看板最新一頁往回翻，抓到指定日期為止，
輸出成和原始檔「完全相同的 JSONL 格式」（stock_{年}_clean.jsonl），所以
volatility_news.load_ptt_board 與 ptt_to_stock.py 都不用改就能讀到新資料。

欄位：article_id, board, url, title, author_id, author_nickname, date_raw,
      date（台北時間 ISO）, date_ts（由文章代號 M.{unix}.A 換算）, content,
      n_push, n_boo, n_comments
以 article_id 去重，已抓過的不重抓。

注意
----
    推噓數會在發文後持續累積。每日抓的是「抓取當下」的數字，比整理檔（事後才抓）少，
    兩段資料的推噓數不能直接比較；主分析本來就不使用推噓數（屬事後資訊）。
    每次請求間隔 1 秒。
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "NEW_pttcc(19-24)" / "pttcc_clean"
BASE = "https://www.ptt.cc"
BOARD = "Stock"
TPE = timezone(timedelta(hours=8))
DELAY = 1.0

S = requests.Session()
S.cookies.set("over18", "1")
S.headers["User-Agent"] = "Mozilla/5.0 (student thesis research)"


def get(url: str) -> str | None:
    for k in range(4):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    return None


def id_time(aid: str) -> datetime | None:
    m = re.match(r"M\.(\d+)\.A", aid)
    return datetime.fromtimestamp(int(m.group(1)), TPE) if m else None


def parse_article(aid: str, html: str) -> dict:
    b = BeautifulSoup(html, "html.parser")
    meta = {m.select_one(".article-meta-tag").text.strip(): m.select_one(".article-meta-value").text.strip()
            for m in b.select("div.article-metaline") if m.select_one(".article-meta-value")}
    author = meta.get("作者", "")
    a_id, nick = (author.split(" (", 1) + [""])[:2]
    pushes = [p.select_one(".push-tag").text.strip() for p in b.select("div.push") if p.select_one(".push-tag")]
    main = b.select_one("#main-content")
    if main:
        for t in main.select("div.article-metaline, div.article-metaline-right, div.push"):
            t.decompose()
        content = main.get_text("\n").split("\n--\n※ 發信站")[0].strip()
    else:
        content = ""
    raw = meta.get("時間", "")
    try:
        dt = datetime.strptime(raw, "%a %b %d %H:%M:%S %Y").isoformat()
    except ValueError:
        it = id_time(aid)
        dt = it.replace(tzinfo=None).isoformat(timespec="seconds") if it else ""
    it = id_time(aid)
    return {"article_id": aid, "board": BOARD, "url": f"{BASE}/bbs/{BOARD}/{aid}.html",
            "title": meta.get("標題", ""), "author_id": a_id, "author_nickname": nick.rstrip(")"),
            "date_raw": raw, "date": dt,
            "date_ts": it.replace(tzinfo=None).isoformat(timespec="seconds") if it else "",
            "content": content, "n_push": pushes.count("推"), "n_boo": pushes.count("噓"),
            "n_comments": len(pushes)}


def known_ids(years: set) -> set:
    ids = set()
    for y in years:
        f = OUT / f"stock_{y}_clean.jsonl"
        if f.exists():
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = re.search(r'"article_id":\s*"([^"]+)"', line)
                    if m:
                        ids.add(m.group(1))
    return ids


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="PTT Stock 板增量爬蟲")
    ap.add_argument("--days", type=float, default=3, help="往回抓幾天（預設 3，涵蓋週末）")
    ap.add_argument("--since", default=None, help="或直接指定起日 YYYY-MM-DD（回補用）")
    a = ap.parse_args(argv)
    since = (datetime.fromisoformat(a.since).replace(tzinfo=TPE) if a.since
             else datetime.now(TPE) - timedelta(days=a.days))
    OUT.mkdir(parents=True, exist_ok=True)
    seen = known_ids({since.year, datetime.now(TPE).year})
    print(f"PTT {BOARD} 板：抓 {since:%Y-%m-%d %H:%M} 之後的文章（已有 {len(seen)} 篇）", flush=True)

    html = get(f"{BASE}/bbs/{BOARD}/index.html")
    if html is None:
        print("看板首頁讀取失敗"); return 1
    m = re.search(rf"/bbs/{BOARD}/index(\d+)\.html\">&lsaquo;", html) or \
        re.search(rf"/bbs/{BOARD}/index(\d+)\.html", html)
    page = int(m.group(1)) + 1 if m else None
    new, n_pages = [], 0
    while html is not None:
        b = BeautifulSoup(html, "html.parser")
        ids = []
        for e in b.select("div.r-ent div.title a"):
            aid = e["href"].rsplit("/", 1)[-1].replace(".html", "")
            t = id_time(aid)
            if t:
                ids.append((aid, t))
        n_pages += 1
        for aid, t in ids:
            if t >= since and aid not in seen:
                art = get(f"{BASE}/bbs/{BOARD}/{aid}.html")
                time.sleep(DELAY)
                if art:
                    new.append(parse_article(aid, art))
                    seen.add(aid)
        # 整頁（不含置底公告）都早於起點就停
        if ids and max(t for _, t in ids) < since:
            break
        if page is None or page <= 1:
            break
        page -= 1
        html = get(f"{BASE}/bbs/{BOARD}/index{page}.html")
        time.sleep(DELAY)

    by_year: dict = {}
    for r in new:
        by_year.setdefault((r["date"] or r["date_ts"])[:4], []).append(r)
    for y, rows in by_year.items():
        with open(OUT / f"stock_{y}_clean.jsonl", "a", encoding="utf-8") as fh:
            for r in sorted(rows, key=lambda r: r["date"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成：翻了 {n_pages} 頁，新增 {len(new)} 篇"
          + "（" + "、".join(f"{y} 年 {len(v)} 篇" for y, v in sorted(by_year.items())) + "）" if new else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
