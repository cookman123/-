#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
國際／總體新聞爬蟲（鉅亨網分類 API）。

與既有台股新聞的差別
--------------------
既有語料是「掛在個股底下的台股新聞」，去重後 44,783 則、只涵蓋 2019 年起，
戰爭類僅 462 則，對主題分析太稀疏。本程式抓的是國際總經、世界股市、美股、
外匯這幾個分類 —— 戰爭、制裁、關稅、新技術這類主題主要出現在這裡。

不抓 headline（綜合）的原因：它每週約 786 則、約 27 頁，單這一類就要一萬五千次
請求（約 6 小時），而且內容與 tw_stock 大量重疊。先抓四個高相關分類，
不夠再補。

輸出：data/intl_news/{分類}_{年}.csv，欄位 date,title,summary,category,news_id
斷點：data/intl_news/checkpoint.json，記錄已完成的 (分類, 週) 組合。
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tw_text_crawler as TX  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "intl_news"
CKPT = OUT / "checkpoint.json"

CATEGORIES = {
    "wd_macro": "國際總經",
    "wd_stock": "世界股市",
    "us_stock": "美股",
    "forex": "外匯",
}


def load_ckpt() -> set:
    return set(json.loads(CKPT.read_text(encoding="utf-8"))) if CKPT.exists() else set()


def save_ckpt(done: set) -> None:
    CKPT.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")


def fetch_window(cat: str, start: date, stop: date) -> list:
    """把一個時間窗的所有頁抓完；API 約在 50 頁後回錯，由 _cnyes_page 回 None 收尾。"""
    rows = []
    for page in range(1, TX.CNYES_MAX_PAGES + 1):
        items = TX._cnyes_page(cat, start, stop, page)
        if not items:
            break
        arr = items.get("data") or []
        if not arr:
            break
        for a in arr:
            ts = a.get("publishAt")
            if not ts:
                continue
            rows.append({
                "date": datetime.fromtimestamp(ts),
                "title": a.get("title") or "",
                "summary": (a.get("summary") or "")[:500],
                "category": cat,
                "news_id": a.get("newsId"),
            })
        if page >= (items.get("last_page") or page):
            break
    return rows


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="國際／總體新聞爬蟲")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)
    windows = list(TX._time_windows(s, e))
    done = load_ckpt()
    todo = [(c, w) for c in args.categories for w in windows
            if f"{c}|{w[0]}" not in done]
    print(f"分類 {len(args.categories)} 個 × 週視窗 {len(windows)} 個 "
          f"= {len(args.categories) * len(windows)} 組，待抓 {len(todo)}")
    print(f"期間 {s} ~ {e}\n")

    buf: dict = {}
    n_rows = 0
    for i, (cat, (w0, w1)) in enumerate(todo, 1):
        rows = fetch_window(cat, w0, w1)
        if rows:
            buf.setdefault((cat, w0.year), []).extend(rows)
            n_rows += len(rows)
        done.add(f"{cat}|{w0}")

        if i % 40 == 0 or i == len(todo):
            for (c, y), rs in buf.items():
                path = OUT / f"{c}_{y}.csv"
                new = pd.DataFrame(rs)
                if path.exists():
                    new = pd.concat([pd.read_csv(path, parse_dates=["date"]), new],
                                    ignore_index=True)
                new = new.drop_duplicates(subset=["news_id"], keep="last")
                new.sort_values("date").to_csv(path, index=False,
                                               encoding="utf-8-sig")
            buf.clear()
            save_ckpt(done)
            print(f"  {i}/{len(todo)}　累計 {n_rows:,} 則", flush=True)

    print("\n完成。各檔：")
    tot = 0
    for f in sorted(OUT.glob("*.csv")):
        if f.name == "checkpoint.json":
            continue
        d = pd.read_csv(f)
        tot += len(d)
        print(f"  {f.name:<22}{len(d):>8,} 則")
    print(f"合計 {tot:,} 則")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
