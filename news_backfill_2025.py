#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
補新聞空缺：2025-01 之後的鉅亨個股新聞。

為什麼不用 Yahoo
----------------
Yahoo 個股新聞頁的伺服器庫存每檔最多約 200 則，翻「載入更多」到底也回不去：
2026-10-15 實測 1101 最早只到 2025-11-04（109 則），2330 只剩最近 3 天（200 則）。
所以 2025-01 ~ 2025-09 的空缺只能用鉅亨的分類 API 回溯（tw_text_crawler 的 cnyes 任務）。

做法
----
直接呼叫 tw_text_crawler.crawl_cnyes_news（同一套節流、robots 檢查、時間窗切法、
strict 個股標記比對），只改兩件事：
  1. 寫檔位置改成另外的補抓檔 data/stocks/{代號}/news/cnyes_news_{代號}_2025fill.csv，
     欄位與既有 cnyes_news_{代號}.csv 相同，另加 backfill_batch 欄標明是哪一批補抓的。
     不直接併進原檔，是為了不影響正在用原檔跑的其他分析；要併的時候用 --merge。
  2. 斷點另存 data/news_backfill_checkpoint.json，不動共用的 data/checkpoint.json。

時間戳沿用爬蟲：publishAt（epoch 秒）轉台北時間、ISO 8601 帶 +08:00。

國際新聞另外直接跑 intl_news_crawler.py --start 2025-01-01（寫成新的 *_2025.csv、*_2026.csv）。

用法：
    python news_backfill_2025.py --start 2025-01-01 --end 2026-10-14
    python news_backfill_2025.py --merge      # 之後要併回原檔時再跑（會改動原檔）
"""
from __future__ import annotations

import argparse
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
STOCKS = ROOT / "data" / "stocks"
CKPT = ROOT / "data" / "news_backfill_checkpoint.json"
SUFFIX = "_2025fill"
BATCH = f"cnyes_backfill_{datetime.now():%Y%m%d}"


def stock_list() -> list:
    """沿用既有資料夾裡的股票（有 price 或 news 子資料夾者）；strict 模式只靠代號標記，名稱僅供日誌。"""
    names = {}
    m = ROOT / "data" / "stock_list_for_pdf.json"
    if m.exists():
        try:
            for x in json.loads(m.read_text(encoding="utf-8")):
                if isinstance(x, dict) and "code" in x:
                    names[str(x["code"])] = x.get("name", "")
        except Exception:
            pass
    codes = sorted(p.name for p in STOCKS.iterdir()
                   if p.is_dir() and ((p / "price").exists() or (p / "news").exists()))
    return [{"code": c, "name": names.get(c, c)} for c in codes]


def _patch_crawler():
    """把 cnyes 個股檔改寫到補抓檔、斷點改存獨立檔。回傳斷點 dict。"""
    orig_write = TX._write_text_csv

    def write_backfill(rows, path, dedup_key):
        if path.name.startswith("cnyes_news_") and SUFFIX not in path.stem:
            path = path.with_name(f"{path.stem}{SUFFIX}.csv")
            rows = [{**r, "backfill_batch": BATCH} for r in rows]
        return orig_write(rows, path, dedup_key)

    TX._write_text_csv = write_backfill
    TX.save_checkpoint = lambda ck: CKPT.write_text(json.dumps(ck, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return json.loads(CKPT.read_text(encoding="utf-8")) if CKPT.exists() else {}


def refetch_truncated(listfile: Path) -> int:
    """
    2026-10 實測鉅亨分類 API 每個查詢最多只給 30 頁（第 31 頁回 422），
    新聞多的週會被截掉「最早的那幾天」（結果由新到舊排）。這裡把被截斷的週改成一天一窗重抓：
    tw_stock 走 crawl_cnyes_news（寫補抓檔），國際分類走 intl_news_crawler.fetch_window 再併進年檔。
    """
    import intl_news_crawler as IN

    ckpt = _patch_crawler()
    sl = stock_list()
    for line in listfile.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        cat, s0, s1 = line.split()
        days = pd.date_range(s0, s1).date
        print(f"重抓 {cat} {s0}~{s1}（{len(days)} 天，一天一窗）", flush=True)
        if cat == "tw_stock":
            # 週窗的斷點鍵是「起日」，跟一天一窗的第一天同名，不清掉會被當成已完成而跳過
            stamps = {f"{dd:%Y%m%d}" for dd in days}
            for k in [k for k in ckpt if "cnyes_news" in k and k[-8:] in stamps]:
                del ckpt[k]
            for dd in days:
                TX.crawl_cnyes_news(sl, ckpt, dd, dd, strict=True)
            continue
        rows = []
        for dd in days:
            got = IN.fetch_window(cat, dd, dd)
            if len(got) >= TX.CNYES_PER_PAGE * 30:
                print(f"  ⚠ {cat} {dd} 單日仍達 30 頁上限，可能還是不完整", flush=True)
            rows += got
        by_year = {}
        for r in rows:
            by_year.setdefault(r["date"].year, []).append(r)
        for y, rs in by_year.items():
            path = IN.OUT / f"{cat}_{y}.csv"
            new = pd.DataFrame(rs)
            if path.exists():
                new = pd.concat([pd.read_csv(path, parse_dates=["date"]), new], ignore_index=True)
            new = new.drop_duplicates(subset=["news_id"], keep="last")
            new.sort_values("date").to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  取回 {len(rows)} 則", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="鉅亨個股新聞補抓（2025 之後）")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--merge", action="store_true", help="把補抓檔併回 cnyes_news_{代號}.csv")
    ap.add_argument("--refetch", default=None,
                    help="被截斷的時間窗清單（每行：分類 起日 迄日），改成一天一窗重抓")
    args = ap.parse_args(argv)

    if args.refetch:
        return refetch_truncated(Path(args.refetch))

    if args.merge:
        n = 0
        for f in STOCKS.glob(f"*/news/cnyes_news_*{SUFFIX}.csv"):
            code = f.parent.parent.name
            rows = pd.read_csv(f, encoding="utf-8-sig").to_dict("records")
            TX._write_text_csv(rows, f.parent / f"cnyes_news_{code}.csv", "news_id")
            n += 1
        print(f"已併回 {n} 檔")
        return 0

    ckpt = _patch_crawler()

    sl = stock_list()
    print(f"股票 {len(sl)} 檔，期間 {args.start} ~ {args.end}，批次 {BATCH}", flush=True)
    TX.crawl_cnyes_news(sl, ckpt, date.fromisoformat(args.start), date.fromisoformat(args.end), strict=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
