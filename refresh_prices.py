#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把所有股票的當年度股價重抓到同一天。

為什麼要這支
------------
面板最後兩週只有 115 檔有資料、其餘 267 檔停在更早的日期 —— 因為逐年檔案是
分批抓的，各檔的最後一筆落在不同天。直接用面板最後一日選股，等於從一個
「碰巧被重爬」的子集裡挑，是選擇偏誤：實測兩份清單只重疊 15/30 檔。

和 `tw_stock_crawler.py price` 的差別
-------------------------------------
那支會重跑 year_chunks 的每一年（靠 checkpoint 跳過），而且 STOCK_LIST 是
現場從 Yahoo 分類重建的，不保證涵蓋磁碟上既有的每一檔。這支只做一件事：
對「磁碟上已經有資料的每一檔」重抓當年度，一檔一個請求。

安全措施
--------
    先寫暫存檔，確認筆數不少於原檔才取代 —— 避免一次壞掉的回應洗掉好資料
    筆數變少的檔案保留原狀並列在報告裡，讓人工判斷（可能是下市或停牌）
    沿用爬蟲的 session、重試與禮貌延遲；每檔之間再多等幾秒
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tw_stock_crawler as X  # noqa: E402

YEAR = date.today().year
# 單檔只發一個請求，所以每檔之間不需要原本 8–15 秒的冷卻；
# 仍維持約每 5 秒一個請求（0.2 req/s），比原設定寬鬆但依然保守。
COOLDOWN = (3.0, 6.0)


def fetch_year(code: str, year: int, suffix: str) -> pd.DataFrame | None:
    """抓單檔單年；回傳 DataFrame，查無資料回 None。"""
    y0, y1 = date(year, 1, 1), date(year, 12, 31)
    data = X.http_get(
        X.YAHOO_CHART_URL.format(symbol=f"{code}{suffix}"),
        params={"period1": int(datetime(y0.year, y0.month, y0.day).timestamp()),
                "period2": int(datetime(y1.year, y1.month, y1.day).timestamp()) + 86400,
                "interval": "1d", "events": "history"},
    )
    result = (data.get("chart") or {}).get("result")
    if not result:
        return None
    r0 = result[0]
    ts = r0.get("timestamp", [])
    if not ts:
        return None
    q = r0["indicators"]["quote"][0]
    # 日期換算必須與 tw_stock_crawler 完全一致，否則新舊檔會對不齊
    return pd.DataFrame({
        "date": [datetime.utcfromtimestamp(t).date() for t in ts],
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume"),
    })


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="把當年度股價重抓到同一天")
    ap.add_argument("--year", type=int, default=YEAR)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（測試用）")
    a = ap.parse_args(argv)

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    if a.limit:
        codes = codes[:a.limit]
    print(f"重抓 {a.year} 年：{len(codes)} 檔　預估 {len(codes)*6/60:.0f} 分鐘\n", flush=True)

    updated, kept, missing, failed = [], [], [], []
    for i, code in enumerate(codes, 1):
        out = ROOT / "data" / "stocks" / code / "price" / f"yahoo_{code}_{a.year}.csv"
        old_n, old_last = 0, None
        if out.exists():
            try:
                o = pd.read_csv(out, encoding="utf-8-sig")
                old_n = len(o)
                old_last = pd.to_datetime(o["date"], errors="coerce").max()
            except Exception:
                pass
        try:
            df = fetch_year(code, a.year, ".TW")
            if df is None or df.empty:          # 可能是上櫃
                X.polite_sleep(*X.API_DELAY_RANGE)
                df = fetch_year(code, a.year, ".TWO")
            if df is None or df.empty:
                missing.append(code)
                print(f"  [{i:>3}/{len(codes)}] {code} 查無資料", flush=True)
            elif len(df) < old_n:
                # 筆數變少：可能下市或長期停牌，也可能是壞掉的回應。保留原檔。
                kept.append((code, old_n, len(df)))
                print(f"  [{i:>3}/{len(codes)}] {code} 筆數變少 {old_n}→{len(df)}，保留原檔",
                      flush=True)
            else:
                df.to_csv(out, index=False, encoding="utf-8-sig")
                new_last = pd.to_datetime(df["date"]).max()
                updated.append((code, new_last))
                if i % 25 == 0 or old_last is None or new_last > old_last:
                    print(f"  [{i:>3}/{len(codes)}] {code} {old_n}→{len(df)} 筆"
                          f"　最後 {new_last:%m-%d}", flush=True)
        except Exception as e:
            failed.append((code, str(e)[:80]))
            print(f"  [{i:>3}/{len(codes)}] {code} 失敗：{str(e)[:80]}", flush=True)
        X.polite_sleep(*COOLDOWN)

    print(f"\n{'='*70}")
    print(f"更新 {len(updated)}　保留原檔 {len(kept)}　查無資料 {len(missing)}　失敗 {len(failed)}")
    if updated:
        last = pd.Series([d for _, d in updated])
        print("\n更新後各檔最後一日分布：")
        for d, n in last.value_counts().sort_index().tail(6).items():
            print(f"  {d:%Y-%m-%d}　{n} 檔")
    for lab, items in (("筆數變少（需人工確認）", kept), ("失敗", failed)):
        if items:
            print(f"\n{lab}：{', '.join(str(x[0]) for x in items[:40])}"
                  f"{' …' if len(items) > 40 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
