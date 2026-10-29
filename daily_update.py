#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日更新：股價、大盤指數、個股新聞。

設計原則
--------
**一次更新到同一天，或者不更新。** 先前的教訓是：各檔最後一日不一致會
安靜地產出一份看似合理、實際有一半是錯的選股清單。所以這支在最後會檢查
對齊，沒對齊就以非零碼結束並印出警告，讓排程記錄看得出來。

**只增不減。** 任何檔案都先在記憶體合併、確認筆數不少於原檔才寫回；
筆數變少一律保留原檔並列進報告，交給人判斷（可能是下市，也可能是壞回應）。

**日期一律正規化到「日」。** Yahoo 的 timestamp 帶時間成分，不 normalize
就會讓去重失效、產生重複列（這個錯誤實際發生過）。

**0050 的假分割要重做。** Yahoo 原始資料在 2014-01-02 有未還原的 4:1 分割，
每次重抓都會重新出現，所以合併後一律重跑整數比偵測。

用法
----
    python daily_update.py                # 股價 ＋ 指數 ＋ 鉅亨新聞（約 40 分）
    python daily_update.py --full-news    # 另加 Yahoo 逐檔新聞（再 +30 分）
    python daily_update.py --no-news      # 只更新股價與指數
    python daily_update.py --quiet        # 少印進度，排程用

結束碼：0 = 全部對齊可用；2 = 各檔最後一日不一致，不要拿去選股
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

MARKET = ROOT / "data" / "market"
STOCKS = ROOT / "data" / "stocks"
LOGDIR = ROOT / "data" / "daily"
# 大盤之外，fetch_index_targets.py 列的 ETF、海外指數、油價全部每日更新
from fetch_index_targets import TARGETS as _TARGETS  # noqa: E402
INDEX_JOBS = [("^TWII", "TWII")] + [(sym, name) for sym, name, _, _ in _TARGETS]
COOLDOWN = (3.0, 6.0)


def fetch_year(symbol: str, year: int) -> pd.DataFrame | None:
    """抓單一標的單一年度的日線；日期正規化到「日」。"""
    y0, y1 = date(year, 1, 1), date(year, 12, 31)
    d = X.http_get(X.YAHOO_CHART_URL.format(symbol=symbol), params={
        "period1": int(datetime(y0.year, y0.month, y0.day).timestamp()),
        "period2": int(datetime(y1.year, y1.month, y1.day).timestamp()) + 86400,
        "interval": "1d", "events": "history"})
    r = (d.get("chart") or {}).get("result")
    if not r or not r[0].get("timestamp"):
        return None
    q = r[0]["indicators"]["quote"][0]
    return pd.DataFrame({
        # .date() 不可省：帶時間成分會讓後面的去重失效
        "date": [datetime.utcfromtimestamp(t).date() for t in r[0]["timestamp"]],
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume")})


def merge_only_grow(path: Path, new: pd.DataFrame) -> tuple[bool, int, int]:
    """合併後筆數不少於原檔才寫回。回傳 (是否寫入, 原筆數, 新筆數)。"""
    old = pd.DataFrame()
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
    n0 = len(old)
    both = pd.concat([old, new], ignore_index=True) if n0 else new.copy()
    both["date"] = pd.to_datetime(both["date"], errors="coerce").dt.normalize()
    both = (both.dropna(subset=["date", "close"]).sort_values("date")
                .drop_duplicates("date", keep="last"))
    if len(both) < n0:
        return False, n0, len(both)
    both["date"] = both["date"].dt.strftime("%Y-%m-%d")
    both.to_csv(path, index=False, encoding="utf-8-sig")
    return True, n0, len(both)


def fix_fake_splits(path: Path) -> int:
    """比值接近整數的單日跳動＝未還原的分割，把之前的區段縮放接上。"""
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    fixed = 0
    while True:
        c = d["close"]
        ratio = c.shift(1) / c
        hit = None
        for i, r in ratio[(ratio > 1.5) | (ratio < 1 / 1.5)].items():
            k = r if r > 1 else 1 / r
            if abs(k - round(k)) / round(k) <= 0.03:
                hit = (i, round(k), r > 1)
                break
        if hit is None:
            break
        i, k, up = hit
        for col in ("open", "high", "low", "close"):
            d.loc[:i - 1, col] = d.loc[:i - 1, col] / (k if up else 1 / k)
        d.loc[:i - 1, "volume"] = d.loc[:i - 1, "volume"] * (k if up else 1 / k)
        fixed += 1
        if fixed > 10:
            break
    if fixed:
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")
        d.to_csv(path, index=False, encoding="utf-8-sig")
    return fixed


def update_prices(year: int, quiet: bool) -> dict:
    codes = sorted(x.name for x in STOCKS.iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    ok, kept, miss, fail = 0, [], [], []
    for i, code in enumerate(codes, 1):
        path = STOCKS / code / "price" / f"yahoo_{code}_{year}.csv"
        try:
            df = fetch_year(f"{code}.TW", year)
            if df is None or df.empty:
                X.polite_sleep(*X.API_DELAY_RANGE)
                df = fetch_year(f"{code}.TWO", year)
            if df is None or df.empty:
                miss.append(code)
            else:
                wrote, n0, n1 = merge_only_grow(path, df)
                if wrote:
                    ok += 1
                else:
                    kept.append((code, n0, n1))
        except Exception as e:
            fail.append((code, str(e)[:60]))
        if not quiet and i % 50 == 0:
            print(f"  股價 {i}/{len(codes)}", flush=True)
        X.polite_sleep(*COOLDOWN)
    return {"n": len(codes), "ok": ok, "kept": kept, "miss": miss, "fail": fail}


def update_indices(year: int) -> list:
    out = []
    for symbol, name in INDEX_JOBS:
        path = MARKET / f"{name}.csv"
        try:
            df = fetch_year(symbol, year)
            if df is None or df.empty:
                out.append((name, "查無資料")); continue
            wrote, n0, n1 = merge_only_grow(path, df)
            if not wrote:
                out.append((name, f"筆數變少 {n0}→{n1}，保留原檔")); continue
            nfix = fix_fake_splits(path) if name == "0050" else 0
            last = pd.read_csv(path, encoding="utf-8-sig")["date"].iloc[-1]
            out.append((name, f"{n0}→{n1} 筆　最後 {last}"
                              + (f"　修正 {nfix} 處假分割" if nfix else "")))
        except Exception as e:
            out.append((name, f"失敗：{str(e)[:60]}"))
        X.polite_sleep(*X.API_DELAY_RANGE)
    return out


MIN_ALIGNED = 0.95     # 至少這個比例的個股要到齊，與 strategy_spec 的九成門檻同一精神


def check_alignment() -> tuple[bool, dict, list]:
    """
    檢查各檔的「有效最後交易日」是否到齊。

    兩個設計要點：

    一、要算有效交易日，不是檔案裡的最後一列。Yahoo 對停牌股會持續吐出
        volume=0 且 OHLC 全等的佔位列，`load_price` 本來就會濾掉它們，
        所以這裡必須用同一條規則，否則檢查的對象和模型看到的資料不一致。

    二、門檻是比例，不是「全部相同」。要求 383 檔全部同一天，只要有一檔
        停牌就會永遠回報失敗 —— 那會把「管線壞了」和「一檔停牌」混為一談，
        警報很快就會被忽略。所以改成九五成到齊即視為可用，同時列出落後名單。
    """
    last = {}
    for d in STOCKS.iterdir():
        files = sorted((d / "price").glob("yahoo_*.csv")) if (d / "price").is_dir() else []
        files = [f for f in files if "merged" not in f.stem]
        if not files:
            continue
        try:
            px = pd.read_csv(files[-1], encoding="utf-8-sig")
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            for c in ("open", "high", "low", "close", "volume"):
                px[c] = pd.to_numeric(px[c], errors="coerce")
            px = px.dropna(subset=["date", "close"])
            flat = ((px["volume"] <= 0) & (px["open"] == px["close"])
                    & (px["high"] == px["low"]))
            px = px.loc[~flat]
            if not px.empty:
                last[d.name] = px["date"].max()
        except Exception:
            continue
    if not last:
        return False, {}, []
    s = pd.Series(last)
    mx = s.max()
    share = float((s == mx).mean())
    lag = sorted((c, d) for c, d in s.items() if d < mx)
    return share >= MIN_ALIGNED, s.value_counts().sort_index().to_dict(), lag


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="每日更新股價、指數與新聞")
    ap.add_argument("--year", type=int, default=date.today().year)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-prices", action="store_true")
    ap.add_argument("--news-days", type=int, default=7,
                    help="鉅亨網回補天數（預設 7，涵蓋週末與斷線）")
    ap.add_argument("--full-news", action="store_true",
                    help="另跑 Yahoo 逐檔新聞快照（383 檔約 30 分鐘）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    print(f"=== 每日更新　{t0:%Y-%m-%d %H:%M} ===", flush=True)

    pr = {}
    if not a.no_prices:
        print("\n[1/3] 股價", flush=True)
        pr = update_prices(a.year, a.quiet)
        print(f"  更新 {pr['ok']}/{pr['n']} 檔"
              f"　保留原檔 {len(pr['kept'])}　查無 {len(pr['miss'])}"
              f"　失敗 {len(pr['fail'])}", flush=True)
        for lab, items in (("筆數變少", pr["kept"]), ("失敗", pr["fail"])):
            if items:
                print(f"  {lab}：{', '.join(str(x[0]) for x in items[:30])}")

        print("\n[2/3] 大盤與海外指數", flush=True)
        for name, msg in update_indices(a.year):
            print(f"  {name:<8}{msg}", flush=True)

    if not a.no_news:
        print("\n[3/3] 新聞", flush=True)
        try:
            import json
            from datetime import timedelta
            from tw_text_crawler import run_text_tasks
            # 鉅亨以 stock 標記分流，非嚴格模式會用到公司名，
            # 所以帶真正的名稱進去，不要拿代號充數
            meta = {r["code"]: r["name"] for r in json.load(
                open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))}
            codes = sorted(x.name for x in STOCKS.iterdir() if x.is_dir())
            stocks = [{"code": c, "name": meta.get(c, c)} for c in codes]
            ck = X.load_checkpoint()
            since = date.today() - timedelta(days=a.news_days)
            # 鉅亨是掃分類頁（快），近幾日重掃就能補上新文章；
            # 回補數日是為了涵蓋週末與前一晚斷線的情況
            for k in [k for k in ck if k.startswith("cnyes:")
                      and k.rsplit(":", 1)[-1] >= since.strftime("%Y-%m-%d")]:
                ck.pop(k, None)
            X.save_checkpoint(ck)
            run_text_tasks(stocks, ck, task="cnyes", start=since, end=date.today())
            print(f"  鉅亨網新聞　{since} ~ {date.today()}　完成", flush=True)
            if a.full_news:
                # Yahoo 是逐檔快照，383 檔約 30 分鐘，所以不預設開啟
                run_text_tasks(stocks, ck, task="yahoo-news",
                               start=since, end=date.today())
                print("  Yahoo 個股新聞快照完成", flush=True)
        except Exception as e:
            print(f"  新聞更新失敗：{str(e)[:120]}", flush=True)

    print("\n[檢查] 各檔有效最後交易日是否到齊", flush=True)
    aligned, dist, lag = check_alignment()
    for d, n in list(dist.items())[-5:]:
        print(f"  {pd.Timestamp(d):%Y-%m-%d}　{n} 檔")
    if lag:
        # 落後多半是停牌或長期無成交，屬正常；列出來讓人判斷而不是逕自當失敗
        tail = " …" if len(lag) > 12 else ""
        print("  落後 " + str(len(lag)) + " 檔："
              + "、".join(f"{c}({pd.Timestamp(d):%m-%d})" for c, d in lag[:12]) + tail)
    mins = (datetime.now() - t0).total_seconds() / 60
    if aligned:
        print(f"\n✓ 到齊（{MIN_ALIGNED:.0%} 門檻），可直接跑 strategy_spec.py"
              f"　（耗時 {mins:.0f} 分）")
        rc = 0
    else:
        print(f"\n✗ 到齊比例低於 {MIN_ALIGNED:.0%} —— 不要拿現在的資料選股，"
              f"請重跑本程式或 refresh_prices.py　（耗時 {mins:.0f} 分）")
        rc = 2
    with open(LOGDIR / f"{date.today():%Y%m%d}.txt", "w", encoding="utf-8") as f:
        f.write(f"{t0:%Y-%m-%d %H:%M}　對齊={aligned}　"
                f"股價更新={pr.get('ok', '-')}/{pr.get('n', '-')}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
