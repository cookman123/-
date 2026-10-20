#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段二：補抓下市公司的歷史股價，重建無生存者偏誤的母體。

為什麼要抓
----------
階段一（`delisted_crawler.py`）確認研究區間內有 126 家下市，佔應有母體的
24.8%。但那只是**家數**佔比。下市公司的報酬通常遠低於存活者，所以報酬偏誤
會比 24.8% 更嚴重 —— 要算出真正的數字，就得有它們下市前的股價。

資料來源
--------
證交所 STOCK_DAY 端點對已下市的代號在其仍上市的日期區間內依然有資料
（公司不在了，歷史成交紀錄還在）。Yahoo 對下市股票通常查不到，所以這裡
用證交所而不是沿用股價主線的 Yahoo。

兩段式抓取，為了不浪費上萬次請求
--------------------------------
    篩選段：每年只抓一個月（1 月），看這檔的成交金額量級。
            完全不可能進入成交金額前 250 名的小型股就不必細抓。
    細抓段：只對通過篩選的代號逐月補齊。

126 家若全部逐月抓，約一萬兩千次請求、數小時；先篩選可砍掉大部分。

斷點續傳
--------
每抓完一個（代號, 年月）就寫 checkpoint，中斷後重跑會跳過已完成的部分。
這支程式預期要跑很久，所以可中斷是必要的而不是加分項。
"""
from __future__ import annotations

import json
import sys
from datetime import date
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

OUT = ROOT / "data" / "delisted"
PRICE = OUT / "price"
#篩選與細抓各用一個 checkpoint 檔。共用一個檔時兩個行程會各自 load/save
# 同一份 JSON，後寫的那個把先寫的紀錄整批吃掉 —— 實際發生過，而且因為
# 兩邊都「成功」執行，沒有任何錯誤訊息。
CKPT = {"screen": OUT / "ckpt_screen.json", "full": OUT / "ckpt_full.json"}
URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
# 成交金額門檻：低於此量級的個股不可能進入前 250 名母體，不必細抓。
# 用 2,000 萬元／日作為下界，比實際門檻寬鬆，寧可多抓不要漏。
SCREEN_MIN_TURNOVER = 2e7


def load_ckpt(mode: str) -> dict:
    f = CKPT[mode]
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}


def save_ckpt(mode: str, c: dict) -> None:
    f = CKPT[mode]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")


def fetch_month(code: str, y: int, m: int) -> pd.DataFrame | None:
    """抓單檔單月。證交所回 data 為列陣列，欄位見 fields。"""
    d = X.http_get(URL, params={"date": f"{y}{m:02d}01", "stockNo": code,
                                "response": "json"})
    if not isinstance(d, dict) or d.get("stat") != "OK" or not d.get("data"):
        return None
    df = pd.DataFrame(d["data"], columns=d.get("fields"))
    # 欄位：日期 成交股數 成交金額 開盤價 最高價 最低價 收盤價 漲跌價差 成交筆數
    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False)
                             .str.replace("--", "", regex=False), errors="coerce")
    out = pd.DataFrame({
        "date": df.iloc[:, 0].map(_roc_day),
        "volume": num(df.iloc[:, 1]),
        "turnover": num(df.iloc[:, 2]),
        "open": num(df.iloc[:, 3]),
        "high": num(df.iloc[:, 4]),
        "low": num(df.iloc[:, 5]),
        "close": num(df.iloc[:, 6]),
    })
    return out.dropna(subset=["date", "close"])


def _roc_day(s):
    """
    證交所的日期欄是民國年，而且**年份位數不固定**：民國 99 年回的是
    ' 99/01/04'（兩位），民國 100 年以後才是 '100/01/04'（三位）。
    只處理三位會把 2011 年之前的每一列靜默丟掉 —— 不會報錯、不會中斷，
    只是資料少了一截，而 2010 年下市的公司會整檔變成「查無資料」。
    """
    t = "".join(ch for ch in str(s) if ch.isdigit())
    if len(t) in (6, 7):                      # yy/mm/dd 或 yyy/mm/dd
        n = len(t) - 4
        return pd.Timestamp(int(t[:n]) + 1911, int(t[n:n + 2]), int(t[n + 2:]))
    if len(t) == 8:                           # 西元 yyyymmdd
        return pd.Timestamp(int(t[:4]), int(t[4:6]), int(t[6:8]))
    return pd.NaT


def months(a: date, b: date):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="補抓下市公司股價")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--mode", choices=["test", "screen", "full"], default="screen")
    ap.add_argument("--only-screened", action="store_true",
                    help="細抓時只做通過成交金額篩選的代號（預設抓齊全部）")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    lst = OUT / "delisted.csv"
    if not lst.exists():
        print(f"找不到 {lst}，請先跑 delisted_crawler.py"); return 1
    df = pd.read_csv(lst, encoding="utf-8-sig", dtype={"code": str},
                     parse_dates=["delist_date"])
    lo = pd.Timestamp(a.start)
    df = df[df["delist_date"] >= lo].sort_values("delist_date")
    if a.limit:
        df = df.head(a.limit)
    PRICE.mkdir(parents=True, exist_ok=True)
    ck = load_ckpt(a.mode) if a.mode != "test" else {}

    if a.mode == "test":
        r = df.iloc[-1]
        print(f"測試 {r['code']} {r['name']}（下市 {r['delist_date']:%Y-%m-%d}）")
        y, m = r["delist_date"].year, max(1, r["delist_date"].month - 1)
        d = fetch_month(r["code"], y, m)
        if d is None or d.empty:
            print(f"  {y}-{m:02d} 查無資料 —— 端點對下市股票可能無效")
            return 1
        print(f"  {y}-{m:02d} 取得 {len(d)} 筆")
        print(d.tail(3).to_string(index=False))
        print(f"  日均成交金額 {d['turnover'].mean():,.0f} 元")
        return 0

    print(f"{'篩選' if a.mode == 'screen' else '細抓'}：{len(df)} 家下市公司"
          f"（區間自 {a.start}）\n", flush=True)
    kept, skipped, nodata = [], [], []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        code, dl = r["code"], r["delist_date"]
        yrs = range(max(lo.year, 2010), dl.year + 1)
        if a.mode == "screen":
            tos = []
            for y in yrs:
                key = f"screen:{code}:{y}"
                if key in ck:
                    if ck[key] is not None:
                        tos.append(ck[key])
                    continue
                try:
                    d = fetch_month(code, y, 1)
                    v = float(d["turnover"].mean()) if d is not None and len(d) else None
                except Exception as e:
                    print(f"    {code} {y} 失敗：{str(e)[:50]}", flush=True)
                    v = None
                ck[key] = v
                save_ckpt(a.mode, ck)
                if v is not None:
                    tos.append(v)
                X.polite_sleep(*X.API_DELAY_RANGE)
            if not tos:
                nodata.append(code)
                mark = "查無"
            elif max(tos) >= SCREEN_MIN_TURNOVER:
                kept.append((code, r["name"], dl, max(tos)))
                mark = f"保留（峰值 {max(tos)/1e8:.2f} 億）"
            else:
                skipped.append(code)
                mark = f"略過（峰值 {max(tos)/1e4:.0f} 萬）"
            print(f"  [{i:>3}/{len(df)}] {code} {str(r['name'])[:8]:<10}"
                  f"下市 {dl:%Y-%m}　{mark}", flush=True)

        else:   # mode == "full"
            # 預設抓齊全部下市公司。篩選門檻原本是為了省請求，但生存者偏誤
            # 要的就是完整性：被門檻擋掉的小型股正是最可能下市、報酬最差的
            # 那一群，漏掉它們等於把要修的偏誤留在資料裡。
            # --only-screened 才改用篩選清單。
            if a.only_screened:
                keep_file = OUT / "screen_kept.csv"
                if not keep_file.exists():
                    print(f"找不到 {keep_file}，請先跑 --mode screen"); return 1
                ok = set(pd.read_csv(keep_file, encoding="utf-8-sig",
                                     dtype={"code": str})["code"])
                if code not in ok:
                    continue
            path = PRICE / f"{code}.csv"
            got = []
            if path.exists():
                got.append(pd.read_csv(path, encoding="utf-8-sig",
                                       parse_dates=["date"]))
            n_new = 0
            for y, m in months(date(max(lo.year, 2010), 1, 1), dl.date()):
                key = f"full:{code}:{y}{m:02d}"
                if key in ck:
                    continue
                try:
                    d = fetch_month(code, y, m)
                    if d is not None and len(d):
                        got.append(d)
                        n_new += len(d)
                except Exception as e:
                    print(f"    {code} {y}-{m:02d} 失敗：{str(e)[:50]}", flush=True)
                    X.polite_sleep(*X.API_DELAY_RANGE)
                    continue
                ck[key] = True
                save_ckpt(a.mode, ck)
                X.polite_sleep(*X.API_DELAY_RANGE)
            if got:
                # 只增不減，且日期正規化後去重 —— 和主線股價同一套規則
                all_ = pd.concat(got, ignore_index=True)
                all_["date"] = pd.to_datetime(all_["date"], errors="coerce").dt.normalize()
                all_ = (all_.dropna(subset=["date", "close"]).sort_values("date")
                            .drop_duplicates("date", keep="last"))
                all_.to_csv(path, index=False, encoding="utf-8-sig")
                kept.append((code, r["name"], dl, float(all_["turnover"].mean())))
                print(f"  [{i:>3}/{len(df)}] {code} {str(r['name'])[:8]:<10}"
                      f"共 {len(all_):,} 筆（新增 {n_new}）"
                      f"　{all_['date'].min():%Y-%m} ~ {all_['date'].max():%Y-%m}",
                      flush=True)
            else:
                nodata.append(code)
                print(f"  [{i:>3}/{len(df)}] {code} 查無資料", flush=True)

    if a.mode == "screen":
        k = pd.DataFrame(kept, columns=["code", "name", "delist_date", "peak_turnover"])
        k.to_csv(OUT / "screen_kept.csv", index=False, encoding="utf-8-sig")
        print(f"\n{'=' * 80}")
        print(f"  需細抓　{len(kept)} 家　（曾達每日成交金額 "
              f"{SCREEN_MIN_TURNOVER/1e4:.0f} 萬以上）")
        print(f"  可略過　{len(skipped)} 家　（規模過小，不可能進前 250 名）")
        print(f"  查無　　{len(nodata)} 家")
        if kept:
            est = sum(len(list(months(date(max(2010, lo.year), 1, 1),
                                      d.date()))) for _, _, d, _ in kept)
            print(f"\n  細抓預估請求數 {est:,} 次"
                  f"　約 {est * 1.5 / 3600:.1f} 小時")
        print(f"\n輸出：{OUT / 'screen_kept.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
