#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓終止上市（下市）公司清單，量化生存者偏誤的規模。

為什麼要做
----------
本研究第七節限制二：股票池是「今天還在市場上的 383 檔」，期間內下市的公司
完全不在樣本裡。這會讓所有歷史績效偏高 —— 因為表現最差、最後下市的公司
被系統性地排除了。先前只能說「這個偏誤存在但無法量化」。

要量化就得先知道：期間內到底有幾家下市、什麼時候下市、規模多大。
這支程式抓的就是這份清單。

兩個階段
--------
    階段一（本程式）：抓清單，算出下市家數與時間分布，據此界定偏誤的上界
    階段二（另案）：逐檔補抓下市前的股價，重建無生存者偏誤的母體再回測
                    估計需要上萬次請求、數小時，所以分開做

資料來源
--------
證交所 OpenAPI 與 rwd 端點（公開政府資料）。沿用 tw_stock_crawler 的
session、重試與禮貌延遲，不另外開連線。
"""
from __future__ import annotations

import json
import sys
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

# 幾個候選端點：證交所改版頻繁，逐一嘗試並記錄哪個有效
SOURCES = [
    ("OpenAPI 終止上市",
     "https://openapi.twse.com.tw/v1/company/suspendListingCsvAndHtml", None),
    ("rwd 終止上市",
     "https://www.twse.com.tw/rwd/zh/company/suspendListingCsvAndHtml",
     {"response": "json"}),
    ("rwd 終止上市（備用路徑）",
     "https://www.twse.com.tw/company/suspendListingCsvAndHtml",
     {"response": "json"}),
]


def try_sources() -> tuple:
    for name, url, params in SOURCES:
        try:
            print(f"  嘗試 {name} …", flush=True)
            d = X.http_get(url, params=params)
            if d:
                print(f"    成功，回傳 {type(d).__name__}"
                      f"（{len(d) if hasattr(d, '__len__') else '?'} 筆）")
                return name, d
        except Exception as e:
            print(f"    失敗：{type(e).__name__}: {str(e)[:70]}")
        X.polite_sleep(*X.API_DELAY_RANGE)
    return "", None


def normalise(raw) -> pd.DataFrame:
    """證交所欄名會改版，所以用關鍵字比對而不是寫死欄名。"""
    rows = raw.get("data") if isinstance(raw, dict) else raw
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if isinstance(raw, dict) and raw.get("fields") and df.shape[1] == len(raw["fields"]):
        df.columns = raw["fields"]

    def pick(*keys):
        for c in df.columns:
            if any(k in str(c) for k in keys):
                return c
        return None

    # 欄名用關鍵字比對而非寫死：OpenAPI 用的是 Code / Company / DelistingDate，
    # rwd 端點用中文欄名，兩者都要能命中
    c_code = pick("公司代號", "股票代號", "Code", "代號")
    c_name = pick("公司名稱", "Company", "Name", "名稱", "簡稱")
    c_date = pick("DelistingDate", "終止上市日期", "終止買賣日期", "下市",
                  "Date", "日期")
    out = pd.DataFrame({
        "code": df[c_code].astype(str).str.strip() if c_code else None,
        "name": df[c_name].astype(str).str.strip() if c_name else None,
        "delist_raw": df[c_date].astype(str).str.strip() if c_date else None,
    })
    return out.dropna(subset=["code"])


def parse_roc_date(s: str):
    """證交所常用民國年（如 1120815 或 112/08/15）。"""
    t = "".join(ch for ch in str(s) if ch.isdigit())
    if len(t) == 7:
        return pd.Timestamp(int(t[:3]) + 1911, int(t[3:5]), int(t[5:7]))
    if len(t) == 8:
        return pd.Timestamp(int(t[:4]), int(t[4:6]), int(t[6:8]))
    return pd.NaT


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="抓終止上市公司清單")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("抓取終止上市公司清單")
    name, raw = try_sources()
    if raw is None:
        print("\n所有端點都失敗。證交所可能已改版，或需要不同的查詢方式。")
        print("這一項因此仍無法量化 —— 不以推測填補。")
        return 1

    json.dump(raw, open(OUT / "raw.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    df = normalise(raw)
    if df.empty:
        print("\n抓到資料但解析不出欄位，原始內容已存到 data/delisted/raw.json")
        return 1

    df["delist_date"] = df["delist_raw"].map(parse_roc_date)
    df = df.dropna(subset=["delist_date"]).sort_values("delist_date")
    df.to_csv(OUT / "delisted.csv", index=False, encoding="utf-8-sig")

    lo, hi = pd.Timestamp(a.start), pd.Timestamp(a.end)
    win = df[(df["delist_date"] >= lo) & (df["delist_date"] <= hi)]

    print(f"\n{'=' * 88}")
    print(f"來源：{name}")
    print(f"清單總筆數　　　　　{len(df):,}")
    print(f"  最早　　　　　　　{df['delist_date'].min():%Y-%m-%d}")
    print(f"  最晚　　　　　　　{df['delist_date'].max():%Y-%m-%d}")
    print(f"研究區間內下市　　　{len(win):,} 家"
          f"（{a.start} ~ {a.end}）")
    print(f"{'=' * 88}")

    if len(win):
        by_year = win.groupby(win["delist_date"].dt.year).size()
        print(f"\n逐年下市家數：")
        for y, n in by_year.items():
            print(f"  {y}　{n:>3} 家　{'█' * n}")
        print(f"\n  平均每年 {by_year.mean():.1f} 家")

        # 偏誤上界：相對於現有 383 檔母體的規模
        print(f"\n生存者偏誤的規模界定")
        print(f"  現有股票池　　　　383 檔（皆為今日仍在市者）")
        print(f"  區間內下市　　　　{len(win)} 家")
        print(f"  若全部可納入　　　母體會變成 {383 + len(win)} 檔，"
              f"下市者佔 {len(win) / (383 + len(win)):.1%}")
        print(f"\n  這是**家數**的佔比，不是報酬偏誤的大小。下市公司的報酬"
              f"通常遠低於存活者，\n  所以報酬偏誤會比家數佔比更嚴重；"
              f"精確數字要等階段二補抓股價才算得出來。")
        print(f"\n  最近 10 家下市：")
        for _, r in win.tail(10).iterrows():
            print(f"    {r['delist_date']:%Y-%m-%d}　{r['code']:<8}{r['name']}")

    print(f"\n輸出：{OUT / 'delisted.csv'}　原始：{OUT / 'raw.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
