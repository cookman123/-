#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
財報與月營收爬蟲：公開資訊觀測站（MOPS）的彙總報表。

為什麼需要這一份
----------------
原本的「基本面」只有證交所每日的本益比、股價淨值比、殖利率（tw_fundamentals_crawler.py），
沒有營收、獲利、資產負債。這支抓兩種一次回傳全市場的彙總表：

  月營收     t21sc03（每月一頁，上市 sii、上櫃 otc），2010-01 起
  季報       ajax_t163sb04（綜合損益表）、ajax_t163sb05（資產負債表），2013Q1 起
             （IFRS 前的 2010–2012 這兩個端點沒有資料）

時點正確（最重要）
------------------
財報期間結束時資料還沒公布。每一筆都附上「最晚何時一定公布了」的 avail_date，
使用時只能在 avail_date 之後的交易日才用：

  月營收     次月 10 日（法定期限）
  季報       各行業法定期限取最晚者：Q1 5/30、Q2 8/31、Q3 11/29、Q4 次年 3/31

用期限而不是實際公告日，是保守的作法：很多公司會提早公布，但用期限絕不會偷看。
另一個限制：彙總表顯示的是「最新版」數字，若公司事後更正財報，這裡拿到的是更正後的值。

輸出
----
  data/financials/raw/         原始網頁（斷點續跑：已存在就不重抓）
  data/financials/revenue.csv  code, ym, revenue（千元）, yoy_pct, avail_date
  data/financials/quarterly.csv
      code, year, q, revenue, gross_profit, op_income, net_income, eps,   ← 單季（已由累計差分）
      total_assets, total_liab, equity, bvps, avail_date

用法：
    python financials_crawler.py                 # 抓全部並解析
    python financials_crawler.py --parse-only    # 只用 raw/ 重新解析
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "financials"
RAW = OUT / "raw"
BASE = "https://mopsov.twse.com.tw"
MARKETS = ("sii", "otc")
SLEEP = 4.0            # MOPS 對密集請求會暫時封鎖，寧可慢

# 各季的公布期限（取所有行業中最晚的一個）
DEADLINE = {1: (0, 5, 30), 2: (0, 8, 31), 3: (0, 11, 29), 4: (1, 3, 31)}

# 不同行業的表格欄名不同，依序取第一個存在的欄位
INCOME_COLS = {
    "revenue": ["營業收入", "收益", "淨收益", "收入", "利息淨收益"],
    "gross_profit": ["營業毛利（毛損）淨額", "營業毛利（毛損）"],
    "op_income": ["營業利益（損失）", "營業利益"],
    "net_income": ["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主",
                   "本期淨利（淨損）", "本期稅後淨利（淨損）"],
    "eps": ["基本每股盈餘（元）"],
}
BALANCE_COLS = {
    "total_assets": ["資產總計", "資產總額"],
    "total_liab": ["負債總計", "負債總額"],
    "equity": ["歸屬於母公司業主之權益合計", "歸屬於母公司業主之權益", "權益總計", "權益總額"],
    "bvps": ["每股參考淨值"],
}

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0"


def fetch(path: Path, url: str, data: dict | None = None, min_bytes: int = 5000) -> bytes | None:
    """抓一頁並存檔；已存在就直接讀。頁面太小代表沒資料或被擋，不存。"""
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            r = (session.post(url, data=data, timeout=60) if data
                 else session.get(url, timeout=60))
            time.sleep(SLEEP)
            if r.status_code == 200 and len(r.content) >= min_bytes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
                return r.content
            if r.status_code == 200:
                return None          # 該期沒有資料
        except requests.RequestException as e:
            print(f"    重試 {attempt + 1}：{e}", flush=True)
        time.sleep(SLEEP * (attempt + 2) * 3)
    print(f"  失敗：{url} {data or ''}", flush=True)
    return None


def months(start: str, end: str):
    p = pd.Period(start, "M")
    while p <= pd.Period(end, "M"):
        yield p.year, p.month
        p += 1


def crawl(start_rev: str, start_q: str, end: str) -> None:
    today = pd.Timestamp(end)
    print("月營收…", flush=True)
    for y, m in months(start_rev, end):
        if pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=10) > today:
            break
        for mk in MARKETS:
            for k in (0, 1):     # 0 國內公司、1 國外公司（KY）
                fetch(RAW / "revenue" / f"{mk}_{y}{m:02d}_{k}.html",
                      f"{BASE}/nas/t21/{mk}/t21sc03_{y - 1911}_{m}_{k}.html",
                      min_bytes=3000 if k else 5000)
        print(f"  {y}-{m:02d}", flush=True)

    print("季報…", flush=True)
    y0, q0 = int(start_q[:4]), int(start_q[-1])
    for y in range(y0, today.year + 1):
        for q in range(1, 5):
            if (y, q) < (y0, q0):
                continue
            dy, dm, dd = DEADLINE[q]
            if pd.Timestamp(y + dy, dm, dd) > today:
                continue
            for mk in MARKETS:
                for ep in ("t163sb04", "t163sb05"):
                    fetch(RAW / ep / f"{mk}_{y}Q{q}.html", f"{BASE}/mops/web/ajax_{ep}",
                          data={"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1,
                                "isQuery": "Y", "TYPEK": mk, "year": y - 1911,
                                "season": f"{q:02d}"})
            print(f"  {y}Q{q}", flush=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip()
                         .replace({"--": None, "": None, "nan": None}), errors="coerce")


def _flat(cols) -> list:
    return [str(c[-1] if isinstance(c, tuple) else c).replace(" ", "") for c in cols]


def parse_revenue() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "revenue").glob("*.html")):
        mk, ym, _ = f.stem.split("_")
        try:
            tabs = pd.read_html(io.BytesIO(f.read_bytes()), encoding="cp950")
        except ValueError:
            continue
        for t in tabs:
            cols = _flat(t.columns)
            if "公司代號" not in cols or "當月營收" not in cols:
                continue
            t.columns = cols
            d = pd.DataFrame({"code": t["公司代號"].astype(str).str.strip(),
                              "revenue": _num(t["當月營收"]),
                              "yoy_pct": _num(t["去年同月增減(%)"])})
            d = d[d["code"].str.fullmatch(r"\d{4,6}")]
            d["ym"] = f"{ym[:4]}-{ym[4:]}"
            d["market"] = mk
            rows.append(d)
    rev = pd.concat(rows, ignore_index=True).drop_duplicates(["code", "ym"], keep="last")
    p = pd.PeriodIndex(rev["ym"], freq="M")
    rev["avail_date"] = ((p + 1).to_timestamp() + pd.Timedelta(days=9)).strftime("%Y-%m-%d")
    return rev.sort_values(["code", "ym"]).reset_index(drop=True)


def _parse_q(ep: str, wanted: dict) -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / ep).glob("*.html")):
        mk, yq = f.stem.split("_")
        y, q = int(yq[:4]), int(yq[-1])
        tabs = pd.read_html(io.BytesIO(f.read_bytes()), encoding="utf-8")
        for t in tabs:
            cols = _flat(t.columns)
            if "公司代號" not in cols:
                continue
            t.columns = cols
            d = pd.DataFrame({"code": t["公司代號"].astype(str).str.strip()})
            for k, cands in wanted.items():
                # 欄位存在但值是 --（例如沒有子公司就沒有「歸屬於母公司業主」），逐列往下一個候選欄補
                v = pd.Series(float("nan"), index=t.index)
                for c in cands:
                    if c in cols:
                        v = v.combine_first(_num(t[c]))
                d[k] = v
            d["year"], d["q"], d["market"] = y, q, mk
            rows.append(d[d["code"].str.fullmatch(r"\d{4,6}")])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["code", "year", "q"], keep="last")


def parse_quarterly() -> pd.DataFrame:
    inc = _parse_q("t163sb04", INCOME_COLS)
    bal = _parse_q("t163sb05", BALANCE_COLS)
    # 損益表是「年初至該季累計」，差分成單季；缺前一季就無法差分，留 NaN
    inc = inc.sort_values(["code", "year", "q"])
    flow = list(INCOME_COLS)
    prev = inc.groupby(["code", "year"])[flow].shift(1)
    prev_q = inc.groupby(["code", "year"])["q"].shift(1)
    single = inc[flow] - prev.where(prev_q == inc["q"] - 1)
    single[inc["q"] == 1] = inc.loc[inc["q"] == 1, flow]
    inc[flow] = single
    df = inc.merge(bal.drop(columns="market"), on=["code", "year", "q"], how="outer")
    df["avail_date"] = [date(y + DEADLINE[q][0], DEADLINE[q][1], DEADLINE[q][2]).isoformat()
                        for y, q in zip(df["year"], df["q"])]
    return df.sort_values(["code", "year", "q"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MOPS 月營收與季報彙總表")
    ap.add_argument("--start-rev", default="2010-01")
    ap.add_argument("--start-q", default="2013Q1")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args(argv)

    if not a.parse_only:
        crawl(a.start_rev, a.start_q, a.end)
    rev = parse_revenue()
    rev.to_csv(OUT / "revenue.csv", index=False, encoding="utf-8-sig")
    print(f"月營收 {len(rev):,} 列　{rev['code'].nunique()} 家　{rev['ym'].min()} ~ {rev['ym'].max()}")
    qt = parse_quarterly()
    qt.to_csv(OUT / "quarterly.csv", index=False, encoding="utf-8-sig")
    print(f"季報 {len(qt):,} 列　{qt['code'].nunique()} 家　"
          f"{qt['year'].min()}Q{qt.loc[qt['year'] == qt['year'].min(), 'q'].min()} ~ "
          f"{qt['year'].max()}Q{qt.loc[qt['year'] == qt['year'].max(), 'q'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
