#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
證交所 T86（三大法人買賣超日報）每日增量抓取。

專案原本的 T86 是一次性提供的原始檔（data/twse/twse/t86/，2015-01 ～ 2024-12），
沒有爬蟲，所以 2025 年起的法人資料一直沒有更新。這支從「已有的最後一天」接著抓到今天，
存成和原始檔完全相同的 JSON（t86_YYYYMMDD.json），之後交給既有的
`python -m davis_tw.prepare_chips` 轉成 data/chips/{代號}.csv，下游程式不用改。

規則
----
    非交易日：證交所回 stat 不是 OK。超過 3 天前的就照存（標記處理過，之後不再問）；
              最近 3 天的不存，因為可能只是當天資料還沒公布（約 15:00 後才有）
    間隔 3 秒，被擋時退避重試
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tw_stock_crawler as X  # noqa: E402

T86 = ROOT / "data" / "twse" / "twse" / "t86"
URL = "https://www.twse.com.tw/rwd/zh/fund/T86"


def last_date() -> date:
    files = sorted(T86.glob("t86_*.json"))
    s = files[-1].stem.split("_")[1] if files else "20141231"
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="證交所 T86 三大法人買賣超（增量）")
    ap.add_argument("--start", default=None, help="預設：已有的最後一天的隔天")
    a = ap.parse_args(argv)
    T86.mkdir(parents=True, exist_ok=True)
    d = date.fromisoformat(a.start) if a.start else last_date() + timedelta(days=1)
    today = date.today()
    n_ok = n_skip = 0
    print(f"T86 抓取 {d} ~ {today}", flush=True)
    while d <= today:
        if d.weekday() < 5:
            f = T86 / f"t86_{d:%Y%m%d}.json"
            r = None
            for k in range(5):
                try:
                    r = X.http_get(URL, params={"date": f"{d:%Y%m%d}", "selectType": "ALLBUT0999",
                                                "response": "json"})
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  {d} 失敗 {e}，{10 * (k + 1)} 秒後重試", flush=True)
                    time.sleep(10 * (k + 1))
            if r is not None:
                ok = isinstance(r, dict) and r.get("stat") == "OK"
                if ok or (today - d).days > 3:
                    f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                n_ok += ok
                n_skip += not ok
            time.sleep(3)
        d += timedelta(days=1)
    print(f"完成：交易日 {n_ok} 天，非交易日或尚未公布 {n_skip} 天；最後一天 {last_date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
