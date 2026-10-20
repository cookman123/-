#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把下市股的證交所原始價換算成和存活股（Yahoo）同一個基準。

問題
----
存活股的 Yahoo 價格對配股、減資做了分割調整（價 ÷ 分割比、量 × 分割比），
現金股利不調整。下市股來自證交所 STOCK_DAY，是完全未調整的原始價。兩者併在
同一個面板（survivorship_fix.py）會出兩種錯：
    配股除權日：原始價機械性下跌（例如配 100 股／千股，價格掉約 9%），
                被當成真實虧損
    減資恢復日：原始價機械性上漲（彌補虧損減資常讓價格翻倍），被當成獲利
而且 60 日波動度訊號會把這些跳動算進去，讓下市股更容易被選進高波動組合。

做法（與 Yahoo 的慣例一致）
--------------------------
    除權（配股、現金增資）：分割比 s = (除權息前收盤 − 現金股利) ÷ 參考價。
          純配股時等於 1 + 每千股配股/1000，與 Yahoo 的 splitRatio 相同
          （1409 在 2011-09-26：1.0557 vs Yahoo 1.05544）
    減資：分割比 s = 停止買賣前收盤 ÷ 恢復買賣參考價
          （退還股款的現金部分 Yahoo 不調整，這裡無法拆開，近似處理；
          下市股只有 2 筆減資，影響很小）
    事件日之前的價格 ÷ s、成交量 × s；成交金額不變。
    現金股利不調整（與存活股一致）。

輸出到 data/delisted/price_adj/，原始檔不動。survivorship_fix.py 預設讀調整後
的版本，--raw 可切回原始價做對照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "delisted" / "price"
DST = ROOT / "data" / "delisted" / "price_adj"
CA = ROOT / "data" / "corp_actions"


def split_events() -> pd.DataFrame:
    e = pd.read_csv(CA / "exright.csv", dtype={"code": str}, parse_dates=["date"])
    e = e[e["split"] > 1.0005]
    s = pd.DataFrame({"date": e["date"], "code": e["code"],
                      "s": e["split"], "kind": "除權"})
    r = pd.read_csv(CA / "reduction.csv", dtype={"code": str}, parse_dates=["date"])
    r = r[(r["prev_close"] > 0) & (r["ref"] > 0)]
    r = pd.DataFrame({"date": r["date"], "code": r["code"],
                      "s": r["prev_close"] / r["ref"], "kind": "減資"})
    return pd.concat([s, r], ignore_index=True)


def adjust(d: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """ev：該檔的 (date, s)。回傳調整後的 OHLCV。"""
    fac = pd.Series(1.0, index=d.index)
    for _, e in ev.iterrows():
        fac[d.index < e["date"]] /= e["s"]
    out = d.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = d[c] * fac
    out["volume"] = (d["volume"] / fac).round()
    return out


def main() -> int:
    ev = split_events()
    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(SRC.glob("*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"]).set_index("date").sort_index()
        e = ev[(ev["code"] == f.stem) & (ev["date"] > d.index.min()) & (ev["date"] <= d.index.max())]
        a = adjust(d, e) if len(e) else d
        a.reset_index().to_csv(DST / f.name, index=False, encoding="utf-8-sig")
        # 驗證：事件日的單日報酬，調整前 vs 調整後
        for _, x in e.iterrows():
            i = d.index.searchsorted(x["date"])
            if 0 < i < len(d):
                r0 = d["close"].iloc[i] / d["close"].iloc[i - 1] - 1
                r1 = a["close"].iloc[i] / a["close"].iloc[i - 1] - 1
                rows.append((f.stem, x["date"].date(), x["kind"], round(x["s"], 4), r0, r1))
    v = pd.DataFrame(rows, columns=["code", "date", "kind", "s", "ret_raw", "ret_adj"])
    v.to_csv(DST / "_events.csv", index=False, encoding="utf-8-sig")
    print(f"下市股 {len(list(SRC.glob('*.csv')))} 檔，有事件的 {v['code'].nunique()} 檔，事件 {len(v)} 筆"
          f"（除權 {int((v['kind'] == '除權').sum())}、減資 {int((v['kind'] == '減資').sum())}）")
    if len(v):
        print(f"事件日單日報酬：原始 中位數 {v['ret_raw'].median():+.2%}、最差 {v['ret_raw'].min():+.2%}、最大 {v['ret_raw'].max():+.2%}")
        print(f"　　　　　　　　調整後 中位數 {v['ret_adj'].median():+.2%}、最差 {v['ret_adj'].min():+.2%}、最大 {v['ret_adj'].max():+.2%}")
        print(f"調整後仍超過 ±10% 的事件：{int((v['ret_adj'].abs() > 0.10).sum())} 筆")
        print(v[v["ret_adj"].abs() > 0.10].to_string(index=False))
    print(f"輸出：{DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
