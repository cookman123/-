#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
證交所公司行動資料：除權息（TWT49U＋明細）與減資（TWTAUU），2010 起。

為什麼需要
----------
補 2011 缺口時發現 Yahoo 股價是「分割調整價」：配股、減資往回調整，現金股利
不調整。兩件事因此需要逐筆的公司行動資料：

    一、下市股來自證交所 STOCK_DAY，是**原始價**。和存活股放在同一個面板裡，
        配股除權日會是一個假的下跌、減資恢復買賣日會是一個假的上漲。
        要換算成和 Yahoo 同一個基準才能比。
    二、現金股利從頭到尾沒算。高波動股通常配息少，加回股利後波動度溢酬可能
        縮小，必須量化。

輸出（data/corp_actions/）
--------------------------
    exright.csv     date, code, kind(權/息/權息), prev_close, ref, cash, stock_per_1000, split
    reduction.csv   date, code, prev_close, ref, reason
    raw/            原始回應快取（可斷點續傳）

明細（每股現金股利、每千股配股）只對「權」「權息」兩類去抓；「息」的現金
股利直接等於 除權息前收盤價 − 參考價，不必多一次請求。
"""
from __future__ import annotations

import json
import sys
import time
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

OUT = ROOT / "data" / "corp_actions"
RAW = OUT / "raw"
BASE = "https://www.twse.com.tw/rwd/zh"


def _get(name: str, url: str, params: dict) -> dict:
    """帶快取的 GET；失敗退避重試。"""
    f = RAW / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for k in range(5):
        try:
            r = X.http_get(url, params={**params, "response": "json"})
            break
        except Exception as e:  # noqa: BLE001
            print(f"  {name} 失敗 {e}，{10 * (k + 1)} 秒後重試", flush=True)
            time.sleep(10 * (k + 1))
    else:
        raise SystemExit(f"{name} 連續失敗")
    f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    time.sleep(3)
    return r


def _roc(s: str):
    t = "".join(ch if ch.isdigit() else " " for ch in str(s)).split()
    if len(t) < 3:
        return pd.NaT
    y = int(t[0])
    return pd.Timestamp(y + 1911 if y < 1911 else y, int(t[1]), int(t[2]))


def _f(s) -> float:
    t = "".join(ch for ch in str(s) if ch.isdigit() or ch in ".-")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def fetch(start: int, end: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    ex, red = [], []
    for y in range(start, end + 1):
        # 一次查一整年會逾時，所以按半年切
        for a, b in (("0101", "0630"), ("0701", "1231")):
            r = _get(f"TWT49U_{y}{a}", f"{BASE}/exRight/TWT49U",
                     {"startDate": f"{y}{a}", "endDate": f"{y}{b}"})
            if r.get("stat") == "OK":
                ex.append(pd.DataFrame(r["data"], columns=r["fields"]))
        r = _get(f"TWTAUU_{y}", f"{BASE}/reducation/TWTAUU",
                 {"startDate": f"{y}0101", "endDate": f"{y}1231"})
        if r.get("stat") == "OK" and r.get("data"):
            red.append(pd.DataFrame(r["data"], columns=r["fields"]))
        print(f"{y} 完成", flush=True)

    e = pd.concat(ex, ignore_index=True)
    e = pd.DataFrame({
        "date": e["資料日期"].map(_roc), "code": e["股票代號"].str.strip(),
        "kind": e["權/息"].str.strip(),
        "prev_close": e["除權息前收盤價"].map(_f), "ref": e["除權息參考價"].map(_f),
    }).drop_duplicates(["date", "code"])
    # 明細一筆一個請求，只抓本研究會用到的代號（存活股＋下市股），否則要好幾個小時
    ours = {x.name for x in (ROOT / "data" / "stocks").iterdir() if x.is_dir()} |         {x.stem for x in (ROOT / "data" / "delisted" / "price").glob("*.csv")}
    e = e[e["code"].isin(ours)].reset_index(drop=True)
    cash, stock = [], []
    need = e["kind"].str.contains("權")
    print(f"除權息 {len(e)} 筆，其中含配股 {int(need.sum())} 筆要抓明細", flush=True)
    for i, r in e.iterrows():
        if not need[i]:
            cash.append(round(r["prev_close"] - r["ref"], 6))
            stock.append(0.0)
            continue
        d = _get(f"DETAIL_{r['code']}_{r['date']:%Y%m%d}", f"{BASE}/exRight/TWT49UDetail",
                 {"STK_NO": r["code"], "T1": f"{r['date']:%Y%m%d}"})
        row = (d.get("data") or [[None] * 6])[0]
        cash.append(_f(row[2]) if row[2] else 0.0)
        # 每千股無償配股（A 欄）。B 欄員工紅利轉增資是「總股數」不是每千股，不能加
        stock.append(_f(row[4]) if row[4] else 0.0)
    e["cash"], e["stock_per_1000"] = cash, stock
    # 分割比：把除權的機械性跳動（配股、現金增資）全部抵掉，但保留現金股利的下跌，
    # 與 Yahoo 的慣例一致。純配股時等於 1 + A/1000（1409 2011-09-26：
    # (10.15 − 0.3528) ÷ 9.28 = 1.0557，Yahoo 給 1.05544）
    e["split"] = 1.0
    w = e["kind"].str.contains("權") & (e["ref"] > 0)
    e.loc[w, "split"] = (e.loc[w, "prev_close"] - e.loc[w, "cash"]) / e.loc[w, "ref"]
    e.sort_values(["code", "date"]).to_csv(OUT / "exright.csv", index=False, encoding="utf-8-sig")

    rd = pd.concat(red, ignore_index=True)
    rd = pd.DataFrame({
        "date": rd["恢復買賣日期"].map(_roc), "code": rd["股票代號"].str.strip(),
        "prev_close": rd["停止買賣前收盤價格"].map(_f), "ref": rd["恢復買賣參考價"].map(_f),
        "reason": rd["減資原因"].str.strip(),
    }).drop_duplicates(["date", "code"])
    rd.sort_values(["code", "date"]).to_csv(OUT / "reduction.csv", index=False, encoding="utf-8-sig")
    print(f"減資 {len(rd)} 筆\n輸出：{OUT}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="抓證交所除權息與減資資料")
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2026)
    a = ap.parse_args(argv)
    fetch(a.start, a.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
