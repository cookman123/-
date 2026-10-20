#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
母體覆蓋稽核：到底有幾檔真的進了分析，沒進的是為什麼。

為什麼需要這支
--------------
特徵庫的建立過程有兩層縮減，兩層都容易被忽略：

    一、`load_panels` 丟掉在分析區間內資料少於 300 日的個股
    二、長表只保留「當日在成交金額前 N 名」的列（時點正確母體遮罩）

結果是：磁碟上有 383 檔，面板算了 382 檔，實際存進特徵庫的平均每日只有 249 檔。
報告「跑了全檔」而實際上是前 250 名，這種落差正是本專題第 3.2 節那個
選股清單錯誤的同一類問題 —— 子集看起來像全集。

本程式把每一檔的去向列清楚，並指出被排除的原因。
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
import davis_tw_train as T  # noqa: E402

OUT = ROOT / "results" / "quarterly"


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="母體覆蓋稽核")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-days", type=int, default=300)
    a = ap.parse_args(argv)

    meta = {x["code"]: x["name"] for x in
            json.load(open(ROOT / "data" / "industry_stock_map.json",
                           encoding="utf-8"))}
    disk = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                  if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"磁碟上有股價資料　　　　{len(disk)} 檔")

    rows = []
    for c in disk:
        px = T.load_price(c)
        if px is None or px.empty:
            rows.append((c, 0, None, None, "讀不到價格"))
            continue
        p = px[(px.index >= pd.Timestamp(a.start)) & (px.index <= pd.Timestamp(a.end))]
        n = len(p)
        why = "" if n >= a.min_days else f"區間內僅 {n} 日（<{a.min_days}）"
        rows.append((c, n, p.index.min() if n else None,
                     p.index.max() if n else None, why))
    df = pd.DataFrame(rows, columns=["code", "日數", "起", "迄", "排除原因"])
    df["名稱"] = df["code"].map(meta).fillna("?")

    dropped = df[df["排除原因"] != ""]
    print(f"進不了面板　　　　　　　{len(dropped)} 檔")
    for _, r in dropped.iterrows():
        rng = (f"{r['起']:%Y-%m-%d} ~ {r['迄']:%Y-%m-%d}"
               if pd.notna(r["起"]) else "無資料")
        print(f"  {r['code']} {r['名稱']}　{r['排除原因']}　{rng}")
    print(f"進入面板　　　　　　　　{len(df) - len(dropped)} 檔")

    store = ROOT / "data" / "features" / f"panel{a.tag}.pkl"
    if store.exists():
        s = pd.read_pickle(store)
        codes = set(s.index.get_level_values("code").unique())
        d = s.index.get_level_values("date")
        print(f"\n特徵庫 {store.name}")
        print(f"  列數　　　　　　　　{len(s):,}")
        print(f"  出現過的檔數　　　　{len(codes)}")
        print(f"  平均每日檔數　　　　{len(s) / d.nunique():.0f}"
              f"　←　這才是模型真正看到的寬度")
        never = sorted(set(df[df['排除原因'] == '']['code']) - codes)
        print(f"  在面板但從未進母體　{len(never)} 檔"
              + ("：" + "、".join(f"{c}{meta.get(c, '')}" for c in never[:20])
                 if never else ""))
        # 每檔被選入母體的天數分布，看母體是穩定的還是一直在換
        cnt = s.groupby(level="code").size().sort_values()
        print(f"  入選天數：中位數 {int(cnt.median()):,}"
              f"　最少 {int(cnt.min()):,}　最多 {int(cnt.max()):,}"
              f"（總交易日 {d.nunique():,}）")
        print(f"  入選超過九成交易日的　{int((cnt > d.nunique() * 0.9).sum())} 檔")

    df.sort_values("日數").to_csv(OUT / f"coverage{a.tag}.csv", index=False,
                                  encoding="utf-8-sig")
    print(f"\n輸出：{OUT / f'coverage{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
