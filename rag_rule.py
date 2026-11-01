#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
候選規則：相似事件排除（RAG 的行為模式，排除版）。

狀態：候選。rag_event_test.py 只在 2020-10 ~ 2024-11 測過，效果集中在後半段，
尚未通過樣本外驗證；正式採用前要先累積新事件，見 results/候選規則_相似事件排除.md。

規則（每一條都只用決策當下已知的資料）
--------------------------------------
事件       跌深事件，與 event_model.py 相同（5 日內跌 8%、等止跌、分批出場、持有上限 20 天）
檢索庫     出場日早於「今天 − 30 天」的過去事件 —— 只收答案已揭曉的，再留 30 天緩衝
           每月第一個交易日更新一次
相似度     26 個技術面、籌碼、新聞統計特徵，以檢索庫的平均與標準差標準化後算歐氏距離
分數       最近 20 個過去事件的淨報酬，以距離倒數加權平均
門檻       過去 365 天內所有事件「當時」分數的第 20 百分位（至少要有 100 筆，不足就不排除）
動作       分數低於門檻 → 不做；其餘照原規則照單全收
           只排除、不挑選：測試裡「挑最好的」換個 k 就消失，「排除最差的」各 k 方向一致

為什麼門檻不能用整段測試期的百分位
----------------------------------
rag_event_test.py 的「後 20%」是拿整段四年的分數排，等於知道未來的分數分布。
實際操作只能和過去比，所以這裡改成滾動門檻，回測重新跑一次，數字以這裡為準。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_strategy as A  # noqa: E402
import event_model as E  # noqa: E402
import rag_event_test as R  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rag_rule"
K, EXCL_Q, LOOKBACK, MIN_HIST, EMBARGO, MIN_LIB = 20, 0.20, 365, 100, 30, 1500


def score(lib: pd.DataFrame, new: pd.DataFrame, feats: list) -> np.ndarray:
    """新事件的檢索分數：最近 K 個庫內事件的距離加權平均淨報酬。"""
    return R.retrieve(lib, new, feats, K)


def threshold(hist: pd.DataFrame, day: pd.Timestamp) -> float:
    """過去 LOOKBACK 天內事件當時分數的第 20 百分位；樣本不足回傳 NaN（不排除）。"""
    h = hist[(hist["entry"] < day) & (hist["entry"] >= day - pd.Timedelta(days=LOOKBACK))]
    return float(h["score"].quantile(EXCL_Q)) if len(h) >= MIN_HIST else np.nan


def decide(lib: pd.DataFrame, new: pd.DataFrame, feats: list,
           hist: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """
    每日流程的進入點：給今天的新事件打分並決定做不做。
    lib 是全部歷史事件（含 exit、net），函式自己挑出答案已揭曉的部分。
    hist 是過去事件當時的分數（entry、score），用來算門檻。
    """
    lib = lib[pd.to_datetime(lib["exit"]) < day - pd.Timedelta(days=EMBARGO)]
    o = new.copy()
    o["score"] = score(lib, new, feats)
    o["門檻"] = threshold(hist, day)
    o["排除"] = o["score"] < o["門檻"]          # 門檻是 NaN 時比較結果為 False
    return o


def backtest(ev: pd.DataFrame, feats: list) -> pd.DataFrame:
    """逐月更新檢索庫，逐事件套滾動門檻；所有分數與門檻都只用當時已知的資料。"""
    ev = ev.sort_values("entry").reset_index(drop=True)
    entry, exit_ = pd.to_datetime(ev["entry"]), pd.to_datetime(ev["exit"])
    months = entry.dt.to_period("M")
    scored = []
    for mo in months.unique():
        m0 = mo.to_timestamp()
        lib = ev[exit_ < m0 - pd.Timedelta(days=EMBARGO)]
        if len(lib) < MIN_LIB:
            continue
        te = ev[months == mo]
        o = te[["code", "entry", "exit", "net"]].copy()
        o["score"] = score(lib, te, feats)
        scored.append(o)
    S = pd.concat(scored, ignore_index=True)
    S["entry"] = pd.to_datetime(S["entry"])
    S["門檻"] = [threshold(S, d) for d in S["entry"]]
    S["排除"] = S["score"] < S["門檻"]
    return S


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="候選規則：相似事件排除")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--topn", type=int, default=120)
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    ev, feats = R.load_events(a.start, a.end, a.topn)
    S = backtest(ev, feats)
    S.to_csv(OUT / "decisions.csv", index=False, encoding="utf-8-sig")

    live = S[S["門檻"].notna()]                  # 門檻生效之後才算規則在運作
    kept, drop = live[~live["排除"]], live[live["排除"]]
    rows = [("照單全收", live), ("套用規則（保留）", kept), ("被排除的", drop)]
    print(f"{'=' * 84}\n候選規則回測　{live.entry.min():%Y-%m} ~ {live.entry.max():%Y-%m}"
          f"　K={K}　排除第 {EXCL_Q:.0%} 百分位以下（滾動 {LOOKBACK} 天門檻）\n{'=' * 84}")
    print(f"  {'':<16}{'筆數':>7}{'勝率':>8}{'平均淨報酬':>12}{'群集t':>8}")
    summ = []
    for name, g in rows:
        m, t, n = A.cluster_t(g)
        summ.append({"組別": name, "筆數": len(g), "勝率": (g.net > 0).mean(),
                     "平均淨報酬": m, "群集t": t})
        print(f"  {name:<16}{len(g):>7,}{(g.net > 0).mean():>8.1%}{m:>+12.2%}{t:>8.2f}")
    d, dt, n = E._diff(live, live["排除"])
    print(f"\n  被排除 − 保留：{d:+.2%}　差異 t {dt:.2f}（{n} 個月）")
    print(f"  實際排除比例 {live['排除'].mean():.1%}（目標 {EXCL_Q:.0%}）")

    early = live["entry"] < R.SPLIT
    for name, g in (("前段", live[early]), ("後段", live[~early])):
        dd, tt, nn = E._diff(g, g["排除"])
        print(f"  {name}（{g.entry.min():%Y-%m} ~ {g.entry.max():%Y-%m}）"
              f"被排除 − 保留 {dd:+.2%}　t {tt:.2f}（{nn} 個月）")
        summ.append({"組別": f"{name} 被排除−保留", "筆數": len(g),
                     "平均淨報酬": dd, "群集t": tt})
    summ.append({"組別": "全期 被排除−保留", "筆數": len(live), "平均淨報酬": d, "群集t": dt})
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n  判讀：差異 t < −2 才算規則在實際可用的門檻下仍有效。")
    print(f"  輸出：{OUT / 'decisions.csv'}、{OUT / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
