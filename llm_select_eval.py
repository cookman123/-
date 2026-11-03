#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 選股（流程圖「實際操作」）有沒有比規則好：只在 gemma3 訓練截止之後的期間評估。

事前固定的規格（跑之前寫死，跑完不改、不加）
------------------------------------------
期間    llm_judge.EVAL_START（2024-10-01）起，每 STEP=5 個交易日一期（持有 5 日，各期不重疊）
候選池  當天 5 日模型（含下市股版）排名前 N=12 名
LLM     llm_judge.select(pit=True)：每檔給新資訊評分（5／20 日模型排名、近期漲跌、最新 3 則新聞標題）
        與 RAG 行為模式（價格走勢最像的 10 段歷史之後 5 日、最像的 2 張已揭曉事件卡），
        由 LLM 挑最多 K=5 檔並給權重；只給當天以前已知的資料
結果    隔日開盤買、第 6 個交易日開盤賣，扣成本 0.3855%
比較    A. LLM 組合（權重照 LLM；那期一檔都沒挑或失敗就放現金，報酬 0）
        B. 規則組合：候選池裡模型排名前 5 名等權（不用 LLM 時的做法）
        C. 候選池 12 檔等權（等於從 12 檔裡隨便挑的期望值）
        每期的差（A − B、A − C）對期數做 t 檢定
採用    A − B > 0 且 t > 2，才算「LLM 選股比規則好」；
        只有 A − C 顯著而 A − B 不顯著，代表 LLM 只是重新發現了模型排名

LLM 判斷存在 llm_decisions 表，中斷後重跑會沿用。輸出：results/llm_select_eval/periods.csv、picks.csv、summary.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import forecast as F  # noqa: E402
import llm_judge as J  # noqa: E402
import paper_trade as P  # noqa: E402

OUT = ROOT / "results" / "llm_select_eval"
STEP, N, K, H = 5, 12, 5, 5
COST = 2 * P.FEE + P.TAX


def periods(con, start: str) -> list:
    days = F.complete_days(con, H).index
    last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    cal = [d for (d,) in con.execute("SELECT date FROM market WHERE symbol = 'TWII' AND date >= ? ORDER BY date",
                                     [start])]
    ok = {d for i, d in enumerate(cal) if i + H + 1 < len(cal) and cal[i + H + 1] <= last}
    return [d for d in days if d >= start and d in ok][::STEP]


def outcome(con, code: str, d: str) -> float:
    r = con.execute("SELECT open FROM prices WHERE code = ? AND date > ? ORDER BY date LIMIT ?",
                    [code, d, H + 1]).fetchall()
    if len(r) < H + 1 or not r[0][0] or not r[H][0]:
        return np.nan
    return r[H][0] / r[0][0] - 1 - COST


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="LLM 選股 vs 規則（訓練截止之後）")
    ap.add_argument("--start", default=J.EVAL_START)
    ap.add_argument("--limit", type=int, default=0, help="只跑前幾期（試跑用）")
    a = ap.parse_args(argv)
    if J.available():
        print(J.available())
        return 1
    con = F.connect()
    J.ensure(con)
    ps = periods(con, a.start)
    if a.limit:
        ps = ps[:a.limit]
    print(f"{len(ps)} 期（{ps[0]} ~ {ps[-1]}），每期候選 {N} 檔、LLM 挑最多 {K} 檔")
    rows, picks, t0 = [], [], time.time()
    for i, d in enumerate(ps, 1):
        c = J.candidates(con, d, N, exclude_restricted=False)
        c["net"] = [outcome(con, x, d) for x in c["code"]]
        r = J.select(con, d, c, K, pit=True, batch="llm_select_eval")
        w = {p["code"]: p["weight"] for p in r["picks"]}
        net = c.set_index("code")["net"]
        llm = sum(w[x] * net[x] for x in w if net[x] == net[x]) / max(sum(w[x] for x in w if net[x] == net[x]), 1e-12) \
            if w else 0.0
        rows.append(dict(date=d, llm=llm, rule5=net.iloc[:K].mean(), all12=net.mean(), n_pick=len(w),
                         overlap=len(set(w) & set(c["code"].iloc[:K])), error=r["error"]))
        picks += [dict(date=d, code=x, weight=w.get(x, 0.0), rank=j + 1, net=net[x], picked=x in w)
                  for j, x in enumerate(c["code"])]
        if i % 5 == 0 or i == len(ps):
            el = time.time() - t0
            print(f"  {i}/{len(ps)}　{el / 60:.0f} 分　預估還要 {el / i * (len(ps) - i) / 60:.0f} 分", flush=True)
    pr, pk = pd.DataFrame(rows), pd.DataFrame(picks)
    OUT.mkdir(parents=True, exist_ok=True)
    pr.to_csv(OUT / "periods.csv", index=False, encoding="utf-8-sig")
    pk.to_csv(OUT / "picks.csv", index=False, encoding="utf-8-sig")
    ab, ac = pr["llm"] - pr["rule5"], pr["llm"] - pr["all12"]
    sel, nsel = pk[pk["picked"]], pk[~pk["picked"]]
    s = {"期間": f"{pr['date'].min()} ~ {pr['date'].max()}", "期數": len(pr),
         "LLM 失敗期數": int(pr["error"].notna().sum()), "平均挑幾檔": float(pr["n_pick"].mean()),
         "和規則前 5 名重疊（檔）": float(pr["overlap"].mean()),
         "A LLM 每期": float(pr["llm"].mean()), "B 規則前 5 每期": float(pr["rule5"].mean()),
         "C 12 檔等權每期": float(pr["all12"].mean()),
         "A−B": float(ab.mean()), "A−B t": tstat(ab), "A−C": float(ac.mean()), "A−C t": tstat(ac),
         "被挑中的平均": float(sel["net"].mean()), "沒被挑的平均": float(nsel["net"].mean()),
         "結論": ("LLM 選股比規則好（A−B > 0 且 t > 2）" if ab.mean() > 0 and tstat(ab) > 2 else
                "沒有證據顯示 LLM 選股比規則好")}
    pd.Series(s).to_frame("值").to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print()
    for k, v in s.items():
        print(f"  {k:<16}{v:+.4f}" if isinstance(v, float) else f"  {k:<16}{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
