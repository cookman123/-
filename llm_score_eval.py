#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 選股第二版（打分＋模型沒有的資訊）有沒有比規則好：只在 gemma3 訓練截止之後的期間評估。

和第一版（llm_select_eval.py）的差別
------------------------------------
一、多給模型沒有的資訊：月營收年增率（最新一個月、近 3 月合計）、營收是否創新高、最近一季毛利率、
    營益率與 EPS 對去年同季；新聞從只有標題改成標題＋內文開頭。全部用公告日判斷，只給當天已公告的。
二、改成每檔打 0～100 分，取前 5 名等權；可以算分數和實際報酬的相關（IC），檢定力比「挑或不挑」高。
候選池和第一版完全相同（5 日模型前 12 名，exclude_restricted=False），兩版可以直接比較。
不排除處置／注意股的原因：data/restricted.parquet 寫規格時還在下載（注意股只到 2017 年左右），
用不完整的名單會讓各期排除的標準不一致。規則組合 B、C、D 也從同一個池子取。

事前固定的規格（跑之前寫死，跑完不改、不加）
------------------------------------------
期間    2024-10-01 起，每 5 個交易日一期（持有 5 日、各期不重疊），和第一版相同
結果    隔日開盤買、第 6 個交易日開盤賣，扣成本 0.3855%
比較    A. LLM 組合：分數前 5 名等權
        B. 規則組合：候選池裡模型排名前 5 名等權
        C. 候選池 12 檔等權
        D. 營收規則：候選池裡最新月營收年增率前 5 名等權（缺值排最後）
           ——LLM 贏了的話，要確定不是「只要看營收年增就做得到」
主檢定  A − B 每期的差對期數做 t 檢定。這是同一段期間的第二次 LLM 選股檢定，
        門檻用 Bonferroni 修正：t > 2.24（單尾 0.025 ÷ 2），而且 A − D > 0，才算第二版有用
輔助    每期 LLM 分數和實際報酬的 Spearman IC，取平均與 t；漏打分數的期數

輸出：results/llm_score_eval/periods.csv、scores.csv、summary.csv
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
from llm_select_eval import outcome, periods, tstat  # noqa: E402

OUT = ROOT / "results" / "llm_score_eval"
N, K = 12, 5
T_ADOPT = 2.24


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="LLM 選股第二版 vs 規則（訓練截止之後）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前幾期（試跑用）")
    a = ap.parse_args(argv)
    if J.available():
        print(J.available())
        return 1
    con = F.connect()
    J.ensure(con)
    ps = periods(con, J.EVAL_START)
    if a.limit:
        ps = ps[:a.limit]
    print(f"{len(ps)} 期（{ps[0]} ~ {ps[-1]}），每期候選 {N} 檔、LLM 打分取前 {K} 名")
    rows, sc, t0 = [], [], time.time()
    for i, d in enumerate(ps, 1):
        c = J.candidates(con, d, N, exclude_restricted=False)
        net = pd.Series([outcome(con, x, d) for x in c["code"]], index=c["code"])
        r = J.score(con, d, c, K, batch="llm_score_eval")
        s = pd.Series(r["scores"]).reindex(net.index)
        rev = pd.Series(r["rev_yoy"]).reindex(net.index)
        rev_top = rev.sort_values(ascending=False, na_position="last").index[:K]
        ok = net.notna()
        ic = s[ok].corr(net[ok], method="spearman") if ok.sum() >= 5 and s[ok].nunique() > 1 else np.nan
        rows.append(dict(date=d, llm=net[r["picks"]].mean(), rule5=net.iloc[:K].mean(), all12=net.mean(),
                         rev5=net[rev_top].mean(), ic=ic, overlap=len(set(r["picks"]) & set(net.index[:K])),
                         n_missing=r["n_missing"], error=r["error"]))
        sc += [dict(date=d, code=x, rank=j + 1, score=s[x], rev_yoy=rev[x], net=net[x], picked=x in r["picks"])
               for j, x in enumerate(net.index)]
        if i % 5 == 0 or i == len(ps):
            el = time.time() - t0
            print(f"  {i}/{len(ps)}　{el / 60:.0f} 分　預估還要 {el / i * (len(ps) - i) / 60:.0f} 分", flush=True)
    pr, sd = pd.DataFrame(rows), pd.DataFrame(sc)
    OUT.mkdir(parents=True, exist_ok=True)
    pr.to_csv(OUT / "periods.csv", index=False, encoding="utf-8-sig")
    sd.to_csv(OUT / "scores.csv", index=False, encoding="utf-8-sig")
    ab, ac, ad = pr["llm"] - pr["rule5"], pr["llm"] - pr["all12"], pr["llm"] - pr["rev5"]
    ok = ab.mean() > 0 and tstat(ab) > T_ADOPT and ad.mean() > 0
    s = {"期間": f"{pr['date'].min()} ~ {pr['date'].max()}", "期數": len(pr),
         "LLM 失敗期數": int(pr["error"].notna().sum()), "有漏打分數的期數": int((pr["n_missing"] > 0).sum()),
         "和規則前 5 名重疊（檔）": float(pr["overlap"].mean()),
         "A LLM 前 5 每期": float(pr["llm"].mean()), "B 規則前 5 每期": float(pr["rule5"].mean()),
         "C 12 檔等權每期": float(pr["all12"].mean()), "D 營收年增前 5 每期": float(pr["rev5"].mean()),
         "A−B": float(ab.mean()), "A−B t": tstat(ab), "A−C": float(ac.mean()), "A−C t": tstat(ac),
         "A−D": float(ad.mean()), "A−D t": tstat(ad),
         "IC 平均": float(pr["ic"].mean()), "IC t": tstat(pr["ic"]),
         "結論": (f"第二版 LLM 選股比規則好（A−B t > {T_ADOPT} 且贏過營收規則）" if ok else
                "沒有證據顯示第二版 LLM 選股比規則好")}
    pd.Series(s).to_frame("值").to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print()
    for k, v in s.items():
        print(f"  {k:<18}{v:+.4f}" if isinstance(v, float) else f"  {k:<18}{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
