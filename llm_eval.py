#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 選股審核有沒有用：只在 gemma3 訓練截止之後的期間評估（llm_judge.EVAL_START 起）。

事前固定的規格（跑之前寫死，跑完不改、不加）
------------------------------------------
候選    5 日模型（含下市股版）每 STEP=5 個交易日取一次，當天預測排名前 TOP=5 名
        （5 日一次、持有 5 日，前後兩期不重疊）
審核    llm_judge.review(pit=True)：只給當天以前已知的資料，不給用整段期間算出來的研究結論
結果    隔日開盤買、第 6 個交易日開盤賣（就是模型的標籤），扣成本 0.3855%
比較    規則組合：5 檔等權
        LLM 組合：權重＝buy 1、half 0.5、skip 0，再標準化；全部 skip 的那期放現金（報酬 0）
        每期的差（LLM − 規則）對期數做 t 檢定（各期不重疊）
採用    差 > 0 且 t > 2 才算 LLM 審核有用；否則只當解說，不拿來改變下單

另外列出：保留（buy／half）與略過（skip）的平均報酬、略過比例、LLM 失敗筆數。
LLM 判斷都存在 llm_decisions 表，中斷後重跑會沿用，不用重算。

輸出：results/llm_eval/decisions.csv、summary.csv
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

OUT = ROOT / "results" / "llm_eval"
STEP, TOP, H = 5, 5, 5
COST = 2 * P.FEE + P.TAX


def candidates(con, start: str) -> pd.DataFrame:
    p = pd.read_sql("SELECT date, code, pred, pctile FROM predictions WHERE h = ? AND date >= ?", con,
                    params=[H, start])
    days = sorted(set(p["date"]) & set(F.complete_days(con, H).index))
    last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    cal = [d for (d,) in con.execute("SELECT date FROM market WHERE symbol = 'TWII' AND date >= ? ORDER BY date",
                                     [start])]
    # 標籤要走完：第 6 個交易日的開盤要在資料裡
    ok = {d for i, d in enumerate(cal) if i + H + 1 < len(cal) and cal[i + H + 1] <= last}
    days = [d for d in days if d in ok][::STEP]
    p = p[p["date"].isin(days)]
    return p.sort_values("pred", ascending=False).groupby("date").head(TOP).sort_values(["date", "pred"],
                                                                                    ascending=[True, False])


def outcome(con, code: str, d: str):
    r = con.execute("SELECT date, open FROM prices WHERE code = ? AND date > ? ORDER BY date LIMIT ?",
                    [code, d, H + 1]).fetchall()
    if len(r) < H + 1 or not r[0][1] or not r[H][1]:
        return np.nan
    return r[H][1] / r[0][1] - 1 - COST


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="LLM 選股審核的事後評估（訓練截止之後）")
    ap.add_argument("--start", default=J.EVAL_START)
    ap.add_argument("--limit", type=int, default=0, help="只跑前幾筆（試跑用）")
    a = ap.parse_args(argv)
    if a.start < J.EVAL_START:
        print(f"警告：{a.start} 早於 {J.EVAL_START}，模型可能記得那段行情，結果會偏樂觀")
    if J.available():
        print(J.available())
        return 1
    con = F.connect()
    J.ensure(con)
    c = candidates(con, a.start)
    if a.limit:
        c = c.head(a.limit)
    print(f"候選 {len(c)} 筆、{c['date'].nunique()} 期（{c['date'].min()} ~ {c['date'].max()}）")
    rows, t0 = [], time.time()
    for i, r in enumerate(c.itertuples(), 1):
        j = J.review(con, r.code, r.date, "h5", pit=True, batch="llm_eval")
        rows.append(dict(date=r.date, code=r.code, pred=r.pred, decision=j["decision"], size=j["size"],
                         confidence=j["confidence"], error=j["error"], cached=j["cached"],
                         net=outcome(con, r.code, r.date), reason=j["reason"]))
        if i % 10 == 0 or i == len(c):
            el = time.time() - t0
            print(f"  {i}/{len(c)}　{el / 60:.0f} 分　預估還要 {el / i * (len(c) - i) / 60:.0f} 分", flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "decisions.csv", index=False, encoding="utf-8-sig")

    d = d.dropna(subset=["net"])
    rule = d.groupby("date")["net"].mean()
    llm = d.groupby("date").apply(lambda g: (g["net"] * g["size"]).sum() / g["size"].sum() if g["size"].sum() > 0
                                  else 0.0, include_groups=False)
    diff = llm - rule
    t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff)) if len(diff) > 2 and diff.std() > 0 else np.nan
    keep, skip = d[d["size"] > 0], d[d["size"] == 0]
    s = {"期間": f"{d['date'].min()} ~ {d['date'].max()}", "期數": len(rule), "筆數": len(d),
         "略過比例": float((d["decision"] == "skip").mean()), "減半比例": float((d["decision"] == "half").mean()),
         "LLM 失敗": int(d["error"].notna().sum()),
         "規則組合每期": float(rule.mean()), "LLM 組合每期": float(llm.mean()),
         "差（LLM−規則）": float(diff.mean()), "差 t": float(t),
         "保留的平均": float(keep["net"].mean()) if len(keep) else np.nan,
         "略過的平均": float(skip["net"].mean()) if len(skip) else np.nan,
         "結論": "LLM 審核有用（差 > 0 且 t > 2）" if diff.mean() > 0 and t > 2 else "沒有證據顯示 LLM 審核有用"}
    pd.Series(s).to_frame("值").to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print()
    for k, v in s.items():
        print(f"  {k:<14}{v:+.4f}" if isinstance(v, float) else f"  {k:<14}{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
