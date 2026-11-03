#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地端 LLM 選股／審核的隨機同檔數對照（事後補充，不改 llm_eval.py、llm_select_eval.py 的採用規則）。

做法取自 jameschen108/tw-stock-quant 的 llm_trader.py／llm_replay.py：LLM 挑了 k 檔，就從同一群候選
隨機挑 k 檔、重複很多次，看 LLM 的成績落在隨機分布的哪裡。原本的 C 組（12 檔等權）只是隨機的期望值，
沒有告訴我們「這個差距在隨機挑的時候常不常見」；也分不出 LLM 的好壞是來自挑股還是給權重。

不呼叫模型，只讀兩個評估已經存好的結果：
  選股  results/llm_select_eval/picks.csv、periods.csv
        每期從當期 12 檔候選隨機挑和 LLM 一樣多檔（有報酬的那幾檔）、等權；LLM 那期一檔都沒挑就兩邊都放現金（0）
        另算「LLM 挑的股票等權」，把挑股和給權重拆開
  審核  results/llm_eval/decisions.csv
        每期把 LLM 給的 buy／half／skip 在 5 檔之間隨機重排（略過幾檔、減半幾檔不變）
p 值 = 隨機的平均每期報酬 ≥ LLM 的比例（單尾）。
輸出 results/llm_random_control/summary.csv
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
RES = ROOT / "results"
OUT = RES / "llm_random_control"
N_RANDOM = 5000


def select_control(rng) -> list[dict]:
    # 沒有報酬的候選（停牌等，只有 1 筆）和原評估一樣不計入
    picks = pd.read_csv(RES / "llm_select_eval" / "picks.csv", dtype={"code": str}).dropna(subset=["net"])
    per = pd.read_csv(RES / "llm_select_eval" / "periods.csv").set_index("date")
    per = per[per["error"].isna()] if "error" in per else per
    groups = {d: g for d, g in picks.groupby("date") if d in per.index}
    dates = sorted(groups)
    llm = per.loc[dates, "llm"].to_numpy()
    eq = np.array([groups[d].loc[groups[d]["picked"], "net"].mean() if groups[d]["picked"].any() else 0.0
                   for d in dates])
    sims = np.zeros(N_RANDOM)
    for d in dates:
        net = groups[d]["net"].to_numpy()
        k = min(int(groups[d]["picked"].sum()), len(net))
        if k == 0:
            continue
        draw = np.argsort(rng.random((N_RANDOM, len(net))), axis=1)[:, :k]
        sims += net[draw].mean(axis=1)
    sims /= len(dates)
    return [
        {"對照": "選股：LLM 組合（照 LLM 權重）", "期數": len(dates), "LLM 每期": llm.mean(),
         "隨機每期（平均）": sims.mean(), "隨機 95% 分位": np.quantile(sims, 0.95),
         "p（隨機 ≥ LLM）": float((sims >= llm.mean()).mean())},
        {"對照": "選股：LLM 挑的股票等權", "期數": len(dates), "LLM 每期": eq.mean(),
         "隨機每期（平均）": sims.mean(), "隨機 95% 分位": np.quantile(sims, 0.95),
         "p（隨機 ≥ LLM）": float((sims >= eq.mean()).mean())},
    ]


def review_control(rng) -> list[dict]:
    dec = pd.read_csv(RES / "llm_eval" / "decisions.csv", dtype={"code": str}).dropna(subset=["net"])
    groups = [g for _, g in dec.groupby("date")]

    def port(size, net):
        w = size / size.sum() if size.sum() > 0 else size * 0
        return float((w * net).sum())

    llm = np.array([port(g["size"].to_numpy(), g["net"].to_numpy()) for g in groups])
    sims = np.zeros(N_RANDOM)
    for g in groups:
        size, net = g["size"].to_numpy(), g["net"].to_numpy()
        if (size == size[0]).all():                 # 全部照買或全部略過：怎麼重排都一樣
            sims += port(size, net)
            continue
        perm = np.argsort(rng.random((N_RANDOM, len(size))), axis=1)
        s = size[perm]
        tot = s.sum(axis=1, keepdims=True)
        sims += np.where(tot > 0, (s / np.where(tot > 0, tot, 1) * net).sum(axis=1, keepdims=True), 0).ravel()
    sims /= len(groups)
    n_adj = int(sum((g["size"] != 1).any() for g in groups))
    return [{"對照": f"審核：LLM 組合（只有 {n_adj} 期有略過或減半）", "期數": len(groups), "LLM 每期": llm.mean(),
             "隨機每期（平均）": sims.mean(), "隨機 95% 分位": np.quantile(sims, 0.95),
             "p（隨機 ≥ LLM）": float((sims >= llm.mean() - 1e-12).mean())}]


def main() -> int:
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = select_control(rng) + review_control(rng)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"隨機同檔數對照（{N_RANDOM} 次，事後補充；採用與否仍照原評估的 t > 2 規則）\n")
    for r in rows:
        print(f"  {r['對照']}")
        print(f"    {r['期數']} 期　LLM 每期 {r['LLM 每期']:+.3%}　隨機平均 {r['隨機每期（平均）']:+.3%}"
              f"　隨機 95% 分位 {r['隨機 95% 分位']:+.3%}　p {r['p（隨機 ≥ LLM）']:.3f}")
    print(f"\n輸出：{OUT / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
