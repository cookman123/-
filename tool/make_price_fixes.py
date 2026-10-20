#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
由 tool/price_jump_scan.py 在修正前的結果（price_jumps_before_fix.csv）產生股價修正表 data/price_fixes.csv。

規則：相鄰交易日之間超過漲跌幅限制的跳動，除了下列兩種，其餘一律視為假跳動，
把該日之前的歷史價格整段乘上一個係數，讓那一天的報酬變成 0（等同 Yahoo 漏做的分割調整）：
    保留：新上市前 5 日（不設漲跌幅）
    保留：除權息日的「下跌」（現金股利造成，Yahoo 本來就不調整現金股利）
另外手動加入一筆：2601 在 2011-08-11 的 −18.6% 是補 2011 缺口時的係數切換造成的
（fill_2011_gap.py，原因不明的係數變化），原始價在那天沒有跳動。

load_price() 讀檔後套用這張表；原始 csv 不動。
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    j = pd.read_csv(ROOT / "results" / "data_fix_2011" / "price_jumps_before_fix.csv", dtype={"code": str})
    keep = j["why"].str.startswith("上市前") | (j["why"].str.startswith("事件日") & (j["ret"] < 0))
    fx = j[~keep].copy()
    fx = pd.concat([fx, pd.DataFrame([{"code": "2601", "date": "2011-08-11", "ret": -0.1855,
                                      "why": "補 2011 缺口時的係數切換（手動）"}])], ignore_index=True)
    # 該日之前的價格乘上 (1 + ret)，當天報酬就變成 0
    fx["factor"] = 1 + fx["ret"]
    out = ROOT / "data" / "price_fixes.csv"
    fx[["code", "date", "factor", "ret", "why"]].sort_values(["code", "date"]) \
        .to_csv(out, index=False, encoding="utf-8-sig")
    print(f"修正 {len(fx)} 筆（保留 {int(keep.sum())} 筆合理跳動），涉及 {fx['code'].nunique()} 檔\n輸出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
