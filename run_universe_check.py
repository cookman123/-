#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
母體敏感度檢查：把三件事一次跑完。

一、覆蓋稽核 —— 到底幾檔進了分析、沒進的是為什麼
二、放寬母體到全部個股，另建一份特徵庫（tag=_full，不覆蓋原本的前 250 名版）
三、用全母體重跑日頻與季頻，和前 250 名版並列對照

為什麼要做
----------
原本所有結論都建立在「成交金額前 250 名」這個母體上。那是存活策略交易的母體，
所以對該策略是正確的；但對「結論有多普遍」這個問題，母體就是一個未檢驗的選擇。

預期方向（寫在跑之前，跑完才知道對不對）：
    日頻會更負 —— 放進小型股，價差與衝擊都變大
    季頻的 beta 會更高 —— 小型股 beta 普遍更大，alpha 大概仍不顯著

把預期先寫下來，是為了避免事後用結果反推說「本來就這樣」。
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    # 輸出被重導向到檔案時 Windows 會退回 cp950，非該編碼的字元會讓整支程式
    # 在 print 當下崩掉（第一次執行就是在第 4 步被「／」打斷的）
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOG = ROOT / "results" / "universe"

STEPS = [
    ("覆蓋稽核（前 250 名版）", ["coverage_audit.py"], "coverage_250.log"),
    ("建全母體特徵庫", ["feature_engineering.py", "--uni", "400",
                        "--tag", "_full", "--quiet"], "fe_full.log"),
    ("覆蓋稽核（全母體版）", ["coverage_audit.py", "--tag", "_full"],
     "coverage_full.log"),
    ("日頻／全母體", ["nextday_v2.py", "--tag", "_full"], "nextday_full.log"),
    ("季頻／全母體", ["quarterly_factors.py", "--tag", "_full"],
     "quarterly_full.log"),
    # 五道驗證兩個母體各跑一次，否則沒有可對照的基準
    ("五道驗證／前 250 名", ["quarterly_verify.py"], "verify_250.log"),
    ("五道驗證／全母體", ["quarterly_verify.py", "--tag", "_full"],
     "verify_full.log"),
]


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    print(f"=== 母體敏感度檢查　{t0:%Y-%m-%d %H:%M} ===\n", flush=True)
    for i, (name, args, log) in enumerate(STEPS, 1):
        t = datetime.now()
        print(f"[{i}/{len(STEPS)}] {name}　→ results/universe/{log}", flush=True)
        with open(LOG / log, "w", encoding="utf-8") as f:
            rc = subprocess.run([PY, "-u", "-W", "ignore", *args],
                                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        mins = (datetime.now() - t).total_seconds() / 60
        print(f"        結束碼 {rc}　耗時 {mins:.1f} 分", flush=True)
        if rc != 0:
            print(f"        ✗ 失敗，後續步驟仍會嘗試（各步驟互相獨立）", flush=True)
    print(f"\n全部完成　總耗時 "
          f"{(datetime.now() - t0).total_seconds() / 60:.0f} 分")
    print(f"對照時請並看：")
    print(f"  前 250 名　results/nextday/v2.csv　　　results/quarterly/factors.csv")
    print(f"  全母體　　 results/nextday/v2_full.csv　results/quarterly/factors_full.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
