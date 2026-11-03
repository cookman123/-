#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日自動跟單：每天股價更新進 DB 之後執行，把設了自動跟單的模擬帳戶從上次跑到的隔天接著跑到最新一天。

建議用法（往後實際比較「只用規則」和「規則＋LLM 審核」）：
    在「模擬交易」頁開兩個帳戶，自動跟單設成同一個策略，只差 LLM 審核開或關，
    之後每天由 daily_all.py 第 13 步自動執行；手動的話：
        python tool/run_surv.py predict_daily --tag _surv
        python build_local_db.py --tables prices market news predictions
        python paper_daily.py
    跑幾個月後比較兩個帳戶。這段期間的行情模型不可能看過，是最乾淨的對照。

每日更新總控（daily_all.py）第 13 步會依序跑 predict_daily.py（模型預測接到當天）、build_local_db.py、
這支程式，所以排程好的話不用手動執行。選「LLM 選股」或勾 LLM 審核的帳戶要開著 Ollama。
"""
import sys

import paper_trade as P

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    con = P.connect()
    res = P.run_auto(con, progress=lambda f, m: print(f"  {m}", flush=True))
    if not res:
        print("沒有帳戶設定每日自動跟單（在「模擬交易」頁設定）。")
    for r in res:
        name = con.execute("SELECT name FROM paper_accounts WHERE id = ?", [r["帳戶"]]).fetchone()[0]
        extra = "".join(f"，{k} {r[k]}" for k in ("進場筆數", "LLM 略過", "LLM 減半") if k in r)
        print(f"#{r['帳戶']} {name}：{r['結果']}{extra}")
    # 跟單完把當天的 LLM 決策存證（attest_local.py），往後的對照紀錄才能證明沒被事後改
    import attest_local
    attest_local.stamp(con)
    # 有帳戶失敗就回傳非 0，每日更新總控的報告才會標成失敗
    sys.exit(1 if any(r.get("失敗") for r in res) else 0)
