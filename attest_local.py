#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地端 LLM 決策存證：證明 db/local.db 的 llm_decisions 在某個時間點已經是這個樣子，之後只有追加、沒有改寫。

做法取自 jameschen108/tw-stock-quant 的 attest.py，改成存 SQLite 的資料列而不是 jsonl 檔。
為什麼需要：paper_daily.py 往後每天累積「只用規則」對「規則＋LLM 審核」的對照，那是最乾淨的證據
（未來的行情模型不可能看過）。但 llm_decisions 是一般的資料表，事後改掉幾筆不會留下痕跡；
有了存證，任何人都能驗證報告裡的 LLM 決策是當時就做好的，不是看完行情才補或改的。

用法
----
    python attest_local.py stamp      # 把目前的決策附到存證；有裝 ots（OpenTimestamps）就順便蓋章
    python attest_local.py verify     # 重算 DB，檢查每一筆存證：舊資料列有沒有被改、刪，存證鏈有沒有斷

原理
----
  - 每一列（含 items：模型看到的資料原文、raw：模型原始輸出）算 sha256，依 id 由小到大串成一個累積雜湊
  - 每筆存證記下列數、最大 id、累積雜湊，以及上一筆存證的雜湊（prev），整串成鏈，改舊存證會斷鏈
  - attest/llm_decisions.rows 逐列記下 id 與雜湊（只追加），verify 失敗時能指出是哪一列被改
  - 時間證明靠第三方：把 attest/ 提交到 git 並推上 GitHub（推送時間），或裝了 ots 時的比特幣時間戳
  - 第一次 stamp 之前的內容無法回溯證明；也不證明 LLM 的判斷對，只證明紀錄沒被事後改
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DB = ROOT / "db" / "local.db"
ATTEST = ROOT / "attest"
LEDGER = ATTEST / "ledger.jsonl"
TABLE = "llm_decisions"
ROWS = ATTEST / f"{TABLE}.rows"
GENESIS = "0" * 64


def sha(x: str | bytes) -> str:
    return hashlib.sha256(x.encode("utf-8") if isinstance(x, str) else x).hexdigest()


def row_hashes(con, max_id: int | None = None) -> list[tuple[int, str]]:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({TABLE})")]
    q = f"SELECT {', '.join(cols)} FROM {TABLE}" + (" WHERE id <= ?" if max_id is not None else "") + " ORDER BY id"
    rows = con.execute(q, [max_id] if max_id is not None else []).fetchall()
    return [(r[0], sha(json.dumps(dict(zip(cols, r)), ensure_ascii=False, sort_keys=True, default=str)))
            for r in rows]


def chain(hashes: list[tuple[int, str]]) -> str:
    acc = GENESIS
    for _, h in hashes:
        acc = sha(acc + h)
    return acc


def ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]


def entry_hash(e: dict) -> str:
    return sha(json.dumps({k: v for k, v in e.items() if k != "hash"}, ensure_ascii=False, sort_keys=True))


def stamp(con) -> int:
    ATTEST.mkdir(exist_ok=True)
    hs = row_hashes(con)
    led = ledger()
    if led and led[-1]["n_rows"] == len(hs) and led[-1]["max_id"] == (hs[-1][0] if hs else 0):
        print(f"沒有新的決策（{len(hs)} 列），不新增存證。")
        return 0
    e = {"seq": len(led) + 1, "time": datetime.now().astimezone().isoformat(timespec="seconds"),
         "table": TABLE, "n_rows": len(hs), "max_id": hs[-1][0] if hs else 0, "digest": chain(hs),
         "prev": led[-1]["hash"] if led else GENESIS}
    e["hash"] = entry_hash(e)
    old = {int(x.split("\t")[0]) for x in ROWS.read_text(encoding="utf-8").splitlines()} if ROWS.exists() else set()
    with open(ROWS, "a", encoding="utf-8") as f:
        f.writelines(f"{i}\t{h}\n" for i, h in hs if i not in old)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"存證 #{e['seq']}：{e['n_rows']} 列（id ≤ {e['max_id']}），digest {e['digest'][:16]}…")
    if shutil.which("ots"):
        dig = ATTEST / f"{e['seq']:04d}.digest"
        dig.write_text(e["hash"] + "\n", encoding="utf-8")
        r = subprocess.run(["ots", "stamp", str(dig)], capture_output=True, text=True)
        print("OpenTimestamps：" + ("已送出，數小時後跑 ots upgrade" if r.returncode == 0 else r.stderr.strip()[:200]))
    else:
        print("沒有裝 ots；時間證明請把 attest/ 提交到 git 並推上 GitHub。")
    return 0


def verify(con) -> int:
    led = ledger()
    if not led:
        print("還沒有存證（先跑 python attest_local.py stamp）。")
        return 1
    saved = dict(tuple(x.split("\t")) for x in ROWS.read_text(encoding="utf-8").splitlines()) if ROWS.exists() else {}
    ok, prev = True, GENESIS
    for e in led:
        msg = []
        if e["prev"] != prev:
            msg.append("鏈結斷了（prev 對不上上一筆）")
        if entry_hash(e) != e["hash"]:
            msg.append("存證本身被改過")
        hs = row_hashes(con, e["max_id"])
        if len(hs) != e["n_rows"]:
            msg.append(f"列數 {len(hs)}，存證時是 {e['n_rows']}（有資料列被刪除或補進舊 id）")
        if chain(hs) != e["digest"]:
            changed = [i for i, h in hs if saved.get(str(i)) not in (None, h)]
            gone = sorted(set(map(int, saved)) - {i for i, _ in hs} - {i for i in map(int, saved) if i > e["max_id"]})
            msg.append("內容和存證不同" + (f"；被改的 id：{changed[:10]}" if changed else "")
                       + (f"；不見的 id：{gone[:10]}" if gone else ""))
        print(f"  #{e['seq']} {e['time']}  {e['n_rows']} 列  " + ("✓" if not msg else "✗ " + "；".join(msg)))
        ok &= not msg
        prev = e["hash"]
    n_now = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    print(("全部通過" if ok else "驗證失敗") + f"。目前 {n_now} 列，最後一次存證涵蓋 {led[-1]['n_rows']} 列。")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="地端 LLM 決策存證")
    ap.add_argument("cmd", choices=["stamp", "verify"])
    ap.add_argument("--db", default=str(DB))
    a = ap.parse_args(argv)
    con = sqlite3.connect(a.db)
    return stamp(con) if a.cmd == "stamp" else verify(con)


if __name__ == "__main__":
    raise SystemExit(main())
