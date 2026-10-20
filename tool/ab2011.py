#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2011 缺口修正的前後對照：同一份程式，只換 2011 年的價格。

用法：python tool/ab2011.py old|new 腳本.py [腳本參數…]

  new  照常執行（修正後的資料）
  old  把 yahoo_*_2011.csv 換回 data/_backup_yahoo_2011/ 的原檔（缺 1～8 月），
       母體遮罩快取改用以舊價格在暫存區重建的版本，其餘資料與程式完全相同。
       這樣兩次執行的差異只來自 2011 年的補值。

old 模式的輸出仍寫在腳本預設的位置，跑完要自行搬走，避免蓋掉修正後的結果。
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import davis_tw_train as T  # noqa: E402
import universe_pit as U  # noqa: E402

BACKUP = ROOT / "data" / "_backup_yahoo_2011"
OLD_CACHE = Path(os.environ.get("AB_OLD_CACHE",
                                ROOT / "results" / "data_fix_2011" / "turnover_rank_old.csv"))


def load_price_old(code: str):
    files = sorted((T.DATA / "stocks" / code / "price").glob("yahoo_*.csv"))
    files = [f for f in files if "merged" not in f.stem]
    files = [BACKUP / f.name if (BACKUP / f.name).exists() else f for f in files]
    if not files:
        return None
    px = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = (px.dropna(subset=["date", "close"])
            .sort_values("date").drop_duplicates("date", keep="last").set_index("date"))
    flat = (px["volume"] <= 0) & (px["open"] == px["close"]) & (px["high"] == px["low"])
    return px.loc[~flat, ["open", "high", "low", "close", "volume"]]


def main() -> int:
    mode, script, *rest = sys.argv[1:]
    if mode == "old":
        T.load_price = load_price_old
        U.CACHE = OLD_CACHE
        if not OLD_CACHE.exists():
            # 與現行快取相同的期間（2003 起），只是價格換回舊版
            U.get_mask("2003-01-01", "2026-12-31", 250, refresh=True)
        print(f"[ab2011] old 模式：2011 用備份價格，母體快取 {OLD_CACHE}", flush=True)
    else:
        print("[ab2011] new 模式：修正後資料", flush=True)
    # strategy_spec 會 refresh=True 重建快取；兩種模式的快取都已用同一期間（2003 起）
    # 建好，重建只會在別的程序讀檔時改寫它，所以對照時一律不重建
    _get_mask = U.get_mask
    U.get_mask = lambda start, end, n, refresh=False: _get_mask(start, end, n, False)
    sys.argv = [script, *rest]
    mod =sys.modules.get(Path(script).stem)
    if mod is not None:
        # 已被打補丁的模組（如 universe_pit）要直接呼叫，run_path 會另建一份沒打補丁的
        return mod.main() or 0
    runpy.run_path(str(ROOT / script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
