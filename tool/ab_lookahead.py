#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_test.backtest 前視篩選的前後對照：同一份資料、同一支程式，只切換
factor_test.LOOKAHEAD_FILTER。

用法：python tool/ab_lookahead.py highvol_controls [其他參數…]
      python tool/ab_lookahead.py highvol_diagnose
      python tool/ab_lookahead.py strategy_spec
輸出到 results/data_fix_2011/lookahead/<程式>_{old,new}.log
"""
import contextlib
import importlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import factor_test as F  # noqa: E402
import universe_pit as U  # noqa: E402

# strategy_spec 每次都 refresh=True 重建共用的成交金額快取（而且只建它自己的期間），
# 對照時一律讀現有快取，兩次執行才是同一個母體
_get_mask = U.get_mask
U.get_mask = lambda start, end, n, refresh=False: _get_mask(start, end, n, refresh=False)

OUT = ROOT / "results" / "data_fix_2011" / "lookahead"


def main() -> int:
    name, args = sys.argv[1], sys.argv[2:]
    mod = importlib.import_module(name)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, flag in (("old", True), ("new", False)):
        F.LOOKAHEAD_FILTER = flag
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main(args)
        (OUT / f"{name}_{tag}.log").write_text(buf.getvalue(), encoding="utf-8")
        print(f"{name} {tag} 完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
