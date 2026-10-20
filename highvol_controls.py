#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高波動因子的兩個對照檢查：等權基準、以及生存者偏誤的量化。

要查什麼
--------
先前測得「高波動/前250/持30檔」年化 +16.28%、超額 +8.03%（對 TWII）、t = 2.11。
但有兩個沒處理的問題，都可能讓超額大幅縮水：

    一、基準比錯了
        策略是 30 檔「等權」，TWII 是「市值加權」。等權本身就帶有小型股效應，
        可能比市值加權賺更多 —— 那不是波動度因子的功勞。
        正確的對照是「同一個選股母體的等權組合」：同樣的股票池、同樣的換倉頻率、
        同樣的成本，唯一差別是選不選高波動。

    二、股票池有生存者偏誤
        選股排名已改成時點正確的成交金額，但股票池本身仍是 383 檔
        「今天市值大的公司」。2003 年流動性好、後來萎縮或下市的公司不在池中，
        而高波動股正是最容易陣亡的一類。
        無法完全修正（需要 point-in-time 上市清單），但可以量化：
        比較「母體等權」與大盤指數的差距 —— 那個差距的一部分就是生存者偏誤。

另外修正一個小的前視
--------------------
原本波動度排名用到換倉日當天的報酬，又在同一天收盤買進，等於邊算邊下單。
改成用前一日為止的資料排名（shift(1)），雖然影響很小，但不該留著。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factor_test as F  # noqa: E402
import universe_pit as U  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "highvol"


def stats(r: pd.Series, br: pd.Series | None = None) -> dict:
    r = r.dropna()
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    d = {"年化": (1 + r).prod() ** (1 / yrs) - 1,
         "夏普": r.mean() / r.std() * np.sqrt(252),
         "最大回撤": float((eq / eq.cummax() - 1).min()), "天數": len(r)}
    if br is not None:
        idx = r.index.intersection(br.index)
        t, p = nw_tstat(r.reindex(idx) - br.reindex(idx))
        byrs = len(idx) / 252
        d["超額"] = d["年化"] - ((1 + br.reindex(idx)).prod() ** (1 / byrs) - 1)
        d["超額t"], d["超額p"] = t, p
    return d


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="高波動因子的對照檢查")
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--uni", type=int, default=250)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"載入 {len(codes)} 檔股價…")
    C = F.load_panel(codes, a.start, a.end)
    mask = U.get_mask(a.start, a.end, a.uni)
    r1 = C.pct_change()

    # 修正前視：排名一律用前一日為止的資料
    vol_now = r1.rolling(60).std()
    vol_lag = vol_now.shift(1)

    tw = pd.read_csv(ROOT / "data" / "market" / "TWII.csv",
                     parse_dates=["date"]).set_index("date")["close"]
    twr = tw.pct_change()

    print("\n回測中…")
    runs = {}
    runs["高波動（原版，排名含當日）"] = F.backtest(vol_now, C, mask, a.top)
    runs["高波動（排名改用前一日）"] = F.backtest(vol_lag, C, mask, a.top)
    # 母體等權：分數給常數，等於在母體中任取 top 檔；改用全母體才是真正的等權基準
    big = 10 ** 6
    runs[f"母體等權（前{a.uni}檔全持）"] = F.backtest(
        pd.DataFrame(1.0, index=C.index, columns=C.columns), C, mask, big)
    runs["低波動（對照）"] = F.backtest(-vol_lag, C, mask, a.top)

    base_tw = twr.reindex(runs["高波動（排名改用前一日）"].index).fillna(0.0)
    ew = runs[f"母體等權（前{a.uni}檔全持）"]

    print(f"\n{'=' * 96}\n一、與兩種基準的比較\n{'=' * 96}")
    print(f"  {'組合':<28}{'年化':>9}{'夏普':>7}{'最大回撤':>10}"
          f"{'超額vs大盤':>11}{'t':>7}{'超額vs等權':>11}{'t':>7}")
    rows = []
    for name, r in runs.items():
        s_tw = stats(r, base_tw)
        s_ew = stats(r, ew)
        rows.append({"組合": name, **s_tw, "超額vs等權": s_ew["超額"], "t_等權": s_ew["超額t"]})
        print(f"  {name:<28}{s_tw['年化']:>+9.2%}{s_tw['夏普']:>7.2f}"
              f"{s_tw['最大回撤']:>+10.2%}{s_tw['超額']:>+11.2%}{s_tw['超額t']:>7.2f}"
              f"{s_ew['超額']:>+11.2%}{s_ew['超額t']:>7.2f}")
    s = stats(base_tw)
    print(f"  {'大盤 TWII（市值加權）':<28}{s['年化']:>+9.2%}{s['夏普']:>7.2f}"
          f"{s['最大回撤']:>+10.2%}")

    print(f"\n{'=' * 96}\n二、生存者偏誤的量化\n{'=' * 96}")
    s_ew_only = stats(ew, base_tw)
    print(f"  母體等權 年化 {s_ew_only['年化']:+.2%}　大盤 {s['年化']:+.2%}"
          f"　差距 {s_ew_only['超額']:+.2%}（t={s_ew_only['超額t']:.2f}）")
    print("  這個差距混合了兩件事：")
    print("    等權效應　　 小型股權重較高，本來就可能贏過市值加權")
    print("    生存者偏誤　 股票池是「今天還在的 383 檔」，當年陣亡的不在裡面")
    print("  兩者無法用現有資料分離，但都不是波動度因子的功勞 ——")
    print("  所以「超額 vs 等權」那一欄，才是波動度因子自己的貢獻。")

    pd.DataFrame(rows).to_csv(OUT / "controls.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出：{OUT / 'controls.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
