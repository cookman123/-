#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加回現金股利後，季頻波動度溢酬還剩多少。

為什麼要測
----------
Yahoo 股價只對配股、減資做分割調整，**現金股利從頭到尾沒算**。如果所有股票
的殖利率差不多，漏掉股利只是把每個組合等量往下平移，不影響比較。但高波動股
通常是成長股或虧損股，配息偏少；低波動、大型股配息偏多。所以加回股利後，
母體等權報酬會上升得比高波動組合多，**波動度溢酬（相對母體的超額）可能縮小**。
這是論文審查一定會問的點，要量化而不是揭露了事。

做法
----
    股利殖利率 = 每股現金股利 ÷ 除息前收盤價（兩者都是證交所原始價，與 Yahoo
    的調整基準無關，所以可以直接用在 Yahoo 價格上）
    含息價 = 開盤價 × 累積(1 + 殖利率)，在除息日把股利再投入
    其餘規則與 quarterly_verify.py 完全相同（同一個 book() 函式、同一套季末、
    同一個母體與成本），只換掉計算季報酬用的價格。

股利資料來自證交所 TWT49U（corp_actions.py），只涵蓋上市股；上櫃股沒有
股利資料，含息與不含息相同（會在輸出中列出佔比）。
大盤基準仍是價格指數，沒有對應的報酬指數，所以這裡只對母體等權（同樣含息）
比較超額，不對大盤。
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
sys.path.insert(0, str(ROOT))
import feature_engineering as FE  # noqa: E402
from portfolio_backtest import nw_tstat  # noqa: E402
from quarterly_factors import forward_returns, quarter_ends  # noqa: E402
from quarterly_verify import SIG, STORE, ann, book, sharpe  # noqa: E402

EXR = ROOT / "data" / "corp_actions" / "exright.csv"
OUT = ROOT / "results" / "quarterly"


def dividend_factor(idx: pd.DatetimeIndex, cols: pd.Index) -> tuple:
    """(累積含息因子面板, 殖利率長表)。因子在除息日（含）起乘上 1 + 殖利率。"""
    e = pd.read_csv(EXR, dtype={"code": str}, parse_dates=["date"])
    e = e[(e["cash"] > 0) & (e["prev_close"] > 0) & e["code"].isin(cols)].copy()
    e["y"] = e["cash"] / e["prev_close"]
    # 除息日若不是面板的交易日（例如停牌），挪到之後第一個交易日
    pos = idx.searchsorted(e["date"].to_numpy())
    e = e[pos < len(idx)].copy()
    e["d"] = idx[idx.searchsorted(e["date"].to_numpy())]
    g = np.log1p(e.groupby(["d", "code"])["y"].sum()).unstack()
    g = g.reindex(index=idx, columns=cols).fillna(0.0)
    return np.exp(g.cumsum()), e


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="現金股利對波動度溢酬的影響")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if not EXR.exists():
        print(f"找不到 {EXR}，請先跑 corp_actions.py"); return 1

    df = pd.read_pickle(STORE.with_name(f"panel{a.tag}.pkl"))
    P = FE.load_panels("2010-01-01", "2026-12-31", quiet=True)
    O = P["open"]
    fac, ev = dividend_factor(O.index, O.columns)
    O_tr = O * fac

    qe = [d for d in quarter_ends(O.index) if pd.Timestamp(d) in df.index.get_level_values("date")]
    dates = df.index.get_level_values("date")
    res = {}
    for name, px in (("不含息", O), ("含息", O_tr)):
        fwd = forward_returns(px, qe)
        qd = [d for d in fwd.index if d in dates]
        sub = df.loc[dates.isin(qd)]
        ew = pd.Series({d: float(fwd.loc[d].reindex(sub.loc[d].index).dropna().mean())
                        for d in qd}).sort_index()
        res[name] = {"ew": ew,
                     "十分位多單": book(sub, fwd, qd, SIG, 10, "多單")["r"],
                     "固定30檔": book(sub, fwd, qd, SIG, 10, "多單", n_fixed=30)["r"],
                     "十分位多空": book(sub, fwd, qd, SIG, 10, "多空")["r"]}

    # 殖利率按波動十分位：直接看「高波動股是不是配息少」
    last = df[SIG].dropna()
    yr = ev.assign(year=ev["d"].dt.year).groupby(["year", "code"])["y"].sum()
    rows = []
    for y in range(2010, 2026):
        d = [x for x in qe if pd.Timestamp(x).year == y - 1 and pd.Timestamp(x).month == 12]
        if not d or d[0] not in last.index.get_level_values("date"):
            continue
        s = last.xs(d[0], level="date")
        dec = pd.qcut(s.rank(method="first"), 10, labels=False)
        dy = yr.xs(y, level="year").reindex(s.index).fillna(0.0) if y in yr.index.get_level_values(0) \
            else pd.Series(0.0, index=s.index)
        rows.append(dy.groupby(dec).mean())
    by_dec = pd.DataFrame(rows).mean()

    n_codes = O.columns.size
    has = ev["code"].nunique()
    print(f"股利事件 {len(ev):,} 筆，涵蓋 {has}/{n_codes} 檔（其餘多為上櫃股，無資料）")
    print(f"\n年度現金殖利率，按前一年底 60 日波動度十分位（0＝最低波動，9＝最高）")
    print("  " + "  ".join(f"{int(k)}:{v:.2%}" for k, v in by_dec.items()))

    print(f"\n{'=' * 90}\n季頻波動度溢酬：不含息 vs 含息（{SIG}）\n{'=' * 90}")
    print(f"  {'組合':<14}{'不含息年化':>12}{'含息年化':>12}{'差':>9}"
          f"{'超額(不含息)':>14}{'超額(含息)':>12}{'超額t(含息)':>12}")
    out = []
    for k in ("十分位多單", "固定30檔", "十分位多空"):
        r0, r1 = res["不含息"][k], res["含息"][k]
        base0, base1 = res["不含息"]["ew"], res["含息"]["ew"]
        if k == "十分位多空":
            x0, x1 = r0, r1
        else:
            x0 = r0 - base0.reindex(r0.index)
            x1 = r1 - base1.reindex(r1.index)
        t1, _ = nw_tstat(x1)
        out.append({"組合": k, "不含息年化": ann(r0), "含息年化": ann(r1),
                    "超額_不含息": ann(r0) - ann(base0.reindex(r0.index)) if k != "十分位多空" else ann(r0),
                    "超額_含息": ann(r1) - ann(base1.reindex(r1.index)) if k != "十分位多空" else ann(r1),
                    "超額t_含息": t1, "夏普_含息": sharpe(r1)})
        o = out[-1]
        print(f"  {k:<14}{o['不含息年化']:>+12.2%}{o['含息年化']:>+12.2%}"
              f"{o['含息年化'] - o['不含息年化']:>+9.2%}{o['超額_不含息']:>+14.2%}"
              f"{o['超額_含息']:>+12.2%}{t1:>12.2f}")
    e0, e1 = res["不含息"]["ew"], res["含息"]["ew"]
    print(f"  {'母體等權':<14}{ann(e0):>+12.2%}{ann(e1):>+12.2%}{ann(e1) - ann(e0):>+9.2%}")
    print("  （多空的「超額」就是本身年化）")

    pd.DataFrame(out).to_csv(OUT / f"dividend{a.tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出：{OUT / f'dividend{a.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
