#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""長期版（2011-2024）：規則式跌深反彈，時點正確的選股範圍。"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
import pandas as pd, numpy as np
import adaptive_strategy as A, universe_pit as U, portfolio_sim as PS
from portfolio_backtest import nw_tstat

ROOT = Path(__file__).resolve().parent
START, END, TOPN = "2011-01-01", "2024-12-31", 120
OUT = ROOT / "results" / "long_rule"; OUT.mkdir(parents=True, exist_ok=True)

codes = sorted(x.name for x in (ROOT/"data"/"stocks").iterdir()
               if x.is_dir() and any((x/"price").glob("*.csv")))
print(f"全部 {len(codes)} 檔進入掃描，之後以「當日成交金額前 {TOPN} 名」逐日篩選")

parts = A.build(codes, START, END, 5, 0.08, 20, news_size=True, scale_out=True,
                early_trim=True, wait_rebound=True, rebound_mode="relative")
print(f"原始批次列 {len(parts):,}")
mask = U.get_mask(START, END, TOPN)
parts = U.filter_trades(parts, mask, "drop_day")
parts.to_csv(OUT/"parts.csv", index=False, encoding="utf-8-sig")
cam = A.to_campaigns(parts)
print(f"時點篩選後：批次列 {len(parts):,}、操作回合 {len(cam):,}")

m, t, n = A.cluster_t(cam)
print(f"\n全期間：{len(cam):,} 回合　勝率 {(cam.net>0).mean():.1%}　"
      f"平均淨報酬 {m:+.2%}　群集 t {t:.2f}（{n} 個月）")

cam["yr"] = pd.to_datetime(cam.entry).dt.year
print(f"\n{'年':<6}{'回合':>6}{'勝率':>8}{'淨報酬':>10}{'加碼比例':>10}")
for y, g in cam.groupby("yr"):
    print(f"{y:<6}{len(g):>6}{(g.net>0).mean():>8.1%}{g.net.mean():>+10.2%}"
          f"{(g.size_mult>1).mean():>10.1%}")

ex = cam[~cam.yr.isin([2020, 2021])]
m2, t2, n2 = A.cluster_t(ex)
print(f"\n排除 2020-2021：{len(ex):,} 回合　勝率 {(ex.net>0).mean():.1%}　"
      f"平均淨報酬 {m2:+.2%}　群集 t {t2:.2f}（{n2} 個月）")

b = pd.read_csv(ROOT/"data"/"market"/"0050.csv", parse_dates=["date"]).set_index("date").sort_index()
b = b[(b.index >= parts.entry.min()) & (b.index <= parts.exit.max())]
cal, idle = b.index, b.close.pct_change().fillna(0.0)
byrs = len(cal)/252; bcagr = (1+idle).prod()**(1/byrs)-1
print(f"\n0050 買進持有（同期）：年化 {bcagr:+.2%}　夏普 {idle.mean()/idle.std()*np.sqrt(252):.2f}"
      f"　最大回撤 {((1+idle).cumprod()/(1+idle).cumprod().cummax()-1).min():+.2%}")
print(f"\n{'格數':>5}{'投入率':>9}{'年化':>10}{'超額':>10}{'夏普':>7}{'最大回撤':>10}{'超額t':>8}{'p':>7}")
for k in (5, 10, 20):
    r = PS.simulate(parts, k, cal, idle)
    c = r["_curve"]; d = c.pct_change().dropna()
    exc = (d - idle.reindex(d.index)).dropna(); tt, pp = nw_tstat(exc)
    print(f"{k:>5}{r['平均資金投入率']:>9.1%}{r['年化']:>+10.2%}{r['年化']-bcagr:>+10.2%}"
          f"{d.mean()/d.std()*np.sqrt(252):>7.2f}{((c/c.cummax())-1).min():>+10.2%}{tt:>8.2f}{pp:>7.3f}")
    c.to_csv(OUT/f"equity_k{k}.csv", encoding="utf-8-sig")
print(f"\n輸出：{OUT}")
