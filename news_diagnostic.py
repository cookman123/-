#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新聞到底跟股價有沒有關係？—— 把「沒有增量貢獻」和「完全不相關」分開檢驗。

前面的結論「新聞沒有貢獻」是在**已經知道有大跌**的前提下得到的：
同樣的跌幅門檻，再加上新聞條件並不會讓報酬變好。
但那不等於新聞與股價無關。本程式分三層檢驗：

  第一層　無條件關聯：出現負面新聞的當天，報酬是不是真的比較差？
          （若這一層就沒關係，代表負面詞表根本沒抓到東西）
  第二層　事前 vs 事後：負面新聞當天跌，是「新聞造成的」還是「跌完才報導」？
          看事件前後的累積報酬形狀。
  第三層　增量貢獻：在大跌事件中，有負面新聞的和沒有的，後續反彈有沒有差別？
          這才是前面那個結論的正確範圍。

顯著性一律用「以月份為群集」的 t 值——全市場同時下跌會讓幾百檔同時觸發，
把每檔每天當獨立樣本會嚴重高估顯著性。
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
import davis_tw_train as T  # noqa: E402
from news_reversal import NEGATIVE_WORDS, daily_news_stats, load_stock  # noqa: E402

ROOT = Path(__file__).resolve().parent


def cluster_t(df: pd.DataFrame, col: str, datecol: str = "date") -> tuple:
    """以月份為群集的 t 值，回傳 (平均, t值, 群集數)。"""
    d = df.dropna(subset=[col])
    if len(d) < 30:
        return np.nan, np.nan, 0
    g = d.groupby(pd.to_datetime(d[datecol]).dt.to_period("M"))[col].mean()
    m = g.to_numpy()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float(d[col].mean()), np.nan, len(m)
    return float(d[col].mean()), float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))), len(m)


def main() -> int:
    start, end = "2019-01-01", "2024-12-31"
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    print(f"標的 {len(codes)} 檔　期間 {start} ~ {end}")
    print(f"負面詞表 {len(NEGATIVE_WORDS)} 個（自建，非 NTUSD 等標準詞典）\n")

    rows = []
    for code in codes:
        px, news = load_stock(code, start, end)
        if px is None:
            continue
        ns = daily_news_stats(news, px.index)
        c = px["close"]
        r0 = c.pct_change()                       # 當日報酬
        d = pd.DataFrame({
            "date": px.index, "code": code,
            "r0": r0.to_numpy(),
            "r1": r0.shift(-1).to_numpy(),        # 次日
            "r5": (c.shift(-5) / c - 1).to_numpy(),
            "r20": (c.shift(-20) / c - 1).to_numpy(),
            "pre5": (c / c.shift(5) - 1).to_numpy(),   # 事件前 5 日
            "n": ns["n"].to_numpy(),
            "neg": ns["neg"].to_numpy(),
        })
        base = ns["n"].rolling(60, min_periods=20).median().shift(1)
        d["spike"] = (ns["n"] >= np.maximum(base * 3.0, 2.0)).to_numpy()
        rows.append(d)

    A = pd.concat(rows, ignore_index=True)
    A = A[A["n"].notna()]
    print(f"樣本 {len(A):,} 個「股票-交易日」，其中有新聞的 {int((A['n'] > 0).sum()):,}、"
          f"有負面標題的 {int((A['neg'] > 0).sum()):,}\n")

    # ---------- 第一層：無條件關聯 ----------
    print("=" * 92)
    print("第一層　無條件關聯：負面新聞當天與之後的報酬")
    print("=" * 92)
    groups = {
        "全部交易日": A,
        "當天無新聞": A[A["n"] == 0],
        "有新聞、無負面詞": A[(A["n"] > 0) & (A["neg"] == 0)],
        "有負面詞 1 則": A[A["neg"] == 1],
        "有負面詞 >=2 則": A[A["neg"] >= 2],
        "新聞放量 + 負面": A[A["spike"] & (A["neg"] >= 1)],
    }
    print(f"  {'分組':<18}{'樣本數':>10}" + "".join(f"{c:>20}" for c in
                                                ("當日報酬", "次日", "後5日", "後20日")))
    for name, g in groups.items():
        cells = []
        for col in ("r0", "r1", "r5", "r20"):
            m, t, _ = cluster_t(g, col)
            cells.append(f"{m:+.3%} (t={t:>5.2f})" if pd.notna(t) else f"{m:+.3%}")
        print(f"  {name:<18}{len(g):>10,}" + "".join(f"{x:>20}" for x in cells))

    # ---------- 第二層：事前 vs 事後 ----------
    print("\n" + "=" * 92)
    print("第二層　新聞是「造成下跌」還是「跌完才報導」")
    print("=" * 92)
    ev = A[A["spike"] & (A["neg"] >= 1)]
    m_pre, t_pre, _ = cluster_t(ev, "pre5")
    m_0, t_0, _ = cluster_t(ev, "r0")
    m_5, t_5, _ = cluster_t(ev, "r5")
    print(f"  負面新聞放量日（{len(ev):,} 筆）")
    print(f"    事件前 5 日累積報酬　{m_pre:+.3%}（t={t_pre:.2f}）")
    print(f"    事件當日報酬　　　　{m_0:+.3%}（t={t_0:.2f}）")
    print(f"    事件後 5 日報酬　　　{m_5:+.3%}（t={t_5:.2f}）")
    print("    判讀：若事件前已大跌、當日續跌、事後回升，代表新聞多半是跟著價格走的，")
    print("          資訊已被價格吸收；若事件前平穩、當日才跌，才是新聞驅動。")

    # ---------- 第三層：大跌事件中的增量貢獻 ----------
    print("\n" + "=" * 92)
    print("第三層　大跌事件中，有無負面新聞的後續報酬差異（這才是前面結論的範圍）")
    print("=" * 92)
    print(f"  {'跌幅條件':<16}{'有負面新聞':>26}{'無負面新聞':>26}{'差異':>22}")
    for thr in (-0.05, -0.08, -0.12):
        sub = A[A["pre5"] <= thr]
        with_n = sub[sub["neg"] >= 1]
        without = sub[sub["neg"] == 0]
        if len(with_n) < 30 or len(without) < 30:
            continue
        mw, tw, _ = cluster_t(with_n, "r20")
        mo, to, _ = cluster_t(without, "r20")
        print(f"  5日跌幅<={thr:.0%}   "
              f"{mw:+.2%} (t={tw:>5.2f}, n={len(with_n):>5,})   "
              f"{mo:+.2%} (t={to:>5.2f}, n={len(without):>6,})   "
              f"{mw - mo:+.2%}")
    print("\n  「差異」那一欄才是新聞的增量貢獻。接近 0 代表：知道跌幅之後，")
    print("  再知道有沒有負面新聞，並不會改變對後續報酬的判斷。")

    print("\n注意：負面詞表是自建的關鍵詞比對，不是情緒模型；"
          "且多數個股新聞稀疏（全期間中位數 131 則），")
    print("因此『負面新聞』的測量本身有誤差，本結果是對這個測量方式的檢驗。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
