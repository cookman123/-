#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合式決策：以「跌深反彈」為主規則，四個面向當作附加判斷條件。

設計理念
--------
不再去找單一萬能訊號。前面的實驗已經一一證明：新聞內容、籌碼、PTT 單獨用都沒有
增量；真正站得住的是「跌深反彈」這條規則（以月份為群集的 t 值 3.9~5.6，扣成本後
每筆仍有 +2% ~ +7.5%）。所以正確的問題不是「哪個訊號能預測」，而是
「在已經決定要買的那些跌深標的裡，哪些條件能幫我篩掉會繼續跌的」。

主規則（使用者實際的做法）
--------------------------
    第 0 日              （負面）新聞或單純起跌，記下當日收盤為基準價
    第 1~watch 日        跌幅自基準達到門檻就準備進場
    次日開盤             進場
    之後                 收盤回到基準價即出場；超過 max_hold 強制出場

四個面向的附加條件（全部只用「決策當下」已知的資料）
----------------------------------------------------
    技術面   KD 低檔、RSI 超賣、以 ATR 標準化後的跌幅、OBV 背離、均線排列
    基本面   本益比／股價淨值比／殖利率相對該檔自身歷史的位置
    籌碼面   下跌期間三大法人（尤其外資、投信）是否買超
    新聞面   是否有負面新聞、屬於哪一類

評估方式
--------
對每個條件，比較「符合」與「不符合」兩組的每筆淨報酬，並以月份為群集計算差異的
t 值。全市場同時下跌會讓幾百檔同時觸發，把每筆交易當獨立樣本會嚴重高估顯著性。

同時報告測了幾個條件——條件測得多，單一顯著就可能只是巧合。
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
import news_reversal as NR  # noqa: E402

ROOT = Path(__file__).resolve().parent
FUND_DIR = ROOT / "data" / "fundamentals"


def load_fundamentals() -> pd.DataFrame:
    """證交所每日本益比／股價淨值比／殖利率；沒抓到就回傳空表（不影響其他面向）。"""
    files = sorted(FUND_DIR.glob("bwibbu_*.csv"))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files],
                  ignore_index=True)
    d["code"] = d["code"].astype(str).str.strip()
    return d[["date", "code", "pe", "pb", "yield_pct"]].dropna(subset=["date", "code"])


def build_trades(codes: list, start: str, end: str, watch: int, drop: float,
                 hold: int, fund: pd.DataFrame) -> pd.DataFrame:
    """產生主規則的交易，並在每筆交易上附掛四個面向的條件值。"""
    out = []
    fund_by_code = {c: g.set_index("date").sort_index()
                    for c, g in fund.groupby("code")} if not fund.empty else {}

    for i, code in enumerate(codes, 1):
        px, news = NR.load_stock(code, start, end)
        if px is None:
            continue
        ns = NR.daily_news_stats(news, px.index)
        tr = NR.find_trades(px, ns, "price", watch, drop, hold)
        if tr.empty:
            continue

        tech = T.step2_price_features(px, quiet=True)
        chips = T.load_chips(code)
        fd = fund_by_code.get(code)

        # 決策當下 = 跌幅成立的那一天（drop_day）收盤後，隔天開盤才進場
        d = tr["drop_day"]
        tr = tr.copy()
        tr["code"] = code

        for col in ("px_k", "px_rsi14", "px_drop_in_atr", "px_obv_dev",
                    "px_ma_align", "px_ma60_dev", "px_bb_z"):
            if col in tech.columns:
                tr[col] = tech[col].reindex(d).to_numpy()

        # 籌碼：下跌期間（基準日到跌幅成立日）三大法人買賣超佔成交量比重
        if chips is not None and not chips.empty:
            vol = px["volume"].replace(0, np.nan)
            for name, col in (("fi", "foreign_investor"), ("it", "investment_trust")):
                if col not in chips.columns:
                    continue
                r = (chips[col].reindex(px.index) / vol).rolling(watch + 1).sum()
                tr[f"chip_{name}"] = r.reindex(d).to_numpy()

        # 基本面：相對該檔自身過去兩年的百分位（只用過去，不含當日之後）
        if fd is not None and len(fd) > 250:
            for col in ("pe", "pb", "yield_pct"):
                s = fd[col].reindex(px.index).ffill(limit=5)
                pct = s.rolling(504, min_periods=120).rank(pct=True)
                tr[f"fund_{col}_pct"] = pct.reindex(d).to_numpy()

        # 新聞面：基準日到跌幅成立日之間有沒有負面新聞
        neg = ns["neg"].rolling(watch + 1).sum()
        tr["news_neg"] = neg.reindex(d).to_numpy()
        tr["news_n"] = ns["n"].rolling(watch + 1).sum().reindex(d).to_numpy()

        out.append(tr)
        if i % 80 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cluster_diff(tr: pd.DataFrame, mask: pd.Series) -> tuple:
    """以月份為群集，比較 mask 為真與為假兩組每筆淨報酬的差異。"""
    d = tr.assign(_m=pd.to_datetime(tr["entry"]).dt.to_period("M"), _g=mask)
    per = d.groupby("_m").apply(
        lambda g: (g.loc[g["_g"], "net"].mean() - g.loc[~g["_g"], "net"].mean())
        if g["_g"].any() and (~g["_g"]).any() else np.nan,
        include_groups=False).dropna()
    if len(per) < 5:
        return np.nan, np.nan, len(per)
    sd = per.std(ddof=1)
    t = float(per.mean() / (sd / np.sqrt(len(per)))) if sd > 0 else np.nan
    return float(per.mean()), t, len(per)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="整合式決策：主規則 + 四面向附加條件")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--watch", type=int, default=5)
    ap.add_argument("--drop", type=float, default=0.08)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=0)
    ap.add_argument("--out", default="results/integrated")
    args = ap.parse_args(argv)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    if args.top_n:
        import json
        mp = json.load(open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))
        rank = {r["code"]: r.get("market_cap") or 0 for r in mp}
        codes = sorted(codes, key=lambda c: -rank.get(c, 0))[:args.top_n]

    fund = load_fundamentals()
    print(f"主規則：盯 {args.watch} 天、跌 {args.drop:.0%}、最長持有 {args.hold} 日")
    print(f"標的 {len(codes)} 檔　期間 {args.start} ~ {args.end}")
    print(f"基本面資料：{'已載入 ' + str(len(fund)) + ' 列' if not fund.empty else '尚未抓取（本次略過基本面條件）'}\n")

    tr = build_trades(codes, args.start, args.end, args.watch, args.drop,
                      args.hold, fund)
    if tr.empty:
        print("沒有產生交易")
        return 1
    tr.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

    net = tr["net"].to_numpy()
    bm, bt, bn = cluster_diff(tr.assign(_all=True), pd.Series(True, index=tr.index))
    g = tr.groupby(pd.to_datetime(tr["entry"]).dt.to_period("M"))["net"].mean()
    base_t = float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))
    print("=" * 96)
    print(f"主規則基準：{len(tr):,} 筆　勝率 {(net > 0).mean():.1%}　"
          f"平均淨報酬 {net.mean():+.2%}　群集 t {base_t:.2f}（{len(g)} 個月）")
    print("=" * 96)

    # 每個附加條件：定義、以及「符合」的判斷
    conds = [
        ("技術面", "KD 的 K 值 < 0.2（低檔）", lambda d: d.get("px_k") < 0.2),
        ("技術面", "RSI14 < 0.3（超賣）", lambda d: d.get("px_rsi14") < 0.3),
        ("技術面", "跌幅 > 2 倍 ATR（相對自身波動算大跌）",
         lambda d: d.get("px_drop_in_atr") < -2),
        ("技術面", "OBV 未同步走弱（賣壓衰竭）", lambda d: d.get("px_obv_dev") > 0),
        ("技術面", "均線仍多頭排列", lambda d: d.get("px_ma_align") > 0),
        ("技術面", "仍在 60 日均線之上", lambda d: d.get("px_ma60_dev") > 0),
        ("籌碼面", "外資於跌勢中買超", lambda d: d.get("chip_fi") > 0),
        ("籌碼面", "投信於跌勢中買超", lambda d: d.get("chip_it") > 0),
        ("基本面", "本益比在自身過去 2 年後 30%（相對便宜）",
         lambda d: d.get("fund_pe_pct") < 0.3),
        ("基本面", "股價淨值比在自身過去 2 年後 30%",
         lambda d: d.get("fund_pb_pct") < 0.3),
        ("基本面", "殖利率在自身過去 2 年前 30%（股價跌出殖利率）",
         lambda d: d.get("fund_yield_pct_pct") > 0.7),
        ("新聞面", "跌勢中有負面新聞", lambda d: d.get("news_neg") >= 1),
        ("新聞面", "跌勢中完全沒有新聞", lambda d: d.get("news_n") == 0),
    ]

    print(f"\n  {'面向':<6}{'條件':<36}{'符合筆數':>9}{'符合報酬':>11}"
          f"{'不符報酬':>11}{'差異':>10}{'群集t':>8}")
    tested = 0
    rows = []
    for pillar, name, fn in conds:
        try:
            mask = fn(tr)
        except Exception:
            continue
        if mask is None or not hasattr(mask, "fillna"):
            continue
        mask = mask.fillna(False).astype(bool)
        n_yes = int(mask.sum())
        if n_yes < 50 or n_yes > len(tr) - 50:
            print(f"  {pillar:<6}{name:<36}{n_yes:>9,}　樣本不足或幾乎全中，略過")
            continue
        tested += 1
        my = float(tr.loc[mask, "net"].mean())
        mn = float(tr.loc[~mask, "net"].mean())
        diff, t, nclu = cluster_diff(tr, mask)
        rows.append({"面向": pillar, "條件": name, "符合筆數": n_yes,
                     "符合報酬": my, "不符報酬": mn, "差異": diff,
                     "群集t": t, "群集數": nclu})
        print(f"  {pillar:<6}{name:<36}{n_yes:>9,}{my:>+11.2%}{mn:>+11.2%}"
              f"{diff:>+10.2%}{t:>8.2f}")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "conditions.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n  共檢定 {tested} 個條件。在 5% 水準下，純機率期望約 "
          f"{tested * 0.05:.1f} 個會顯著，因此單一 t>2 不足以採信；")
    print("  要看的是同一面向內多個條件是否一致同向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
