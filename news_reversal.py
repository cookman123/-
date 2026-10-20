#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利空過度反應／均值回歸策略：復刻「看到壞消息 → 預期接下來幾天會跌 → 跌到一定
程度後買入 → 放一週或更久 → 等價格回到跌前水準或更高才出貨」。

與前面所有實驗的差別
--------------------
前面是拿新聞「內容」去預測報酬（TF-IDF → 主題 → 回歸），全部失敗。
這裡把新聞當成「事件發生了」的訊號，賭的是價格對利空反應過度、之後回歸
（De Bondt & Thaler 的 overreaction hypothesis）。用法完全不同，
所以前面的負面結果不構成對這個策略的否證。

事件順序（忠於實際做法：新聞在前，跌幅在後）
--------------------------------------------
  第 0 日              出現（負面）新聞，記下當日收盤為基準價 ref
  第 1~watch_days 日   盯著看，收盤價自 ref 的跌幅一旦達到 drop_pct 就準備進場
  次日開盤             進場（訊號收盤後才成立，只能用次日開盤價）
  之後                 收盤價回到 ref 以上即觸發，次日開盤出場；
                       超過 max_hold 個交易日仍未回到 ref，強制出場
  同一檔同時只持有一個部位

三種事件定義（用來分離「到底是跌幅有用還是新聞有用」）
------------------------------------------------------
  price    ：不看新聞，每個交易日都當第 0 日（對照組）
  news     ：當日新聞則數放量（>= 3 倍於近 60 日中位數，且至少 2 則）
  negative ：news 的條件 + 標題含負面關鍵詞
若 negative 沒有明顯優於 price，就代表新聞沒有貢獻，是「跌深就反彈」在做事。

必須揭露的偏誤
--------------
標的清單是「當期市值排名」的快照，不是 point-in-time。跌下去沒再回來、
甚至已下市的股票不在樣本裡，而這正是本策略最容易受害的方向。
因此本回測的報酬應視為**上界**，不是可實現值。
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

ROOT = Path(__file__).resolve().parent

# 自建財經負面詞表（沒有 NTUSD 等現成詞典可用）。只比對標題，標題比內文乾淨。
NEGATIVE_WORDS = [
    "下修", "下調", "下滑", "衰退", "虧損", "轉虧", "減資", "跌停", "重挫",
    "暴跌", "大跌", "利空", "警訊", "示警", "疲弱", "疲軟", "不如預期",
    "低於預期", "砍單", "減單", "遞延", "延後", "停工", "停產", "斷鏈",
    "召回", "瑕疵", "起火", "爆炸", "意外", "裁罰", "訴訟", "求償", "違約",
    "跳票", "掏空", "舞弊", "檢調", "搜索", "起訴", "調查", "裁員", "解僱",
    "關廠", "撤資", "出清", "賣壓", "賣超", "調降", "降評", "看壞",
    "縮減", "萎縮", "流失", "衝擊", "認列損失", "減值", "呆帳", "違約金",
]

# 交易標的的成本結構差很多：同一個訊號拿去做個股、ETF、期貨，來回成本差 30~60 倍
INSTRUMENTS = {
    "stock": (0.001425 * 0.3, 0.001425 * 0.3 + 0.003),    # 現股，證交稅 0.3%
    "etf": (0.001425 * 0.3, 0.001425 * 0.3 + 0.001),      # 股票型 ETF（0050），稅 0.1%
    "futures": (0.00003, 0.00003 + 0.00002),              # 台指期，期交稅 0.002%
}
COST_BUY, COST_SELL = INSTRUMENTS["stock"]


def load_index(start: str, end: str):
    """
    大盤（發行量加權股價指數）。

    做大盤而非個股有三個好處，對這個策略特別重要：
      1. 沒有生存者偏誤——指數不會下市，而「買跌等反彈」最怕的就是這個
      2. 樣本長——TWII 自 2010 年就有，15 年；個股新聞只有 2019 年起
      3. 成本低——0050 證交稅 0.1%、台指期約 0.01% 來回，個股是 0.3%
    指數為價格指數（不含股利），實際操作 0050 還會多領約 3~4% 的股利。
    """
    f = ROOT / "data" / "market" / "TWII.csv"
    px = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    return px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]


def market_news_stats(start: str, end: str, calendar) -> pd.DataFrame:
    """
    全市場的新聞強度：把所有個股的新聞按交易日加總。

    大盤層級的「利空」沒有現成資料，用「全市場新聞則數放量 + 負面標題則數」
    當代理變數。收盤後發布者一樣順延到下一個交易日。
    """
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and (x / "news").exists())
    total = pd.DataFrame(0.0, index=calendar, columns=["n", "neg"])
    for code in codes:
        news = T.load_news(code)
        if news is None or len(news) == 0:
            continue
        n = news.copy()
        n["day"] = T.TextFeatures._to_trading_day(n["ts"], calendar)
        n = n.dropna(subset=["day"])
        if n.empty:
            continue
        total = total.add(daily_news_stats(n, calendar), fill_value=0.0)
    return total


def load_stock(code: str, start: str, end: str):
    px = T.load_price(code)
    if px is None or px.empty:
        return None, None
    px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
    if len(px) < 250:
        return None, None
    news = T.load_news(code)
    if news is None or len(news) == 0:
        return px, None
    n = news.copy()
    n["day"] = T.TextFeatures._to_trading_day(n["ts"], px.index)
    return px, n.dropna(subset=["day"])


def daily_news_stats(news, calendar) -> pd.DataFrame:
    """每個交易日的新聞則數與負面標題則數（收盤後發布者已順延至次日）。"""
    out = pd.DataFrame(index=calendar)
    if news is None or news.empty:
        out["n"] = 0.0
        out["neg"] = 0.0
        return out
    out["n"] = news.groupby("day").size().reindex(calendar).fillna(0.0)
    title = news["title"].fillna("")
    is_neg = title.str.contains("|".join(NEGATIVE_WORDS), case=False, regex=True)
    out["neg"] = (news.assign(_neg=is_neg.astype(float))
                      .groupby("day")["_neg"].sum().reindex(calendar).fillna(0.0))
    return out


def _start_days(px, ns, mode: str, news_mult: float):
    if mode == "price":
        return range(len(px.index))
    base = ns["n"].rolling(60, min_periods=20).median().shift(1)
    # 平常就沒什麼新聞的股票，門檻至少 2 則，否則一則就算「放量」
    cond = ns["n"] >= np.maximum(base * news_mult, 2.0)
    if mode == "negative":
        cond = cond & (ns["neg"] >= 1)
    return np.flatnonzero(cond.fillna(False).to_numpy())


def find_trades(px, ns, mode: str, watch_days: int, drop_pct: float,
                max_hold: int, news_mult: float = 3.0) -> pd.DataFrame:
    """依上述事件順序產生交易明細。"""
    idx = px.index
    o = px["open"].to_numpy(float)
    c = px["close"].to_numpy(float)
    n = len(idx)

    trades, busy_until = [], -1
    for i0 in _start_days(px, ns, mode, news_mult):
        i0 = int(i0)
        if i0 <= busy_until:
            continue
        ref = c[i0]
        if not np.isfinite(ref) or ref <= 0:
            continue

        hit = None
        for k in range(1, watch_days + 1):
            i = i0 + k
            if i >= n - 1:
                break
            if np.isfinite(c[i]) and c[i] / ref - 1 <= -abs(drop_pct):
                hit = i
                break
        if hit is None:
            continue

        entry_i = hit + 1
        if entry_i >= n:
            continue
        entry = o[entry_i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        exit_i, reason = None, "時間到"
        for j in range(entry_i, min(entry_i + max_hold, n - 1)):
            if np.isfinite(c[j]) and c[j] >= ref:
                exit_i, reason = j + 1, "回到基準"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold, n - 1)
        px_out = o[exit_i]
        if not np.isfinite(px_out) or px_out <= 0:
            continue

        trades.append({
            "news_day": idx[i0], "drop_day": idx[hit], "entry": idx[entry_i],
            "exit": idx[exit_i], "wait": hit - i0, "hold": exit_i - entry_i,
            "drop": c[hit] / ref - 1,
            "gross": px_out / entry - 1,
            "net": (px_out * (1 - COST_SELL)) / (entry * (1 + COST_BUY)) - 1,
            "reason": reason,
        })
        busy_until = exit_i
    return pd.DataFrame(trades)


def clustered_t(trades: pd.DataFrame) -> tuple:
    """
    以「進場月份」為群集計算 t 值，取代把每筆交易當獨立樣本的做法。

    為什麼非改不可：全面性下跌（例如 2020 年 3 月）會在幾百檔股票同時觸發訊號，
    這些交易共用同一個市場衝擊，完全不是獨立樣本。直接對 1,977 筆做 t 檢定，
    分母的 sqrt(n) 會把標準誤壓到不合理的小，t 值因此嚴重灌水。
    改成先算每個月的平均報酬，再對「月平均」做檢定，有效樣本就變成
    真正獨立的事件期數，而不是交易筆數。

    回傳 (群集 t 值, 群集數, 最大單月交易佔比)。最後一項用來看集中度——
    若過半交易擠在同一個月，就算群集 t 值好看也只是押注單一事件。
    """
    if trades.empty:
        return np.nan, 0, np.nan
    g = trades.groupby(pd.to_datetime(trades["entry"]).dt.to_period("M"))["net"]
    means = g.mean().to_numpy()
    sizes = g.size().to_numpy()
    n = len(means)
    if n < 3:
        return np.nan, n, float(sizes.max() / sizes.sum()) if len(sizes) else np.nan
    sd = means.std(ddof=1)
    t = float(means.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan
    return t, n, float(sizes.max() / sizes.sum())


def equity_metrics(trades: pd.DataFrame, px: pd.DataFrame) -> dict:
    """
    單一標的的資金曲線指標。

    為什麼非算不可：這個策略大部分時間空手，而大盤長期是上漲的。
    勝率 70% 看起來很好，但若整段期間只進場 20 次、每次賺 1.8%，
    總報酬可能遠輸給「一直持有」。要比的是總報酬與曝險，不是勝率。
    """
    idx = px.index
    c = px["close"]
    days = len(idx)
    if days < 2 or trades.empty:
        return {}
    yrs = days / 252
    eq = float(np.prod(1 + trades["net"].to_numpy()))
    in_mkt = int(trades["hold"].sum())
    bh = float(c.iloc[-1] / c.iloc[0])
    return {
        "策略總報酬": eq - 1,
        "策略年化": eq ** (1 / yrs) - 1,
        "買進持有總報酬": bh - 1,
        "買進持有年化": bh ** (1 / yrs) - 1,
        "曝險比例": in_mkt / days,
        # 只看「有部位的那段時間」的報酬率，排除空手期間的稀釋
        "投入期間年化": (eq ** (252 / max(in_mkt, 1)) - 1) if eq > 0 else np.nan,
    }


def event_study(px, day_idx, pre: int = 5, post: int = 30) -> np.ndarray:
    """以新聞日收盤為 0，前後的平均累積報酬——看有沒有「先跌過頭再回來」。"""
    c = px["close"].to_numpy(float)
    rows = []
    for i in day_idx:
        i = int(i)
        if i - pre < 0 or i + post >= len(c) or not np.isfinite(c[i]) or c[i] <= 0:
            continue
        rows.append(c[i - pre:i + post + 1] / c[i] - 1)
    return np.array(rows) if rows else np.zeros((0, pre + post + 1))


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="利空過度反應／均值回歸策略")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--watch-days", type=int, nargs="+", default=[3, 5],
                    help="新聞後盯幾天等跌幅出現")
    # 大盤的跌幅遠小於個股：指數跌 5% 已是重大事件，個股 5% 只是一般波動
    ap.add_argument("--drop-pct", type=float, nargs="+", default=None,
                    help="預設：大盤 2/3/5/8%%、個股 5/8/12%%")
    ap.add_argument("--max-hold", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--modes", nargs="+", default=["price", "news", "negative"])
    ap.add_argument("--indices", nargs="+", default=["TWII"],
                    help="指數模式要跑哪些標的（data/market/ 下的檔名），"
                         "或用 all 跑全部")
    ap.add_argument("--top-n", type=int, default=0,
                    help="只取市值前 N 大（用來檢驗生存者偏誤：大型股幾乎不會下市，"
                         "若效果在大型股仍成立，偏誤就無法解釋這個結果）")
    ap.add_argument("--target", choices=["index", "stock"], default="index",
                    help="index=大盤（預設，無生存者偏誤）、stock=個股")
    ap.add_argument("--instrument", choices=list(INSTRUMENTS), default=None,
                    help="成本結構；預設大盤用 etf、個股用 stock")
    ap.add_argument("--out", default="results/reversal")
    args = ap.parse_args(argv)

    global COST_BUY, COST_SELL
    inst = args.instrument or ("etf" if args.target == "index" else "stock")
    COST_BUY, COST_SELL = INSTRUMENTS[inst]
    if args.drop_pct is None:
        args.drop_pct = ([0.02, 0.03, 0.05, 0.08] if args.target == "index"
                         else [0.05, 0.08, 0.12])
    # 大盤只有一個標的，事件數天生就少；個股有幾百檔，可以要求多一點
    min_trades = 15 if args.target == "index" else 50

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"標的類型 {args.target}　交易工具 {inst}　"
          f"來回成本 {COST_BUY + COST_SELL:.4%}")
    print(f"期間 {args.start} ~ {args.end}　負面詞表 {len(NEGATIVE_WORDS)} 個")

    if args.target == "index" and args.indices != ["TWII"]:
        # 多標的指數模式：每個 ETF／指數各自跑，最後合併統計。
        # 指數型標的不會下市，所以加再多標的都不會引入生存者偏誤，
        # 這是在不犧牲乾淨度的前提下把樣本數撐起來的唯一辦法。
        names = args.indices
        if names == ["all"]:
            names = sorted(f.stem for f in (ROOT / "data" / "market").glob("*.csv"))
        data = {}
        for nm in names:
            f = ROOT / "data" / "market" / f"{nm}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
            d = d[(d.index >= pd.Timestamp(args.start)) & (d.index <= pd.Timestamp(args.end))]
            d = d.dropna(subset=["close"])
            if len(d) < 250:
                continue
            data[nm] = (d, pd.DataFrame(0.0, index=d.index, columns=["n", "neg"]))
        if not data:
            print("找不到任何指數資料")
            return 1
        print(f"指數型標的 {len(data)} 個：{'、'.join(data)}")
        for nm, (d, _) in sorted(data.items()):
            print(f"  {nm:<8}{len(d):>6} 筆　{d.index.min():%Y-%m-%d} ~ {d.index.max():%Y-%m-%d}")
        args.modes = ["price"]          # 海外指數沒有對應的台股新聞，只跑價格事件
        min_trades = 15
    elif args.target == "index":
        px = load_index(args.start, args.end)
        if px.empty:
            print("找不到大盤資料")
            return 1
        need_news = any(m != "price" for m in args.modes)
        ns = (market_news_stats(args.start, args.end, px.index) if need_news
              else pd.DataFrame(0.0, index=px.index, columns=["n", "neg"]))
        data = {"TWII": (px, ns)}
        print(f"大盤 {len(px)} 個交易日　"
              f"{px.index.min():%Y-%m-%d} ~ {px.index.max():%Y-%m-%d}")
        if need_news:
            has = ns[ns["n"] > 0]
            print(f"全市場新聞 {int(ns['n'].sum()):,} 則、負面標題 "
                  f"{int(ns['neg'].sum()):,} 則；有新聞的交易日 {len(has)}／{len(ns)}"
                  f"（新聞資料自 {has.index.min():%Y-%m-%d} 起）")
    else:
        codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                       if x.is_dir() and any((x / "price").glob("*.csv")))
        if args.top_n:
            import json
            mp = json.load(open(ROOT / "data" / "industry_stock_map.json",
                                encoding="utf-8"))
            rank = {r["code"]: r.get("market_cap") or 0 for r in mp}
            codes = [c for c in sorted(codes, key=lambda c: -rank.get(c, 0))
                     ][:args.top_n]
            print(f"限定市值前 {args.top_n} 大（檢驗生存者偏誤）")
        data = {}
        for code in codes:
            p, news = load_stock(code, args.start, args.end)
            if p is None:
                continue
            data[code] = (p, daily_news_stats(news, p.index))
        tot_news = sum(int(x["n"].sum()) for _, x in data.values())
        tot_neg = sum(int(x["neg"].sum()) for _, x in data.values())
        print(f"可用 {len(data)} 檔　新聞 {tot_news:,} 則，"
              f"標題含負面詞 {tot_neg:,} 則（{tot_neg / max(tot_news, 1):.1%}）")
    print()

    rows = []
    for mode in args.modes:
        for W in args.watch_days:
            for X in args.drop_pct:
                for H in args.max_hold:
                    parts = [find_trades(px, ns, mode, W, X, H)
                             for px, ns in data.values()]
                    parts = [p for p in parts if len(p)]
                    if not parts:
                        continue
                    tr = pd.concat(parts, ignore_index=True)
                    if len(tr) < min_trades:
                        continue
                    net = tr["net"].to_numpy()
                    sd = net.std(ddof=1)
                    row = {
                        "事件": mode, "盯盤天數": W, "跌幅門檻": X, "最長持有": H,
                        "交易數": len(tr), "勝率": float((net > 0).mean()),
                        "平均等待日": float(tr["wait"].mean()),
                        "平均持有日": float(tr["hold"].mean()),
                        "平均跌幅": float(tr["drop"].mean()),
                        "毛報酬": float(tr["gross"].mean()),
                        "淨報酬": float(net.mean()),
                        "t值": float(net.mean() / (sd / np.sqrt(len(net)))) if sd > 0 else np.nan,
                        "回到基準比例": float((tr["reason"] == "回到基準").mean()),
                    }
                    ct, nclu, conc = clustered_t(tr)
                    row["群集t值"] = ct
                    row["群集數"] = nclu
                    row["最大單月佔比"] = conc
                    if len(data) == 1:          # 單一標的（大盤）才有意義的資金曲線
                        px0 = next(iter(data.values()))[0]
                        row.update(equity_metrics(tr, px0))
                    else:
                        # 多標的：報「資金投入期間」的年化，避免用勝率誤導。
                        # 這不是組合總報酬——要談組合還得處理同時持倉數與部位配置。
                        h = max(float(tr["hold"].mean()), 1.0)
                        row["投入期間年化"] = (1 + net.mean()) ** (252 / h) - 1
                    rows.append(row)
                    tr.to_csv(out_dir / f"trades_{mode}_W{W}_X{int(X * 100)}_H{H}.csv",
                              index=False, encoding="utf-8-sig")
        print(f"  {mode} 完成")

    if not rows:
        print("沒有產生任何交易")
        return 1
    res = pd.DataFrame(rows).sort_values("淨報酬", ascending=False)
    res.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 108)
    print("每筆交易的平均報酬（已扣來回成本）　前 20 名")
    print("=" * 108)
    show = ["事件", "盯盤天數", "跌幅門檻", "最長持有", "交易數", "勝率",
            "平均持有日", "淨報酬", "t值", "群集t值", "群集數", "最大單月佔比"]
    for extra in ("策略總報酬", "買進持有總報酬", "曝險比例", "投入期間年化"):
        if extra in res.columns:
            show.append(extra)
    show = [c for c in show if c in res.columns]
    v = res.head(20)[show].copy()
    for c in ("勝率", "曝險比例", "最大單月佔比"):
        if c in v:
            v[c] = v[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    for c in ("淨報酬", "策略總報酬", "買進持有總報酬", "投入期間年化"):
        if c in v:
            v[c] = v[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
    for c in ("t值", "群集t值"):
        if c in v:
            v[c] = v[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    v["平均持有日"] = v["平均持有日"].map(lambda x: f"{x:.1f}")
    print(v.to_string(index=False))
    if "策略總報酬" in res.columns:
        print("\n  注意：勝率高不等於贏過買進持有——策略大部分時間空手，"
              "要看『策略總報酬』與『買進持有總報酬』的對照。")

    print("\n" + "=" * 108)
    print("各事件定義的最佳結果對照（分辨新聞到底有沒有貢獻）")
    print("=" * 108)
    for mode in args.modes:
        m = res[res["事件"] == mode]
        if m.empty:
            continue
        b = m.iloc[0]
        print(f"  {mode:<9} 最佳淨報酬 {b['淨報酬']:+.2%}／筆"
              f"（盯 {int(b['盯盤天數'])} 天、跌 {b['跌幅門檻']:.0%}、"
              f"最長持有 {int(b['最長持有'])} 日；{int(b['交易數'])} 筆，"
              f"勝率 {b['勝率']:.1%}，t={b['t值']:.2f}）")

    print("\n" + "=" * 108)
    print("事件研究：新聞日收盤 = 0，之後各日的平均累積報酬")
    print("=" * 108)
    pts = [-5, -1, 0, 1, 2, 3, 5, 10, 20, 30]
    print("  " + "事件".ljust(12) + "樣本數".rjust(9) + "".join(f"{p:>9}" for p in pts))
    for mode in args.modes:
        if mode == "price":
            continue
        segs = []
        for px, ns in data.values():
            s = event_study(px, _start_days(px, ns, mode, 3.0))
            if len(s):
                segs.append(s)
        if not segs:
            continue
        arr = np.vstack(segs)
        m = arr.mean(axis=0)
        print("  " + mode.ljust(12) + f"{len(arr):>9,}"
              + "".join(f"{m[5 + p]:>+9.2%}" for p in pts))

    print(f"\n判讀：淨報酬要明顯為正且 t 值 > 2 才算數（來回成本 "
          f"{COST_BUY + COST_SELL:.2%} 已扣除）。")
    print("關鍵對照：若 negative 沒有明顯優於 price，代表新聞沒有貢獻，"
          "是「跌深就反彈」本身在做事。")
    if args.target == "index":
        print("\n標的為大盤指數：沒有生存者偏誤（指數不會下市），"
              "但指數為價格指數不含股利，實際做 0050 還會多領約 3~4% 股利。")
    else:
        print("\n重要偏誤：標的清單為當期市值快照，跌了沒回來或已下市的股票不在樣本中，"
              "本策略最易受此影響，結果應視為上界。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
