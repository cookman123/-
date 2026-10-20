#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主題熱度對產業類股的領先性檢定。

問題
----
「戰爭新聞變多 → 航運要漲了」這種判讀，在資料上站不站得住？

三個必須處理的陷阱
------------------
1. 反向因果：航運漲了新聞才報導。因此一律用第 t 日（含）之前的新聞，
   預測第 t+1 日起的報酬；盤後（13:30 後）發布的新聞順延到下一個交易日。
2. 必須跟「純價格動能」對照：若主題熱度只是價格動能的影子，就沒有增量。
   所以除了單變量相關，另做「控制該產業自身動能後」的偏相關。
   先前六種新聞用法全都敗在這一關。
3. 多重檢定：55 組配對在 5% 水準下純機率就有約 2.8 組顯著。
   因此主要證據是「符合預期方向的比例」（二項檢定），不是挑最漂亮的那一組。

顯著性用以月份為群集的 t 值：同一則大事件會同時影響許多產業與許多天，
把每天當獨立樣本會高估顯著性。
"""
from __future__ import annotations

import glob
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
from themes import THEMES, THEME_SECTOR  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "theme" / "theme_daily.csv"


def build_theme_series(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    每個交易日、每個主題的新聞則數。

    同一篇新聞常被掛到多檔股票，先依標題去重，否則熱門股的新聞會被重複計數。
    時間對齊沿用 13:30 規則：盤後發布者歸到下一個交易日，避免用到尚未公開的資訊。
    """
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)

    pats = {t: "|".join(ws) for t, ws in THEMES.items()}
    frames, seen = [], set()
    files = sorted(glob.glob(str(ROOT / "data/stocks/*/news/cnyes_news_*.csv")))
    files += sorted(glob.glob(str(ROOT / "data/intl_news/*.csv")))
    print(f"掃描 {len(files)} 個新聞檔…")
    for i, f in enumerate(files, 1):
        try:
            d = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        if "date" not in d.columns or "title" not in d.columns:
            continue
        d["title"] = d["title"].fillna("")
        d = d[~d["title"].isin(seen)]
        seen.update(d["title"].tolist())
        if d.empty:
            continue
        ts = pd.to_datetime(d["date"], errors="coerce", utc=True)
        try:
            ts = ts.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        body = d["summary"].fillna("") if "summary" in d.columns else ""
        txt = d["title"] + " " + body
        day = T.TextFeatures._to_trading_day(ts, calendar)
        sub = pd.DataFrame({"day": day})
        for t, p in pats.items():
            sub[t] = txt.str.contains(p, case=False, regex=True, na=False).astype(float)
        frames.append(sub.dropna(subset=["day"]))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")
    if not frames:
        return pd.DataFrame(index=calendar)
    allp = pd.concat(frames, ignore_index=True)
    out = allp.groupby("day").sum().reindex(calendar).fillna(0.0)
    out["_total"] = allp.groupby("day").size().reindex(calendar).fillna(0.0)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, encoding="utf-8-sig")
    print(f"已快取 {CACHE}")
    return out


def heat(series: pd.Series, win: int = 60) -> pd.Series:
    """
    熱度 = 相對自身近 win 日的標準化水準，全部用回看視窗。

    用相對值而非絕對則數：不同主題的基礎聲量差十倍以上，
    絕對則數無法比較，而「比平常多」才是訊號。
    """
    mu = series.rolling(win, min_periods=20).mean()
    sd = series.rolling(win, min_periods=20).std()
    return ((series - mu) / sd.replace(0, np.nan)).clip(-5, 5)


def cluster_t(x: pd.Series, dates) -> tuple:
    d = pd.DataFrame({"v": x, "m": pd.to_datetime(dates).to_period("M")}).dropna()
    if len(d) < 60:
        return np.nan, np.nan
    g = d.groupby("m")["v"].mean()
    if len(g) < 6 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan
    return float(d["v"].mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="主題熱度 → 產業類股領先性")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    args = ap.parse_args(argv)

    imap = T.load_industry_map()
    sectors = {}
    print("合成產業類股指數…")
    for name, codes in imap.items():
        idx, used = T.build_sector_index(codes, args.start, args.end)
        if idx is not None and not idx.empty and len(used) >= 3:
            sectors[name] = idx["close"]
    print(f"  可用 {len(sectors)} 個產業\n")
    if not sectors:
        return 1

    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in sectors.values()])))
    th = build_theme_series(cal)
    heats = {t: heat(th[t]) for t in THEMES if t in th.columns}

    rows = []
    for theme, pairs in THEME_SECTOR.items():
        if theme not in heats:
            continue
        h = heats[theme]
        for sec, direction, why in pairs:
            if sec not in sectors:
                continue
            px = sectors[sec]
            own = px.pct_change(20)                      # 該產業自身動能（控制變數）
            for H in args.horizons:
                fwd = px.shift(-H) / px - 1
                d = pd.DataFrame({"h": h, "fwd": fwd, "own": own}).dropna()
                if len(d) < 200:
                    continue
                # 單變量：熱度與後續報酬的相關（方向已按預期調整符號）
                raw = d["h"].corr(d["fwd"]) * direction
                # 控制自身動能後的偏相關
                hr = d["h"] - d["h"].corr(d["own"]) * d["own"] * (d["h"].std() / d["own"].std())
                part = hr.corr(d["fwd"]) * direction
                # 熱度高的日子（z>1）之後的報酬，按預期方向取符號
                sel = d[d["h"] > 1.0]
                m, t = cluster_t(sel["fwd"] * direction, sel.index) if len(sel) > 60 \
                    else (np.nan, np.nan)
                base, _ = cluster_t(d["fwd"] * direction, d.index)
                rows.append({"主題": theme, "產業": sec, "方向": direction, "期間": H,
                             "單變量相關": raw, "控動能後": part,
                             "熱度高時報酬": m, "全期平均": base,
                             "超額": (m - base) if pd.notna(m) else np.nan,
                             "群集t": t, "熱日數": len(sel)})
    if not rows:
        print("沒有可檢定的配對")
        return 1
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "theme"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "theme_sector.csv", index=False, encoding="utf-8-sig")

    print("=" * 96)
    print("整體證據：符合預期方向的比例（主要判讀依據）")
    print("=" * 96)
    from scipy import stats
    for H in args.horizons:
        s = res[res["期間"] == H]
        for col in ("單變量相關", "控動能後", "超額"):
            v = s[col].dropna()
            if v.empty:
                continue
            k = int((v > 0).sum())
            p = stats.binomtest(k, len(v), 0.5).pvalue
            print(f"  h={H:>2}　{col:<8}符合預期 {k}/{len(v)}（{k/len(v):.0%}）"
                  f"　二項檢定 p={p:.3f}　平均 {v.mean():+.4f}")
    print("\n  判讀：若符合比例接近 50%、p 不顯著，代表主題熱度沒有領先性；")
    print("  『控動能後』比『單變量』掉很多，代表訊號其實只是價格動能的影子。")

    print("\n" + "=" * 96)
    print("個別配對（依控制動能後的相關排序，僅列前 15 與後 5）")
    print("=" * 96)
    s = res[res["期間"] == args.horizons[-1]].sort_values("控動能後", ascending=False)
    v = pd.concat([s.head(15), s.tail(5)])[
        ["主題", "產業", "方向", "單變量相關", "控動能後", "超額", "群集t", "熱日數"]]
    for c in ("單變量相關", "控動能後"):
        v[c] = v[c].map(lambda x: f"{x:+.3f}")
    v["超額"] = v["超額"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
    v["群集t"] = v["群集t"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(v.to_string(index=False))
    print(f"\n完整結果：{out / 'theme_sector.csv'}")
    print(f"共檢定 {len(res)} 組（{len(args.horizons)} 種期間 × 配對），"
          f"5% 水準下純機率約有 {len(res) * 0.05:.0f} 組會顯著。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
