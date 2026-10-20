#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主題訊號的改良規格：事件起點 + 方向極性 + 來源分離。

為什麼要改，以及改的是什麼
--------------------------
前一版（theme_sector_test.py）用「熱度水準」測 55 組配對，2014-2024 的結果是
30/55（55%、p=0.59），跟擲骰子無異。診斷出三個具體缺陷：

  1. 水準 vs 變化：戰爭開打後報導會持續數月高檔，但資訊在第一天就定價完了。
     用水準會把「持續報導」誤當成「持續訊號」。改抓熱度首次突破的事件起點。
  2. 沒有方向性：「停火」「和談」與「開戰」「增兵」都被算成戰爭熱度，
     方向相反卻同號，彼此抵消。加入極性詞後分開計算。
  3. 語料混雜：台股新聞提到戰爭多半是「某公司受影響」（已反映在該股價格），
     國際新聞才是資訊源頭。兩者分開測。

誠實聲明
--------
本程式是在看過前一版結果後才寫的，屬於同一份資料上的反覆修改（資料窺探）。
因此即使結果改善，也只能視為「待用新資料驗證的假設」，不構成證據。
為限制窺探程度，只測一個改良規格，不做參數網格搜尋。
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

# 極性詞：同一主題下，事態升級與緩和的方向相反，不能混為一談
POLARITY = {
    "戰爭衝突": (["開戰", "升級", "擴大", "增兵", "進攻", "空襲", "砲擊", "動員",
                  "入侵", "報復", "襲擊"],
                 ["停火", "和談", "撤軍", "協議", "緩和", "談判", "休戰", "釋放"]),
    "關稅貿易戰": (["加徵", "課稅", "調高", "報復", "擴大", "祭出", "開徵"],
                   ["豁免", "取消", "調降", "排除", "協議", "延後", "暫緩"]),
    "制裁管制": (["新增", "加碼", "擴大", "列入", "祭出", "收緊", "升級"],
                 ["解除", "豁免", "放寬", "鬆綁", "移出"]),
    "利率通膨": (["升息", "鷹派", "上修", "加息", "緊縮", "高於預期"],
                 ["降息", "鴿派", "下修", "寬鬆", "低於預期", "放緩"]),
    "能源價格": (["飆漲", "大漲", "上看", "減產", "供應中斷", "斷供"],
                 ["大跌", "回落", "增產", "釋出儲備", "供給充足"]),
}


def build_series(calendar: pd.DatetimeIndex, intl_only: bool) -> pd.DataFrame:
    """每交易日、每主題的（升級則數、緩和則數、總則數）。"""
    pats = {t: "|".join(ws) for t, ws in THEMES.items()}
    pol = {t: ("|".join(u), "|".join(d)) for t, (u, d) in POLARITY.items()}
    files = sorted(glob.glob(str(ROOT / "data/intl_news/*.csv")))
    if not intl_only:
        files += sorted(glob.glob(str(ROOT / "data/stocks/*/news/cnyes_news_*.csv")))
    frames, seen = [], set()
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
        sub = pd.DataFrame({"day": T.TextFeatures._to_trading_day(ts, calendar)})
        for t, p in pats.items():
            hit = txt.str.contains(p, case=False, regex=True, na=False)
            if t in pol:
                up, dn = pol[t]
                sub[f"{t}|up"] = (hit & txt.str.contains(up, regex=True, na=False)).astype(float)
                sub[f"{t}|dn"] = (hit & txt.str.contains(dn, regex=True, na=False)).astype(float)
            sub[f"{t}|all"] = hit.astype(float)
        frames.append(sub.dropna(subset=["day"]))
        if i % 120 == 0:
            print(f"  {i}/{len(files)}")
    if not frames:
        return pd.DataFrame(index=calendar)
    return pd.concat(frames, ignore_index=True).groupby("day").sum()\
             .reindex(calendar).fillna(0.0)


def heat(s: pd.Series, win: int = 60) -> pd.Series:
    mu = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-5, 5)


def onset(h: pd.Series, thr: float = 1.5, quiet: int = 10) -> pd.Series:
    """
    事件起點：熱度突破門檻，且前 quiet 日都還沒突破過。

    這是本規格的核心修正。持續高熱度代表「還在報導」，不代表「有新資訊」；
    只有從平靜跳到高檔的那一刻，才是市場尚未消化的部分。
    """
    hot = h > thr
    was_quiet = ~hot.shift(1).rolling(quiet, min_periods=1).max().astype(bool)
    return (hot & was_quiet).fillna(False)


def cluster_t(x: pd.Series, idx) -> tuple:
    d = pd.DataFrame({"v": x.to_numpy()},
                     index=pd.DatetimeIndex(idx)).dropna()
    if len(d) < 30:
        return np.nan, np.nan
    g = d.groupby(d.index.to_period("M"))["v"].mean()
    if len(g) < 5 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan
    return float(d["v"].mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))


def run(sectors: dict, ser: pd.DataFrame, horizons: list, label: str) -> pd.DataFrame:
    rows = []
    for theme, pairs in THEME_SECTOR.items():
        cols = {k: f"{theme}|{k}" for k in ("up", "dn", "all")}
        if cols["all"] not in ser.columns:
            continue
        # 有極性詞的主題用「升級 − 緩和」，沒有的用總則數
        if cols["up"] in ser.columns:
            raw = ser[cols["up"]] - ser[cols["dn"]]
        else:
            raw = ser[cols["all"]]
        h = heat(raw)
        on = onset(h)
        for sec, direction, _why in pairs:
            if sec not in sectors:
                continue
            px = sectors[sec]
            for H in horizons:
                fwd = (px.shift(-H) / px - 1).reindex(on.index)
                sel = fwd[on & fwd.notna()]
                base = fwd.dropna()
                if len(sel) < 25 or len(base) < 200:
                    continue
                m, t = cluster_t(sel * direction, sel.index)
                b, _ = cluster_t(base * direction, base.index)
                rows.append({"規格": label, "主題": theme, "產業": sec,
                             "期間": H, "起點數": len(sel),
                             "起點後報酬": m, "全期平均": b, "超額": m - b,
                             "群集t": t})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse
    from scipy import stats

    ap = argparse.ArgumentParser(description="主題事件起點 × 極性")
    ap.add_argument("--start", default="2014-01-01")
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
    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in sectors.values()])))
    print(f"  {len(sectors)} 個產業、{len(cal):,} 個交易日\n")

    out = ROOT / "results" / "theme"
    out.mkdir(parents=True, exist_ok=True)
    allres = []
    for intl_only, label in ((True, "僅國際新聞"), (False, "國際＋台股新聞")):
        cache = out / f"onset_series_{'intl' if intl_only else 'all'}.csv"
        if cache.exists():
            ser = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            print(f"建立【{label}】主題序列…")
            ser = build_series(cal, intl_only)
            ser.to_csv(cache, encoding="utf-8-sig")
        r = run(sectors, ser, args.horizons, label)
        allres.append(r)

    res = pd.concat(allres, ignore_index=True)
    res.to_csv(out / "theme_onset.csv", index=False, encoding="utf-8-sig")

    print("=" * 92)
    print("改良規格的結果：事件起點後的超額報酬，符合預期方向的比例")
    print("=" * 92)
    for label in res["規格"].unique():
        for H in args.horizons:
            s = res[(res["規格"] == label) & (res["期間"] == H)]["超額"].dropna()
            if len(s) < 10:
                continue
            k = int((s > 0).sum())
            p = stats.binomtest(k, len(s), 0.5).pvalue
            print(f"  {label:<14}h={H:>2}　符合 {k}/{len(s)}（{k/len(s):.0%}）"
                  f"　p={p:.3f}　平均超額 {s.mean():+.3%}")

    print("\n" + "=" * 92)
    print("個別配對（僅國際新聞、依超額排序，前 12）")
    print("=" * 92)
    s = res[(res["規格"] == "僅國際新聞")].sort_values("超額", ascending=False).head(12)
    v = s[["主題", "產業", "期間", "起點數", "起點後報酬", "全期平均", "超額", "群集t"]].copy()
    for c in ("起點後報酬", "全期平均", "超額"):
        v[c] = v[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
    v["群集t"] = v["群集t"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(v.to_string(index=False))
    print("\n提醒：本規格是看過前一版結果後才設計的，屬同一份資料上的修改。")
    print("即使改善，也只能當作待用新資料驗證的假設。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
