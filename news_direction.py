#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新聞 → 方向預測：看到新聞之後，預期接下來會漲還是會跌。

與先前所有做法的差別
--------------------
1. 先前是「已經看到跌幅」再用新聞當過濾條件；這裡新聞在前，是用來預測後續方向。
2. 先前的文本特徵每個交易日都算，但 87% 的日子根本沒有新聞
   （52.3 萬個股票-交易日中只有 6.7 萬有新聞），等於用大量空白稀釋訊號。
   這裡**只在有新聞的日子上建模**，樣本雖然小但每一筆都有內容。
3. 先前用 TF-IDF → PCA → Ward 產生上百欄高維特徵，逐檔擬合、樣本嚴重不足。
   這裡改用低維、可解釋的關鍵詞類別計數——先前 PTT 的實驗顯示
   低維表示不會造成傷害，高維表示會。

兩段分析
--------
  第一段　可解釋的分組統計：每一類新聞（財報好／壞、訂單、法人升降評、
          法律案件、增減資…）出現後，未來 1／5／20 日的平均報酬。
          這一段不訓練模型，直接看關鍵詞類別與後續報酬的關係。
  第二段　預測模型：把這些類別計數加上價格情境當特徵，
          walk-forward 訓練，看方向準確率能否勝過基準比例。

顯著性一律用以月份為群集的 t 值：全市場同時波動會讓幾百檔同時出現新聞，
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
from news_reversal import load_stock  # noqa: E402

ROOT = Path(__file__).resolve().parent

# 關鍵詞類別：刻意設計成可解釋、低維，而不是幾百欄的 TF-IDF
KEYWORD_GROUPS = {
    "財報佳": ["創新高", "新高", "大增", "成長", "優於預期", "上修", "轉盈",
               "獲利增", "營收增", "年增", "月增", "季增", "賺"],
    "財報差": ["下修", "下調", "衰退", "虧損", "轉虧", "不如預期", "低於預期",
               "年減", "月減", "季減", "減少", "下滑", "萎縮"],
    "訂單增": ["訂單", "拉貨", "急單", "追加", "擴產", "新廠", "量產", "投產"],
    "訂單減": ["砍單", "減單", "delay", "遞延", "延後", "停工", "停產", "缺料",
               "斷鏈", "庫存調整", "去化"],
    "法人升": ["調升", "升評", "目標價", "看好", "買進評等", "加碼", "買超"],
    "法人降": ["調降", "降評", "看壞", "賣出評等", "減碼", "賣超"],
    "法律案件": ["訴訟", "求償", "裁罰", "罰款", "檢調", "搜索", "起訴", "調查",
                 "掏空", "舞弊", "違約", "跳票"],
    "股利": ["配息", "除息", "股利", "股息", "配股", "殖利率"],
    "增減資": ["減資", "增資", "私募", "可轉債", "發行新股"],
    "人事異動": ["董事長", "總經理", "請辭", "辭任", "接任", "改組", "裁員"],
    "併購投資": ["併購", "收購", "合併", "入股", "策略聯盟", "投資案", "設廠"],
    "產業利多": ["漲價", "報價上漲", "供不應求", "缺貨", "旺季"],
    "產業利空": ["跌價", "報價下跌", "供過於求", "價格戰", "淡季"],
}


def cluster_t(x: pd.Series, dates: pd.Series) -> tuple:
    """以月份為群集的平均與 t 值。"""
    d = pd.DataFrame({"v": x, "m": pd.to_datetime(dates).dt.to_period("M")}).dropna()
    if len(d) < 30:
        return np.nan, np.nan, 0
    g = d.groupby("m")["v"].mean()
    if len(g) < 3 or g.std(ddof=1) == 0:
        return float(d["v"].mean()), np.nan, len(g)
    return float(d["v"].mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), len(g)


def build_news_panel(start: str, end: str) -> pd.DataFrame:
    """只保留「有新聞的股票-交易日」，每一列是一個新聞事件。"""
    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    pats = {k: "|".join(v) for k, v in KEYWORD_GROUPS.items()}
    rows = []
    for i, code in enumerate(codes, 1):
        px, news = load_stock(code, start, end)
        if px is None or news is None or news.empty:
            continue
        title = news["title"].fillna("")
        body = news.get("text", pd.Series("", index=news.index)).fillna("").str[:300]
        txt = title + " " + body
        feat = pd.DataFrame({"day": news["day"]})
        for k, pat in pats.items():
            feat[k] = txt.str.contains(pat, case=False, regex=True).astype(float)
        agg = feat.groupby("day").sum()
        agg["n_news"] = feat.groupby("day").size()

        c = px["close"]
        ctx = pd.DataFrame(index=px.index)
        ctx["pre1"] = c.pct_change()
        ctx["pre5"] = c / c.shift(5) - 1
        ctx["pre20"] = c / c.shift(20) - 1
        ctx["vol20"] = c.pct_change().rolling(20).std()
        ctx["from_hi20"] = c / c.rolling(20).max() - 1
        ctx["fwd1"] = c.shift(-1) / c - 1
        ctx["fwd5"] = c.shift(-5) / c - 1
        ctx["fwd20"] = c.shift(-20) / c - 1

        d = agg.join(ctx, how="inner")
        d["code"] = code
        d = d.reset_index().rename(columns={"day": "date", "index": "date"})
        rows.append(d)
        if i % 80 == 0:
            print(f"  已處理 {i}/{len(codes)} 檔")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="新聞 → 方向預測")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default="results/news_direction")
    args = ap.parse_args(argv)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"期間 {args.start} ~ {args.end}　關鍵詞類別 {len(KEYWORD_GROUPS)} 組")
    A = build_news_panel(args.start, args.end)
    if A.empty:
        print("沒有資料")
        return 1
    A = A.dropna(subset=["fwd1", "fwd5", "fwd20"])
    print(f"\n新聞事件 {len(A):,} 筆（股票-交易日），"
          f"{A['code'].nunique()} 檔、{A['date'].nunique():,} 個交易日\n")

    # ---------- 第一段：可解釋的分組統計 ----------
    print("=" * 100)
    print("第一段　各類新聞出現後的平均報酬（以月份為群集的 t 值）")
    print("=" * 100)
    base = {}
    for h in ("fwd1", "fwd5", "fwd20"):
        m, t, _ = cluster_t(A[h], A["date"])
        base[h] = m
    print(f"  {'基準（所有新聞日）':<14}{len(A):>8,}"
          + "".join(f"{base[h]:>+14.3%}" for h in ("fwd1", "fwd5", "fwd20")))
    print(f"  {'類別':<14}{'出現次數':>8}"
          + "".join(f"{c:>14}" for c in ("次日超額", "5日超額", "20日超額")))
    stats_rows = []
    for k in KEYWORD_GROUPS:
        sub = A[A[k] > 0]
        if len(sub) < 200:
            continue
        cells, rec = [], {"類別": k, "次數": len(sub)}
        for h in ("fwd1", "fwd5", "fwd20"):
            m, t, _ = cluster_t(sub[h] - base[h], sub["date"])
            cells.append(f"{m:+.3%}({t:>5.2f})" if pd.notna(t) else f"{m:+.3%}")
            rec[f"{h}_超額"], rec[f"{h}_t"] = m, t
        stats_rows.append(rec)
        print(f"  {k:<14}{len(sub):>8,}" + "".join(f"{x:>14}" for x in cells))
    pd.DataFrame(stats_rows).to_csv(out_dir / "keyword_stats.csv",
                                    index=False, encoding="utf-8-sig")
    print("\n  「超額」= 該類新聞日的報酬減去所有新聞日的平均，已扣除大盤與時期效應的一部分。")
    print("  t 值 > 2 才算數；13 個類別同時檢定，單一顯著可能是巧合。")

    # ---------- 第二段：預測模型 ----------
    print("\n" + "=" * 100)
    print("第二段　用新聞類別 + 價格情境預測方向（walk-forward，含 purging）")
    print("=" * 100)
    from sklearn.ensemble import HistGradientBoostingRegressor

    feats = list(KEYWORD_GROUPS) + ["n_news", "pre1", "pre5", "pre20",
                                    "vol20", "from_hi20"]
    A = A.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(A["date"].unique()))

    for h, hor in (("fwd1", 1), ("fwd5", 5), ("fwd20", 20)):
        folds = T.step5_split(dates, quiet=True, horizon=hor)
        if not folds:
            continue
        preds = []
        for f in folds:
            tr = A[A["date"].isin(f.train)]
            te = A[A["date"].isin(f.test)]
            if len(tr) < 3000 or len(te) < 300:
                continue
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=200,
                                              learning_rate=0.05,
                                              early_stopping=False, random_state=42)
            m.fit(tr[feats].to_numpy(), tr[h].to_numpy())
            o = te[["date", "code", h]].copy()
            o["pred"] = m.predict(te[feats].to_numpy())
            preds.append(o)
        if not preds:
            continue
        P = pd.concat(preds, ignore_index=True)
        da = float(((P["pred"] > 0) == (P[h] > 0)).mean())
        base_rate = float((P[h] > 0).mean())
        # 預測最高／最低十分位的實際報酬差
        q = P["pred"].quantile([0.1, 0.9])
        hi = P[P["pred"] >= q.iloc[1]][h]
        lo = P[P["pred"] <= q.iloc[0]][h]
        mh, th, _ = cluster_t(hi, P[P["pred"] >= q.iloc[1]]["date"])
        ml, tl, _ = cluster_t(lo, P[P["pred"] <= q.iloc[0]]["date"])
        print(f"  {h:<6} 測試樣本 {len(P):>7,}　方向準確率 {da:.4f}"
              f"（基準 always_long {base_rate:.4f}，差 {100 * (da - base_rate):+.2f} 個百分點）")
        print(f"         預測前 10% 實際報酬 {mh:+.3%}（t={th:.2f}）　"
              f"後 10% {ml:+.3%}（t={tl:.2f}）　價差 {mh - ml:+.3%}")
        P.to_csv(out_dir / f"preds_{h}.csv", index=False, encoding="utf-8-sig")

    print("\n判讀：方向準確率要跟 always_long 基準比，不是跟 50% 比。")
    print("「價差」是最實用的指標：預測最看好與最看壞的十分位，實際報酬差多少；"
          "要大於來回成本才有交易價值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
