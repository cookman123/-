#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徵工程：建立面板特徵庫（date × code）。

為什麼要重做
------------
原有的 43 個 `px_*` 全部是「每檔股票自己的時間序列」。但橫斷面模型要回答的是
「今天這 250 檔裡哪幾檔會相對強」，而單檔的 RSI 或均線乖離完全沒有告訴模型
**這檔股票今天在同儕裡站在什麼位置**。本模組補上四類原本缺的資訊：

    cs_   橫斷面百分位 —— 同一天把每個訊號在母體內排名，相對強弱的直接表述
    rel_  相對市場與產業 —— 超額報酬、beta、殘差波動、產業內相對位置
    liq_  流動性與微結構 —— Amihud 非流動性、零報酬日比例、價差代理
    trd_  可交易性 —— 距漲跌停的距離、連續漲跌停天數

最後一類是上一輪的教訓換來的。隔日模型跑出年化 +161% 是因為它押在漲跌停
（極端分位佔 9.4%，母體只有 1.1%），那些部位買不到也賣不掉。把可交易性做成
明確的欄位，評估時才能**排除**，而不是無意間去利用它。

不偷看未來
----------
    所有 rolling 一律向後看，沒有任何 shift(-N)
    橫斷面排名只用同一天的資料 —— 當天的橫斷面在當天是已知的
    產業平均採 leave-one-out，不把自己算進去
    母體遮罩沿用 universe_pit 的時點正確成交金額排名
    美股隔夜報酬是台股開盤前就已知的值，屬合法事前資訊

用法
----
    python feature_engineering.py                 # 建庫（預設 2010 起）
    python feature_engineering.py --audit         # 只跑前視稽核，不建庫
    python feature_engineering.py --start 2006-01-01

輸出
----
    data/features/panel.pkl   長表 (date, code) × 特徵，float32
    data/features/meta.json   特徵清單、家族說明、建庫參數
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
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
import fe_families as FF  # noqa: E402
import universe_pit as U  # noqa: E402

OUT = ROOT / "data" / "features"
FAMILY_DOC = {
    "ts_": "單檔時間序列：報酬、波動（含 Parkinson／Garman-Klass）、均線、"
           "KD／RSI／MACD／布林、量能、波動結構與量價背離",
    "rel_": "相對市場與產業：超額報酬、beta、相關、殘差波動、產業 leave-one-out 超額",
    "liq_": "流動性與微結構：成交金額、Amihud 非流動性、零報酬日比例、"
            "Corwin-Schultz 價差代理",
    "trd_": "可交易性：漲跌停旗標與距離、連續鎖死天數、近期漲跌停次數",
    "ovn_": "美股隔夜：費半／那斯達克／標普前一夜報酬與個股 beta 交互項",
    "cs_": "橫斷面百分位：每日在母體內把各訊號排成 0~1",
}


def load_panels(start: str, end: str, quiet: bool = False) -> dict:
    """OHLCV 四張寬表（date × code）。逐檔讀檔只做一次，之後全是面板運算。"""
    import davis_tw_train as T

    codes = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
                   if x.is_dir() and any((x / "price").glob("*.csv")))
    cols = {k: {} for k in ("open", "high", "low", "close", "volume")}
    for i, code in enumerate(codes, 1):
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        if len(px) < 300:
            continue
        for k in cols:
            cols[k][code] = px[k]
        if not quiet and i % 100 == 0:
            print(f"  讀取 {i}/{len(codes)}", flush=True)
    P = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    idx, cs = P["close"].index, P["close"].columns
    return {k: v.reindex(index=idx, columns=cs).astype("float64")
            for k, v in P.items()}


def load_market(start: str, end: str) -> pd.DataFrame:
    out = {}
    for name in ("TWII", "SOX", "IXIC", "GSPC"):
        p = ROOT / "data" / "market" / f"{name}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
            out[name] = d["close"][~d.index.duplicated(keep="last")]
    m = pd.DataFrame(out)
    return m[(m.index >= pd.Timestamp(start)) & (m.index <= pd.Timestamp(end))]


def industry_map() -> dict:
    r = json.load(open(ROOT / "data" / "industry_stock_map.json", encoding="utf-8"))
    return {x["code"]: x.get("category", "其他") for x in r}


def build(start: str, end: str, uni: int, quiet: bool = False) -> tuple:
    print(f"載入價格面板　{start} ~ {end}", flush=True)
    P = load_panels(start, end, quiet)
    C = P["close"]
    print(f"  面板 {C.shape[0]:,} 日 × {C.shape[1]} 檔", flush=True)

    M = load_market(start, end)
    ind = industry_map()
    mask = U.get_mask(start, end, uni).reindex(
        index=C.index, columns=C.columns).fillna(False).astype(bool)
    print(f"  母體遮罩：平均每日 {mask.sum(axis=1).mean():.0f} 檔", flush=True)

    f = {}
    for name, fn in (("時間序列", lambda: FF.timeseries(P)),
                     ("相對市場與產業", lambda: FF.relative(P, M, ind, mask)),
                     ("流動性與微結構", lambda: FF.liquidity(P)),
                     ("可交易性", lambda: FF.tradability(P)),
                     ("美股隔夜", lambda: FF.overnight(P, M))):
        g = fn()
        f.update(g)
        print(f"  {name:<10}{len(g):>4} 個", flush=True)
    g = FF.cross_sectional(f, mask)
    f.update(g)
    print(f"  {'橫斷面百分位':<10}{len(g):>4} 個", flush=True)
    print(f"  合計 {len(f)} 個特徵", flush=True)
    return f, P, mask


def to_long(f: dict, mask: pd.DataFrame) -> pd.DataFrame:
    """
    疊成長表並只保留母體內的列。
    逐特徵 stack 後 join，而不是先把 120 張面板同時放在記憶體裡再合併。
    """
    keep = mask.stack()
    keep = keep[keep].index
    keep.names = ["date", "code"]
    out = None
    for i, (name, panel) in enumerate(sorted(f.items()), 1):
        s = panel.stack(future_stack=True)
        s.index.names = ["date", "code"]
        s = s.reindex(keep).astype("float32").rename(name)
        out = s.to_frame() if out is None else out.join(s, how="left")
        if i % 30 == 0:
            print(f"  疊表 {i}/{len(f)}", flush=True)
    return out


def audit(f: dict, P: dict) -> int:
    """
    前視稽核：把每個特徵和「未來一日報酬」比相關。
    任何特徵若與未來報酬的相關係數異常高，就是洩漏的徵兆。
    這不能證明沒有洩漏，但能抓到最常見的那種（不小心用了 shift(-N)）。
    """
    C = P["close"]
    fwd = C.pct_change().shift(-1)
    print(f"\n{'=' * 92}\n前視稽核：各特徵與未來一日報酬的相關係數\n{'=' * 92}")
    rows = []
    for name, panel in f.items():
        a = panel.to_numpy().ravel()
        b = fwd.to_numpy().ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10000:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        rows.append((name, c, int(ok.sum())))
    rows.sort(key=lambda x: -abs(x[1]))
    print(f"  {'特徵':<26}{'相關':>9}{'樣本數':>12}")
    for name, c, n in rows[:12]:
        print(f"  {name:<26}{c:>+9.4f}{n:>12,}")
    bad = [r for r in rows if abs(r[1]) > 0.15]
    if bad:
        print(f"\n  ⚠ {len(bad)} 個特徵與未來報酬相關超過 0.15，請逐一檢查：")
        for name, c, _ in bad:
            print(f"    {name}　{c:+.4f}")
    else:
        print(f"\n  ✓ 最高僅 {abs(rows[0][1]):.4f}，沒有洩漏的跡象"
              f"（單一特徵對隔日報酬本來就該幾乎無相關）")
    return len(bad)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="建立面板特徵庫")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--uni", type=int, default=250)
    ap.add_argument("--audit", action="store_true", help="只跑前視稽核，不建庫")
    ap.add_argument("--tag", default="",
                    help="輸出檔名後綴，例如 --tag _full 會寫成 panel_full.pkl；"
                         "放寬母體時務必指定，否則會覆蓋既有的特徵庫")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    f, P, mask = build(a.start, a.end, a.uni, a.quiet)
    nbad = audit(f, P)
    if a.audit:
        return 1 if nbad else 0

    print(f"\n疊成長表…", flush=True)
    long = to_long(f, mask)
    # 標籤一起存：收盤→收盤與開盤→開盤兩種，後者是誠實的成交假設
    C, O = P["close"], P["open"]
    for nm, panel in (("y_cc", C.pct_change().shift(-1)),
                      ("y_oo", O.shift(-2) / O.shift(-1) - 1)):
        s = panel.stack(future_stack=True)
        s.index.names = ["date", "code"]
        long[nm] = s.reindex(long.index).astype("float32")

    path = OUT / f"panel{a.tag}.pkl"
    if path.exists() and not a.tag:
        # 不帶 tag 等於要覆蓋正式特徵庫；放寬母體時忘記加 tag 會把前 250 名
        # 那份蓋掉，而兩份的列數差一倍，事後很難看出是哪一份
        print(f"  覆蓋既有 {path.name}（{path.stat().st_size / 1e6:.0f} MB）")
    long.to_pickle(path)
    feats = [c for c in long.columns if not c.startswith("y_")]
    meta = {"建立時間": t0.strftime("%Y-%m-%d %H:%M"),
            "區間": [a.start, a.end], "母體": a.uni,
            "列數": int(len(long)), "特徵數": len(feats),
            "特徵": feats, "家族說明": FAMILY_DOC,
            "標籤": {"y_cc": "t 收盤→t+1 收盤（寬鬆）",
                     "y_oo": "t+1 開盤→t+2 開盤（誠實）"}}
    json.dump(meta, open(OUT / f"meta{a.tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    mins = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'=' * 92}")
    print(f"  長表 {len(long):,} 列 × {len(feats)} 特徵"
          f"　{long.index.get_level_values('date').min():%Y-%m}"
          f" ~ {long.index.get_level_values('date').max():%Y-%m}")
    print(f"  缺值比例 {long[feats].isna().mean().mean():.2%}"
          f"　檔案 {path.stat().st_size / 1e6:.0f} MB　耗時 {mins:.1f} 分")
    for pre, doc in FAMILY_DOC.items():
        n = sum(1 for c in feats if c.startswith(pre))
        print(f"  {pre:<6}{n:>4} 個　{doc[:52]}")
    print(f"\n輸出：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
