#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
補 Yahoo 股價在 2011-01 ~ 2011-09 的全市場缺口。

問題
----
data/stocks/*/price/yahoo_*_2011.csv 全部 316 檔都從 2011-09-05 或
2011-09-19 才開始（2330 只有 83 列）。合併後面板在 2010-12-31 之後直接
跳到 9 月，quarterly_factors.quarter_ends() 會漏掉 2011Q1、Q2，季頻回測裡
出現一個橫跨 9 個月的「季」。其他年份沒有全市場性的缺口（2000 年與個股
停牌造成的空白都在研究期 2010 起之外，或是真的停牌）。

資料來源
--------
證交所 MI_INDEX（type=ALLBUT0999）一天一次請求就回全部上市個股的 OHLCV，
約 180 個交易日 ≈ 180 次請求；逐檔 STOCK_DAY 要 316 × 9 ≈ 2,800 次。
上櫃股（13 檔）用櫃買中心 dailyQuotes，同樣一天一次。

Yahoo 的價格不是原始價（這是補資料時才發現的）
------------------------------------------------
原本以為 Yahoo 給的是未還原價，因為 2330 與證交所一模一樣。但 2330 是
2010 年後沒配過股的特例；9 月重疊日比對時 302 檔有 219 檔差超過 0.5%，
2408 甚至差 8.7 倍。原因是 Yahoo 把**配股與減資當成股票分割**往回調整：
價格 × f、成交量 ÷ f，成交金額不變（1101：28.65 × 18.77M ≈ 38.95 × 13.82M）。
現金股利則不調整。所以直接貼交易所原始價會在接縫處製造假跳動。

做法
----
    1. 逐日抓 2010-12 ~ 2011-09 的交易所行情，原始回應快取到
       data/twse_mi_index_2011/、data/tpex_daily_2011/（可斷點續傳）
    2. 每檔估 f = Yahoo 收盤 ÷ 交易所收盤：缺口之前用 2010-12、之後用 2011-09
       的重疊日取中位數。兩側相同就整段乘同一個 f；不同代表缺口內有除權，
       在證交所除權日（TWT49U）切換
    3. 每檔只補「2011-01-01 到該檔 Yahoo 2011 檔第一天之前」的日期，
       寫回 yahoo_{code}_2011.csv；原檔先備份到 data/_backup_yahoo_2011/
       （重跑一律從備份重建，不會重複補）
    load_price() 不用改，照常讀逐年檔。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
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
import tw_stock_crawler as X  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "twse_mi_index_2011"
BACKUP = DATA / "_backup_yahoo_2011"
REPORT = ROOT / "results" / "data_fix_2011" / "fill_report.csv"
URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_CACHE = DATA / "tpex_daily_2011"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
LO, HI = pd.Timestamp("2011-01-01"), pd.Timestamp("2011-09-30")
# 2010-12 也抓：用來算缺口「之前」那一側的 Yahoo／交易所比值
FETCH_LO = pd.Timestamp("2010-12-01")


def _get(src: str, d: pd.Timestamp) -> dict:
    if src == "twse":
        return X.http_get(URL, params={"date": f"{d:%Y%m%d}", "type": "ALLBUT0999",
                                       "response": "json"})
    return X.http_get(TPEX_URL, params={"date": f"{d:%Y/%m/%d}", "response": "json"})


def fetch_all(src: str) -> None:
    cache = CACHE if src == "twse" else TPEX_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    for d in pd.bdate_range(FETCH_LO, HI):
        f = cache / f"{d:%Y%m%d}.json"
        if f.exists():
            continue
        for k in range(5):
            try:
                r = _get(src, d)
                break
            except Exception as e:  # noqa: BLE001
                print(f"{d:%Y-%m-%d} 失敗 {e}，{10 * (k + 1)} 秒後重試", flush=True)
                time.sleep(10 * (k + 1))
        else:
            raise SystemExit(f"{d:%Y-%m-%d} 連續失敗")
        # 休市日 stat 不是 OK，也照樣存，下次就不會再問
        f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"{d:%Y-%m-%d} {r.get('stat')}", flush=True)
        time.sleep(3)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


# 兩個交易所的欄名不同：證交所「證券代號／收盤價／成交股數」，櫃買「代號／收盤／成交股數」
COLS = {"twse": ("證券代號", "開盤價", "最高價", "最低價", "收盤價", "成交股數"),
        "tpex": ("代號", "開盤", "最高", "最低", "收盤", "成交股數")}


def load_cache(src: str = "twse") -> pd.DataFrame:
    """把快取的每日 JSON 攤成 (date, code, open, high, low, close, volume)。"""
    cache = CACHE if src == "twse" else TPEX_CACHE
    kc, ko, kh, kl, kcl, kv = COLS[src]
    out = []
    for f in sorted(cache.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if str(r.get("stat")).upper() != "OK":
            continue
        tab = [t for t in r.get("tables", []) if t.get("fields") and kc in t["fields"]
               and kcl in t["fields"] and t.get("data")]
        if not tab:
            continue
        t = tab[0]
        df = pd.DataFrame(t["data"], columns=t["fields"])
        out.append(pd.DataFrame({
            "date": pd.Timestamp(f.stem),
            "code": df[kc].astype(str).str.strip(),
            "open": _num(df[ko]), "high": _num(df[kh]),
            "low": _num(df[kl]), "close": _num(df[kcl]),
            "volume": _num(df[kv]),
        }))
    if not out:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume"])
    m = pd.concat(out, ignore_index=True)
    # 整天沒成交的個股收盤是 "--"，轉成 NaN 後丟掉（與 load_price 剔除假列一致）
    return m.dropna(subset=["close"])


def load_exrights() -> pd.DataFrame:
    """證交所 TWT49U：2010-12 ~ 2011-09 的除權息日，只用來交叉核對換算係數的切換日。"""
    f = DATA / "twse_mi_index_2011" / "TWT49U_201012_201109.json"
    if not f.exists():
        r = X.http_get("https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
                       params={"startDate": "20101201", "endDate": "20110930", "response": "json"})
        f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    r = json.loads(f.read_text(encoding="utf-8"))
    df = pd.DataFrame(r["data"], columns=r["fields"])
    return pd.DataFrame({"date": df["資料日期"].map(_roc), "code": df["股票代號"].str.strip(),
                         "kind": df["權/息"].str.strip()})


def _roc(s: str) -> pd.Timestamp:
    t = "".join(ch if ch.isdigit() else " " for ch in s).split()
    return pd.Timestamp(int(t[0]) + 1911, int(t[1]), int(t[2]))


def _read_yahoo(code: str, year: int) -> pd.DataFrame:
    f = DATA / "stocks" / code / "price" / f"yahoo_{code}_{year}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="date"), dtype="float64")
    y = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    return y[["open", "high", "low", "close", "volume"]]


def _ratio(y: pd.DataFrame, ex: pd.DataFrame, side: str, stop=None) -> tuple:
    """
    Yahoo 收盤 ÷ 交易所收盤，取最貼近缺口的 3 個重疊日的中位數。

    不能用整段重疊期：有 14 檔在 2011-09 的重疊期內就除權（例如 2609 在
    09-22），整月取中位數會把除權前後兩個係數混在一起。
    """
    r = (y["close"] / ex["close"].reindex(y.index)).dropna()
    r = r[(r > 0) & y["volume"].reindex(r.index).gt(0)].sort_index()
    if stop is not None:                     # 重疊期內的除權日（含）之後不算
        r = r[r.index < stop] if (r.index < stop).any() else r.head(1)
    r = r.tail(3) if side == "pre" else r.head(3)
    if r.empty:
        return np.nan, np.nan, 0
    med = float(r.median())
    return med, float((r / med - 1).abs().max()), len(r)


def plan(code: str, ex: pd.DataFrame, xr: pd.DataFrame) -> dict:
    """
    決定這一檔要補哪些日期、每天乘多少。

    Yahoo 的價格不是原始價：配股、減資被當成分割往回調整（價 × f、量 ÷ f，
    成交金額不變），所以交易所原始價必須乘上同一個 f 才接得上。f 用缺口兩側
    的重疊日估：之前用 2010-12，之後用 2011-09。兩側不同代表缺口內有除權，
    切換日優先取證交所除權日，否則取「套用後當日跳動縮最多」的那天。
    """
    y10, y11 = _read_yahoo(code, 2010), _read_yahoo(code, 2011)
    first = y11.index.min()
    out = {"code": code, "yahoo_first": first.date()}
    fill = ex[(ex.index >= LO) & (ex.index < first)].dropna(subset=["close"])
    fill = fill[fill["volume"] > 0]
    out["added"] = len(fill)
    if fill.empty:
        return {**out, "fill": fill}
    f_pre, dev_pre, n_pre = _ratio(y10[y10.index >= FETCH_LO], ex, "pre")
    xc = xr[(xr["code"] == code) & xr["kind"].str.contains("權")]["date"]
    nxt = xc[xc >= first]
    f_post, dev_post, n_post = _ratio(y11[y11.index <= HI], ex, "post",
                                      nxt.min() if len(nxt) else None)
    out.update(f_pre=f_pre, n_pre=n_pre, dev_pre=dev_pre,
               f_post=f_post, n_post=n_post, dev_post=dev_post)
    if np.isnan(f_post) and np.isnan(f_pre):
        return {**out, "added": 0, "note": "兩側都無重疊日，無法換算", "fill": fill.iloc[:0]}
    if np.isnan(f_pre) or np.isnan(f_post) or abs(f_pre / f_post - 1) < 0.005:
        f = f_post if not np.isnan(f_post) else f_pre
        fac = pd.Series(f, index=fill.index)
        out["switch"] = ""
    else:
        cand = xc[(xc >= LO) & (xc < first)]
        anyx = xr[(xr["code"] == code) & (xr["date"] >= LO) & (xr["date"] < first)]["date"]
        if len(cand):
            d, how = cand.iloc[-1], "證交所除權日"
        elif len(anyx):
            # 缺口內只有除息、沒有除權，但 Yahoo 係數仍變了（2601）：原因不明，
            # 切在唯一的事件日，並在報告中標記；這一跳本來就存在於原資料的缺口裡
            d, how = anyx.iloc[-1], "僅除息日，原因不明"
        else:
            raw = ex["close"].reindex(fill.index)
            jump = (np.log(raw / raw.shift()) + np.log(f_post / f_pre)).abs() - \
                np.log(raw / raw.shift()).abs()
            d, how = jump.idxmin(), "推估"
        fac = pd.Series(np.where(fill.index < d, f_pre, f_post), index=fill.index)
        out["switch"] = f"{d:%Y-%m-%d}（{how}）"
    adj = fill[["open", "high", "low", "close"]].mul(fac, axis=0)
    adj["volume"] = (fill["volume"] / fac).round()
    return {**out, "fill": adj}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="補 Yahoo 2011 缺口")
    ap.add_argument("--fetch", action="store_true", help="抓證交所與櫃買每日行情到快取")
    ap.add_argument("--write", action="store_true", help="實際寫回 2011 檔（否則只做檢查）")
    a = ap.parse_args(argv)

    if a.fetch:
        fetch_all("twse")
        fetch_all("tpex")
        return 0
    tw, tp = load_cache("twse"), load_cache("tpex")
    print(f"快取：證交所 {tw['date'].nunique()} 日／{tw['code'].nunique()} 碼，"
          f"櫃買 {tp['date'].nunique()} 日／{tp['code'].nunique()} 碼")
    src = {c: ("twse", g) for c, g in tw.groupby("code")}
    for c, g in tp.groupby("code"):
        # 同一代號兩邊都有（轉市），取缺口內日數較多的那邊
        if c not in src or len(g[g["date"] < "2011-09-01"]) > \
                len(src[c][1][src[c][1]["date"] < "2011-09-01"]):
            src[c] = ("tpex", g)
    xr = load_exrights()

    rows = []
    for f in sorted(DATA.glob("stocks/*/price/yahoo_*_2011.csv")):
        code = f.parent.parent.name
        if code not in src:
            rows.append({"code": code, "added": 0, "note": "兩個交易所都查無此代號"})
            continue
        s, g = src[code]
        p = plan(code, g.set_index("date").sort_index(), xr)
        fill = p.pop("fill")
        rows.append({**p, "source": s})
        if a.write and len(fill):
            BACKUP.mkdir(parents=True, exist_ok=True)
            bk = BACKUP / f.name
            if not bk.exists():
                shutil.copy2(f, bk)
            y = pd.read_csv(bk, encoding="utf-8-sig")
            new = fill.reset_index()
            new["date"] = new["date"].dt.strftime("%Y-%m-%d")
            pd.concat([new, y[["date", "open", "high", "low", "close", "volume"]]],
                      ignore_index=True).to_csv(f, index=False, encoding="utf-8-sig")

    r = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(REPORT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(f"檔案 {len(r)}，補列總數 {int(r['added'].sum())}，補 0 列 {int((r['added'] == 0).sum())} 檔")
    print(r[r["added"] == 0][["code", "yahoo_first", "note"]].to_string(index=False))
    ok = r.dropna(subset=["f_post"])
    print(f"係數 = 1（Yahoo 恰為原始價）：{int((ok['f_post'] - 1).abs().lt(0.005).sum())}/{len(ok)} 檔")
    print(f"重疊日內係數不穩（3 日內偏離 >1%）：之前 {int((r['dev_pre'] > 0.01).sum())}、"
          f"之後 {int((r['dev_post'] > 0.01).sum())} 檔")
    sw = r[r.get("switch", pd.Series(dtype=str)).fillna("").ne("")]
    print(f"缺口內係數有切換（除權）：{len(sw)} 檔")
    print(sw[["code", "f_pre", "f_post", "switch"]].to_string(index=False))
    print(f"報告：{REPORT}")
    print("已寫回" if a.write else "（檢查模式，未寫檔）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
