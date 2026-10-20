#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把下市股放進日頻母體後執行任一支程式（生存者偏誤檢查）。

日頻的股票池（data/stocks）只有今天仍上市的公司，研究期間下市的 71 家
（data/delisted/price_adj，已換算成與 Yahoo 同一價格基準）不在裡面。
這支工具在記憶體裡替換三個載入點，不改任何共用檔案：
    davis_tw_train.load_price     下市代號改讀 price_adj
    feature_engineering.load_panels  代號清單加入下市股
    universe_pit.get_mask         成交金額排名含下市股一起排（不寫入共用快取）

用法（輸出務必加 tag，避免蓋掉正式特徵庫）：
    python tool/run_surv.py feature_engineering --tag _surv --quiet
    python tool/run_surv.py nextday_horizon --tag _surv
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import davis_tw_train as T  # noqa: E402
import feature_engineering as FE  # noqa: E402
import universe_pit as U  # noqa: E402

DEL = ROOT / "data" / "delisted" / "price_adj"
ALIVE = sorted(x.name for x in (ROOT / "data" / "stocks").iterdir()
               if x.is_dir() and any((x / "price").glob("*.csv")))
DEAD = sorted(f.stem for f in DEL.glob("[0-9]*.csv") if f.stem not in set(ALIVE))
_load_price = T.load_price


def load_price(code: str):
    if code not in DEAD:
        return _load_price(code)
    px = pd.read_csv(DEL / f"{code}.csv", encoding="utf-8-sig", parse_dates=["date"])
    px = px.dropna(subset=["date", "close"]).sort_values("date") \
        .drop_duplicates("date", keep="last").set_index("date")
    flat = (px["volume"] <= 0) & (px["open"] == px["close"]) & (px["high"] == px["low"])
    return px.loc[~flat, ["open", "high", "low", "close", "volume"]]


def load_panels(start: str, end: str, quiet: bool = False) -> dict:
    cols = {k: {} for k in ("open", "high", "low", "close", "volume")}
    for code in ALIVE + DEAD:
        px = load_price(code)
        if px is None or px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        # 存活股沿用原本的 300 日門檻；下市股只要夠算 60 日特徵即可，否則短命的會全被濾掉
        if len(px) < (300 if code in ALIVE else 80):
            continue
        for k in cols:
            cols[k][code] = px[k]
    P = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    idx, cs = P["close"].index, P["close"].columns
    if not quiet:
        print(f"  面板含下市股 {len([c for c in cs if c in DEAD])} 檔")
    return {k: v.reindex(index=idx, columns=cs).astype("float64") for k, v in P.items()}


def get_mask(start: str, end: str, n: int, refresh: bool = False):
    t = U.build_turnover_panel(ALIVE + DEAD, start, end)
    return U.top_n_mask(t, n)


T.load_price = load_price
FE.load_panels = load_panels
U.get_mask = get_mask


def main() -> int:
    name, args = sys.argv[1], sys.argv[2:]
    if "--tag" not in args:
        print("必須指定 --tag（例如 _surv），避免蓋掉正式輸出"); return 1
    print(f"存活 {len(ALIVE)} 檔＋下市 {len(DEAD)} 檔", flush=True)
    return importlib.import_module(name).main(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
