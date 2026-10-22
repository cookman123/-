# -*- coding: utf-8 -*-
"""
沙盒的資料層：把本 repo 的價格檔讀成原始價面板，並提供防偷看未來的資料視圖。

為什麼要還原成原始價
--------------------
Yahoo 存活股的價格是分割調整價：配股、減資往回調整，現金股利不調整
（見 corp_actions.py、adjust_delisted.py）。撮合需要真實能成交的價格：
整股 1000 股的金額、升降單位、漲跌停價都以原始價為準。所以這裡先照
davis_tw_train.load_price 的做法讀出調整價，再用 data/corp_actions/ 的分割比
把事件日之前的價格乘回去，也就是 adjust_delisted.py 的反向操作。
下市股來自證交所 STOCK_DAY，本來就是原始價，不需還原。

本模組不 import 根目錄腳本。與 load_price 重複的那段邏輯，由
tests/sandbox/test_data.py 在本機有 data/ 時逐檔對照。

命令列：python -m sandbox.data [--data 資料夾]
    對全部標的做資料稽核（§2.2 的驗證），輸出有問題的事件。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import tick_size

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("open", "high", "low", "close", "volume")

# 可交易的 ETF，取自 fetch_index_targets.py 的 TARGETS。
# 槓桿、反向型（00631L、00632R）的漲跌幅規則不同，第一版不納入。
ETFS = ("0050", "006208", "0051", "0056", "00878", "00713", "00850",
        "00692", "0052", "0053", "0055", "00881")

# 分割比與 1 的差距小於這個值時視為沒有分割，與 adjust_delisted.py 一致
_SPLIT_EPS = 0.0005


class LookaheadError(Exception):
    """策略查詢了資料視圖日期之後的資料。"""


# ============================== 讀檔 ==============================


def _read_ohlcv(files: list[Path]) -> pd.DataFrame | None:
    """讀一或多個 CSV 並串成 date 索引、已排序去重的 OHLCV。"""
    if not files:
        return None
    px = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()
    for c in FIELDS:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = (px.dropna(subset=["date", "close"])
            .sort_values("date").drop_duplicates("date", keep="last").set_index("date"))
    return px.loc[:, list(FIELDS)]


def _drop_flat(px: pd.DataFrame) -> pd.DataFrame:
    """Yahoo 在停牌／補值日會給 volume=0 且 OHLC 全等的假列（同 load_price）。"""
    flat = (px["volume"] <= 0) & (px["open"] == px["close"]) & (px["high"] == px["low"])
    return px.loc[~flat]


def apply_price_fixes(px: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    """套用 data/price_fixes.csv 中該檔的修正（同 davis_tw_train._apply_price_fixes）。"""
    if fixes.empty:
        return px
    px = px.copy()
    for _, r in fixes.iterrows():
        before = px.index < r["date"]
        px.loc[before, ["open", "high", "low", "close"]] *= r["factor"]
        px.loc[before, "volume"] /= r["factor"]
    return px


def unadjust(px: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    把分割調整價還原成原始價。events：該檔的 (date, s)。

    adjust_delisted.adjust 的反向：事件日之前的價格乘以 s、成交量除以 s。
    只處理落在資料期間內的事件；資料第一天之前的事件沒有「之前」的價格可調，
    最後一天之後的事件 Yahoo 下載當時還沒發生。
    """
    if events.empty or px.empty:
        return px
    lo, hi = px.index.min(), px.index.max()
    fac = pd.Series(1.0, index=px.index)
    for _, e in events.iterrows():
        if lo < e["date"] <= hi:
            fac[px.index < e["date"]] *= e["s"]
    out = px.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = px[c] * fac
    out["volume"] = (px["volume"] / fac).round()
    return out


class RepoData:
    """
    本 repo data/ 資料夾的轉接器。路徑與欄位名稱只出現在這裡（計劃 §1）。

    標的分三類：
        stock     data/stocks/<代號>/price/yahoo_<代號>_<年>.csv（調整價，需還原）
        delisted  data/delisted/price/<代號>.csv（證交所原始價）
        etf       data/market/<代號>.csv，只取 ETFS 名單（調整價，需還原）
    """

    def __init__(self, data_dir: Path | str | None = None):
        self.dir = Path(data_dir) if data_dir else ROOT / "data"
        self._fixes = self._read_fixes()
        self._events = None

    # ---------- 標的清單 ----------

    def codes(self) -> dict[str, str]:
        """{代號: 類別}。同一代號同時出現在存活股與下市股時以存活股為準，並記錄警告。"""
        out: dict[str, str] = {}
        stocks = self.dir / "stocks"
        if stocks.exists():
            for d in sorted(stocks.iterdir()):
                if d.is_dir() and any((d / "price").glob("yahoo_*.csv")):
                    out[d.name] = "stock"
        delisted = self.dir / "delisted" / "price"
        if delisted.exists():
            for f in sorted(delisted.glob("[0-9]*.csv")):
                if f.stem in out:
                    logger.warning("[%s] 同時有存活股與下市股價格檔，採用存活股", f.stem)
                    continue
                out[f.stem] = "delisted"
        for code in ETFS:
            if (self.dir / "market" / f"{code}.csv").exists():
                out[code] = "etf"
        return out

    # ---------- 價格 ----------

    def load_adjusted(self, code: str, kind: str) -> pd.DataFrame | None:
        """還原前的價格。存活股與 ETF 是套用 price_fixes 後的調整價；下市股是原始價。"""
        if kind == "stock":
            files = sorted((self.dir / "stocks" / code / "price").glob("yahoo_*.csv"))
            files = [f for f in files if "merged" not in f.stem]
        elif kind == "etf":
            files = [self.dir / "market" / f"{code}.csv"]
        elif kind == "delisted":
            return _read_ohlcv([self.dir / "delisted" / "price" / f"{code}.csv"])
        else:
            raise ValueError(f"未知的類別：{kind}")
        px = _read_ohlcv([f for f in files if f.exists()])
        if px is None:
            return None
        px = _drop_flat(px)
        return apply_price_fixes(px, self._fixes[self._fixes["code"] == code])

    def load(self, code: str, kind: str) -> pd.DataFrame | None:
        """原始價 OHLCV。"""
        px = self.load_adjusted(code, kind)
        if px is None or kind == "delisted":
            return px
        ev = self.split_events()
        return unadjust(px, ev[ev["code"] == code])

    # ---------- 公司行動與日曆 ----------

    def exright(self) -> pd.DataFrame:
        return self._read_ca("exright.csv")

    def reduction(self) -> pd.DataFrame:
        return self._read_ca("reduction.csv")

    def split_events(self) -> pd.DataFrame:
        """(code, date, s, kind)，定義與 adjust_delisted.split_events 相同。"""
        if self._events is not None:
            return self._events
        parts = []
        e = self.exright()
        if not e.empty:
            e = e[e["split"] > 1 + _SPLIT_EPS]
            parts.append(pd.DataFrame({"code": e["code"], "date": e["date"],
                                       "s": e["split"], "kind": "除權"}))
        r = self.reduction()
        if not r.empty:
            r = r[(r["prev_close"] > 0) & (r["ref"] > 0)]
            parts.append(pd.DataFrame({"code": r["code"], "date": r["date"],
                                       "s": r["prev_close"] / r["ref"], "kind": "減資"}))
        self._events = (pd.concat(parts, ignore_index=True) if parts
                        else pd.DataFrame(columns=["code", "date", "s", "kind"]))
        return self._events

    def calendar(self) -> pd.DatetimeIndex:
        """交易日曆：data/market/TWII.csv 的日期（本 repo 沒有獨立的日曆檔）。"""
        f = self.dir / "market" / "TWII.csv"
        if not f.exists():
            raise FileNotFoundError(f"找不到交易日曆來源 {f}；請先執行 tw_market_index.py")
        d = pd.read_csv(f, encoding="utf-8-sig", usecols=["date"])
        return pd.DatetimeIndex(pd.to_datetime(d["date"]).dt.normalize().unique()).sort_values()

    # ---------- 內部 ----------

    def _read_fixes(self) -> pd.DataFrame:
        f = self.dir / "price_fixes.csv"
        if not f.exists():
            return pd.DataFrame(columns=["code", "date", "factor"])
        return pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str}, parse_dates=["date"])

    def _read_ca(self, name: str) -> pd.DataFrame:
        f = self.dir / "corp_actions" / name
        if not f.exists():
            return pd.DataFrame(columns=["code", "date"])
        return pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str}, parse_dates=["date"])


# ============================== 稽核 ==============================


def audit(data: RepoData, codes: dict[str, str] | None = None) -> pd.DataFrame:
    """
    還原後的原始價稽核（計劃 §2.2）。回傳有問題的列：

        prev_close  除權息前一日收盤與 exright.csv 的 prev_close（證交所原始價）
                    相差超過一個升降單位
        jump        非事件日的收盤漲跌超過 ±10.5%，可能是沒記錄到的分割或錯價
    """
    codes = codes if codes is not None else data.codes()
    ex = data.exright()
    ev = data.split_events()
    rows = []
    for code, kind in codes.items():
        px = data.load(code, kind)
        if px is None or len(px) < 2:
            continue
        etf = kind == "etf"
        for _, e in ex[ex["code"] == code].iterrows():
            i = px.index.searchsorted(e["date"])
            if not 0 < i < len(px) or not e["prev_close"] > 0:
                continue
            got = px["close"].iloc[i - 1]
            if abs(got - e["prev_close"]) > tick_size(e["prev_close"], etf) + 1e-9:
                rows.append((code, kind, e["date"], "prev_close", got, e["prev_close"]))
        ret = px["close"].pct_change()
        event_days = set(ev.loc[ev["code"] == code, "date"]) | set(ex.loc[ex["code"] == code, "date"])
        for d, r in ret[ret.abs() > 0.105].items():
            if d not in event_days:
                rows.append((code, kind, d, "jump", r, np.nan))
    return pd.DataFrame(rows, columns=["code", "kind", "date", "check", "got", "expected"])


# ============================== 面板與資料視圖 ==============================


class Panel:
    """原始價寬表：每個欄位一張 日期 × 代號 的表，索引為交易日曆。"""

    def __init__(self, fields: dict[str, pd.DataFrame], kinds: dict[str, str]):
        self.fields = fields
        self.kinds = kinds
        self.calendar = fields["close"].index

    @classmethod
    def build(cls, data: RepoData, start: str, end: str,
              codes: list[str] | None = None) -> "Panel":
        cal = data.calendar()
        cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
        allk = data.codes()
        kinds = {c: allk[c] for c in (codes if codes is not None else allk)}
        cols: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
        for code, kind in kinds.items():
            px = data.load(code, kind)
            if px is None:
                continue
            off = px.index[(px.index >= cal.min()) & (px.index <= cal.max())].difference(cal)
            if len(off):
                logger.warning("[%s] %d 筆價格不在交易日曆上，已略過", code, len(off))
            px = px.reindex(cal)
            for f in FIELDS:
                cols[f][code] = px[f]
        fields = {f: pd.DataFrame(cols[f], index=cal) for f in FIELDS}
        return cls(fields, {c: k for c, k in kinds.items() if c in fields["close"]})


class DataView:
    """
    策略在 asof 日收盤後看到的唯讀資料（計劃 §3）。

    查詢 asof 之後的資料直接報 LookaheadError，不回傳空值：
    回傳空值的話，偷看未來的程式只會安靜地少一段資料，不會被發現。
    回傳的都是複本，策略改了也不影響沙盒。
    """

    def __init__(self, panel: Panel, asof: pd.Timestamp):
        asof = pd.Timestamp(asof)
        if asof not in panel.calendar:
            raise ValueError(f"{asof.date()} 不是交易日")
        self._panel = panel
        self._asof = asof

    @property
    def asof(self) -> pd.Timestamp:
        return self._asof

    def calendar(self) -> pd.DatetimeIndex:
        cal = self._panel.calendar
        return cal[cal <= self._asof]

    def field(self, name: str, start=None, end=None) -> pd.DataFrame:
        """日期 × 代號 的寬表，期間預設到 asof 為止。"""
        if name not in self._panel.fields:
            raise KeyError(f"沒有欄位 {name}；可用：{', '.join(FIELDS)}")
        end = self._asof if end is None else pd.Timestamp(end)
        if end > self._asof:
            raise LookaheadError(f"查詢到 {end.date()}，但資料視圖只到 {self._asof.date()}")
        df = self._panel.fields[name]
        lo = df.index.min() if start is None else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= end)].copy()

    def universe(self) -> list[str]:
        """asof 當天有收盤價的標的。之後才上市或已下市的不在裡面。"""
        row = self._panel.fields["close"].loc[self._asof]
        return sorted(row.index[row.notna()])

    def kind(self, code: str) -> str:
        return self._panel.kinds[code]


# ============================== 命令列 ==============================


def main(argv=None) -> int:
    import argparse

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="沙盒資料稽核：原始價還原是否與證交所一致")
    ap.add_argument("--data", default=None, help="資料夾（預設 repo 的 data/）")
    ap.add_argument("--out", default=None, help="問題清單輸出 CSV")
    a = ap.parse_args(argv)

    data = RepoData(a.data)
    codes = data.codes()
    if not codes:
        print(f"{data.dir} 底下沒有任何價格檔；請先照 README 建好資料")
        return 1
    print(f"稽核 {len(codes)} 檔（"
          + "、".join(f"{k} {sum(v == k for v in codes.values())}" for k in ("stock", "delisted", "etf"))
          + "）…")
    bad = audit(data, codes)
    n_ex = int(data.exright()["code"].isin(codes.keys()).sum())
    print(f"除權息事件 {n_ex} 筆，前日收盤不符 {int((bad['check'] == 'prev_close').sum())} 筆；"
          f"非事件日漲跌超過 ±10.5% {int((bad['check'] == 'jump').sum())} 筆")
    if a.out:
        bad.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"輸出：{a.out}")
    elif len(bad):
        print(bad.head(30).to_string(index=False))
    return 0 if bad.empty else 2


if __name__ == "__main__":
    sys.exit(main())
