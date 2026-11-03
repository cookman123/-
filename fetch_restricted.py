"""下載處置、注意、變更交易方法（全額交割）資料，原始 JSON 存到 data/raw/restricted/{source}/，
整理成 data/restricted.parquet（格式與判斷規則見 restricted.py）。

    python fetch_restricted.py --source twse    # 處置 punish + 注意 notice + 變更交易 TWT85U（每月最後交易日的快照）
    python fetch_restricted.py --source tpex    # 處置 disposal + 注意 attention + 變更交易（OpenAPI 目前名單，存當天一份）
    python fetch_restricted.py --build          # 只用已下載的檔案重建 restricted.parquet

取自 jameschen108/tw-stock-quant；交易日曆改用本專案的 data/market/TWII.csv。

已存在的檔案會跳過；涵蓋到今天的那一段每次都重抓。被擋時多半回 HTML，連續失敗 5 次就停，重跑會接續。
上櫃變更交易查不到歷史，只能從開始執行的那天往後累積。
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date

import pandas as pd

import restricted as R

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}
START = "2010-01-01"
TWII = R.ROOT / "data" / "market" / "TWII.csv"


def chunks(freq: str):
    today = pd.Timestamp(date.today())
    for s in pd.date_range(START, today, freq=freq):
        e = min(s + pd.tseries.frequencies.to_offset(freq) - pd.Timedelta(days=1), today)
        yield s, e, e == today

# (名稱, 網址, 分段頻率)；分段太長時交易所會截斷筆數
SOURCES = {
    "twse": [
        ("twse_punish", "https://www.twse.com.tw/rwd/zh/announcement/punish?startDate={s:%Y%m%d}&endDate={e:%Y%m%d}&response=json", "QS"),
        ("twse_notice", "https://www.twse.com.tw/rwd/zh/announcement/notice?startDate={s:%Y%m%d}&endDate={e:%Y%m%d}&response=json", "MS"),
    ],
    "tpex": [
        ("tpex_disposal", "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?startDate={s:%Y}%2F{s:%m}%2F{s:%d}&endDate={e:%Y}%2F{e:%m}%2F{e:%d}&response=json", "QS"),
        ("tpex_attention", "https://www.tpex.org.tw/www/zh-tw/bulletin/attention?startDate={s:%Y}%2F{s:%m}%2F{s:%d}&endDate={e:%Y}%2F{e:%m}%2F{e:%d}&response=json", "MS"),
    ],
}
TWSE_CMODE = "https://www.twse.com.tw/exchangeReport/TWT85U?response=json&date={d:%Y%m%d}"
TPEX_CMODE = "https://www.tpex.org.tw/openapi/v1/tpex_cmode"


class Fetcher:
    def __init__(self, sleep: float):
        self.sleep, self.fails = sleep, 0

    def get(self, url: str, label: str) -> bytes | None:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                raw = r.read()
            json.loads(raw)
        except Exception as ex:
            self.fails += 1
            print(f"{label} 失敗（第 {self.fails} 次）：{type(ex).__name__} {str(ex)[:80]}", flush=True)
            if self.fails >= 5:
                sys.exit("連續失敗 5 次，可能被暫時封鎖；稍後重跑會接續")
            time.sleep(60 * self.fails)
            return None
        self.fails = 0
        time.sleep(self.sleep)
        return raw


def fetch_ranges(f: Fetcher, source: str):
    for name, url_fmt, freq in SOURCES[source]:
        out_dir = R.RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for s, e, is_open in chunks(freq):
            path = out_dir / f"{s:%Y%m%d}_{e:%Y%m%d}.json"
            if path.exists() and not is_open:
                continue
            raw = f.get(url_fmt.format(s=s, e=e), f"{name} {s.date()}")
            if raw is None:
                continue
            for old in out_dir.glob(f"{s:%Y%m%d}_*.json"):          # 開放中的那一段可能每次不同，舊檔先刪掉
                old.unlink()
            path.write_bytes(raw)
            fields, data = R._rows(json.loads(raw))
            print(f"{name} {s.date()}~{e.date()}: {len(data)} 筆", flush=True)
        print(f"{name}: 完成", flush=True)


def month_ends() -> list[pd.Timestamp]:
    """每月最後一個交易日（加權指數有行情的日子）＋最新交易日。"""
    days = pd.DatetimeIndex(sorted(pd.read_csv(TWII, parse_dates=["date"])["date"].unique()))
    days = days[days >= START]
    return sorted(set(pd.Series(days, index=days).groupby(days.to_period("M")).max()) | {days[-1]})


def fetch_twse_cmode(f: Fetcher):
    out_dir = R.RAW / "twse_cmode"
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = month_ends()
    for d in dates:
        path = out_dir / f"{d:%Y%m%d}.json"
        if path.exists() and d != dates[-1]:
            continue
        raw = f.get(TWSE_CMODE.format(d=d), f"twse_cmode {d.date()}")
        if raw is None:
            continue
        path.write_bytes(raw)
        print(f"twse_cmode {d.date()}: {len(json.loads(raw).get('data') or [])} 筆", flush=True)
    print("twse_cmode: 完成", flush=True)


def fetch_tpex_cmode(f: Fetcher):
    out_dir = R.RAW / "tpex_cmode"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = f.get(TPEX_CMODE, "tpex_cmode")
    if raw is None:
        return
    rows = json.loads(raw)
    snap = R.parse_tpex_cmode(rows)
    d = snap["start"].max() if len(snap) else pd.Timestamp(date.today())
    (out_dir / f"{d:%Y%m%d}.json").write_bytes(raw)
    print(f"tpex_cmode {d.date()}: {len(snap)} 筆", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES))
    ap.add_argument("--build", action="store_true", help="只重建 restricted.parquet，不下載")
    ap.add_argument("--sleep", type=float, default=6)
    a = ap.parse_args()
    if not a.build:
        if not a.source:
            ap.error("要指定 --source 或 --build")
        f = Fetcher(a.sleep)
        fetch_ranges(f, a.source)
        (fetch_twse_cmode if a.source == "twse" else fetch_tpex_cmode)(f)
    df = R.build()
    df.to_parquet(R.OUT, index=False)
    print(f"{R.OUT.name}: {len(df)} 筆")
    print(df.groupby(["kind", "market", df["announce_date"].dt.year]).size().unstack("kind").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
