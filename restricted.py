"""處置股、注意股、全額交割（變更交易方法）股：整理交易所原始資料，判斷某訊號日能不能進場。

取自 jameschen108/tw-stock-quant 的 restricted.py（判斷規則不變）。放在量化端而不是交給地端 LLM：
「能不能成交」是交易所公告的硬條件，用規則查表就有確定答案，回測也能一起套用；
讓 LLM 從新聞去猜一檔股票是不是處置股，只會多一個出錯的地方。

資料由 fetch_restricted.py 下載到 data/raw/restricted/{source}/，整理成 data/restricted.parquet：
  kind = disposal       處置，announce_date 公布日、start / end 處置起迄
  kind = attention      注意，announce_date 公布日（start = end = 公布日）
  kind = full_delivery  變更交易方法（全額交割），交易所某一天的名單快照，announce_date = start = end = 快照日

訊號日 s、成交日 t = s 之後第 lag 個交易日，下列任一成立就不買（已持有的不受影響）:
  處置      公布日 <= s，且處置期間涵蓋 t（處置多半在公布隔日生效，訊號日收盤後已經知道）
  注意      公布日 = s（只看訊號日當天）
  全額交割  s 當天或之前最近一次快照（不超過 SNAP_MAX_AGE 天）有列入；上市、上櫃各自取快照

涵蓋範圍: 處置、注意 2010 年起；全額交割上市 2010 年起（每月最後交易日的快照），
上櫃只有交易所目前名單，fetch_restricted.py 每次執行存一份，往後才累積，過去沒有。
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "restricted"
OUT = ROOT / "data" / "restricted.parquet"
SNAP_MAX_AGE = 45
COLS = ["code", "market", "kind", "announce_date", "start", "end"]

_ROC = re.compile(r"(\d{2,3})[/.](\d{1,2})[/.](\d{1,2})")


def roc_dates(s) -> list[pd.Timestamp]:
    """字串裡所有民國日期（101/01/13、101.01.05、1151030 不算），依出現順序。"""
    return [pd.Timestamp(int(y) + 1911, int(m), int(d)) for y, m, d in _ROC.findall(str(s or ""))]


def _rows(js: dict) -> tuple[list, list]:
    if "tables" in js:
        t = js["tables"][0] if js["tables"] else {}
        return t.get("fields") or [], t.get("data") or []
    return js.get("fields") or [], js.get("data") or []


def _col(fields: list, *names: str) -> int:
    for n in names:
        if n in fields:
            return fields.index(n)
    raise KeyError(f"找不到欄位 {names}：{fields}")


def parse_disposal(js: dict, market: str) -> pd.DataFrame:
    fields, data = _rows(js)
    if not data:
        return pd.DataFrame(columns=COLS)
    i_code, i_ann = _col(fields, "證券代號"), _col(fields, "公布日期")
    i_per = _col(fields, "處置起迄時間", "處置起訖時間")
    out = []
    for r in data:
        code = str(r[i_code]).strip()
        ann, per = roc_dates(r[i_ann]), roc_dates(r[i_per])
        if not code or not ann or len(per) < 2:          # 上櫃「本日無處置資料」的空列
            continue
        out.append((code, market, "disposal", ann[0], per[0], per[-1]))
    return pd.DataFrame(out, columns=COLS)


def parse_attention(js: dict, market: str) -> pd.DataFrame:
    fields, data = _rows(js)
    if not data:
        return pd.DataFrame(columns=COLS)
    i_code, i_date = _col(fields, "證券代號"), _col(fields, "日期", "公告日期")
    out = []
    for r in data:
        code = str(r[i_code]).strip()
        d = roc_dates(r[i_date])
        if not code or not d:
            continue
        out.append((code, market, "attention", d[0], d[0], d[0]))
    return pd.DataFrame(out, columns=COLS)


def parse_twse_cmode(js: dict) -> pd.DataFrame:
    """上市變更交易名單（TWT85U，某一天的快照）。名單空的視為缺資料，不當成「沒有全額交割股」。"""
    data = js.get("data") or []
    if not data or "date" not in js:
        return pd.DataFrame(columns=COLS)
    d = pd.Timestamp(js["date"])
    return pd.DataFrame([(str(r[0]).strip(), "TWSE", "full_delivery", d, d, d) for r in data], columns=COLS)


def parse_tpex_cmode(rows: list) -> pd.DataFrame:
    """上櫃變更交易名單（OpenAPI tpex_cmode，目前名單；Date 為民國 1151030 格式）。"""
    out = []
    for r in rows:
        if str(r.get("AlteredTrading", "")).strip() not in ("Y", "Ｙ"):
            continue
        raw = str(r.get("Date", "")).strip()
        if len(raw) < 7:
            continue
        d = pd.Timestamp(int(raw[:-4]) + 1911, int(raw[-4:-2]), int(raw[-2:]))
        out.append((str(r["SecuritiesCompanyCode"]).strip(), "TPEx", "full_delivery", d, d, d))
    return pd.DataFrame(out, columns=COLS)


PARSERS = {
    "twse_punish": lambda js: parse_disposal(js, "TWSE"),
    "twse_notice": lambda js: parse_attention(js, "TWSE"),
    "twse_cmode": parse_twse_cmode,
    "tpex_disposal": lambda js: parse_disposal(js, "TPEx"),
    "tpex_attention": lambda js: parse_attention(js, "TPEx"),
    "tpex_cmode": parse_tpex_cmode,
}


def build(raw: Path = RAW) -> pd.DataFrame:
    parts = []
    for name, parse in PARSERS.items():
        for f in sorted((raw / name).glob("*.json")):
            parts.append(parse(json.loads(f.read_bytes())))
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    # 上櫃處置用期間查詢，相鄰兩段會重複；注意股同一天多個款別也是多列
    return df.drop_duplicates().sort_values(["kind", "announce_date", "code"]).reset_index(drop=True)


def load(path: Path = OUT) -> pd.DataFrame:
    if not path.exists():
        print(f"警告：沒有 {path.name}（先跑 fetch_restricted.py），處置／注意／全額交割股不會排除")
        return pd.DataFrame(columns=COLS)
    return pd.read_parquet(path)


def trade_dates_after(signal: pd.Series, days: pd.DatetimeIndex, lag: int = 1) -> pd.Series:
    """每個訊號日之後第 lag 個交易日；超出資料範圍（最新訊號日）用平日推估。"""
    sig = pd.DatetimeIndex(signal.unique())
    pos = days.searchsorted(sig, side="right") + lag - 1
    t = [days[p] if p < len(days) else s + pd.offsets.BDay(lag) for s, p in zip(sig, pos)]
    return signal.map(dict(zip(sig, t)))


def flags(keys: pd.DataFrame, days: pd.DatetimeIndex, lag: int = 1, ev: pd.DataFrame | None = None) -> pd.Series:
    """keys 需有 date（訊號日）、code；回傳和 keys 同索引的布林值，True = 這天不能進場。"""
    ev = load() if ev is None else ev
    out = pd.Series(False, index=keys.index)
    if ev.empty or keys.empty:
        return out
    k = keys[["date", "code"]].assign(t=trade_dates_after(keys["date"], pd.DatetimeIndex(days), lag))
    k = k.reset_index(names="_i")

    dsp = k.merge(ev[ev["kind"] == "disposal"], on="code")
    hit = dsp[(dsp["announce_date"] <= dsp["date"]) & (dsp["start"] <= dsp["t"]) & (dsp["end"] >= dsp["t"])]
    out[hit["_i"].unique()] = True

    att = ev.loc[ev["kind"] == "attention", ["code", "announce_date"]].drop_duplicates()
    hit = k.merge(att, left_on=["code", "date"], right_on=["code", "announce_date"])
    out[hit["_i"].unique()] = True

    snap = ev.loc[ev["kind"] == "full_delivery", ["market", "code", "start"]]
    for mkt, g in snap.groupby("market"):
        sd = pd.DatetimeIndex(sorted(g["start"].unique()))
        sig = pd.DatetimeIndex(k["date"].unique())
        p = sd.searchsorted(sig, side="right") - 1
        use = {s: sd[i] for s, i in zip(sig, p) if i >= 0 and (s - sd[i]).days <= SNAP_MAX_AGE}
        lists = g.groupby("start")["code"].apply(set)
        m = k["date"].map(use)
        inside = [c in lists[d] if pd.notna(d) else False for c, d in zip(k["code"], m)]
        out[k.loc[inside, "_i"].unique()] = True
    return out
