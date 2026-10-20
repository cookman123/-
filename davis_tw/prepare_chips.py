# -*- coding: utf-8 -*-
"""
把證交所 T86（三大法人買賣超日報）的逐日 JSON 轉成單股籌碼面 CSV。

輸入：data/twse/twse/t86/t86_YYYYMMDD.json
        每檔是「某一交易日、全市場所有個股」的一張表
輸出：data/chips/{code}.csv
        欄位 date, foreign_investor, investment_trust, dealer,
             dealer_self, dealer_hedge, inst_total（單位：股）

用法：
    python -m davis_tw.prepare_chips              # 全部股票
    python -m davis_tw.prepare_chips 2330 1101    # 只轉指定股票

備註：
  - 欄位以 JSON 內的 fields 名稱動態對應，不用固定索引，
    避免證交所改版欄位順序造成錯位。
  - 非交易日的檔案 stat 不是 OK，直接略過。
  - 買賣超為「當日盤後」公開資訊，用於預測 t+1 無前視問題。
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config as C

logger = logging.getLogger(__name__)

T86_DIR = C.DATA_DIR / "twse" / "twse" / "t86"

# 證交所欄位名稱 → 專案統一欄名
FIELD_MAP = {
    "外資買賣超股數": "foreign_investor",
    "外陸資買賣超股數(不含外資自營商)": "foreign_investor",
    "投信買賣超股數": "investment_trust",
    "自營商買賣超股數": "dealer",
    "自營商買賣超股數(自行買賣)": "dealer_self",
    "自營商買賣超股數(避險)": "dealer_hedge",
    "三大法人買賣超股數": "inst_total",
}


def _to_number(x: str) -> float:
    s = str(x).replace(",", "").replace("--", "").strip()
    if s in ("", "-", "X"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_t86_dir(t86_dir: Path = T86_DIR, codes: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """掃過所有日檔，累積成 {股票代號: DataFrame}。"""
    files = sorted(t86_dir.glob("t86_*.json"))
    if not files:
        raise FileNotFoundError(f"找不到 T86 檔案：{t86_dir}")

    rows: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("讀取失敗 %s：%s", f.name, exc)
            continue
        if payload.get("stat") != "OK":
            skipped += 1
            continue

        date = pd.to_datetime(payload.get("date") or f.stem.split("_")[-1], format="%Y%m%d")
        fields = payload.get("fields") or []
        # 欄位索引：同名欄位取第一個（例如新舊版外資欄名並存時）
        idx_map = {}
        for i, name in enumerate(fields):
            std = FIELD_MAP.get(name.strip())
            if std and std not in idx_map:
                idx_map[std] = i
        if not idx_map:
            logger.warning("%s 無可辨識欄位：%s", f.name, fields)
            continue

        for row in payload.get("data", []):
            code = str(row[0]).strip()
            if codes is not None and code not in codes:
                continue
            rec = {"date": date}
            for std, i in idx_map.items():
                rec[std] = _to_number(row[i]) if i < len(row) else float("nan")
            rows[code].append(rec)

    logger.info("T86 解析完成：%d 個交易日檔、%d 檔股票（略過非交易日 %d 檔）",
                len(files) - skipped, len(rows), skipped)
    return {code: pd.DataFrame(recs).sort_values("date") for code, recs in rows.items()}


def write_chip_csvs(out_dir: Path = C.CHIPS_DIR, codes=None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_set = set(codes) if codes else None
    tables = parse_t86_dir(codes=codes_set)

    written = []
    for code, df in tables.items():
        if len(df) < 100:      # 資料過短的（下市／新上市）不輸出
            continue
        path = out_dir / f"{code}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    logger.info("已輸出 %d 檔籌碼面 CSV → %s", len(written), out_dir)
    return written


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])
    paths = write_chip_csvs(codes=argv or None)
    if paths:
        sample = pd.read_csv(paths[0], encoding="utf-8-sig")
        print(f"\n範例 {paths[0].name}：{len(sample)} 列 "
              f"{sample['date'].min()} ~ {sample['date'].max()}")
        print(sample.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
