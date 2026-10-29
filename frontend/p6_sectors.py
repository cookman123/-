"""產業：各產業的強弱排行、產業指數走勢、成分股。

版面與指標取自 jameschen108/tw-stock-quant 的主題頁（p6_themes.py、themes.py）。
他的主題成分大多不在本研究的 383 檔裡，所以改用 industry_stock_map.json 的產業分類：
分類是現成的、不是看過漲跌才挑的，沒有手挑主題的事後偏差，但用的是現行分類。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common

SINCE = "2025-06-01"          # 最長看 120 日，只讀近一年多的行情
HORIZONS = (1, 5, 20, 60)
WINDOWS = {"20 日": 20, "60 日": 60, "120 日": 120}
PALETTE = ["#3b82f6", "#e5484d", "#18a058", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#64748b",
           "#84cc16", "#f97316", "#14b8a6"]


def window_return(daily: pd.Series, n: int) -> float:
    """最近 n 個交易日的複利報酬。"""
    w = daily.dropna().iloc[-n:]
    return float((1 + w).prod() - 1) if len(w) == n else np.nan


def index_levels(daily: pd.Series, n: int) -> pd.Series:
    """最近 n 個交易日的累積指數。"""
    return (1 + daily.dropna().iloc[-n:]).cumprod()


def summarize(groups: dict[str, list[str]], rets: pd.DataFrame, val: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    mkt_val = val.sum(axis=1)
    rows = []
    for name, codes in groups.items():
        have = [c for c in codes if c in rets.columns and rets[c].notna().iloc[-20:].any()]
        d = rets[have].mean(axis=1, skipna=True)        # 等權、每天再平衡
        row = {"產業": name, "成分數": len(have)}
        for h in HORIZONS:
            row[f"{h}日"] = window_return(d, h)
        for h in (20, 60):
            row[f"超額{h}日"] = row[f"{h}日"] - window_return(bench, h)
        r20 = rets[have].iloc[-20:].add(1).prod() - 1
        row["20日上漲比例"] = float((r20 > 0).mean())
        share = val[have].sum(axis=1) / mkt_val
        row["成交值占比"] = float(share.iloc[-5:].mean())
        row["成交值倍數"] = float(share.iloc[-5:].mean() / share.iloc[-60:].mean()) if share.iloc[-60:].mean() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sig = common.data_sig()
    common.page_title("產業", "依產業分類的等權指數：強弱排行與資金輪動")
    nm = common.names(sig)
    rets, val = common.all_prices(sig, SINCE)
    tw = common.twii(sig)
    bench = tw.pct_change().reindex(rets.index)
    groups = {k: list(v.index) for k, v in nm.groupby("category")}
    st.caption(f"行情到 {rets.index.max():%Y-%m-%d}，383 檔分成 {len(groups)} 個產業。超額 = 產業報酬 − 加權指數（價格指數）；"
               "成交值倍數 = 近 5 日 / 近 60 日平均占這 383 檔的比重，大於 1 代表資金在流入。")

    s = summarize(groups, rets, val, bench)
    sort_by = common.segmented("排序", ["20日", "5日", "60日", "超額20日", "成交值倍數"], "20日", "sector_sort",
                               label_visibility="collapsed")
    s = s.sort_values(sort_by, ascending=False).reset_index(drop=True)
    ret_cols = ["1日", "5日", "20日", "60日", "超額20日", "超額60日"]
    ev = common.table(common.styled(s, fmt={**{c: "{:+.1%}" for c in ret_cols}, "20日上漲比例": "{:.0%}",
                                            "成交值占比": "{:.1%}", "成交值倍數": "{:.2f}x"}, signed=ret_cols),
                      hide_index=True, height=min(450, 38 + 35 * len(s)), on_select="rerun",
                      selection_mode="single-row", key="sector_tbl")
    sel = ev.selection.rows if ev and ev.selection else []
    if not sel:
        st.caption("點一列看成分股。")

    n = WINDOWS[common.segmented("走勢區間", list(WINDOWS), "60 日", "sector_win")]
    chosen = st.multiselect("走勢圖產業", list(s["產業"]), default=list(s["產業"].head(3)), key="sector_pick")
    series = {c: index_levels(rets[groups[c]].mean(axis=1, skipna=True), n) for c in chosen}
    series["加權指數"] = index_levels(bench, n)
    if chosen:
        xs = series[chosen[0]].index
        colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(chosen)} | {"加權指數": common.C_ETF}
        common.chart(common.line_fig(xs, {k: v.reindex(xs) for k, v in series.items()},
                                     f"累積指數（起點 = 1，近 {n} 日）", colors, 380))

    if sel:
        pick = s.iloc[sel[0]]["產業"]
        st.subheader(pick)
        m = pd.DataFrame([{"代號": c, "名稱": nm.at[c, "name"],
                           **{f"{h}日": window_return(rets[c], h) if c in rets else np.nan for h in HORIZONS}}
                          for c in groups[pick]]).sort_values("20日", ascending=False).reset_index(drop=True)
        ev2 = common.table(common.styled(m, fmt={c: "{:+.1%}" for c in ["1日", "5日", "20日", "60日"]},
                                         signed=["1日", "5日", "20日", "60日"]),
                           hide_index=True, on_select="rerun", selection_mode="single-row", key="sector_members")
        common.pick_row(ev2, m)

    st.markdown(f'<div class="disc">{common.esc(common.DISCLAIMER)} 研究裡用新聞主題熱度預測產業報酬沒有通過檢定'
                "（結論第三節），這頁只看當下的相對強弱，不是訊號。</div>", unsafe_allow_html=True)


main()
