"""營收研究：月營收公告的事件研究（四種基準）、財報因子、中性化驗證。"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common

BASES = ["母體", "成交值×波動", "產業", "大盤"]
BASE_COLOR = dict(zip(BASES, [common.C_POOL, common.C_STRAT, common.UP, common.C_ETF]))


def _event(sig: str) -> None:
    b = common.result_csv(sig, "revenue_event/bench.csv")
    if b.empty:
        common.empty("還沒有 results/revenue_event/bench.csv，先跑 python revenue_event_study.py。")
        return
    st.caption("事件日 = 營收可用日（次月 10 日），隔一個交易日開盤進場，持有 h 個交易日後開盤出場；只看進場當天在母體內的股票。"
               "t 值：事件依進場月份平均後做 Newey-West。做法取自 tw-stock-quant 的 event_study.py。")
    evs = list(dict.fromkeys(b["事件"]))
    c1, c2 = st.columns([2, 1])
    ev = c1.selectbox("事件", evs, index=evs.index("創新高 且 年增≥30%") if "創新高 且 年增≥30%" in evs else 0)
    with c2:
        h = common.segmented("持有（交易日）", [5, 20, 60], 60, "ev_h")
    d = b[(b["事件"] == ev) & (b["持有"] == h)]

    full = d[d["區間"] == "全期"].set_index("基準")
    common.kpis([(f"對{k}", common.signed_html(full.at[k, "平均超額"], 2) if k in full.index else "—",
                  f"t {common.num(full.at[k, 't'])}・勝率 {common.num(full.at[k, '勝率'] * 100, 0)}%"
                  if k in full.index else "", k == "成交值×波動") for k in BASES])

    fig = go.Figure()
    periods = list(dict.fromkeys(d["區間"]))
    for k in BASES:
        x = d[d["基準"] == k].set_index("區間").reindex(periods)
        fig.add_trace(go.Bar(x=periods, y=x["平均超額"] * 100, name=k, marker_color=BASE_COLOR[k],
                             customdata=x["t"], hovertemplate="%{y:+.2f}%（t %{customdata:.2f}）"))
    fig.update_yaxes(ticksuffix="%")
    common.chart(common.fig_layout(fig, f"{ev}：持有 {h} 日的平均超額，依區間與基準", 360))

    show = d[["區間", "基準", "事件數", "月數", "平均超額", "中位超額", "扣成本", "勝率", "t"]]
    fmt = {"事件數": "{:,.0f}", "月數": "{:.0f}", "平均超額": "{:+.2%}", "中位超額": "{:+.2%}", "扣成本": "{:+.2%}",
           "勝率": "{:.0%}", "t": "{:.2f}"}
    common.table(common.styled(show, fmt=fmt, signed=["平均超額", "中位超額", "扣成本", "t"]), hide_index=True)

    y = common.result_csv(sig, "revenue_event/yearly.csv")
    if not y.empty:
        w = y.pivot(index="年", columns="基準", values="平均超額")
        n = y[y["基準"] == "母體"].set_index("年")["事件數"]
        fig = go.Figure()
        for k in BASES:
            if k in w:
                fig.add_trace(go.Scatter(x=w.index, y=w[k] * 100, name=k, mode="lines+markers",
                                         line=dict(color=BASE_COLOR[k], width=1.6)))
        fig.add_trace(go.Bar(x=n.index, y=n, name="事件數（右軸）", marker_color="rgba(128,128,128,.25)", yaxis="y2"))
        fig.update_layout(yaxis=dict(ticksuffix="%"), yaxis2=dict(overlaying="y", side="right", showgrid=False))
        common.chart(common.fig_layout(fig, "逐年：創新高且年增≥30%，60 日平均超額", 340))
    st.caption("要注意：2024 年以後對母體的超額大多是小型、高波動股一起漲（換成配對基準就小很多）；"
               "中位數多半是負的、勝率不到一半；2012–2013 的「歷史新高」只跟兩三年比。詳見結論 15.5。")


def _factors(sig: str) -> None:
    f = common.result_csv(sig, "fundamental/factors_with_price_extra.csv")
    if f.empty:
        f = common.result_csv(sig, "fundamental/factors_with_price.csv")
    if f.empty:
        common.empty("還沒有財報因子檢定，先跑 python fundamental_factors.py --with-price --extra。")
        return
    d = f[f["特徵"].str.startswith("fin_")].sort_values("IC_t", key=abs, ascending=False)
    fmt = {"IC平均": "{:+.4f}", "IC_t": "{:.2f}", "IC_p": "{:.4f}", "多空年化_扣成本": "{:+.2%}", "勝率": "{:.0%}",
           "前十分位對母體_季均": "{:+.2%}"}
    common.table(common.styled(d, fmt=fmt, signed=["IC平均", "IC_t", "多空年化_扣成本", "前十分位對母體_季均"]),
                 hide_index=True)
    st.caption(f"FDR 是和全部 {len(f)} 個特徵（含 105 個價量特徵）一起做的。fin_rev_vs_high、fin_rev_yoy12、"
               "fin_asset_growth 取自 tw-stock-quant 的因子定義（--extra）。")


def _verify(sig: str) -> None:
    p = common.RES / "fundamental" / "verify_neutral.log"
    if not p.exists():
        common.empty("還沒有 results/fundamental/verify_neutral.log，先跑 python fundamental_verify.py。")
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    parts = text.split("=" * 96)   # 開頭說明、訊號名、內容、訊號名、內容…
    names = [x.strip() for x in parts[1::2]]
    bodies = parts[2::2]
    if not names:
        st.code(text, language="text")
        return
    pick = common.segmented("訊號", names, names[0], "verify_sig")
    st.code(bodies[names.index(pick)].strip("\n"), language="text")
    st.caption("第 5 步的中性化做法取自 tw-stock-quant 的 backtest.neutralize；本專案沒有時點正確的市值，用 60 日均成交金額代替。")


def main():
    sig = common.data_sig()
    common.page_title("營收研究", "唯一不是 beta 的候選：事件研究、財報因子、五道驗證")
    t1, t2, t3 = st.tabs(["事件研究", "財報因子", "五道驗證"])
    with t1:
        _event(sig)
    with t2:
        _factors(sig)
    with t3:
        _verify(sig)


main()
