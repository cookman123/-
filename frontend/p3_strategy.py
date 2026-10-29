"""策略：高波動＋趨勢過濾的規格回測、beta 分解、隔日預測、105 個價量特徵的季頻檢定。"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common


def _spec(sig: str) -> None:
    s = common.result_csv(sig, "spec/summary.csv")
    if s.empty:
        common.empty("還沒有 results/spec/summary.csv，先跑 python strategy_spec.py。")
        return
    st.caption("每 42 個交易日在母體（60 日均成交金額前 250 名）取近 60 日波動最高的 30 檔；舊持股掉出前 40 名才賣、"
               "續抱的不拉回等權；大盤收在 120 日線之上才持有。舊規則（63 天、每次重設等權）列在最後兩列對照。"
               "手續費 3 折、證交稅 0.3%。大盤是價格指數，不含股利。")
    fmt = {"年化": "{:+.2%}", "夏普": "{:.2f}", "最大回撤": "{:.1%}", "報酬回撤比": "{:.2f}", "年數": "{:.1f}",
           "超額": "{:+.2%}", "超額t": "{:.2f}", "超額p": "{:.3f}"}
    common.table(common.styled(s, fmt=fmt, signed=["年化", "超額"]), hide_index=True)
    y = common.result_csv(sig, "spec/yearly.csv", index_col=0)
    if y.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=y.index, y=y["策略"] * 100, name="高波動＋過濾", marker_color=common.C_STRAT))
    fig.add_trace(go.Bar(x=y.index, y=y["大盤"] * 100, name="大盤", marker_color=common.C_POOL))
    fig.update_yaxes(ticksuffix="%")
    common.chart(common.fig_layout(fig, f"逐年報酬（勝過大盤 {int((y['差'] > 0).sum())}/{len(y)} 年）", 340))
    st.caption("趨勢過濾把最大回撤從 −74% 降到 −44%，但夏普沒有變高，對大盤的超額也不再顯著（t 1.51，規格文件第四、八節）。")


def _alpha(sig: str) -> None:
    a = common.result_csv(sig, "quarterly/alpha.csv")
    if a.empty:
        common.empty("還沒有 results/quarterly/alpha.csv，先跑 python quarterly_alpha.py。")
        return
    st.caption("季頻版（2011 起，62 季）。重點是最後三列：沖掉 beta 之後不顯著，對同波動的槓桿大盤只差 −1.4%。")
    fmt = {"年化": "{:+.2%}", "夏普": "{:.2f}", "季均": "{:+.2%}", "t": "{:.2f}", "p": "{:.3f}", "季數": "{:.0f}"}
    common.table(common.styled(a, fmt=fmt, signed=["年化", "季均"]), hide_index=True)


def _nextday(sig: str) -> None:
    surv = st.toggle("含下市股（稽核後的主要版本）", value=True)
    c = common.result_csv(sig, f"nextday/cumulative{'_surv' if surv else ''}.csv", index_col=0, parse_dates=True)
    if c.empty:
        common.empty("還沒有 results/nextday/cumulative*.csv，先跑 python cumulative_returns.py。")
        return
    colors = dict(zip(c.columns, [common.C_STRAT, common.UP, common.C_POOL, common.C_ETF]))
    common.chart(common.line_fig(c.index, {k: c[k] for k in c.columns},
                                 f"5 日標籤＋緩衝帶的累積淨值（扣成本，{'含' if surv else '只有存活股，不含'}下市股）",
                                 colors=colors, height=380))
    st.caption("累積倍數大，主要是 beta 1.1～1.24 加上 2023 年後的行情。稽核後含下市股、對沖大盤之後："
               "10 檔 +10.33%（t 1.65）、30 檔 +7.85%（t 1.97），12 組一起檢定 RC p 0.087、SPA p 0.087，不通過（結論 14.5、14.7）。")


def _factors(sig: str) -> None:
    f = common.result_csv(sig, "fundamental/factors_with_price_extra.csv")
    if f.empty:
        f = common.result_csv(sig, "fundamental/factors_with_price.csv")
    if f.empty:
        common.empty("還沒有財報＋價量的季頻檢定，先跑 python fundamental_factors.py --with-price --extra。")
        return
    c1, c2 = st.columns([1, 1])
    fam = c1.multiselect("特徵族", ["fin_", "ts_", "rel_", "liq_", "trd_", "ovn_", "cs_"],
                         default=["fin_", "ts_", "rel_", "liq_", "trd_", "ovn_", "cs_"])
    only = c2.toggle("只看通過 FDR 的", value=False)
    d = f[f["特徵"].str.startswith(tuple(fam))] if fam else f.iloc[0:0]
    if only:
        d = d[d["FDR通過"]]
    d = d.sort_values("IC_t", key=abs, ascending=False)
    fmt = {"IC平均": "{:+.4f}", "IC_t": "{:.2f}", "IC_p": "{:.4f}", "多空年化_扣成本": "{:+.2%}", "勝率": "{:.0%}",
           "前十分位對母體_季均": "{:+.2%}"}
    common.table(common.styled(d, fmt=fmt, signed=["IC平均", "IC_t", "多空年化_扣成本", "前十分位對母體_季均"]),
                 hide_index=True, height=440)
    st.caption(f"{len(f)} 個特徵一起做 BH FDR（q = 0.10）。季末訊號、隔日開盤進出、前 250 名母體、方向依 IC 符號。"
               "IC 顯著但多空扣成本後為負的，是排序相關而不是可交易的訊號。")


def main():
    sig = common.data_sig()
    common.page_title("策略", "回測與檢定結果（由各腳本產生，這頁只讀）")
    t1, t2, t3, t4 = st.tabs(["高波動＋趨勢過濾", "beta 分解", "隔日預測", "季頻特徵檢定"])
    with t1:
        _spec(sig)
    with t2:
        _alpha(sig)
    with t3:
        _nextday(sig)
    with t4:
        _factors(sig)


main()
