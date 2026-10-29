"""個股：價格、月營收、季財報、法人、營收事件紀錄。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common

SECTIONS = ["價格", "月營收", "季財報", "法人", "營收事件"]


def _picker(sig: str) -> str:
    nm = common.names(sig).reset_index().sort_values("code")
    labels = (nm["code"] + " " + nm["name"].fillna("")).tolist()
    by_code = {lab.split()[0]: lab for lab in labels}
    goto = st.session_state.pop("goto_code", None)
    if goto and goto in by_code:
        st.session_state["stock_pick"] = by_code[goto]
    elif "stock_pick" not in st.session_state:
        q = st.query_params.get("code")
        st.session_state["stock_pick"] = by_code.get(q) or by_code.get("2330") or labels[0]
    chosen = st.selectbox("股票（打代號或名稱篩選，本研究的 383 檔）", labels, key="stock_pick")
    code = chosen.split()[0]
    if st.query_params.get("code") != code:
        st.query_params["code"] = code
    return code


def _hero(sig: str, code: str, px: pd.DataFrame, rv: pd.DataFrame) -> None:
    nm = common.names(sig)
    name, cat = nm.at[code, "name"], nm.at[code, "category"]
    close = px["close"].iloc[-1]
    chg = px["close"].iloc[-1] / px["close"].iloc[-2] - 1 if len(px) >= 2 else None
    last = px["close"].iloc[-252:]
    lo, hi = last.min(), last.max()
    pos = max(0.0, min(1.0, (close - lo) / (hi - lo))) * 100 if hi > lo else 50
    pool_date, pool = common.pool(sig)
    picks = common.picks(sig)
    in_pool = common.badge("在母體內", "info") if code in pool else common.badge("不在母體", "muted")
    in_pick = (common.badge("高波動清單", "bull") if not picks.empty and code in set(common.held(picks)["代號"])
               else common.badge("不在清單", "muted"))
    rev_row = ""
    if len(rv):
        r = rv.iloc[-1]
        flag = common.badge("創歷史新高", "bull") if bool(r["is_high"]) else ""
        rev_row = (f'<div class="row"><span class="k">月營收</span>{common.esc(r["ym"])} 年增 '
                   f'{common.signed_html(r["yoy"])} {flag}</div>')
    st.markdown(
        f'<div class="hero"><div>'
        f'<div><span class="nm">{common.esc(name)}</span><span class="cd">{common.esc(code)} · {common.esc(cat)}</span></div>'
        f'<div><span class="px">{common.num(close)}</span><span class="chg">{common.signed_html(chg, 2)}</span></div>'
        f'<div class="meta">資料日 {px.index[-1]:%Y-%m-%d}・Yahoo 還原價（已套用修正表）</div>'
        f'<div class="row"><span class="k">52 週</span>{common.num(lo)} – {common.num(hi)}'
        f'<div class="range"><i style="left:calc({pos:.0f}% - 1px)"></i></div></div></div>'
        f'<div class="right"><div class="row"><span class="k">母體</span>{in_pool}'
        f'<span style="opacity:.6;font-size:.78rem">{common.esc(pool_date or "")}</span></div>'
        f'<div class="row"><span class="k">策略</span>{in_pick}</div>{rev_row}</div>'
        f'</div>', unsafe_allow_html=True)


def _ret(px: pd.Series, n: int):
    return px.iloc[-1] / px.iloc[-n - 1] - 1 if len(px) > n else None


def _price(sig: str, px: pd.DataFrame) -> None:
    c = px["close"]
    r = c.pct_change()
    val = (px["close"] * px["volume"]).iloc[-20:].mean() / 1e8
    common.kpis([
        ("近 1 月", common.signed_html(_ret(c, 21)), "還原價"),
        ("近 3 月", common.signed_html(_ret(c, 63)), "還原價"),
        ("近 6 月", common.signed_html(_ret(c, 126)), "還原價"),
        ("近 12 月", common.signed_html(_ret(c, 252)), "還原價"),
        ("距 52 週高點", common.signed_html(c.iloc[-1] / c.iloc[-252:].max() - 1), ""),
        ("相對 120 日均線", common.signed_html(c.iloc[-1] / c.rolling(120).mean().iloc[-1] - 1), "趨勢過濾用的長度"),
        ("60 日年化波動", common.pct(r.iloc[-60:].std() * np.sqrt(252)).lstrip("+"), "高波動清單的排序依據"),
        ("20 日均成交值", common.num(val) + " 億", ""),
    ])
    s = c.iloc[-756:]
    tw = common.twii(sig).reindex(s.index).ffill()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=s.index, y=s, name="還原收盤", line=dict(color=common.C_STRAT, width=1.8)))
    fig.add_trace(go.Scatter(x=s.index, y=c.rolling(120).mean().reindex(s.index), name="120 日均線",
                             line=dict(color=common.C_ETF, width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=tw.index, y=tw, name="加權指數（右軸）",
                             line=dict(color=common.C_POOL, width=1)), secondary_y=True)
    fig.update_xaxes(rangeselector=dict(x=0.5, xanchor="center", y=1.02, yanchor="bottom", buttons=[
        dict(count=3, label="3 月", step="month", stepmode="backward"),
        dict(count=6, label="6 月", step="month", stepmode="backward"),
        dict(count=1, label="1 年", step="year", stepmode="backward"),
        dict(step="all", label="全部")]))
    common.chart(common.fig_layout(fig, "還原收盤（近 3 年）", 400))


def _revenue(rv: pd.DataFrame) -> None:
    if rv.empty:
        common.empty("這檔沒有月營收資料。")
        return
    d = rv.tail(48)
    x = pd.PeriodIndex(d["ym"], freq="M").to_timestamp()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [common.UP if h else common.C_POOL for h in d["is_high"]]
    fig.add_trace(go.Bar(x=x, y=d["revenue"] / 1e5, name="單月營收（億）", marker_color=colors))
    fig.add_trace(go.Scatter(x=x, y=d["yoy"] * 100, name="年增率 %（右軸）",
                             line=dict(color=common.C_STRAT, width=1.8)), secondary_y=True)
    common.chart(common.fig_layout(fig, "月營收（紅色 = 創歷史新高）", 360))
    show = rv.iloc[::-1].head(24)[["ym", "revenue", "yoy", "yoy3", "cum_yoy", "is_high", "avail_date"]].rename(columns={
        "ym": "月份", "revenue": "單月營收(千元)", "yoy": "年增", "yoy3": "近3月年增", "cum_yoy": "累計年增",
        "is_high": "創新高", "avail_date": "可用日"})
    show["創新高"] = show["創新高"].map(lambda v: "★" if v else "")
    show["可用日"] = pd.to_datetime(show["可用日"]).dt.strftime("%Y-%m-%d")
    common.table(common.styled(show, fmt={"單月營收(千元)": "{:,.0f}", "年增": "{:+.1%}", "近3月年增": "{:+.1%}",
                                          "累計年增": "{:+.1%}"}, signed=["年增", "近3月年增", "累計年增"]),
                 hide_index=True)
    st.caption("可用日是法定期限（次月 10 日），研究裡只在這天之後才使用這筆營收。")


def _quarterly(q: pd.DataFrame) -> None:
    if q.empty:
        common.empty("這檔沒有季報資料（季報 2013 年起）。")
        return
    q = q.sort_values(["year", "q"]).copy()
    q["季"] = q["year"].astype(int).astype(str) + "Q" + q["q"].astype(int).astype(str)
    q["毛利率"] = q["gross_profit"] / q["revenue"]
    q["營益率"] = q["op_income"] / q["revenue"]
    q["淨利率"] = q["net_income"] / q["revenue"]
    q["負債比"] = q["total_liab"] / q["total_assets"]
    q["EPS年差"] = q["eps"] - q["eps"].shift(4)
    d = q.tail(20)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=d["季"], y=d["eps"], name="EPS", marker_color=common.C_STRAT))
    for col, color in (("毛利率", common.UP), ("營益率", common.C_ETF)):
        fig.add_trace(go.Scatter(x=d["季"], y=d[col] * 100, name=f"{col} %（右軸）", line=dict(color=color, width=1.6)),
                      secondary_y=True)
    common.chart(common.fig_layout(fig, "EPS 與利潤率", 340))
    show = q.iloc[::-1].head(12)[["季", "revenue", "eps", "EPS年差", "毛利率", "營益率", "淨利率", "負債比", "bvps"]].rename(
        columns={"revenue": "營收(千元)", "eps": "EPS", "bvps": "每股淨值"})
    common.table(common.styled(show, fmt={"營收(千元)": "{:,.0f}", "EPS": "{:.2f}", "EPS年差": "{:+.2f}",
                                          "毛利率": "{:.1%}", "營益率": "{:.1%}", "淨利率": "{:.1%}",
                                          "負債比": "{:.1%}", "每股淨值": "{:.2f}"}, signed=["EPS年差"]),
                 hide_index=True)


def _chips(sig: str, code: str) -> None:
    ch = common.chips(sig, code)
    if ch.empty:
        common.empty("這檔沒有三大法人資料。")
        return
    d = ch.iloc[-250:] / 1000
    cols = {"foreign_investor": "外資", "investment_trust": "投信", "dealer": "自營商"}
    common.kpis([(f"{n} 近 {k} 日", common.signed_html(d[c].iloc[-k:].sum(), 0, 1, " 張"), "")
                 for c, n in cols.items() for k in (5, 20)])
    fig = common.line_fig(d.index, {n: d[c].cumsum() for c, n in cols.items()}, "近一年累計買賣超（張）",
                          colors={"外資": common.C_STRAT, "投信": common.UP, "自營商": common.C_ETF})
    common.chart(fig)
    st.caption(f"證交所 T86，資料至 {ch.index[-1]:%Y-%m-%d}。研究裡籌碼面沒有增量（結論第三節）。")


def _events(sig: str, code: str) -> None:
    ev = common.events(sig)
    if ev.empty:
        common.empty("還沒有 results/revenue_event/events.csv，先跑 python revenue_event_study.py。")
        return
    e = ev[ev["code"] == code].copy()
    e = e[e["is_high"] | (e["yoy"] >= 0.30) | (e["yoy"] <= -0.20)]
    if e.empty:
        common.empty("這檔在母體內沒有營收事件紀錄（創新高、年增≥30% 或年增≤−20%）。")
        return
    e["事件"] = np.select([e["is_high"] & (e["yoy"] >= 0.30), e["yoy"] >= 0.30, e["is_high"], e["yoy"] <= -0.20],
                        ["創新高且年增≥30%", "年增≥30%", "創新高", "年增≤−20%"], "")
    show = e.sort_values("entry", ascending=False)[["ym", "事件", "yoy", "entry", "uni5", "uni20", "uni60", "sv60"]].rename(
        columns={"ym": "月份", "yoy": "年增", "entry": "進場日", "uni5": "5日對母體", "uni20": "20日對母體",
                 "uni60": "60日對母體", "sv60": "60日對配對"})
    show["進場日"] = show["進場日"].dt.strftime("%Y-%m-%d")
    fmt = {c: "{:+.1%}" for c in ["年增", "5日對母體", "20日對母體", "60日對母體", "60日對配對"]}
    common.table(common.styled(show, fmt=fmt, signed=list(fmt)), hide_index=True)
    st.caption("進場日 = 可用日之後第一個交易日開盤；超額未扣成本。只列進場當天在母體內的事件。")


def main():
    sig = common.data_sig()
    common.page_title("個股", "價格、營收、財報、法人與事件紀錄")
    code = _picker(sig)
    px = common.price(sig, code)
    if px.empty:
        common.empty("讀不到這檔的價格。")
        return
    flags = common.revenue_flags(sig)
    rv = flags[flags["code"] == code].sort_values("ym")
    _hero(sig, code, px, rv)
    tabs = st.tabs(SECTIONS)
    with tabs[0]:
        _price(sig, px)
    with tabs[1]:
        _revenue(rv)
    with tabs[2]:
        q = common.quarterly_all(sig)
        _quarterly(q[q["code"] == code])
    with tabs[3]:
        _chips(sig, code)
    with tabs[4]:
        _events(sig, code)


main()
