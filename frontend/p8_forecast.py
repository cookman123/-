"""個股預測：選策略看方向、預測區間、建議進出場點與歷史相似走勢，可以直接模擬下單（forecast.py 的前端）。"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import forecast as F
import paper_trade as P
import paper_ui

KEYS = list(F.STRATEGIES)
NAMES = {k: v["name"] for k, v in F.STRATEGIES.items()}
SHOW_DAYS = 60
FADE = "rgba(148,163,184,.55)"


@st.cache_data(show_spinner="載入全市場價格…", ttl=3600)
def _panel(_sig: str):
    return F.load_panel(F.connect())


@st.cache_data(show_spinner=False, max_entries=64)
def _px(_sig: str, code: str) -> pd.DataFrame:
    return F.load_px(F.connect(), code)


@st.cache_data(show_spinner="計算中…", max_entries=64)
def _run(_sig: str, code: str, asof: str, key: str, opt: tuple) -> dict:
    opt = dict(opt)
    need_panel = key == "highvol" or (key == "analog" and opt.get("pool") == "market")
    C, V = _panel(_sig) if need_panel else (None, None)
    if key == "analog" and opt.get("pool") == "market":
        opt["selftest"] = False
    return F.forecast(F.connect(), code, asof, key, px=_px(_sig, code), closes=C, volumes=V, **opt)


def _controls(sig: str, nm: pd.DataFrame):
    labels = (nm["code"] + " " + nm["name"].fillna("")).tolist()
    default = st.session_state.get("stock_code", "2330")
    idx = next((i for i, l in enumerate(labels) if l.split()[0] == default), 0)
    c1, c2, c3 = st.columns([1.6, 1, 2.6])
    with c1:
        code = st.selectbox("個股", labels, index=idx, key="fc_code").split()[0]
    px = _px(sig, code)
    last = px.index[-1].date() if len(px) else date.today()
    with c2:
        asof = st.date_input("視角日期（只用這天收盤以前的資料）", value=last, min_value=date(2012, 1, 1),
                             max_value=last, key="fc_asof")
    with c3:
        key = common.segmented("策略", KEYS, "h5", "fc_strategy", format_func=NAMES.get)
    opt = {}
    if key == "analog":
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            pool = st.radio("比對範圍", ["self", "market"], horizontal=True, key="fc_pool",
                            format_func={"self": "這檔自己", "market": "全市場"}.get)
        with a2:
            L = st.slider("比對最近幾天", 10, 60, 20, 5, key="fc_L")
        with a3:
            H = st.slider("往後看幾天", 5, 60, 20, 5, key="fc_H")
        with a4:
            k = st.slider("取幾段最像的", 5, 30, 10, key="fc_k")
        opt = dict(pool=pool, L=L, H=H, k=k)
    mode = common.segmented("出場方式", list(F.EXIT_MODES), "fixed", "fc_exit", format_func=F.EXIT_MODES.get)
    return code, asof.strftime("%Y-%m-%d"), key, opt, px, mode


def _exit_test(sig: str, key: str, mode: str) -> str:
    """forecast_check.py 對這個出場方式的實測結果（只在 5、20 日模型的前 10 名上測過）。"""
    t = common.result_csv(sig, "forecast_check/stop_target_summary.csv")
    h = F.STRATEGIES[key]["h"] if F.STRATEGIES[key]["kind"] == "model" else None
    if t.empty:
        return "還沒跑 forecast_check.py，沒有實測結果。"
    rows = t[t["出場方式"] == F.EXIT_COLS[mode]]
    parts = []
    for r in rows.itertuples():
        s = f"{r.h} 日模型前 10 名 {r.筆數:,} 筆：每筆平均 {r.平均淨報酬:+.2%}、勝率 {r.勝率:.0%}"
        if mode != "fixed":
            s += f"，比固定持有 {r.減固定:+.2%}（t {r.差異t:.2f}）"
        parts.append(s)
    note = "" if h in (5, 20) else f"這套策略（{F.STRATEGIES[key]['name']}）沒有單獨測過，下面是模型策略的結果供參考。"
    return f"「{F.EXIT_MODES[mode]}」的實測（2014 年起、扣成本）：{note}" + "；".join(parts) + "。"


def _price(r: dict, t: int) -> float:
    return r["close"] * float(np.exp(r["path"]["mid"].iloc[t]))


def _kpis(r: dict, mode: str) -> None:
    p, path = r["plan"], r["path"]
    end = len(path) - 1
    tone = {"bull": "up", "bear": "down"}.get(r["tone"], "")
    items = [("方向", f'<span class="{tone}">{common.esc(r["direction"])}</span>', r["name"], True),
             (f"第 {end} 個交易日中線", common.signed_html(np.expm1(path["mid"].iloc[-1])), "預測區間的中間"),
             ("80% 區間", f'{common.pct(np.expm1(path["lo80"].iloc[-1]))} ~ {common.pct(np.expm1(path["hi80"].iloc[-1]))}',
              f"{end} 天後")]
    if p and p["enter"]:
        items += [("建議進場", f"{p['entry_date']:%m/%d} 開盤", f"約 {p['entry_est']:.2f}（以收盤估）"),
                  ("建議出場", f"{p['exit_date']:%m/%d} 開盤", f"固定持有 {p['exit_t'] - 1} 個交易日")]
        stop, tgt = F.exit_levels(p, mode)
        if mode != "fixed":
            items.append((F.EXIT_MODES[mode], " ／ ".join(f"{v:.2f}" for v in (stop, tgt) if v is not None),
                          "碰到就出場；都沒碰到照出場日"))
    elif p:
        items.append(("建議", "不進場", f"持有者 {p['exit_date']:%m/%d} 開盤出場"))
    else:
        items.append(("建議", "觀望", "沒有進場訊號"))
    common.kpis(items)


def _chart(r: dict, px: pd.DataFrame, mode: str) -> None:
    asof = r["asof"]
    hist = px.loc[:asof].iloc[-SHOW_DAYS:]
    fut_d = pd.DatetimeIndex(r["dates"])
    after = px.loc[(px.index > asof) & (px.index <= fut_d[-1])]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"],
                                 close=hist["close"], name="K 線", increasing_line_color=common.UP,
                                 decreasing_line_color=common.DOWN))
    for n, col in ((20, common.C_ETF), (60, common.C_POOL)):
        ma = px["close"].loc[:asof].rolling(n).mean().reindex(hist.index)
        fig.add_trace(go.Scatter(x=hist.index, y=ma, name=f"{n} 日均線", line=dict(color=col, width=1)))
    if len(after):
        fig.add_trace(go.Candlestick(x=after.index, open=after["open"], high=after["high"], low=after["low"],
                                     close=after["close"], name="實際走勢（事後對答案）",
                                     increasing_line_color=FADE, decreasing_line_color=FADE,
                                     increasing_fillcolor=FADE, decreasing_fillcolor=FADE))
    path = r["path"]
    c = r["close"]
    xs = list(fut_d[:len(path)])
    for lo, hi, a, nm in (("lo80", "hi80", .10, "80% 區間"), ("lo50", "hi50", .18, "50% 區間")):
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(c * np.exp(path[hi])) + list(c * np.exp(path[lo]))[::-1],
                                 fill="toself", fillcolor=f"rgba(59,130,246,{a})", line=dict(width=0),
                                 name=nm, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=c * np.exp(path["mid"]), name="預測中線",
                             line=dict(color=common.C_STRAT, width=2, dash="dash")))
    p = r["plan"]
    if p and p["enter"]:
        ex, en = p["exit_date"], p["entry_date"]
        fig.add_trace(go.Scatter(x=[en], y=[_price(r, 1)], mode="markers+text", name="建議進場",
                                 marker=dict(symbol="triangle-up", size=15, color=common.UP),
                                 text=["進場"], textposition="bottom center"))
        fig.add_trace(go.Scatter(x=[ex], y=[_price(r, p["exit_t"])], mode="markers+text", name="建議出場",
                                 marker=dict(symbol="triangle-down", size=15, color=common.DOWN),
                                 text=["出場"], textposition="top center"))
        stop, tgt = F.exit_levels(p, mode)
        for y, nm, col in ((stop, "停損", common.DOWN), (tgt, "目標價", common.UP)):
            if y is not None:
                fig.add_trace(go.Scatter(x=[en, ex], y=[y, y], mode="lines", name=nm,
                                         line=dict(color=col, width=1.4, dash="dot")))
        hit = F.simulate_exit(px, en, ex, stop, tgt) if len(after) else None
        if hit and mode != "fixed":
            fig.add_trace(go.Scatter(x=[hit[0]], y=[hit[1]], mode="markers+text", name=f"實際出場（{hit[2]}）",
                                     marker=dict(symbol="x", size=13, color="#a855f7"),
                                     text=[hit[2]], textposition="top center"))
    elif p:
        fig.add_trace(go.Scatter(x=[p["exit_date"]], y=[_price(r, 1)], mode="markers+text", name="持有者出場",
                                 marker=dict(symbol="triangle-down", size=15, color=common.DOWN),
                                 text=["出場"], textposition="top center"))
    fig.add_vline(x=asof, line=dict(color="rgba(128,128,128,.6)", dash="dot", width=1))
    fig.update_layout(xaxis_rangeslider_visible=False,
                      xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]))
    common.fig_layout(fig, f"近 {SHOW_DAYS} 個交易日 K 線＋未來 {len(path) - 1} 天預測", 500)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0))
    common.chart(fig)
    if p and p["enter"] and len(after) and after.index[-1] >= p["exit_date"]:
        fee = P.FEE * 2 + P.TAX
        entry = px["open"].get(p["entry_date"], np.nan)
        fixed = F.simulate_exit(px, p["entry_date"], p["exit_date"], None, None)
        if fixed and np.isfinite(entry):
            msg = (f"事後對答案：{p['entry_date']:%Y-%m-%d} 開盤買。固定持有到 {fixed[0]:%Y-%m-%d} 開盤賣，"
                   f"扣成本 {fixed[1] / entry - 1 - fee:+.2%}")
            if mode != "fixed" and hit:
                msg += (f"；{F.EXIT_MODES[mode]}在 {hit[0]:%Y-%m-%d} {hit[2]}出場，"
                        f"扣成本 {hit[1] / entry - 1 - fee:+.2%}")
            st.info(msg + "。")


def _analog(r: dict, nm: pd.DataFrame) -> None:
    a = r.get("analog")
    if not a:
        return
    L, H, m = a["L"], a["H"], a["matches"]
    st.subheader("歷史相似走勢")
    fig = go.Figure()
    tx = np.arange(-L + 1, H + 1)
    for i, row in m.iterrows():
        y = np.concatenate([row["x"] - row["x"][-1], row["f"]])
        fig.add_trace(go.Scatter(x=tx, y=np.expm1(y), mode="lines", line=dict(width=1, color=FADE),
                                 name=f"{row['code']} {row['end']:%Y-%m-%d}", showlegend=False))
    q = a["query"]
    fig.add_trace(go.Scatter(x=np.arange(-L + 1, 1), y=np.expm1(q - q[-1]), name="這檔最近走勢",
                             line=dict(color=common.C_STRAT, width=3)))
    fig.add_trace(go.Scatter(x=np.arange(0, H + 1), y=np.expm1(r["path"]["mid"]), name="相似片段中位數",
                             line=dict(color=common.C_ETF, width=2.5, dash="dash")))
    fig.add_vline(x=0, line=dict(color="rgba(128,128,128,.6)", dash="dot", width=1))
    fig.update_yaxes(tickformat="+.0%")
    fig.update_xaxes(title="交易日（0 = 視角日期／相似片段的最後一天）")
    common.chart(common.fig_layout(fig, "以最後一天為基準的累積報酬：藍線是這檔，灰線是歷史上最像的片段", 380))
    names = nm.set_index("code")["name"]
    tb = pd.DataFrame({"代號": m["code"], "名稱": m["code"].map(names), "相似片段": m["start"].dt.strftime("%Y-%m-%d")
                       + " ~ " + m["end"].dt.strftime("%Y-%m-%d"), "距離": m["dist"], f"之後 {H} 日": m["fwd"]})
    common.table(common.styled(tb, fmt={"距離": "{:.3f}", f"之後 {H} 日": "{:+.2%}"}, signed=[f"之後 {H} 日"]),
                 hide_index=True)
    st.caption("距離＝兩段「每天相對起點的累積 log 報酬」的歐氏距離，越小越像。同一檔的兩段至少隔 L 天，"
               "而且每一段「之後 H 天」都在視角日期以前就走完，沒有偷看。")


def _order(r: dict, nm: pd.DataFrame, mode: str) -> None:
    st.subheader("模擬下單")
    con = P.connect()
    acct = paper_ui.pick(con, "forecast", "下單到哪個模擬帳戶")
    held = P.portfolio(con, acct)["positions"]
    have = int(held.loc[held["code"] == r["code"], "shares"].sum())
    p = r["plan"]
    st.caption(f"視角日期 {r['asof']:%Y-%m-%d} 收盤後下單、下一個交易日開盤成交。目前模擬帳戶持有這檔 {have:,} 股。")
    with st.form("fc_order", border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            side = st.radio("買賣", ["buy", "sell"], horizontal=True, format_func={"buy": "買進", "sell": "賣出"}.get,
                            index=0 if not p or p["enter"] else 1)
        with c2:
            shares = st.number_input("股數（1 張 = 1000 股）", min_value=1, value=1000 if side == "buy" else max(have, 1),
                                     step=1000)
        with c3:
            auto_exit = st.checkbox(f"照出場方式預約賣出（{F.EXIT_MODES[mode]}）", value=bool(p and p["enter"]),
                                    disabled=not (p and p["enter"]),
                                    help="固定持有：出場日開盤賣出。停損／目標價：每天檢查，碰到就賣，"
                                         "都沒碰到就在出場日開盤賣出")
        ok = st.form_submit_button("送出模擬單", type="primary")
    if ok:
        oid = P.place(con, r["asof"].strftime("%Y-%m-%d"), r["code"], side, int(shares), r["strategy"],
                      f"{r['name']}：{r['direction']}", acct=acct)
        msg = [f"#{oid} {'買進' if side == 'buy' else '賣出'} {int(shares):,} 股"]
        if side == "buy" and auto_exit and p and p["enter"]:
            if mode == "fixed":
                pre = r["dates"][p["exit_t"] - 1].strftime("%Y-%m-%d")
                oid2 = P.place(con, pre, r["code"], "sell", int(shares), r["strategy"], "照建議出場日預約賣出",
                               parent=oid, acct=acct)
                msg.append(f"#{oid2} 預約 {p['exit_date']:%Y-%m-%d} 開盤賣出")
            else:
                stop, tgt = F.exit_levels(p, mode)
                oid2 = P.place(con, r["asof"].strftime("%Y-%m-%d"), r["code"], "sell", int(shares), r["strategy"],
                               F.EXIT_MODES[mode], stop=stop, target=tgt,
                               expire_date=p["exit_date"].strftime("%Y-%m-%d"), parent=oid, acct=acct)
                lv = "、".join(f"{n} {v:.2f}" for n, v in (("停損", stop), ("目標", tgt)) if v is not None)
                msg.append(f"#{oid2} 條件賣出（{lv}，最晚 {p['exit_date']:%Y-%m-%d} 開盤）")
        od = P.orders(con, acct).set_index("id")
        st.success("已送出：" + "；".join(f"{m}（{P.STATUS[od.loc[i, 'status']]}）"
                                       for m, i in zip(msg, [oid] + ([oid2] if len(msg) > 1 else []))))
        bad = od.loc[[oid], "reason"].dropna()
        if len(bad):
            st.error(f"#{oid} 退單：{bad.iloc[0]}")
    st.page_link("p9_paper.py", label="看模擬帳戶", icon=":material/account_balance_wallet:")


def main():
    common.page_title("個股預測", "選策略看方向與預測區間 · 只用視角日期以前的資料 · 回測不保證未來")
    if not F.DB.exists():
        common.empty("還沒有 db/local.db，先跑 python build_local_db.py。")
        return
    con = F.connect()
    if not con.execute("SELECT name FROM sqlite_master WHERE name = 'predictions'").fetchone():
        common.empty("DB 裡還沒有預測，先跑 python build_local_db.py --tables predictions。")
        return
    sig = common.data_sig()
    nm = common.names(sig).reset_index().sort_values("code")
    code, asof, key, opt, px, mode = _controls(sig, nm)
    if len(px) < 30:
        common.empty(f"{code} 的價格資料不足。")
        return
    try:
        r = _run(sig, code, asof, key, tuple(sorted(opt.items())))
    except ValueError as e:
        common.empty(str(e))
        return
    _kpis(r, mode)
    for w in r["warn"]:
        st.warning(w)
    _chart(r, px, mode)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**判斷依據**")
        st.markdown("\n".join(f"- {common.esc(f)}" for f in r["facts"]))
        if r["plan"] and r["plan"].get("note"):
            st.caption(r["plan"]["note"])
        st.caption(_exit_test(sig, key, mode))
    with c2:
        st.markdown("**這套策略的研究結論**")
        st.markdown(f'<div class="card">{common.esc(r["evidence"])}</div>', unsafe_allow_html=True)
    _analog(r, nm)
    _order(r, nm, mode)
    st.markdown(f'<div class="disc">{common.esc(common.DISCLAIMER)} 預測區間是常態或歷史分位數的近似，'
                f'停損與目標價是預測區間換算出來的，實測照它們出場平均比固定持有差。</div>', unsafe_allow_html=True)


main()
