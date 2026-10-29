"""總覽：今天要看什麼。大盤趨勢狀態、本月營收事件、高波動清單、研究重點。"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common

MA = 120  # strategy_spec.py 的趨勢過濾


def _bench_row(bench: pd.DataFrame, event: str, base: str, h: int = 60, period: str = "全期") -> dict:
    if bench.empty:
        return {}
    r = bench[(bench["事件"] == event) & (bench["基準"] == base) & (bench["持有"] == h) & (bench["區間"] == period)]
    return r.iloc[0].to_dict() if len(r) else {}


def _trend(sig: str) -> tuple[bool | None, float | None, float | None, str]:
    tw = common.twii(sig)
    if len(tw) < MA:
        return None, None, None, "—"
    now, ma = float(tw.iloc[-1]), float(tw.rolling(MA).mean().iloc[-1])
    return now > ma, now, ma, f"{tw.index[-1]:%Y-%m-%d}"


def _month_events(sig: str) -> tuple[str | None, pd.DataFrame, pd.DataFrame]:
    flags = common.revenue_flags(sig)
    if flags.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    ym = flags["ym"].max()
    pool_date, pool = common.pool(sig)
    cur = flags[(flags["ym"] == ym) & flags["code"].isin(pool)].copy()
    nm = common.names(sig)
    cur["名稱"] = cur["code"].map(nm["name"])
    cur["產業"] = cur["code"].map(nm["category"])
    return ym, cur, flags


def _event_table(df: pd.DataFrame, key: str) -> None:
    show = df.sort_values("yoy", ascending=False)[
        ["code", "名稱", "產業", "revenue", "yoy", "cum_yoy", "avail_date"]].rename(columns={
            "code": "代號", "revenue": "單月營收(千元)", "yoy": "年增", "cum_yoy": "累計年增", "avail_date": "可用日"})
    show["可用日"] = pd.to_datetime(show["可用日"]).dt.strftime("%Y-%m-%d")
    show = show.reset_index(drop=True)
    ev = common.table(common.styled(show, fmt={"單月營收(千元)": "{:,.0f}", "年增": "{:+.1%}", "累計年增": "{:+.1%}"},
                                    signed=["年增", "累計年增"]),
                      hide_index=True, on_select="rerun", selection_mode="single-row", key=key)
    common.pick_row(ev, show)


def main():
    sig = common.data_sig()
    common.page_title("總覽", "大盤趨勢、本月營收事件、高波動清單")

    on, now, ma, tw_date = _trend(sig)
    ym, cur, _ = _month_events(sig)
    bench = common.result_csv(sig, "revenue_event/bench.csv")
    picks = common.picks(sig)
    hi = cur[cur["is_high"] & (cur["yoy"] >= 0.30)] if len(cur) else cur
    neg = cur[cur["yoy"] <= -0.20] if len(cur) else cur
    sv = _bench_row(bench, "創新高 且 年增≥30%", "成交值×波動")

    common.kpis([
        ("大盤趨勢", common.badge("均線之上・持有", "bull") if on else
         (common.badge("跌破均線・空手", "bear") if on is False else "—"),
         f"加權 {common.num(now, 0)}／{MA} 日線 {common.num(ma, 0)}（{tw_date}）", bool(on)),
        ("高波動清單", f"{len(common.held(picks))} 檔", "strategy_spec.py 的最新輸出"),
        ("本月營收公布（母體內）", f"{len(cur)} 家", f"{ym or '—'} 營收"),
        ("創新高且年增≥30%", f"{len(hi)} 家", f"年增≤−20%：{len(neg)} 家", len(hi) > 0),
        ("此事件 60 日超額（配對）", common.signed_html(sv.get("平均超額"), 2),
         f"t {common.num(sv.get('t'))}・勝率 {common.num((sv.get('勝率') or 0) * 100, 0)}%（全期）"),
    ])

    st.subheader(f"本月營收事件　{ym or ''}")
    st.caption("母體：最新一天 60 日均成交金額前 250 名。事件定義同 revenue_event_study.py；營收在可用日（次月 10 日）"
               "之後才算已知。點一列打開個股頁。")
    t1, t2 = st.tabs([f"創新高且年增≥30%（{len(hi)}）", f"年增≤−20%（{len(neg)}）"])
    with t1:
        if hi.empty:
            common.empty("這個月母體內沒有營收創歷史新高且年增 30% 以上的公司。")
        else:
            _event_table(hi, "ev_hi")
            st.caption(f"歷史上這類事件持有 60 日，對成交值×波動配對的平均超額 {common.pct(sv.get('平均超額'), 2)}，"
                       f"但中位數 {common.pct(sv.get('中位超額'), 2)}、勝率不到一半：平均是被少數大漲股撐起來的。")
    with t2:
        if neg.empty:
            common.empty("這個月母體內沒有營收年減 20% 以上的公司。")
        else:
            _event_table(neg, "ev_neg")
            r = _bench_row(bench, "年增≤−20%", "母體")
            st.caption(f"歷史上持有 60 日對母體平均 {common.pct(r.get('平均超額'), 2)}（t {common.num(r.get('t'))}）。")

    st.subheader("高波動清單")
    if picks.empty:
        common.empty("還沒有 results/spec/current_picks.csv，先跑 python strategy_spec.py。")
    else:
        rule = "規則為持有" if on else "大盤跌破 120 日線，規則為空手"
        st.caption(f"每 42 個交易日在母體內取近 60 日波動最高的 30 檔，舊持股在前 40 名內續抱；{rule}。這是 beta 大於 1 的風險補償，不是選股能力（結論第八、九節）。")
        show = picks.copy()
        ev = common.table(common.styled(show, fmt={"年化波動": "{:.1%}"}), hide_index=True,
                          on_select="rerun", selection_mode="single-row", key="picks_tbl")
        common.pick_row(ev, show)

    st.subheader("研究重點")
    spec = common.result_csv(sig, "spec/summary.csv")
    hv = spec[spec["設定"] == "高波動（無過濾）"].iloc[0].to_dict() if len(spec) else {}
    cards = [
        ("波動度溢酬", "唯一存活但是 beta", [
            f"高波動 30 檔 2003 起年化 {common.pct(hv.get('年化'))}，對大盤超額 {common.pct(hv.get('超額'))}（t {common.num(hv.get('超額t'))}）",
            "beta 大於 1，沖掉 beta 後不顯著，也贏不過同波動的槓桿大盤",
            "趨勢過濾降回撤，但不提高夏普"]),
        ("營收年增率", "唯一不是 beta 的候選", [
            "季頻十分位多空 alpha 年化約 +9%，多空 beta 接近 0",
            "產業＋規模中性化後多空 +12.8% → +8.0%，仍顯著",
            "與全部價量特徵一起檢定 RC p 0.086、SPA p 0.022，屬邊緣"]),
        ("方法論", "會讓結果虛胖的錯誤", [
            "walk-forward 沒 purging：h=60 的 t 從 3.94 掉到 1.25",
            "用當期市值選股：超額從 +7.2% 變 −4.3%",
            "訊號與成交用同一個收盤價：多空 +162% 變 −19%"]),
    ]
    html = "".join(
        f'<div class="card"><div class="hd"><b>{common.esc(t)}</b>{common.badge(b, "info")}</div>'
        f'<ul>{"".join(f"<li>{common.esc(x)}</li>" for x in items)}</ul></div>' for t, b, items in cards)
    st.markdown(f'<div class="cgrid">{html}</div>', unsafe_allow_html=True)
    st.caption("完整內容見「研究結論」頁。")


main()
