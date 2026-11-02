"""模擬交易：多個模擬帳戶的總覽、持股、委託紀錄與策略自動下單（paper_trade.py 的前端）。"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import forecast as F
import paper_trade as P
import paper_ui

STRATS = list(F.STRATEGIES)


def _money(v: float) -> str:
    cls = "up" if v > 0 else ("down" if v < 0 else "")
    return f'<span class="{cls}">{v:+,.0f}</span>'


def _cond(od: pd.DataFrame) -> pd.Series:
    """條件賣單的停損／目標價與最晚出場日；一般單留空。"""
    def one(r):
        if pd.isna(r["expire_date"]):
            return ""
        lv = [f"{n} {r[k]:.2f}" for n, k in (("停損", "stop"), ("目標", "target")) if pd.notna(r[k])]
        return "、".join(lv + [f"最晚 {r['expire_date']}"])
    return od.apply(one, axis=1)


def _auto(con, initial: float, acct: int) -> None:
    """策略自動下單：選策略、出場方式、期間，逐日照訊號下單。"""
    nm = common.names(common.data_sig()).reset_index().sort_values("code")
    labels = (nm["code"] + " " + nm["name"].fillna("")).tolist()
    last = pd.Timestamp(con.execute("SELECT MAX(date) FROM prices").fetchone()[0]).date()
    with st.form("auto", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            key = st.selectbox("策略", STRATS, format_func=lambda k: F.STRATEGIES[k]["name"])
        with c2:
            mode = st.selectbox("出場方式", list(F.EXIT_MODES), format_func=F.EXIT_MODES.get)
        with c3:
            scope = st.radio("範圍", ["codes", "top"], horizontal=True,
                             format_func={"codes": "指定個股", "top": "模型每天前 N 名"}.get,
                             help="前 N 名只適用 5 日、20 日、隔日模型")
        c1, c2 = st.columns([3, 1])
        with c1:
            picks = st.multiselect("個股（範圍選「指定個股」時用）", labels,
                                   default=[l for l in labels if l.startswith("2330 ")])
        with c2:
            top_n = st.number_input("前 N 名", 1, 50, 10)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            start = st.date_input("起日", value=last - pd.Timedelta(days=365), min_value=pd.Timestamp("2014-06-01").date(),
                                  max_value=last)
        with c2:
            end = st.date_input("迄日", value=last, min_value=pd.Timestamp("2014-06-01").date(), max_value=last)
        with c3:
            max_pos = st.number_input("最多同時持有幾檔", 1, 50, 10)
        with c4:
            amount = st.number_input("每筆金額", 1_000, 10_000_000, int(initial / 10 * 0.95), step=10_000,
                                     help="預設是起始資金 ÷ 10 檔再留 5% 給手續費")
        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            fresh = st.checkbox("先清空帳戶再跑", value=True,
                                help="同一個帳戶裡的各批次共用現金，混在一起時後跑的會影響先跑的成交")
        with c2:
            sweep = st.checkbox("閒置資金買 0050（含息）", value=P.sweep_on(con, acct),
                                help="沒有用在個股上的錢一律放 0050；個股要買時賣掉剛好夠的 0050，"
                                     "賣出後立刻買回。賣 0050 付手續費＋ETF 證交稅 0.1%")
        with c3:
            cash = st.number_input("起始資金（清空時用）", 10_000, 1_000_000_000, int(initial), step=100_000)
        go = st.form_submit_button("開始自動下單", type="primary")
    if not go:
        return
    if scope == "top" and F.STRATEGIES[key]["kind"] != "model":
        st.error("「模型每天前 N 名」只適用 5 日、20 日、隔日模型。")
        return
    if scope == "codes" and not picks:
        st.error("請至少選一檔個股。")
        return
    if fresh:
        P.reset(con, cash, sweep=sweep, acct=acct)
    else:
        P.set_setting(con, "sweep", int(sweep), acct)
    bar = st.progress(0.0, "準備中…")
    try:
        r = P.run_strategy(con, key, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), mode,
                           codes=[l.split()[0] for l in picks] if scope == "codes" else None,
                           top_n=int(top_n) if scope == "top" else None, amount=float(amount),
                           max_pos=int(max_pos), progress=lambda f, m: bar.progress(min(f, 1.0), m), acct=acct)
    except ValueError as e:
        st.error(str(e))
        return
    st.session_state["auto_done"] = r
    st.rerun()


def _batches(con, acct: int) -> None:
    b = P.batch_summary(con, acct)
    if b.empty:
        return
    st.subheader("自動下單批次")
    common.table(common.styled(b, fmt={"已實現損益": "{:+,.0f}", "勝率": "{:.0%}", "平均每筆": "{:+,.0f}"},
                               signed=["已實現損益", "平均每筆"]), hide_index=True)
    st.caption("已實現損益與勝率只算買賣都成交的那幾筆，已扣手續費與證交稅。"
               "現金不夠時買單會退單，對應的賣單也一起退。")
    c1, c2 = st.columns([3, 1])
    with c1:
        sel = st.selectbox("刪除一批", b["批次"].tolist(), key="del_batch")
    with c2:
        st.write("")
        if st.button("刪除這批"):
            P.delete_batch(con, sel, acct)
            st.rerun()


def _bar(con) -> int:
    """帳戶選單＋新增帳戶。"""
    c1, c2, c3 = st.columns([2.2, 1, 3])
    with c1:
        acct = paper_ui.pick(con, "paper")
    with c2:
        st.write("")
        with st.popover("新增帳戶", icon=":material/person_add:"):
            with st.form("new_acct", border=False):
                name = st.text_input("帳戶名稱", placeholder="例如：小明、5 日模型測試")
                cash = st.number_input("起始資金", 10_000, 1_000_000_000, P.INITIAL_CASH, step=100_000)
                sweep = st.checkbox("閒置資金買 0050（含息）", value=True)
                if st.form_submit_button("建立", type="primary"):
                    try:
                        st.session_state["paper_acct"] = P.create_account(con, name, cash, sweep)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    with c3:
        st.caption("每個帳戶的現金、持股、委託和自動下單批次都分開算。"
                   "多台電腦連進來時，各自選自己的帳戶就不會互相影響。")
    return acct


def _manage(con, acct: int, p: dict) -> None:
    """改名、重設、刪除這個帳戶。"""
    name = P.accounts(con).set_index("id").loc[acct, "name"]
    with st.expander(f"管理帳戶「{name}」"):
        c1, c2 = st.columns([3, 1])
        with c1:
            new = st.text_input("改名", value=name, key=f"rename_{acct}")
        with c2:
            st.write("")
            if st.button("儲存名稱", key=f"rename_btn_{acct}", disabled=new.strip() == name):
                try:
                    P.rename_account(con, acct, new)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        st.markdown("**重設**：清除這個帳戶的所有委託，其他帳戶不受影響")
        cash = st.number_input("起始資金", min_value=10_000, value=int(p["initial"]), step=100_000,
                               key=f"reset_cash_{acct}")
        sw = st.checkbox("閒置資金買 0050（含息）", value=P.sweep_on(con, acct), key=f"reset_sweep_{acct}")
        sure = st.checkbox("確定清除這個帳戶的所有模擬委託", key=f"reset_sure_{acct}")
        if st.button("重設", disabled=not sure, key=f"reset_btn_{acct}"):
            P.reset(con, cash, sweep=sw, acct=acct)
            st.rerun()
        st.markdown("**刪除帳戶**：帳戶和它的委託一起刪掉")
        sure_del = st.checkbox(f"確定刪除「{name}」", key=f"del_sure_{acct}")
        if st.button("刪除帳戶", disabled=not sure_del, key=f"del_btn_{acct}"):
            try:
                P.delete_account(con, acct)
                st.session_state.pop("paper_acct", None)
                st.rerun()
            except ValueError as e:
                st.error(str(e))


def main():
    common.page_title("模擬交易", "下單日收盤後送出、下一個交易日開盤成交 · 手續費 3 折＋證交稅 0.3%")
    if not P.DB.exists():
        common.empty("還沒有 db/local.db，先跑 python build_local_db.py。")
        return
    con = P.connect()
    acct = _bar(con)
    n = P.settle(con, acct)
    if n:
        st.toast(f"新成交 {n} 筆")
    p = P.portfolio(con, acct)
    names = pd.concat([common.names(common.data_sig())["name"], pd.Series({P.ETF: "元大台灣50（含息還原價）"})])
    ret = p["equity"] / p["initial"] - 1
    c = p["curve"]
    bench = None
    if len(c) > 1:
        e = pd.read_sql("SELECT date, close FROM etf_prices WHERE code = '0050'", con, parse_dates=["date"]) \
            .set_index("date")["close"].reindex(c.index).ffill()
        # 對照：同一筆錢第一天全部買 0050，買進成本和帳戶一樣算
        bench = e / e.iloc[0] * p["initial"] * (1 - P.FEE)
    items = [("淨值", f"{p['equity']:,.0f}", f"起始 {p['initial']:,.0f}", True),
             ("總報酬", common.signed_html(ret), ""),
             ("現金", f"{p['cash']:,.0f}", ""),
             ("已實現損益", _money(p["realized"]), "已扣手續費與證交稅"),
             ("未實現損益", _money(p["unrealized"]), "假設現在賣出，已扣成本")]
    if bench is not None:
        b_ret = bench.iloc[-1] / p["initial"] - 1          # 和帳戶淨值一樣用收盤市值，不扣賣出成本
        gap = ret - b_ret
        items.append(("對照：全部買 0050 抱著", common.signed_html(b_ret),
                      f"帳戶{'多' if gap >= 0 else '少'} {abs(gap):.1%}"))
    common.kpis(items)
    if bench is not None:
        common.chart(common.line_fig(c.index, {"淨值": c, "全部買 0050 抱著（含息、同起點）": bench},
                                     "逐日淨值（收盤評價）",
                                     colors={"淨值": common.C_STRAT, "全部買 0050 抱著（含息、同起點）": common.C_POOL}))
        st.caption(f"閒置資金買 0050：{'開' if P.sweep_on(con, acct) else '關'}。對照線是同一筆錢第一天全部買 0050 一路抱著"
                   "（研究 14.8 的投資人 A）；開著的話，帳戶就是投資人 B。")

    done = st.session_state.pop("auto_done", None)
    if done:
        st.success(f"自動下單完成：{done['批次']}，進場 {done.get('進場筆數', 0)} 筆，"
                   f"已實現損益 {done.get('已實現損益', 0):+,.0f}。")
    with st.expander("策略自動下單", expanded=P.orders(con, acct).empty):
        st.caption("從起日到迄日逐個交易日，每天只用當天收盤以前的資料判斷有沒有進場訊號（和「個股預測」頁同一套），"
                   "有就在當天收盤後下買單、隔天開盤成交，再照出場方式掛賣單。同一檔持有中不重複進場。")
        _auto(con, p["initial"], acct)
    _batches(con, acct)

    st.subheader("持股")
    pos = p["positions"]
    if pos.empty:
        common.empty("目前沒有持股。到「個股預測」頁選策略、送出模擬單。")
    else:
        tb = pd.DataFrame({"代號": pos["code"], "名稱": pos["code"].map(names), "股數": pos["shares"],
                           "平均成本": pos["avg_cost"], "最新收盤": pos["last"], "日期": pos["last_date"],
                           "市值": pos["value"], "未實現": pos["unrealized"],
                           "報酬率": pos["unrealized"] / (pos["avg_cost"] * pos["shares"])})
        common.table(common.styled(tb, fmt={"股數": "{:,.0f}", "平均成本": "{:.2f}", "最新收盤": "{:.2f}",
                                            "市值": "{:,.0f}", "未實現": "{:+,.0f}", "報酬率": "{:+.2%}"},
                                   signed=["未實現", "報酬率"]), hide_index=True)

    st.subheader("委託紀錄")
    od = P.orders(con, acct)
    n_sw = int((od["batch"] == P.SWEEP).sum())
    if n_sw and not st.toggle(f"顯示閒置資金買賣 0050 的系統單（{n_sw} 筆）", value=False):
        od = od[od["batch"] != P.SWEEP]
    if od.empty:
        st.caption("還沒有委託。")
    else:
        tb = pd.DataFrame({"#": od["id"], "下單日": od["order_date"], "代號": od["code"],
                           "名稱": od["code"].map(names), "買賣": od["side"].map({"buy": "買進", "sell": "賣出"}),
                           "股數": od["shares"], "狀態": od["status"].map(P.STATUS), "成交日": od["fill_date"],
                           "成交價": od["fill_price"], "費用": od["fee"].fillna(0) + od["tax"].fillna(0),
                           "條件": _cond(od), "觸發": od["trigger"], "批次": od["batch"],
                           "策略": od["strategy"], "備註": od["note"].fillna("") + od["reason"].fillna("")})
        common.table(common.styled(tb, fmt={"股數": "{:,.0f}", "成交價": "{:.2f}", "費用": "{:,.0f}"}),
                     hide_index=True)
        pend = od[od["status"] == "pending"]
        if len(pend):
            c1, c2 = st.columns([3, 1])
            with c1:
                oid = st.selectbox("取消待成交的單", pend["id"].tolist(),
                                   format_func=lambda i: f"#{i} {pend.set_index('id').loc[i, 'order_date']} "
                                                         f"{pend.set_index('id').loc[i, 'code']}")
            with c2:
                st.write("")
                if st.button("取消這筆"):
                    P.cancel(con, int(oid), acct)
                    st.rerun()
            st.caption("待成交：下單日之後還沒有開盤價（例如預約的出場日還沒到）。股價更新後重新整理就會成交。")

    _manage(con, acct, p)
    st.markdown(f'<div class="disc">{common.esc(common.DISCLAIMER)} 沒有模擬每筆最低手續費、'
                '漲跌停買不到賣不掉與滑價。</div>', unsafe_allow_html=True)


main()
