"""模擬交易：多個模擬帳戶的總覽、持股、委託紀錄與策略自動下單（paper_trade.py 的前端）。"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import forecast as F
import intraday
import llm_judge as J
import paper_trade as P
import paper_ui

STRATS = list(P.LLM_NAMES) + list(F.STRATEGIES)
PICK_HELP = ("流程圖「實際操作」：候選池是 5 日模型前 N 名，由 LLM 依 RAG 行為模式（歷史上最像的情境後來怎麼走）"
             "和新資訊評分（模型排名、近期漲跌、最新新聞）挑最多 K 檔（K＝最多同時持有幾檔），持有 5 日。"
             "每期約 20 秒；N、K 用下面的「前 N 名」和「最多同時持有幾檔」。\n\n"
             "「LLM 自主操盤」：LLM 自己選股、自己組投資組合。每 5 個交易日讀一次研究結論（從知識庫 RAG 檢索）、"
             "DB 裡各策略當天的輸出（5／20 日模型、營收創新高事件、高波動，各取前 N 名）、大盤與目前持股，"
             "直接決定持有哪些、各佔總資產多少、哪些賣掉；沒有固定持有期，剩下的錢放 0050。"
             "「最多同時持有幾檔」是上限、單檔最多 20%，「出場方式」和「每筆金額」不用。每次約 1～2 分鐘。")
PM_WARN = ("研究結論是用到 2026-10 的資料算出來的，拿它回測更早的期間等於讓 LLM 先知道哪些訊號後來有效，"
           "結果會偏樂觀；這個策略要看的是從今天往後的每日自動跟單紀錄。")
SCOPES = {"codes": "指定個股", "all": "全部股票", "top": "模型每天前 N 名"}
LLM_HELP = (f"每個訊號先給地端模型（{J.MODEL}）審核：照買、買一半或這次不買，一筆約 10～20 秒。"
            f"模型的訓練資料大約到 {J.CUTOFF[:7]}，期間早於那之後的回測結果可能偏樂觀。")


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
            key = st.selectbox("策略", STRATS, format_func=P.strategy_name, help=PICK_HELP)
        with c2:
            mode = st.selectbox("出場方式", list(F.EXIT_MODES), format_func=F.EXIT_MODES.get)
        with c3:
            scope = st.radio("範圍", list(SCOPES), horizontal=True, format_func=SCOPES.get,
                             help="前 N 名只適用 5 日、20 日、隔日模型；全部股票每天要掃 500 檔，適合跌深反彈規則")
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
        use_llm = st.checkbox("LLM 審核每個訊號", value=False, help=LLM_HELP)
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
    if key not in P.LLM_NAMES and scope == "top" and F.STRATEGIES[key]["kind"] != "model":
        st.error("「模型每天前 N 名」只適用 5 日、20 日、隔日模型。")
        return
    if key not in P.LLM_NAMES and scope == "codes" and not picks:
        st.error("請至少選一檔個股。")
        return
    if fresh:
        P.reset(con, cash, sweep=sweep, acct=acct)
    else:
        P.set_setting(con, "sweep", int(sweep), acct)
    if (use_llm or key in P.LLM_NAMES) and start.strftime("%Y-%m-%d") < J.EVAL_START:
        st.warning(f"起日早於 {J.EVAL_START}：模型可能記得那段行情，LLM 審核的效果會被高估。")
    if key == P.LLM_PM:
        st.warning(PM_WARN)
    bar = st.progress(0.0, "準備中…")
    try:
        if key == P.LLM_PM:
            r = P.run_llm_pm(con, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), n=int(top_n), k=int(max_pos),
                             progress=lambda f, m: bar.progress(min(f, 1.0), m), acct=acct)
            st.session_state["auto_done"] = r
            st.rerun()
        if key in P.LLM_NAMES:
            r = P.run_llm_pick(con, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), n=int(top_n), k=int(max_pos),
                               amount=float(amount), progress=lambda f, m: bar.progress(min(f, 1.0), m), acct=acct,
                               strategy=key)
            st.session_state["auto_done"] = r
            st.rerun()
        r = P.run_strategy(con, key, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), mode,
                           codes=_codes(scope, picks), top_n=int(top_n) if scope == "top" else None,
                           amount=float(amount), max_pos=int(max_pos),
                           progress=lambda f, m: bar.progress(min(f, 1.0), m), acct=acct, llm=use_llm)
    except ValueError as e:
        st.error(str(e))
        return
    st.session_state["auto_done"] = r
    st.rerun()


def _codes(scope: str, picks: list):
    return [l.split()[0] for l in picks] if scope == "codes" else ["*"] if scope == "all" else None


def _daily(con, acct: int, initial: float) -> None:
    """每日自動跟單：存設定，每天跑 python paper_daily.py（或按這裡的按鈕）接著跑到最新一天。"""
    cfg = P.auto_config(con, acct) or {}
    nm = common.names(common.data_sig()).reset_index().sort_values("code")
    labels = (nm["code"] + " " + nm["name"].fillna("")).tolist()
    last = pd.Timestamp(con.execute("SELECT MAX(date) FROM prices").fetchone()[0]).date()
    if cfg:
        st.caption(f"目前設定：{P.strategy_name(cfg['strategy'])}、{F.EXIT_MODES[cfg['mode']]}、"
                   f"{'全部股票' if cfg['codes'] == '*' else '前 %d 名' % cfg['top_n'] if cfg['top_n'] else cfg['codes']}、"
                   f"LLM 審核{'開' if cfg['llm'] else '關'}，已經跑到 {cfg['last_run'] or '（還沒跑）'}"
                   f"{'' if cfg['enabled'] else '（已停用）'}。")
    with st.form(f"daily_{acct}", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            key = st.selectbox("策略", STRATS, index=STRATS.index(cfg.get("strategy", "dip")),
                               format_func=P.strategy_name,
                               help="每天 20:00 的每日更新會把模型預測接到當天、再自動執行這裡的設定；"
                                    "選 LLM 選股或勾 LLM 審核的要開著 Ollama。" + PICK_HELP)
        with c2:
            mode = st.selectbox("出場方式", list(F.EXIT_MODES), index=list(F.EXIT_MODES).index(cfg.get("mode", "fixed")),
                                format_func=F.EXIT_MODES.get)
        with c3:
            cur = "all" if cfg.get("codes") == "*" else "top" if cfg.get("top_n") else "codes" if cfg else "all"
            scope = st.radio("範圍", list(SCOPES), index=list(SCOPES).index(cur), horizontal=True,
                             format_func=SCOPES.get)
        c1, c2 = st.columns([3, 1])
        with c1:
            have = (cfg.get("codes") or "").split(",")
            picks = st.multiselect("個股（指定個股時用）", labels, default=[l for l in labels if l.split()[0] in have])
        with c2:
            top_n = st.number_input("前 N 名", 1, 50, int(cfg.get("top_n") or 10))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            start = st.date_input("從哪天開始", value=last, max_value=last,
                                  help="含這天；之前的日子不補。之後每天接著跑")
        with c2:
            max_pos = st.number_input("最多同時持有幾檔", 1, 50, int(cfg.get("max_pos") or 10))
        with c3:
            amount = st.number_input("每筆金額", 1_000, 10_000_000, int(cfg.get("amount") or initial / 10 * 0.95),
                                     step=10_000)
        with c4:
            use_llm = st.checkbox("LLM 審核", value=bool(cfg.get("llm", 0)), help=LLM_HELP)
            enabled = st.checkbox("啟用", value=bool(cfg.get("enabled", 1)))
        save = st.form_submit_button("儲存設定", type="primary")
    if save:
        if key not in P.LLM_NAMES and scope == "top" and F.STRATEGIES[key]["kind"] != "model":
            st.error("「模型每天前 N 名」只適用 5 日、20 日、隔日模型。")
        elif key not in P.LLM_NAMES and scope == "codes" and not picks:
            st.error("請至少選一檔個股。")
        else:
            P.set_auto(con, acct, key, mode, codes=_codes(scope, picks), top_n=int(top_n) if scope == "top" else None,
                       amount=float(amount), max_pos=int(max_pos), llm=use_llm, start=start.strftime("%Y-%m-%d"),
                       enabled=enabled)
            st.rerun()
    if cfg and st.button("現在執行到最新一天", key=f"daily_run_{acct}"):
        bar = st.progress(0.0, "準備中…")
        try:
            res = P.run_auto(con, acct, progress=lambda f, m: bar.progress(min(f, 1.0), m))
            st.session_state["auto_done"] = res[0] if res else None
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    st.caption("每天股價更新後執行 `python paper_daily.py`，所有啟用的帳戶都會接著跑。"
               "想比較 LLM 有沒有用：開兩個帳戶，設定一樣，只差 LLM 審核開或關。")


def _switch(con, acct: int) -> None:
    """換策略：目前持股依新策略分類，該賣的由 LLM（或規則預設）決定出場節奏，可以手動改再下單。"""
    last = pd.Timestamp(con.execute("SELECT MAX(date) FROM prices").fetchone()[0]).date()
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        new = st.selectbox("換成哪個策略", list(F.STRATEGIES), format_func=lambda k: F.STRATEGIES[k]["name"],
                           key=f"sw_new_{acct}")
    with c2:
        asof = st.date_input("哪天收盤後換", value=last, max_value=last, key=f"sw_asof_{acct}")
    with c3:
        use_llm = st.checkbox("LLM 決定出場節奏", value=True, key=f"sw_llm_{acct}",
                              help="不勾就用規則預設：新策略中性的先不賣、偏空或沒訊號的明天賣")
    if st.button("產生換股計畫", key=f"sw_go_{acct}"):
        if use_llm and J.available():
            st.error(J.available())
            return
        bar = st.progress(0.0, "準備中…")
        try:
            plan = J.switch_plan(con, acct, new, asof.strftime("%Y-%m-%d"), use_llm=use_llm,
                                 progress=lambda f, m: bar.progress(min(f, 1.0), m))
        except Exception as e:
            st.error(f"產生計畫失敗：{e}")
            return
        st.session_state[f"sw_plan_{acct}"] = (new, asof.strftime("%Y-%m-%d"), plan)
    saved = st.session_state.get(f"sw_plan_{acct}")
    if not saved:
        st.caption("續抱：新策略也看多。緩衝：新策略中性。該賣：新策略偏空或沒有訊號。"
                   "緩衝和該賣的由 LLM 從「明天全賣／先不賣／分 3 天／分 5 天」挑一個，理由會附上資料編號。")
        return
    new, sday, plan = saved
    if plan.empty:
        st.info("這個帳戶目前沒有個股持股（0050 閒置資金不算）。")
        return
    names = common.names(common.data_sig())["name"]
    view = pd.DataFrame({"代號": plan["code"], "名稱": plan["code"].map(names), "股數": plan["shares"],
                         "新策略判斷": plan["view"], "分組": plan["group"], "節奏": plan["pace"],
                         "理由": plan["reason"]})
    opts = ["hold", "wait", "now", "stagger3", "stagger5"]
    edited = st.data_editor(view, hide_index=True, disabled=["代號", "名稱", "股數", "新策略判斷", "分組", "理由"],
                            column_config={"節奏": st.column_config.SelectboxColumn(
                                "節奏（可以改）", options=opts, required=True,
                                help="hold 續抱、wait 先不賣、now 明天開盤全賣、stagger3／5 分 3／5 天賣")},
                            key=f"sw_edit_{acct}")
    with st.expander("每檔的判斷依據（送給 LLM 的資料）"):
        for r in plan.itertuples():
            if r.items:
                st.markdown(f"**{r.code}**：{common.esc(r.reason)}")
                st.markdown("\n".join(f"- [{i}] {common.esc(t)}" for i, t in enumerate(r.items, 1)))
    if st.button("照計畫下賣單", type="primary", key=f"sw_exec_{acct}"):
        plan = plan.assign(pace=edited["節奏"].to_numpy())
        ids = J.switch_execute(con, acct, plan, sday, new)
        st.session_state.pop(f"sw_plan_{acct}", None)
        st.success(f"已下 {len(ids)} 張賣單（這些股票原本排好的賣單已取消）。續抱和先不賣的，原本策略排好的賣單還在，"
                   "要改可以到委託紀錄取消。新策略要買的股票，用上面的「策略自動下單」或「每日自動跟單」換成新策略。")
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
    # 盤中（intraday.py 在抓即時報價）每 15 秒重畫淨值與持股；成交、停損停利由 intraday.py 結算
    live = st.fragment(run_every=15 if intraday.is_open() else None)
    live(_summary)(con, acct)
    done = st.session_state.pop("auto_done", None)
    if done and "批次" in done:
        llm = f"，LLM 略過 {done['LLM 略過']}、減半 {done['LLM 減半']}" if "LLM 略過" in done else ""
        st.success(f"自動下單完成：{done['批次']}，進場 {done.get('進場筆數', 0)} 筆{llm}，"
                   f"已實現損益 {done.get('已實現損益', 0):+,.0f}。")
    elif done:
        st.info(f"每日自動跟單：{done.get('結果', '')}")
    with st.expander("策略自動下單", expanded=P.orders(con, acct).empty):
        st.caption("從起日到迄日逐個交易日，每天只用當天收盤以前的資料判斷有沒有進場訊號（和「個股預測」頁同一套），"
                   "有就在當天收盤後下買單、隔天開盤成交，再照出場方式掛賣單。同一檔持有中不重複進場。")
        _auto(con, p["initial"], acct)
    with st.expander("每日自動跟單"):
        _daily(con, acct, p["initial"])
    with st.expander("換策略（LLM 決定舊持股的出場節奏）"):
        _switch(con, acct)
    _batches(con, acct)
    live(_holdings)(con, acct, names)
    _orders(con, acct, names)
    _manage(con, acct, p)
    st.markdown(f'<div class="disc">{common.esc(common.DISCLAIMER)} 沒有模擬每筆最低手續費、'
                '漲跌停買不到賣不掉與滑價。</div>', unsafe_allow_html=True)


def _summary(con, acct: int) -> None:
    """淨值、報酬、現金、損益與逐日淨值圖。"""
    p = P.portfolio(con, acct)
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


def _holdings(con, acct: int, names: pd.Series) -> None:
    st.subheader("持股")
    pos = P.portfolio(con, acct)["positions"]
    if pos.empty:
        common.empty("目前沒有持股。到「個股預測」頁選策略、送出模擬單。")
        return
    tb = pd.DataFrame({"代號": pos["code"], "名稱": pos["code"].map(names), "股數": pos["shares"],
                       "平均成本": pos["avg_cost"], "現價": pos["last"], "日期": pos["last_date"],
                       "市值": pos["value"], "未實現": pos["unrealized"],
                       "報酬率": pos["unrealized"] / (pos["avg_cost"] * pos["shares"])})
    common.table(common.styled(tb, fmt={"股數": "{:,.0f}", "平均成本": "{:.2f}", "現價": "{:.2f}",
                                        "市值": "{:,.0f}", "未實現": "{:+,.0f}", "報酬率": "{:+.2%}"},
                               signed=["未實現", "報酬率"]), hide_index=True)
    if pos["last_date"].astype(str).str.endswith("盤中").any():
        st.caption("日期寫「盤中」的是證交所即時價（換算成還原價口徑），其他是 DB 最新收盤。盤中每 15 秒更新。")


def _orders(con, acct: int, names: pd.Series) -> None:
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
            st.caption("待成交：下單日之後還沒有開盤價（例如預約的出場日還沒到）。股價更新後重新整理就會成交；"
                       "盤中報價在跑的話，開盤價出來和碰到停損／目標時就會成交。")


main()
