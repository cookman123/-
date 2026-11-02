"""地端問答：用本機 Ollama 模型，根據 db/local.db 回答問題（local_ask.py 的前端）。"""
import sys
from datetime import date
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import local_ask as L

EXAMPLES = [
    "跌深反彈策略有效嗎？RAG 排除規則可以直接用嗎？",
    "這檔現在適合做反彈嗎？規則怎麼判斷？",
    "最近有哪些新聞可能影響這檔股票？請照原文列出標題。",
]
NO_STOCK = "（不指定個股）"


@st.cache_data(show_spinner=False, ttl=30)
def _health() -> list:
    return L.health()


def _controls(sig: str) -> tuple:
    nm = common.names(sig).reset_index().sort_values("code")
    labels = [NO_STOCK] + (nm["code"] + " " + nm["name"].fillna("")).tolist()
    ex = st.pills("範例", EXAMPLES, label_visibility="collapsed")
    if ex and st.session_state.get("ask_q") != ex:
        st.session_state["ask_q"] = ex
    q = st.text_area("問題", key="ask_q", height=90,
                     placeholder="例如：這檔現在適合做反彈嗎？規則怎麼判斷？")
    c1, c2, c3 = st.columns([2, 1.2, 1])
    with c1:
        pick = st.selectbox("個股（選了會附上價格、RAG 分數、最近新聞）", labels, key="ask_code")
    with c2:
        asof = st.date_input("視角日期（只用這天以前已知的資料）", value=date.today(),
                             min_value=date(2011, 1, 1), max_value=date.today(), key="ask_asof")
    with c3:
        k = st.slider("檢索段數", 4, 12, 8, key="ask_k")
    code = None if pick == NO_STOCK else pick.split()[0]
    return q.strip(), code, asof.strftime("%Y-%m-%d"), k


def _show(r: dict) -> None:
    tags = " ".join(([common.badge(r["code"], "info")] if r["code"] else [])
                    + [common.badge(f"視角 {r['asof']}", "muted")])
    st.markdown(f"**{common.esc(r['q'])}** {tags}", unsafe_allow_html=True)
    # 回答裡的「0.05~0.06」會被 Markdown 當成刪除線
    st.markdown(r["answer"].replace("~", r"\~"))
    if r["bad"]:
        st.warning(f"回答仍引用不存在的出處 {', '.join(f'[{b}]' for b in r['bad'])}，這些句子請勿採信。")
    with st.expander(f"出處 {len(r['items'])} 項：規則定義 {r['n_rule']}、結構化資料 {r['n_ctx']}、"
                     f"檢索 {r['n_hits']}　·　{r['secs']:.0f} 秒"):
        for i, t in enumerate(r["items"], 1):
            st.markdown(f"**[{i}]** {common.esc(t)}".replace("\n", "  \n").replace("~", r"\~"))


def main():
    common.page_title("地端問答", f"Ollama {L.CHAT_MODEL} · 只用本機資料 · 模型只解釋，買賣以數字規則為準")
    problems = _health()
    if problems:
        common.empty("地端模型還沒就緒：" + "；".join(problems))
        st.caption("步驟見 地端部署說明.md。處理完重新整理頁面。")
        return

    sig = common.data_sig()
    q, code, asof, k = _controls(sig)
    if st.button("提問", type="primary", disabled=not q):
        import time

        t0 = time.time()
        with st.spinner("本機模型回答中，大約 30 到 60 秒…"):
            try:
                r = L.ask(q, code=code, asof=asof, k=k)
            except Exception as e:                       # Ollama 中途關掉、模型被卸載等
                st.error(f"問答失敗：{e}")
                return
        r |= {"q": q, "code": code, "secs": time.time() - t0}
        st.session_state.setdefault("ask_hist", []).insert(0, r)

    hist = st.session_state.get("ask_hist", [])
    if not hist:
        st.caption("回答會附上出處編號，展開「出處」可以對照原始資料。重要的數字請回頭核對。")
    for i, r in enumerate(hist):
        if i:
            st.divider()
        _show(r)
    st.caption("這是研究規則的輸出，不是投資建議。")


main()
