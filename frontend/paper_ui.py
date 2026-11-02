"""模擬帳戶的共用選單：「個股預測」與「模擬交易」頁選的是同一個帳戶（存在 session_state）。

每個瀏覽器分頁各自記得自己選的帳戶，所以多台電腦連進來時，各人選各人的帳戶就不會互相影響。
"""
import streamlit as st

import paper_trade as P


def pick(con, where: str, label: str = "模擬帳戶") -> int:
    acc = P.accounts(con)
    ids = [int(i) for i in acc["id"]]
    names = dict(zip(ids, acc["name"]))
    cur = st.session_state.get("paper_acct")
    if cur not in ids:
        cur = ids[0]
    # 兩頁的選單 key 不同，值一律跟著 paper_acct 走。使用者改選時 on_change 先更新 paper_acct
    # （callback 在這段程式之前執行），所以這裡覆寫不會蓋掉使用者的選擇；新增帳戶後也會自動切過去
    key = f"acct_{where}"
    st.session_state["paper_acct"] = cur
    st.session_state[key] = cur

    def changed():
        st.session_state["paper_acct"] = st.session_state[key]

    return st.selectbox(label, ids, format_func=lambda i: f"{names[i]}（#{i}）", key=key, on_change=changed)
