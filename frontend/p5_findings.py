"""研究結論：研究結論總整理.md，依章節切開閱讀。"""
import re
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common


@st.cache_data(show_spinner=False, max_entries=2)
def _sections(mtime: int) -> list[tuple[str, str]]:
    text = common.FINDINGS.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", text)
    head, rest = parts[0], parts[1:]
    out = [("開頭", head.split("\n", 1)[1] if "\n" in head else head)]
    for p in rest:
        title, _, body = p.partition("\n")
        out.append((title.strip(), body.strip().rstrip("-").strip()))
    return out


def main():
    common.page_title("研究結論", "results/研究結論總整理.md")
    if not common.FINDINGS.exists():
        common.empty("找不到 results/研究結論總整理.md。")
        return
    secs = _sections(common.FINDINGS.stat().st_mtime_ns)
    titles = [t for t, _ in secs]
    q = st.query_params.get("sec")
    nav, body = st.columns([1, 3.2], gap="large")
    with nav:
        pick = st.radio("章節", titles, index=titles.index(q) if q in titles else 1 if len(titles) > 1 else 0,
                        label_visibility="collapsed")
        st.caption(f"最後修改 {common._mdate(common.FINDINGS)}")
    if st.query_params.get("sec") != pick:
        st.query_params["sec"] = pick
    with body:
        if pick != "開頭":
            st.markdown(f"## {pick}")
        # 文件用「0.053~0.061」表示區間，Streamlit 會把成對的 ~ 當成刪除線
        st.markdown(dict(secs)[pick].replace("~", r"\~"))


main()
