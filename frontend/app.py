"""本機研究介面。只綁 127.0.0.1，見 .streamlit/config.toml。

架構與樣式沿用 jameschen108/tw-stock-quant 的前端，內容換成本專案的研究結果。
"""
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common

st.set_page_config(page_title="台股 DAViS 延伸研究（本機）", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="collapsed")
common.setup()

page = st.navigation([
    st.Page("p1_overview.py", title="總覽", icon=":material/dashboard:", default=True),
    st.Page("p2_stock.py", title="個股", icon=":material/query_stats:", url_path="stock"),
    st.Page("p3_strategy.py", title="策略", icon=":material/science:", url_path="strategy"),
    st.Page("p4_revenue.py", title="營收研究", icon=":material/insights:", url_path="revenue"),
    st.Page("p6_sectors.py", title="產業", icon=":material/category:", url_path="sectors"),
    st.Page("p5_findings.py", title="研究結論", icon=":material/menu_book:", url_path="findings"),
], position="top")
common.header()
page.run()
