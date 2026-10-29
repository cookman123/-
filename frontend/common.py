"""本機介面的共用部分：路徑、樣式、資料狀態列、快取、圖表。

版面、樣式與狀態列沿用 jameschen108/tw-stock-quant 的 frontend/common.py，
資料改接本專案的 data/ 與 results/。前端只讀不寫，例外是「執行」視窗跑的固定指令。
"""
import html
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA, RES = ROOT / "data", ROOT / "results"
FIN = DATA / "financials"
PICKS = RES / "spec" / "current_picks.csv"
SPEC_LOG = RES / "spec" / "run.log"
EVENT_DIR = RES / "revenue_event"
FINDINGS = RES / "研究結論總整理.md"

DISCLAIMER = "學術研究用。清單與事件是規則的輸出，回測與歷史統計不保證未來報酬，不構成投資建議。"

# 台股慣例：漲紅跌綠
UP, DOWN = "#e5484d", "#18a058"
C_STRAT, C_POOL, C_ETF = "#3b82f6", "#94a3b8", "#f59e0b"

CSS = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1440px;}
[data-testid="stBottomBlockContainer"] {max-width: 1440px; margin: 0 auto;}
h1, h2, h3 {letter-spacing: .01em;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.sbar {display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; font-size:.78rem; margin:0; opacity:.75;}
.sbar b {font-weight:600;}
.sbar i {font-style:normal; opacity:.45;}
.sbar + .disc {margin:.1rem 0 0;}
[data-testid="stHorizontalBlock"]:has(.sbar) {align-items:center; margin-bottom:.4rem;}
.fresh {padding:.4rem .8rem; border-radius:.5rem; font-size:.85rem; background:rgba(234,179,8,.12);
        border:1px solid rgba(234,179,8,.45);}
.fresh ul {margin:0; padding-left:1.1rem;}
[data-testid="stHorizontalBlock"]:has(.fresh) {align-items:center; margin-bottom:.4rem;}
[data-testid="stHorizontalBlock"]:has(.sbar) [data-testid="stButton"] button,
[data-testid="stHorizontalBlock"]:has(.fresh) [data-testid="stButton"] button {
  border: 1px solid rgba(128,128,128,.22); background: transparent; min-height: 1.7rem;
  padding: .05rem .7rem; font-size: .8rem; border-radius: 999px; white-space: nowrap;}
.chip {padding:.12rem .6rem; border-radius:999px; background:rgba(128,128,128,.10);
       border:1px solid rgba(128,128,128,.22); white-space:nowrap;}
.chip b {font-weight:600;}
.chip.warn {background:rgba(234,179,8,.14); border-color:rgba(234,179,8,.55);}
.chip.bad {background:rgba(229,72,77,.14); border-color:rgba(229,72,77,.55);}
.disc {font-size:.76rem; opacity:.62; margin:.15rem 0 .6rem;}
.ptitle {display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; margin:.4rem 0 .6rem;}
.ptitle h1 {font-size:1.9rem; margin:0; padding:0;}
.ptitle .sub {opacity:.65; font-size:.9rem;}
.kgrid {display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.7rem; margin:.3rem 0 1rem;}
.kpi {border:1px solid rgba(128,128,128,.22); border-radius:.8rem; padding:.7rem .95rem; background:rgba(128,128,128,.05);}
.kpi .t {font-size:.78rem; opacity:.68;}
.kpi .v {font-size:1.5rem; font-weight:650; line-height:1.35; font-variant-numeric:tabular-nums;}
.kpi .s {font-size:.76rem; opacity:.62;}
.kpi.accent {border-color:rgba(59,130,246,.5); background:rgba(59,130,246,.06);}  /* 紅綠留給漲跌 */
.badge {display:inline-block; padding:.05rem .5rem; border-radius:.4rem; font-size:.78rem; font-weight:600;
        margin:0 .25rem .2rem 0; white-space:nowrap;}
.badge.bull {background:rgba(229,72,77,.15); color:#e5484d;}
.badge.bear {background:rgba(24,160,88,.15); color:#18a058;}
.badge.neutral, .badge.muted {background:rgba(128,128,128,.14);}
.badge.info {background:rgba(59,130,246,.14); color:#3b82f6;}
.up {color:#e5484d;} .down {color:#18a058;}
.cgrid {display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:.7rem; margin:.3rem 0 1rem;}
.card {border:1px solid rgba(128,128,128,.22); border-radius:.8rem; padding:.7rem .95rem;}
.card .hd {display:flex; justify-content:space-between; align-items:baseline; gap:.5rem; margin-bottom:.3rem;}
.card .hd b {font-size:1rem;}
.card ul {margin:0; padding-left:1.1rem; font-size:.86rem; line-height:1.55;}
.card li.alert::marker {color:#e5484d;}
.hero {display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; border:1px solid rgba(128,128,128,.22);
       border-radius:.9rem; padding:1rem 1.2rem; margin:.4rem 0 .9rem; background:rgba(128,128,128,.04);}
.hero .nm {font-size:1.5rem; font-weight:650;}
.hero .cd {opacity:.6; font-size:.95rem; margin-left:.4rem;}
.hero .px {font-size:2.1rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.2;}
.hero .chg {font-size:1.05rem; font-weight:600; margin-left:.5rem;}
.hero .meta {opacity:.6; font-size:.8rem;}
.hero .right {display:flex; flex-direction:column; gap:.35rem; align-items:flex-start; min-width:280px;}
.hero .row {font-size:.85rem;}
.hero .row .k {opacity:.6; display:inline-block; width:4.2rem;}
.range {position:relative; height:6px; border-radius:3px; background:rgba(128,128,128,.25); width:220px; margin-top:.3rem;}
.range i {position:absolute; top:-4px; width:3px; height:14px; background:currentColor; border-radius:2px;}
.empty {border:1px dashed rgba(128,128,128,.35); border-radius:.8rem; padding:1rem 1.2rem; opacity:.8; font-size:.9rem;}
</style>
"""


# ---------------------------------------------------------------- 版面

def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def badge(label, kind: str = "muted") -> str:
    return f'<span class="badge {kind}">{esc(label)}</span>'


def signed_html(x, nd=1, scale=100.0, suffix="%") -> str:
    v = _num(x)
    if v is None:
        return "—"
    cls = "up" if v > 0 else ("down" if v < 0 else "")
    return f'<span class="{cls}">{v * scale:+.{nd}f}{suffix}</span>'


def page_title(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="ptitle"><h1>{esc(title)}</h1><span class="sub">{esc(sub)}</span></div>',
                unsafe_allow_html=True)


def segmented(label: str, options: list, default, key: str, **kw):
    """單選的 segmented_control。原生的再點一次已選的會變成 None，這裡改回上一個選項。"""
    last = f"_{key}_last"
    if st.session_state.get(key) not in options:
        st.session_state[key] = st.session_state.get(last) if st.session_state.get(last) in options else default

    def keep():
        if st.session_state.get(key) is None:
            st.session_state[key] = st.session_state.get(last, default)
        st.session_state[last] = st.session_state[key]

    st.session_state.setdefault(last, st.session_state[key])
    return st.segmented_control(label, options, key=key, on_change=keep, **kw) or default


def table(data, **kw):
    """st.dataframe，缺值顯示「—」。原生預設顯示 None，Styler 的 na_rep 管不到。"""
    return st.dataframe(data, placeholder="—", **kw)


def kpis(items: list[tuple]) -> None:
    """items: (標題, 值 HTML, 小字, accent?)。值已經是 HTML，呼叫端負責 escape。"""
    cells = []
    for it in items:
        t, v, s = it[0], it[1], it[2] if len(it) > 2 else ""
        acc = " accent" if len(it) > 3 and it[3] else ""
        cells.append(f'<div class="kpi{acc}"><div class="t">{esc(t)}</div><div class="v">{v}</div>'
                     f'<div class="s">{esc(s)}</div></div>')
    st.markdown(f'<div class="kgrid">{"".join(cells)}</div>', unsafe_allow_html=True)


def empty(msg: str) -> None:
    st.markdown(f'<div class="empty">{esc(msg)}</div>', unsafe_allow_html=True)


def _mdate(p: Path) -> str | None:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d") if p.exists() else None


@st.cache_data(show_spinner=False, max_entries=2)
def status(_sig: str) -> dict:
    """狀態列用的日期。前端不維護獨立的狀態檔，直接看資料本身。"""
    tw = twii(_sig)
    rv = revenue_all(_sig)
    q = quarterly_all(_sig)
    return {
        "price": f"{tw.index[-1]:%Y-%m-%d}" if len(tw) else None,
        "revenue": rv["ym"].max() if len(rv) else None,
        "quarterly": f"{int(q['year'].max())}Q{int(q[q['year'] == q['year'].max()]['q'].max())}" if len(q) else None,
        "pool": pool(_sig)[0],
        "picks": _mdate(PICKS),
        "spec_log": _mdate(SPEC_LOG),
        "event": _mdate(EVENT_DIR / "bench.csv"),
    }


def data_sig() -> str:
    """資料或結果改過，快取就失效。"""
    paths = [DATA / "market" / "TWII.csv", FIN / "revenue.csv", FIN / "quarterly.csv", PICKS,
             EVENT_DIR / "bench.csv", RES / "fundamental" / "factors_with_price_extra.csv"]
    return "|".join(str(p.stat().st_mtime_ns) if p.exists() else "0" for p in paths)


# ---------------------------------------------------------------- 執行

JOBS = [
    ("更新股價與指數", "daily_update.py --no-news，約數十分鐘；各檔沒對齊會以結束碼 2 結束",
     ["daily_update.py", "--no-news"]),
    ("重算高波動清單", "strategy_spec.py：回測加上「今天換倉會買哪 30 檔」", ["strategy_spec.py"]),
    ("重跑營收事件研究", "revenue_event_study.py：要讀全部價格，約數分鐘", ["revenue_event_study.py"]),
]


def _run_job(title: str, args: list[str]) -> None:
    lock = ROOT / "logs" / ".frontend_job.lock"
    lock.parent.mkdir(exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        st.warning(f"已經有一個指令在跑。確定沒有的話，刪掉 {lock.relative_to(ROOT)} 再試。")
        return
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        with st.status(f"{title}…", expanded=True) as box:
            proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", env=env)
            lines: list[str] = []
            slot = st.empty()
            assert proc.stdout is not None
            for raw in proc.stdout:
                lines.append(raw.rstrip("\n"))
                slot.code("\n".join(lines[-24:]), language="text")
            code = proc.wait()
            st.session_state["job_last"] = {"title": title, "code": code,
                                            "text": "\n".join(lines[-80:]) or "（沒有輸出）"}
            if code == 0:
                box.update(label=f"{title} 完成", state="complete", expanded=False)
                st.cache_data.clear()
            else:
                box.update(label=f"{title} 結束碼 {code}", state="error")
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def _open_jobs() -> None:
    st.session_state.jobs_open = True


def _close_jobs() -> None:
    st.session_state.jobs_open = False


@st.dialog("執行", width="small", on_dismiss=_close_jobs)
def _jobs_dialog() -> None:
    st.caption("只跑固定指令。滑過按鈕可看它做什麼。關掉視窗後，頁面會讀到新結果。")
    for i, (title, note, args) in enumerate(JOBS):
        if st.button(title, key=f"job_{i}", help=note, width="stretch"):
            _run_job(title, args)
    last = st.session_state.get("job_last")
    if last:
        st.caption(f"上次：{last['title']}　結束碼 {last['code']}")
        st.code(last["text"], language="text")


def _trading_lag(last: str) -> int:
    """價格日之後到昨天為止的平日數（今天的行情收盤前還不會有）。國定假日沒有扣，所以是上限。"""
    import numpy as np
    return int(np.busday_count(pd.Timestamp(last).date() + pd.Timedelta(days=1), pd.Timestamp.today().date()))


def header() -> None:
    """每頁頂端：一行資料日期小字；有過舊或沒對齊才另外提醒，旁邊放更新按鈕。"""
    s = status(data_sig())
    items = [f'<b>價格</b> {esc(s["price"] or "—")}',
             f'<b>月營收</b> {esc(s["revenue"] or "—")}',
             f'<b>季報</b> {esc(s["quarterly"] or "—")}',
             f'<b>母體</b> {esc(s["pool"] or "—")}',
             f'<b>高波動清單</b> {esc(s["picks"] or "—")}',
             f'<b>事件研究</b> {esc(s["event"] or "—")}']
    warns = []
    if s["price"]:
        lag = _trading_lag(s["price"])
        if lag > 2:
            warns.append(f"股價停在 {s['price']}，之後已經過了約 {lag} 個交易日")
    if s["picks"] and s["spec_log"] and s["picks"] < s["spec_log"]:
        warns.append("高波動清單比 strategy_spec 最後一次的執行紀錄舊，清單可能不是最新規則的輸出，建議重算")
    if s["pool"] and s["price"] and s["pool"] < s["price"]:
        warns.append(f"母體（成交金額排名快取）只到 {s['pool']}，比股價舊")
    if warns:
        items.append('<span class="chip warn">有提醒</span>')
    bar, act = st.columns([1, 0.16], gap="small", vertical_alignment="center")
    bar.markdown(f'<div class="sbar">{"<i>·</i>".join(items)}</div>'
                 f'<div class="disc">{DISCLAIMER}</div>', unsafe_allow_html=True)
    act.button("更新 / 重算", type="tertiary", key="open_jobs_btn", on_click=_open_jobs, width="stretch",
               help="更新資料或重算。只在這個視窗裡跑固定指令")
    if st.session_state.get("jobs_open"):
        _jobs_dialog()
    if warns:
        msg, btn = st.columns([1, 0.16], gap="small", vertical_alignment="center")
        body = esc(warns[0]) if len(warns) == 1 else "<ul>" + "".join(f"<li>{esc(w)}</li>" for w in warns) + "</ul>"
        msg.markdown(f'<div class="fresh">{body}</div>', unsafe_allow_html=True)
        btn.button("更新資料", key="fresh_update_btn", on_click=_open_jobs, width="stretch")


# ---------------------------------------------------------------- 格式

def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(x, nd=1, scale=100.0) -> str:
    v = _num(x)
    return "—" if v is None else f"{v * scale:+.{nd}f}%"


def num(x, nd=2) -> str:
    v = _num(x)
    return "—" if v is None else f"{v:,.{nd}f}"


def color_signed(v) -> str:
    v = _num(v)
    if v is None or v == 0:
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}"


def styled(df: pd.DataFrame, fmt: dict | None = None, signed: list | None = None):
    s = df.style
    if signed:
        s = s.map(color_signed, subset=[c for c in signed if c in df.columns])
    if fmt:
        s = s.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="—")
    return s


# ---------------------------------------------------------------- 資料（快取）
# 第一個參數是 data_sig()：資料檔一改，快取就重讀。

@st.cache_data(show_spinner=False, max_entries=1)
def names(_sig: str) -> pd.DataFrame:
    r = json.load(open(DATA / "industry_stock_map.json", encoding="utf-8"))
    return pd.DataFrame(r).astype({"code": str}).set_index("code")


@st.cache_data(show_spinner=False, max_entries=64)
def price(_sig: str, code: str) -> pd.DataFrame:
    import davis_tw_train as T
    px = T.load_price(code)
    return pd.DataFrame() if px is None else px


@st.cache_data(show_spinner=False, max_entries=1)
def twii(_sig: str) -> pd.Series:
    d = pd.read_csv(DATA / "market" / "TWII.csv", parse_dates=["date"]).set_index("date").sort_index()
    return d["close"][~d.index.duplicated(keep="last")]


@st.cache_data(show_spinner=False, max_entries=1)
def revenue_all(_sig: str) -> pd.DataFrame:
    return pd.read_csv(FIN / "revenue.csv", dtype={"code": str}, parse_dates=["avail_date"])


@st.cache_data(show_spinner=False, max_entries=1)
def quarterly_all(_sig: str) -> pd.DataFrame:
    return pd.read_csv(FIN / "quarterly.csv", dtype={"code": str}, parse_dates=["avail_date"])


@st.cache_data(show_spinner=False, max_entries=64)
def chips(_sig: str, code: str) -> pd.DataFrame:
    p = DATA / "chips" / f"{code}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()


@st.cache_data(show_spinner=False, max_entries=1)
def pool(_sig: str) -> tuple[str | None, list[str]]:
    """最新一天的時點正確母體（60 日均成交金額前 250 名，universe_pit 的快取）。"""
    tr = DATA / "market" / "turnover_rank.csv"
    if not tr.exists():
        return None, []
    t = pd.read_csv(tr, index_col=0, parse_dates=True).tail(30)
    # 同 strategy_spec.py：最後幾天可能只有少數幾檔被重抓過，取九成以上都有資料的最後一天
    n = t.notna().sum(axis=1)
    last = n[n >= n.max() * 0.9].index[-1]
    row = t.loc[last].dropna().sort_values(ascending=False)
    return f"{last:%Y-%m-%d}", list(row.index[:250].astype(str))


@st.cache_data(show_spinner=False, max_entries=1)
def revenue_flags(_sig: str) -> pd.DataFrame:
    """每筆月營收加上年增、累計年增、是否創歷史新高（同 revenue_event_study 的定義）。"""
    import revenue_event_study as RE
    r = RE.revenue_events()
    r["ym"] = r["p"].astype(str)
    return r.drop(columns=["p"])


@st.cache_data(show_spinner=False, max_entries=1)
def picks(_sig: str) -> pd.DataFrame:
    return pd.read_csv(PICKS, dtype={"代號": str}, encoding="utf-8-sig") if PICKS.exists() else pd.DataFrame()


def held(p: pd.DataFrame) -> pd.DataFrame:
    """清單裡換倉後仍持有的部分（緩衝帶版會多列「賣出」的舊持股）。"""
    return p[p["動作"] != "賣出"] if "動作" in p.columns else p


@st.cache_data(show_spinner=False, max_entries=8)
def result_csv(_sig: str, rel: str, **kw) -> pd.DataFrame:
    p = RES / rel
    return pd.read_csv(p, encoding="utf-8-sig", **kw) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=1)
def events(_sig: str) -> pd.DataFrame:
    p = EVENT_DIR / "events.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={"code": str}, parse_dates=["avail_date", "entry"])


@st.cache_resource(show_spinner="讀取 383 檔行情…", max_entries=1)
def all_prices(_sig: str, since: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """since 之後的日報酬與成交金額寬表（date × code）。"""
    import davis_tw_train as T
    rets, val = {}, {}
    for code in names(_sig).index:
        px = T.load_price(code)
        if px is None or px.empty:
            continue
        px = px[px.index >= pd.Timestamp(since)]
        rets[code] = px["close"].pct_change(fill_method=None)
        val[code] = px["close"] * px["volume"]
    rets, val = pd.DataFrame(rets).sort_index(), pd.DataFrame(val).sort_index()
    # 尾端只有少數幾檔被重抓過的日子切掉（同 strategy_spec.py 的全覆蓋日）
    n = val.notna().sum(axis=1)
    last = n[n >= n.max() * 0.9].index[-1]
    return rets.loc[:last], val.loc[:last]


def goto_stock(code: str) -> None:
    st.session_state["goto_code"] = code
    st.switch_page("p2_stock.py")


def pick_row(ev, df: pd.DataFrame, col: str = "代號") -> None:
    sel = ev.selection.rows if ev and ev.selection else []
    if sel:
        goto_stock(str(df.iloc[sel[0]][col]))


# ---------------------------------------------------------------- 圖表

def fig_layout(fig, title: str = "", height: int = 340, legend: bool = True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        height=height, margin=dict(l=8, r=8, t=40 if title else 10, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1) if legend else None,
        showlegend=legend, hovermode="x unified",
    )
    return fig


def line_fig(xs, series: dict[str, pd.Series], title: str, colors: dict | None = None, height: int = 340):
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, y in series.items():
        line = dict(width=1.8)
        if colors and name in colors:
            line["color"] = colors[name]
        fig.add_trace(go.Scatter(x=list(xs), y=list(y), name=name, mode="lines", line=line))
    return fig_layout(fig, title, height)


def chart(fig) -> None:
    st.plotly_chart(fig, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
