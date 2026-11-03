#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模擬下單：流程圖「實際操作 → 回傳至 DB 留存」的模擬版，存在 db/local.db 的 paper_orders。

可以開多個模擬帳戶（paper_accounts 表）：每個帳戶有自己的名稱、起始資金、閒置資金設定，
委託、現金、持股、自動下單批次都分開算。所有函式的 acct 參數就是帳戶編號，預設 1（預設帳戶）。
舊版只有一個帳戶，設定存在 paper_account（key／value）表；第一次執行時自動搬成 1 號帳戶。

成交規則和所有回測一樣：下單日（視角日期）收盤後才下單，**下一個交易日開盤**成交。
下單日可以是過去（回到當時的視角模擬），也可以是未來（預約賣出：到那天收盤後送出，
隔天開盤成交）。還沒有隔天開盤價的單，狀態是「待成交」，DB 更新股價後再重新結算就會成交。

條件賣單（給了 stop／target／expire_date）：從下單日的下一個交易日起，每天檢查停損與目標價，
觸發規則和 forecast.simulate_exit 相同（開盤跳空用開盤價、盤中碰到用該價位、同一天都碰到算停損），
都沒碰到就在 expire_date 開盤賣出。

策略自動下單（run_strategy）：選一套 forecast.py 的策略和出場方式，從起日到迄日逐個交易日，
每天只用當天收盤以前的資料判斷有沒有進場訊號（和「個股預測」頁同一套），有就下買單和對應的賣單。
同一檔持有中不重複進場，同時持有不超過 max_pos 檔。同一批的委託共用一個 batch，可以整批刪除。
llm=True 時每個訊號先給地端 LLM 審核（llm_judge.review）：buy 照買、half 買一半、skip 這次不買。

LLM 選股（run_llm_pick，策略代號 llm_pick）：流程圖「實際操作」那一格。規則只給候選池（5 日模型前 N 名），
由 LLM 依「RAG 行為模式＋新資訊評分」挑最多 K 檔、給權重（llm_judge.select），持有 5 日、開盤進出。
上一期的部位出場後才開下一期，所以各期不重疊；每日自動跟單也能用。

LLM 自主操盤（run_llm_pm，策略代號 llm_pm）：LLM 自己選股、自己組投資組合。每 every 個交易日一次，
llm_judge.manage 讀研究結論（RAG）、DB 裡各策略的輸出、大盤與目前持股，給出目標權重（佔總資產）；
這裡把目標和目前持股的差下成隔日開盤的買賣單。沒有固定持有期，什麼時候賣也由 LLM 決定。
差距小於總資產 min_trade 的不動，避免為了微調付成本；剩下的錢照帳戶設定放 0050。

每日自動跟單（paper_auto 表、run_auto）：每個帳戶可以存一組自動下單設定，每天資料更新後執行
python paper_daily.py，從上次跑到的那天接著跑到最新一天。開兩個帳戶，一個只用規則、一個加 LLM 審核，
往後累積的紀錄就是兩者最乾淨的比較（未來的行情模型不可能看過）。

閒置資金買 0050（帳戶設定 sweep，預設開）：和 reversal_two_investors.py 的投資人 B 同一個做法，
沒有用在個股上的錢一律放 0050（含息還原價，etf_prices 表）。每個有成交的交易日開盤：先處理賣單，
買個股錢不夠就賣掉剛好夠的 0050，當天的單都處理完，剩下的現金再買回 0050。
賣 0050 付手續費＋ETF 證交稅 0.1%，買付手續費。這些換手是系統產生的（batch = SWEEP），
每次結算都從頭重算，不要手動改。

成本同 strategy_spec.py：手續費 0.1425% × 3 折、買賣各一次，賣出另加證交稅 0.3%。
實際券商有每筆最低 20 元手續費、漲跌停鎖死買不到賣不掉，這裡都沒有模擬。

paper_orders、paper_accounts 和 ops_log 一樣，build_local_db.py 重建時不會清掉。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / "db" / "local.db"
FEE, TAX = 0.001425 * 0.3, 0.003
ETF, ETF_TAX, SWEEP = "0050", 0.001, "__0050__"
INITIAL_CASH = 1_000_000
STATUS = {"pending": "待成交", "filled": "已成交", "rejected": "退單", "cancelled": "已取消"}
LLM_PICK, LLM_SCORE, LLM_PM, REV_PICK = "llm_pick", "llm_score", "llm_pm", "rev_pick"
LLM_NAMES = {LLM_PM: "LLM 自主操盤（研究結論 RAG＋DB 分析，自己組投資組合）",
             LLM_PICK: "LLM 選股（RAG＋新資訊評分）", LLM_SCORE: "LLM 打分第二版（加基本面與新聞內文）"}
# 不用 LLM 的對照組：和 LLM 選股同一個候選池、同樣持有 5 日，只換成營收年增率排序（llm_score_eval.py 的 D 組）
PICK_NAMES = LLM_NAMES | {REV_PICK: "規則對照：營收年增前 K 名（5 日模型前 N 名裡）"}


def strategy_name(key: str) -> str:
    import forecast as F
    return PICK_NAMES.get(key) or F.STRATEGIES[key]["name"]
SETTINGS = ("initial_cash", "sweep", "sweep_start")          # paper_accounts 裡可以用 setting() 讀寫的欄位


def connect(db: str | Path = DB) -> sqlite3.Connection:
    # 盤中報價（intraday.py）同時在寫，等鎖久一點
    con = sqlite3.connect(db, timeout=30, check_same_thread=False)
    ensure(con)
    return con


def ensure(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS paper_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        order_date TEXT NOT NULL, code TEXT NOT NULL, side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
        shares INTEGER NOT NULL CHECK (shares > 0), strategy TEXT, note TEXT,
        status TEXT NOT NULL DEFAULT 'pending', fill_date TEXT, fill_price REAL,
        fee REAL, tax REAL, reason TEXT)""")
    # 後來加的欄位；舊表補上
    have = {r[1] for r in con.execute("PRAGMA table_info(paper_orders)")}
    for col, typ in (("stop", "REAL"), ("target", "REAL"), ("expire_date", "TEXT"), ("trigger", "TEXT"),
                     ("batch", "TEXT"), ("parent", "INTEGER"), ("account", "INTEGER NOT NULL DEFAULT 1")):
        if col not in have:
            con.execute(f"ALTER TABLE paper_orders ADD COLUMN {col} {typ}")
    con.execute("""CREATE TABLE IF NOT EXISTS paper_auto (
        account INTEGER PRIMARY KEY, strategy TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'fixed',
        codes TEXT, top_n INTEGER, amount REAL NOT NULL, max_pos INTEGER NOT NULL, llm INTEGER NOT NULL DEFAULT 0,
        last_run TEXT, enabled INTEGER NOT NULL DEFAULT 1)""")
    con.execute("""CREATE TABLE IF NOT EXISTS paper_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
        initial_cash REAL NOT NULL, sweep INTEGER NOT NULL DEFAULT 1, sweep_start TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')))""")
    if not con.execute("SELECT 1 FROM paper_accounts").fetchone():
        # 舊版單一帳戶的設定搬過來當 1 號帳戶
        old = {}
        if con.execute("SELECT name FROM sqlite_master WHERE name = 'paper_account'").fetchone():
            old = dict(con.execute("SELECT key, value FROM paper_account"))
        con.execute("INSERT INTO paper_accounts (id, name, initial_cash, sweep, sweep_start) VALUES (1, ?, ?, ?, ?)",
                    ["預設帳戶", float(old.get("initial_cash", INITIAL_CASH)), int(old.get("sweep", "1")),
                     old.get("sweep_start") or None])
    con.commit()


# ---------------------------------------------------------------- 帳戶

def accounts(con) -> pd.DataFrame:
    return pd.read_sql("SELECT a.id, a.name, a.initial_cash, a.sweep, a.created_at, COUNT(o.id) AS n_orders "
                       "FROM paper_accounts a LEFT JOIN paper_orders o ON o.account = a.id AND o.batch IS NOT ? "
                       "GROUP BY a.id ORDER BY a.id", con, params=[SWEEP])


def create_account(con, name: str, cash: float = INITIAL_CASH, sweep: bool = True) -> int:
    name = name.strip()
    if not name:
        raise ValueError("帳戶名稱不能是空的")
    if con.execute("SELECT 1 FROM paper_accounts WHERE name = ?", [name]).fetchone():
        raise ValueError(f"已經有叫「{name}」的帳戶")
    cur = con.execute("INSERT INTO paper_accounts (name, initial_cash, sweep) VALUES (?, ?, ?)",
                      [name, float(cash), int(sweep)])
    con.commit()
    return cur.lastrowid


def rename_account(con, acct: int, name: str) -> None:
    name = name.strip()
    if not name:
        raise ValueError("帳戶名稱不能是空的")
    if con.execute("SELECT 1 FROM paper_accounts WHERE name = ? AND id != ?", [name, acct]).fetchone():
        raise ValueError(f"已經有叫「{name}」的帳戶")
    con.execute("UPDATE paper_accounts SET name = ? WHERE id = ?", [name, acct])
    con.commit()


def delete_account(con, acct: int) -> None:
    """連同委託一起刪掉；至少要留一個帳戶。"""
    if con.execute("SELECT COUNT(*) FROM paper_accounts").fetchone()[0] <= 1:
        raise ValueError("至少要留一個帳戶")
    con.execute("DELETE FROM paper_orders WHERE account = ?", [acct])
    con.execute("DELETE FROM paper_auto WHERE account = ?", [acct])
    con.execute("DELETE FROM paper_accounts WHERE id = ?", [acct])
    con.commit()


def setting(con, key: str, default: str = "", acct: int = 1) -> str:
    if key not in SETTINGS:
        raise KeyError(key)
    r = con.execute(f"SELECT {key} FROM paper_accounts WHERE id = ?", [acct]).fetchone()
    return default if not r or r[0] is None else str(r[0])


def set_setting(con, key: str, value, acct: int = 1) -> None:
    if key not in SETTINGS:
        raise KeyError(key)
    con.execute(f"UPDATE paper_accounts SET {key} = ? WHERE id = ?", [value, acct])
    con.commit()


def sweep_on(con, acct: int = 1) -> bool:
    return setting(con, "sweep", "1", acct) == "1"


def initial_cash(con, acct: int = 1) -> float:
    return float(setting(con, "initial_cash", str(INITIAL_CASH), acct))


def _table(code: str) -> str:
    return "etf_prices" if code == ETF else "prices"


# ---------------------------------------------------------------- 下單

def place(con, order_date: str, code: str, side: str, shares: int, strategy: str | None = None,
          note: str | None = None, stop: float | None = None, target: float | None = None,
          expire_date: str | None = None, batch: str | None = None, settle_now: bool = True,
          parent: int | None = None, acct: int = 1) -> int:
    """stop／target／expire_date 只給賣單：條件賣出，見檔頭。parent 是和這張賣單一起下的買單，
    那張買單沒成交（退單或取消），這張賣單也跟著退，不會去賣到別筆的持股。"""
    if expire_date and side != "sell":
        raise ValueError("條件單只能是賣單")
    cur = con.execute("INSERT INTO paper_orders (order_date, code, side, shares, strategy, note, stop, target, "
                      "expire_date, batch, parent, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      [order_date, code, side, int(shares), strategy, note, stop, target, expire_date, batch,
                       parent, acct])
    con.commit()
    if settle_now:
        settle(con, acct)
    return cur.lastrowid


def cancel(con, order_id: int, acct: int = 1) -> bool:
    n = con.execute("UPDATE paper_orders SET status = 'cancelled' WHERE id = ? AND account = ? AND status = 'pending'",
                    [order_id, acct]).rowcount
    con.commit()
    return n > 0


def run_strategy(con, strategy: str, start: str, end: str, mode: str = "fixed", codes: list | None = None,
                 top_n: int | None = None, amount: float = 100_000, max_pos: int = 10, opt: dict | None = None,
                 progress=None, acct: int = 1, llm: bool = False, batch: str | None = None) -> dict:
    """策略自動下單。codes 給個股清單；top_n 改成每天取模型排名前 N 名（只限模型策略）。
    llm=True：每個訊號先給地端 LLM 審核。batch 給了就接在同一批後面（每日自動跟單用）。
    progress(已完成比例, 訊息) 給前端畫進度條。回傳這一批的摘要。"""
    import time
    import forecast as F

    kind = F.STRATEGIES[strategy]["kind"]
    if top_n and kind != "model":
        raise ValueError("每天取前 N 名只適用模型策略")
    days = pd.DatetimeIndex(pd.read_sql("SELECT DISTINCT date FROM market WHERE symbol = 'TWII' AND date >= ? "
                                        "AND date <= ? ORDER BY date", con, params=[start, end],
                                        parse_dates=["date"])["date"])
    if llm:
        import llm_judge as J
        why = J.available()
        if why:
            raise ValueError(f"LLM 審核需要地端模型：{why}")
    batch = batch or (f"{F.STRATEGIES[strategy]['name']}{'＋LLM 審核' if llm else ''}｜{F.EXIT_MODES[mode]}｜"
                      f"{start}~{end}｜{time.strftime('%m%d-%H%M%S')}{int(time.time() * 1000) % 1000:03d}")
    n_skip = n_half = 0
    tag, note = f"auto:{strategy}", f"策略自動下單：{F.STRATEGIES[strategy]['name']}"
    closes = volumes = None
    if kind == "highvol":
        closes, volumes = F.load_panel(con)
    top = None
    if top_n:
        h = F.STRATEGIES[strategy]["h"]
        p = pd.read_sql("SELECT date, code, pred FROM predictions WHERE h = ? AND date >= ? AND date <= ?",
                        con, params=[h, start, end])
        p = p[p["date"].isin(F.complete_days(con, h).index)]
        top = {pd.Timestamp(d): list(g.nlargest(top_n, "pred")[["code", "pred"]].itertuples(index=False))
               for d, g in p.groupby("date")}
    pxs, busy, n_buy = {}, {}, 0
    if codes == ["*"]:                      # 全部股票（含下市股；當天沒有價格的會自動跳過）
        codes = [c for (c,) in con.execute("SELECT DISTINCT code FROM prices ORDER BY code")]
    codes = codes or []
    if len(days) and not setting(con, "sweep_start", acct=acct):
        set_setting(con, "sweep_start", days[0].strftime("%Y-%m-%d"), acct)

    def px_of(code):
        if code not in pxs:
            pxs[code] = F.load_px(con, code)
        return pxs[code]

    for i, d in enumerate(days):
        if progress and (i % 10 == 0 or llm):
            progress(i / len(days), f"{d:%Y-%m-%d}　已進場 {n_buy} 筆"
                     + (f"，LLM 略過 {n_skip}、減半 {n_half}" if llm else ""))
        held = sum(1 for v in busy.values() if v > d)
        cands = top.get(d, []) if top is not None else [(c, None) for c in codes]
        for code, pred in cands:
            if held >= max_pos:
                break
            if busy.get(code, d) > d:          # 持有中，出場日（含）以後才能再進場
                continue
            px = px_of(code)
            hist = px.loc[:d]
            if len(hist) < 60 or hist.index[-1] != d:
                continue
            if pred is not None:
                plan = F.model_plan(hist, px.index, pred, F.STRATEGIES[strategy]["h"])
            else:
                try:
                    r = F.forecast(con, code, d, strategy, px=px, closes=closes, volumes=volumes,
                                   calib=False, **(opt or {}))
                except ValueError:
                    continue
                plan = r["plan"]
                if not plan or not plan["enter"]:
                    continue
            shares = int(amount // plan["entry_est"])
            if shares < 1:
                continue
            note_i = note
            if llm:
                # 回測期間在模型訓練截止以前的，判斷可能被記憶污染；llm_eval.py 只評估截止之後
                j = J.review(con, code, d.strftime("%Y-%m-%d"), strategy, plan, pit=True, account=acct, batch=batch)
                note_i = f"{note}｜LLM {j['decision']}：{(j['reason'] or '')[:60]}"
                if j["decision"] == "skip":
                    n_skip += 1
                    busy[code] = plan["exit_date"]          # 這一筆不做，持有期間內不再問
                    continue
                if j["decision"] == "half":
                    n_half += 1
                    shares = max(1, shares // 2)
            kw = dict(batch=batch, settle_now=False, acct=acct)
            bid = place(con, d.strftime("%Y-%m-%d"), code, "buy", shares, tag, note_i, **kw)
            stop, tgt = F.exit_levels(plan, mode)
            ex = plan["exit_date"]
            if mode == "fixed":
                # 出場日前一個交易日收盤後送出，出場日開盤成交
                place(con, plan["pre_exit"].strftime("%Y-%m-%d"), code, "sell", shares, tag, note_i,
                      parent=bid, **kw)
                free = ex
            else:
                place(con, d.strftime("%Y-%m-%d"), code, "sell", shares, tag, note_i, stop=stop, target=tgt,
                      expire_date=ex.strftime("%Y-%m-%d"), parent=bid, **kw)
                hit = F.simulate_exit(px, plan["entry_date"], ex, stop, tgt)
                free = hit[0] if hit else ex
            busy[code] = free
            held += 1
            n_buy += 1
    settle(con, acct)
    if progress:
        progress(1.0, f"完成：{n_buy} 筆進場")
    sm = batch_summary(con, acct)
    sm = sm[sm["批次"] == batch]
    out = sm.iloc[0].to_dict() if len(sm) else {"批次": batch, "進場筆數": 0}
    return out | ({"LLM 略過": n_skip, "LLM 減半": n_half} if llm else {})


def run_llm_pick(con, start: str, end: str, n: int = 12, k: int = 5, amount: float = 100_000,
                 progress=None, acct: int = 1, batch: str | None = None, pit: bool = True,
                 strategy: str = LLM_PICK) -> dict:
    """LLM 選股：每期從 5 日模型前 n 名裡由 LLM 挑最多 k 檔，每檔投入 amount × k × 權重，持有 5 日。
    strategy=LLM_SCORE 用第二版（llm_judge.score：加基本面與新聞內文、每檔打分取前 k 名等權）。
    pit=True 和 llm_select_eval.py 一樣只給當天以前已知的資料，往後實際跑的紀錄才能和評估比較。"""
    import time
    import forecast as F
    import llm_judge as J

    why = None if strategy == REV_PICK else J.available()
    if why:
        raise ValueError(f"LLM 選股需要地端模型：{why}")
    complete = set(F.complete_days(con, 5).index)
    days = [d for (d,) in con.execute("SELECT date FROM market WHERE symbol = 'TWII' AND date >= ? AND date <= ? "
                                      "ORDER BY date", [start, end]) if d in complete]
    batch = batch or (f"{strategy_name(strategy)}｜{start}~{end}｜{time.strftime('%m%d-%H%M%S')}"
                      f"{int(time.time() * 1000) % 1000:03d}")
    tag, note = f"auto:{strategy}", {LLM_PICK: "LLM 選股", LLM_SCORE: "LLM 打分"}.get(strategy, "營收規則")
    if days and not setting(con, "sweep_start", acct=acct):
        set_setting(con, "sweep_start", days[0], acct)
    # 上一期還沒出場就不開新的一期（也接得上之前幾次執行留下的部位）
    busy_to = con.execute("SELECT MAX(order_date) FROM paper_orders WHERE account = ? AND strategy = ? "
                          "AND side = 'sell' AND status != 'cancelled'", [acct, tag]).fetchone()[0] or ""
    n_period = n_buy = 0
    for i, d in enumerate(days):
        if progress:
            progress(i / max(len(days), 1), f"{d}　第 {n_period} 期、已進場 {n_buy} 筆")
        if d <= busy_to:
            continue
        c = J.candidates(con, d, n)
        if c.empty:
            continue
        if strategy == REV_PICK:
            yoy = {x: J._rev(con, x, d)[1] for x in c["code"]}
            top = sorted(c["code"], key=lambda x: -yoy[x] if yoy[x] == yoy[x] else np.inf)[:k]   # 缺值排最後
            r = {"picks": [dict(code=x, weight=1 / len(top), reason=f"營收年增 {yoy[x]:+.0f}%")
                           for x in top]}
        elif strategy == LLM_SCORE:
            s = J.score(con, d, c, k, account=acct, batch=batch)
            r = {"picks": [dict(code=x, weight=1 / len(s["picks"]),
                                reason=f"{s['scores'][x]:.0f} 分，{s['reasons'].get(x, '')}") for x in s["picks"]]}
        else:
            r = J.select(con, d, c, k, pit=pit, account=acct, batch=batch)
        n_period += 1
        pred = c.set_index("code")["pred"]
        exits = []
        for p in r["picks"]:
            px = F.load_px(con, p["code"])
            hist = px.loc[:pd.Timestamp(d)]
            if len(hist) < 60 or hist.index[-1] != pd.Timestamp(d):
                continue
            plan = F.model_plan(hist, px.index, float(pred[p["code"]]), 5)
            shares = int(amount * k * p["weight"] // plan["entry_est"])
            if shares < 1:
                continue
            why = f"{note}（權重 {p['weight']:.0%}）：{p['reason'][:60]}"
            bid = place(con, d, p["code"], "buy", shares, tag, why, batch=batch, settle_now=False, acct=acct)
            pre = plan["pre_exit"].strftime("%Y-%m-%d")
            place(con, pre, p["code"], "sell", shares, tag, why, batch=batch, settle_now=False, acct=acct, parent=bid)
            exits.append(pre)
            n_buy += 1
        if exits:
            busy_to = max(exits)
    settle(con, acct)
    if progress:
        progress(1.0, f"完成：{n_period} 期、{n_buy} 筆進場")
    sm = batch_summary(con, acct)
    sm = sm[sm["批次"] == batch]
    out = sm.iloc[0].to_dict() if len(sm) else {"批次": batch, "進場筆數": 0}
    return out | {"LLM 選股期數": n_period}


def _pm_holdings(con, acct: int, tag: str, d: str, equity: float, cal: list) -> tuple:
    """d 收盤時這個策略的持股：（{代號: 股數}, {代號: dict(w, pnl, days)} 給 LLM 看）。
    待成交的單也算進去（已經送出、會照下）；持有天數從這次部位建立那天算起。"""
    od = pd.read_sql("SELECT code, side, shares, fill_price, fee, COALESCE(fill_date, order_date) AS day "
                     "FROM paper_orders WHERE account = ? AND strategy = ? AND status IN ('filled', 'pending') "
                     "AND order_date <= ? ORDER BY day, id", con, params=[acct, tag, d])
    shares, info = {}, {}
    for code, g in od.groupby("code"):
        sh, cost, opened = 0, 0.0, None
        for r in g.itertuples():
            if r.side == "buy":
                opened = opened if sh > 0 else r.day
                sh += r.shares
                cost += r.shares * (r.fill_price if pd.notna(r.fill_price) else 0) + (r.fee or 0)
            else:
                cost -= cost / sh * r.shares if sh else 0
                sh -= r.shares
        if sh <= 0:
            continue
        px = con.execute("SELECT close FROM prices WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                         [code, d]).fetchone()[0]
        shares[code] = sh
        info[code] = dict(w=sh * px / equity, pnl=sh * px / cost - 1 if cost > 0 else 0.0,
                          days=sum(1 for x in cal if opened < x <= d))
    return shares, info


def run_llm_pm(con, start: str, end: str, n: int = 8, k: int = 10, max_w: float = 0.2, every: int = 5,
               min_trade: float = 0.02, progress=None, acct: int = 1, batch: str | None = None) -> dict:
    """LLM 自主操盤：start～end 之間每 every 個交易日問一次 LLM 目標投資組合，照差額下單。
    n：每個策略來源放進候選池的檔數；k：最多持有幾檔；max_w：單檔上限。"""
    import time
    import forecast as F
    import llm_judge as J

    why = J.available()
    if why:
        raise ValueError(f"LLM 自主操盤需要地端模型：{why}")
    tag = f"auto:{LLM_PM}"
    complete = set(F.complete_days(con, 5).index)
    cal = [d for (d,) in con.execute("SELECT date FROM market WHERE symbol = 'TWII' AND date <= ? ORDER BY date",
                                     [end])]
    days = [d for d in cal if d >= start and d in complete]
    batch = batch or f"{LLM_NAMES[LLM_PM]}｜{start}~{end}｜{time.strftime('%m%d-%H%M%S')}"
    if days and not setting(con, "sweep_start", acct=acct):
        set_setting(con, "sweep_start", days[0], acct)
    # 接著之前的執行：這個帳戶上一次做決定的日子（帳戶清空後就不算）
    first = con.execute("SELECT MIN(order_date) FROM paper_orders WHERE account = ? AND strategy = ?",
                        [acct, tag]).fetchone()[0]
    last = con.execute("SELECT MAX(asof) FROM llm_decisions WHERE kind = 'manage' AND account = ? AND asof >= ? "
                       "AND asof < ? AND error IS NULL", [acct, first, start]).fetchone()[0] if first else None
    n_dec = n_buy = n_sell = 0
    for i, d in enumerate(days):
        if progress:
            progress(i / max(len(days), 1), f"{d}　已決定 {n_dec} 次、買 {n_buy} 筆、賣 {n_sell} 筆")
        if last and sum(1 for x in cal if last < x <= d) < every:
            continue
        settle(con, acct)
        cv = _curve(con, pd.read_sql("SELECT * FROM paper_orders WHERE account = ? AND status = 'filled' "
                                     "ORDER BY fill_date, id", con, params=[acct]), initial_cash(con, acct))
        cv = cv[cv.index <= pd.Timestamp(d)] if len(cv) else cv
        equity = float(cv.iloc[-1]) if len(cv) else initial_cash(con, acct)
        shares, held = _pm_holdings(con, acct, tag, d, equity, cal)
        r = J.manage(con, d, held, k=k, max_w=max_w, n=n, account=acct, batch=batch)
        last, n_dec = d, n_dec + 1
        tgt = {x["code"]: x for x in r["targets"]}
        sold = {x["code"]: f"賣出：{x['reason']}" for x in r.get("sells", [])}
        sells, buys = [], []
        for code in sorted(set(shares) | set(tgt)):
            px = con.execute("SELECT close FROM prices WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                             [code, d]).fetchone()[0]
            have = shares.get(code, 0)
            w = tgt[code]["weight"] if code in tgt else 0.0
            diff = w * equity - have * px
            why = f"LLM 自主操盤（目標 {w:.0%}）：{(tgt[code]['reason'] if code in tgt else sold.get(code, '超過檔數上限，賣出'))[:80]}"
            if w == 0 and have:
                sells.append((code, have, why))
            elif abs(diff) >= min_trade * equity:
                if diff < 0:
                    sells.append((code, min(have, int(round(-diff / px))), why))
                elif int(diff // px) >= 1:
                    buys.append((code, int(diff // px), why))
        # 先賣後買，買的錢不夠時 settle 會賣 0050
        for side, rows in (("sell", sells), ("buy", buys)):
            for code, sh, why in rows:
                if sh >= 1:
                    place(con, d, code, side, sh, tag, why, batch=batch, settle_now=False, acct=acct)
        n_sell += len(sells)
        n_buy += len(buys)
    settle(con, acct)
    if progress:
        progress(1.0, f"完成：決定 {n_dec} 次、買 {n_buy} 筆、賣 {n_sell} 筆")
    sm = batch_summary(con, acct)
    sm = sm[sm["批次"] == batch]
    out = sm.iloc[0].to_dict() if len(sm) else {"批次": batch, "進場筆數": 0}
    return out | {"LLM 決定次數": n_dec, "買單": n_buy, "賣單": n_sell}


def auto_config(con, acct: int) -> dict | None:
    r = pd.read_sql("SELECT * FROM paper_auto WHERE account = ?", con, params=[acct])
    # 空欄位讀出來是 NaN，而 NaN 在 if 裡算 True；統一換成 None
    return None if r.empty else {k: (None if pd.isna(v) else v) for k, v in r.iloc[0].to_dict().items()}


def set_auto(con, acct: int, strategy: str, mode: str = "fixed", codes: list | None = None,
             top_n: int | None = None, amount: float = 100_000, max_pos: int = 10, llm: bool = False,
             start: str | None = None, enabled: bool = True) -> None:
    """存這個帳戶的每日自動跟單設定；start 是從哪一天開始（含當天，之前的日子不補）。
    last_run 記的是「已經跑完的最後一天」，所以存 start 的前一個交易日。"""
    prev = con.execute("SELECT MAX(date) FROM market WHERE symbol = 'TWII' AND date < ?",
                       [start]).fetchone()[0] if start else None
    con.execute("INSERT OR REPLACE INTO paper_auto (account, strategy, mode, codes, top_n, amount, max_pos, llm, "
                "last_run, enabled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [acct, strategy, mode, ",".join(codes) if codes else None, top_n, float(amount), int(max_pos),
                 int(llm), prev, int(enabled)])
    con.commit()


def run_auto(con, acct: int | None = None, progress=None) -> list:
    """每日自動跟單：從上次跑到的隔天接著跑到 DB 最新一天。acct=None 跑所有啟用的帳戶。"""
    cfgs = pd.read_sql("SELECT * FROM paper_auto WHERE enabled = 1" + (" AND account = ?" if acct else ""), con,
                       params=[acct] if acct else [])
    last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    out = []
    for c in cfgs.itertuples():
        start = c.last_run or last
        if c.last_run:
            nxt = con.execute("SELECT MIN(date) FROM market WHERE symbol = 'TWII' AND date > ?",
                              [c.last_run]).fetchone()[0]
            if not nxt or nxt > last:
                out.append({"帳戶": c.account, "結果": f"已經跑到 {c.last_run}，沒有新的交易日"})
                continue
            start = nxt
        name = con.execute("SELECT name FROM paper_accounts WHERE id = ?", [c.account]).fetchone()[0]
        top_n = int(c.top_n) if pd.notna(c.top_n) and c.top_n else None
        # 一個帳戶失敗（例如 Ollama 沒開）不影響其他帳戶；last_run 不前進，下次會從同一天補跑
        try:
            if c.strategy == LLM_PM:
                r = run_llm_pm(con, start, last, n=top_n or 8, k=int(c.max_pos), progress=progress,
                               acct=int(c.account), batch=f"每日自動跟單｜{name}")
            elif c.strategy in PICK_NAMES:
                r = run_llm_pick(con, start, last, n=top_n or 12, k=int(c.max_pos), amount=c.amount,
                                 progress=progress, acct=int(c.account), batch=f"每日自動跟單｜{name}",
                                 strategy=c.strategy)
            else:
                r = run_strategy(con, c.strategy, start, last, c.mode,
                                 codes=c.codes.split(",") if isinstance(c.codes, str) else None, top_n=top_n,
                                 amount=c.amount, max_pos=int(c.max_pos), progress=progress, acct=int(c.account),
                                 llm=bool(c.llm), batch=f"每日自動跟單｜{name}")
        except Exception as e:
            out.append({"帳戶": c.account, "結果": f"失敗（{type(e).__name__}: {e}），下次從 {start} 補跑",
                        "失敗": True})
            continue
        con.execute("UPDATE paper_auto SET last_run = ? WHERE account = ?", [last, c.account])
        con.commit()
        out.append({"帳戶": c.account, "結果": f"{start} ~ {last}", **r})
    return out


def batch_summary(con, acct: int = 1) -> pd.DataFrame:
    """每一批自動下單：筆數、已平倉損益（扣成本）、勝率、還沒出場的筆數。"""
    od = pd.read_sql("SELECT * FROM paper_orders WHERE account = ? AND batch IS NOT NULL AND batch != ? ORDER BY id",
                     con, params=[acct, SWEEP])
    cols = ["批次", "期間", "進場筆數", "已平倉", "持有中", "未成交或退單", "已實現損益", "勝率", "平均每筆"]
    rows = []
    for b, g in od.groupby("batch", sort=False):
        buys, sells = g[g["side"] == "buy"], g[g["side"] == "sell"]
        pnl, open_n = [], 0
        pm = (g["strategy"] == f"auto:{LLM_PM}").all()
        if pm:
            # 自主操盤會加碼、減碼，買賣不成對：每檔用平均成本法，每筆賣出算一筆已平倉
            for _, h in g[g["status"] == "filled"].sort_values(["fill_date", "id"]).groupby("code"):
                sh, cost = 0, 0.0
                for r in h.itertuples():
                    if r.side == "buy":
                        sh, cost = sh + r.shares, cost + r.shares * r.fill_price + r.fee
                    elif sh:
                        avg = cost / sh
                        pnl.append(r.shares * r.fill_price - r.fee - r.tax - avg * r.shares)
                        sh, cost = sh - r.shares, cost - avg * r.shares
                open_n += sh > 0
        # 其他自動下單一定是一買一賣成對、依序寫入
        for bu, se in () if pm else zip(buys.itertuples(), sells.itertuples()):
            if bu.status != "filled":
                continue
            if se.status != "filled":
                open_n += 1
                continue
            pnl.append(se.shares * se.fill_price - se.fee - se.tax - bu.shares * bu.fill_price - bu.fee)
        pnl = np.array(pnl)
        rows.append([b, f"{g['order_date'].min()} ~ {g['order_date'].max()}", len(buys), len(pnl), open_n,
                     int((buys["status"] != "filled").sum()), float(pnl.sum()),
                     float((pnl > 0).mean()) if len(pnl) else np.nan, float(pnl.mean()) if len(pnl) else np.nan])
    return pd.DataFrame(rows, columns=cols)


def delete_batch(con, batch: str, acct: int = 1) -> int:
    n = con.execute("DELETE FROM paper_orders WHERE batch = ? AND account = ?", [batch, acct]).rowcount
    con.commit()
    settle(con, acct)
    return n


def reset(con, cash: float = INITIAL_CASH, sweep: bool | None = None, acct: int = 1) -> None:
    """清空這個帳戶的委託，重設起始資金；其他帳戶不受影響。"""
    con.execute("DELETE FROM paper_orders WHERE account = ?", [acct])
    con.execute("UPDATE paper_accounts SET initial_cash = ?, sweep_start = NULL WHERE id = ?", [float(cash), acct])
    if sweep is not None:
        con.execute("UPDATE paper_accounts SET sweep = ? WHERE id = ?", [int(sweep), acct])
    con.commit()


def _next_open(con, code: str, after: str, live: dict | None = None):
    r = con.execute(f"SELECT date, open, close FROM {_table(code)} WHERE code = ? AND date > ? "
                    "ORDER BY date LIMIT 1", [code, after]).fetchone()
    lb = (live or {}).get(code)
    if not r and lb and lb["date"] > after:
        return lb["date"], lb["open"], lb["close"]
    return r


def _live(con) -> dict:
    """今天的盤中 K 棒（intraday.live_bars）；盤中報價沒在跑或 DB 還沒接上昨天就是空的。"""
    try:
        import intraday
        return intraday.live_bars(con)
    except Exception:
        return {}


# ---------------------------------------------------------------- 結算

def settle(con, acct: int | None = 1) -> int:
    """把待成交的單依成交日排進帳本，從頭重放現金與持股；錢不夠或股數不夠的退單。回傳新成交筆數。
    acct=None 會依序結算所有帳戶。"""
    if acct is None:
        return sum(settle(con, int(a)) for a in accounts(con)["id"])
    con.execute("DELETE FROM paper_orders WHERE batch = ? AND account = ?", [SWEEP, acct])
    od = pd.read_sql("SELECT * FROM paper_orders WHERE account = ? AND status IN ('pending', 'filled')", con,
                     params=[acct])
    if od.empty:
        con.commit()
        return 0
    live = _live(con)                      # 盤中：今天的 K 棒接在日線後面，開盤成交、停損停利盤中觸發
    for i, r in od[od["status"] == "pending"].iterrows():
        if pd.notna(r.get("expire_date")):
            hit = _conditional(con, r, live)
            if hit:
                od.loc[i, ["fill_date", "fill_price", "trigger"]] = [hit[0].strftime("%Y-%m-%d"), hit[1], hit[2]]
            continue
        nx = _next_open(con, r["code"], r["order_date"], live)
        if nx:
            px = nx[1] if nx[1] and nx[1] > 0 else nx[2]
            od.loc[i, ["fill_date", "fill_price"]] = [nx[0], float(px)]
    # 同一天：開盤的賣單、開盤的買單、盤中碰到停損／目標的賣單。開盤賣出的錢當天開盤就能拿去買；
    # 盤中觸發的賣單排在買單後面，進場當天就碰到停損才賣得到剛買的股票
    od = od[od["fill_date"].notna()]
    od = od.assign(_b=np.where(od["side"] == "buy", 1, np.where(od["trigger"].isin(["停損", "目標"]), 2, 0))) \
        .sort_values(["fill_date", "_b", "id"])
    cash, pos, n_new, upd = initial_cash(con, acct), {}, 0, []
    state = dict(con.execute("SELECT id, status FROM paper_orders WHERE account = ?", [acct]))
    sweep = sweep_on(con, acct)
    etf = dict(con.execute("SELECT date, open FROM etf_prices WHERE code = ?", [ETF])) if sweep else {}
    if sweep and ETF in live:
        etf.setdefault(live[ETF]["date"], live[ETF]["open"])
    units, swp = 0, []

    def etf_trade(day: str, n: int) -> None:
        """n > 0 買、n < 0 賣 0050；記成系統單。"""
        nonlocal cash, units
        px = etf[day]
        amt = abs(n) * px
        fee, tax = amt * FEE, (amt * ETF_TAX if n < 0 else 0.0)
        cash += (-amt - fee) if n > 0 else (amt - fee - tax)
        units += n
        swp.append((day, ETF, "buy" if n > 0 else "sell", abs(n), "sweep", "閒置資金", "filled", day, px,
                    fee, tax, SWEEP, acct))

    def park(day: str) -> None:
        if sweep and day in etf:
            n = int(cash // (etf[day] * (1 + FEE)))
            if n > 0:
                etf_trade(day, n)

    days = sorted(od["fill_date"].unique())
    start = setting(con, "sweep_start", acct=acct)
    if sweep and start and days and start < days[0] and start in etf:
        park(start)                                # 自動下單的起日就先全部買 0050
    prev = None
    for r in od.itertuples():
        if prev is not None and r.fill_date != prev:
            park(prev)
        prev = r.fill_date
        if sweep and units and r.fill_date in etf and r.side == "buy":
            need = r.shares * r.fill_price * (1 + FEE) - cash
            if need > 0:
                n = min(units, int(np.ceil(need / (etf[r.fill_date] * (1 - FEE - ETF_TAX)))))
                etf_trade(r.fill_date, -n)
        if r.side == "sell" and pd.notna(r.parent) and state.get(int(r.parent)) not in ("filled", "pending"):
            upd.append(("rejected", None, None, None, None, "對應的買單沒有成交", None, r.id))
            state[r.id] = "rejected"
            continue
        amt = r.shares * r.fill_price
        fee = amt * FEE
        tax = amt * TAX if r.side == "sell" else 0.0
        if r.side == "buy" and amt + fee > cash + 1e-6:
            upd.append(("rejected", None, None, None, None, f"現金不足（需要 {amt + fee:,.0f}，剩 {cash:,.0f}）", None, r.id))
            state[r.id] = "rejected"
            continue
        if r.side == "sell" and r.shares > pos.get(r.code, 0):
            upd.append(("rejected", None, None, None, None, f"持股不足（持有 {pos.get(r.code, 0):,} 股）", None, r.id))
            state[r.id] = "rejected"
            continue
        cash += -amt - fee if r.side == "buy" else amt - fee - tax
        pos[r.code] = pos.get(r.code, 0) + (r.shares if r.side == "buy" else -r.shares)
        n_new += r.status == "pending"
        upd.append(("filled", r.fill_date, r.fill_price, fee, tax, None, r.trigger, r.id))
        state[r.id] = "filled"
    if prev is not None:
        park(prev)
    con.executemany("INSERT INTO paper_orders (order_date, code, side, shares, strategy, note, status, fill_date, "
                    "fill_price, fee, tax, batch, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", swp)
    con.executemany("UPDATE paper_orders SET status = ?, fill_date = ?, fill_price = ?, fee = ?, tax = ?, "
                    "reason = ?, trigger = ? WHERE id = ?", upd)
    con.commit()
    return n_new


def _conditional(con, r, live: dict | None = None):
    """條件賣單：下單日的下一個交易日起照停損／目標檢查，回傳（成交日, 價, 原因）或 None（還沒觸發）。
    live 是今天的盤中 K 棒：高低是到目前為止的，碰到就成交，同一根碰到兩邊照日線規則算停損。"""
    import forecast
    nx = _next_open(con, r["code"], r["order_date"], live)
    if not nx:
        return None
    px = pd.read_sql("SELECT date, open, high, low, close FROM prices WHERE code = ? AND date >= ? AND date <= ? "
                     "ORDER BY date", con, params=[r["code"], nx[0], r["expire_date"]],
                     parse_dates=["date"]).set_index("date")
    lb = (live or {}).get(r["code"])
    if lb and lb["date"] <= r["expire_date"] and (px.empty or px.index[-1] < pd.Timestamp(lb["date"])):
        px.loc[pd.Timestamp(lb["date"])] = [lb["open"], lb["high"], lb["low"], lb["close"]]
    stop = r["stop"] if pd.notna(r["stop"]) else None
    tgt = r["target"] if pd.notna(r["target"]) else None
    return forecast.simulate_exit(px, nx[0], r["expire_date"], stop, tgt)


def orders(con, acct: int = 1) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM paper_orders WHERE account = ? ORDER BY id DESC", con, params=[acct])


def portfolio(con, acct: int = 1) -> dict:
    """現金、持股（平均成本法）、已實現損益、逐日淨值。持股用 DB 裡最新收盤評價，盤中用即時價（換成還原價口徑）。"""
    live = _live(con)
    od = pd.read_sql("SELECT * FROM paper_orders WHERE account = ? AND status = 'filled' ORDER BY fill_date, id",
                     con, params=[acct])
    cash0 = initial_cash(con, acct)
    cash, book, realized = cash0, {}, 0.0
    for r in od.itertuples():
        sh, cost = book.get(r.code, (0, 0.0))
        amt = r.shares * r.fill_price
        if r.side == "buy":
            book[r.code] = (sh + r.shares, cost + amt + r.fee)
            cash -= amt + r.fee
        else:
            avg = cost / sh
            realized += amt - r.fee - r.tax - avg * r.shares
            book[r.code] = (sh - r.shares, cost - avg * r.shares)
            cash += amt - r.fee - r.tax
    rows = []
    for code, (sh, cost) in book.items():
        if sh <= 0:
            continue
        last = con.execute(f"SELECT date, close FROM {_table(code)} WHERE code = ? ORDER BY date DESC LIMIT 1",
                           [code]).fetchone()
        if code in live:
            last = (f"{live[code]['date']} 盤中", live[code]["close"])
        mv = sh * last[1]
        # 未實現損益扣掉假設現在賣出的手續費與證交稅，和已實現損益同一個口徑
        rows.append(dict(code=code, shares=sh, avg_cost=cost / sh, last_date=last[0], last=last[1],
                         value=mv, unrealized=mv * (1 - FEE - TAX) - cost))
    pos = pd.DataFrame(rows, columns=["code", "shares", "avg_cost", "last_date", "last", "value", "unrealized"])
    equity = cash + pos["value"].sum()
    return dict(initial=cash0, cash=cash, positions=pos, realized=realized,
                unrealized=float(pos["unrealized"].sum()), equity=equity, curve=_curve(con, od, cash0, live))


def _curve(con, od: pd.DataFrame, cash0: float, live: dict | None = None) -> pd.Series:
    """逐日淨值：每天收盤時的現金＋持股市值。盤中多一天，用即時價評價。"""
    if od.empty:
        return pd.Series(dtype=float)
    codes = sorted(od["code"].unique())
    q = ",".join("?" * len(codes))
    px = pd.read_sql(f"SELECT date, code, close FROM prices WHERE code IN ({q}) AND date >= ? UNION ALL "
                     f"SELECT date, code, close FROM etf_prices WHERE code IN ({q}) AND date >= ?", con,
                     params=codes + [od["fill_date"].min()] + codes + [od["fill_date"].min()],
                     parse_dates=["date"]) \
        .pivot(index="date", columns="code", values="close").sort_index()
    today = {c: live[c]["close"] for c in codes if c in (live or {})}
    if today:
        d = pd.Timestamp(next(iter(live.values()))["date"])
        if not len(px) or px.index[-1] < d:
            px = pd.concat([px, pd.DataFrame(today, index=[d])])
    px = px.reindex(columns=codes).ffill()
    od = od.assign(fill_date=pd.to_datetime(od["fill_date"]))
    sgn = np.where(od["side"] == "buy", 1, -1)
    flow = -(sgn * od["shares"] * od["fill_price"]) - od["fee"] - od["tax"].fillna(0)
    cash = cash0 + pd.Series(flow.to_numpy(), index=od["fill_date"]).groupby(level=0).sum() \
        .reindex(px.index, fill_value=0).cumsum()
    sh = pd.DataFrame({"d": od["fill_date"], "c": od["code"], "s": sgn * od["shares"]}) \
        .pivot_table(index="d", columns="c", values="s", aggfunc="sum") \
        .reindex(px.index, fill_value=0).fillna(0).cumsum()
    return cash + (sh * px[sh.columns]).sum(axis=1)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    con = connect()
    print(f"新成交 {settle(con, None)} 筆")
    for a in accounts(con).itertuples():
        p = portfolio(con, a.id)
        print(f"\n#{a.id} {a.name}：淨值 {p['equity']:,.0f}　現金 {p['cash']:,.0f}　"
              f"已實現 {p['realized']:+,.0f}　未實現 {p['unrealized']:+,.0f}")
        if len(p["positions"]):
            print(p["positions"].to_string(index=False))
