#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模擬下單：流程圖「實際操作 → 回傳至 DB 留存」的模擬版，存在 db/local.db 的 paper_orders。

成交規則和所有回測一樣：下單日（視角日期）收盤後才下單，**下一個交易日開盤**成交。
下單日可以是過去（回到當時的視角模擬），也可以是未來（預約賣出：到那天收盤後送出，
隔天開盤成交）。還沒有隔天開盤價的單，狀態是「待成交」，DB 更新股價後再重新結算就會成交。

條件賣單（給了 stop／target／expire_date）：從下單日的下一個交易日起，每天檢查停損與目標價，
觸發規則和 forecast.simulate_exit 相同（開盤跳空用開盤價、盤中碰到用該價位、同一天都碰到算停損），
都沒碰到就在 expire_date 開盤賣出。

策略自動下單（run_strategy）：選一套 forecast.py 的策略和出場方式，從起日到迄日逐個交易日，
每天只用當天收盤以前的資料判斷有沒有進場訊號（和「個股預測」頁同一套），有就下買單和對應的賣單。
同一檔持有中不重複進場，同時持有不超過 max_pos 檔。同一批的委託共用一個 batch，可以整批刪除。

閒置資金買 0050（帳戶設定 sweep，預設開）：和 reversal_two_investors.py 的投資人 B 同一個做法，
沒有用在個股上的錢一律放 0050（含息還原價，etf_prices 表）。每個有成交的交易日開盤：先處理賣單，
買個股錢不夠就賣掉剛好夠的 0050，當天的單都處理完，剩下的現金再買回 0050。
賣 0050 付手續費＋ETF 證交稅 0.1%，買付手續費。這些換手是系統產生的（batch = SWEEP），
每次結算都從頭重算，不要手動改。

成本同 strategy_spec.py：手續費 0.1425% × 3 折、買賣各一次，賣出另加證交稅 0.3%。
實際券商有每筆最低 20 元手續費、漲跌停鎖死買不到賣不掉，這裡都沒有模擬。

paper_orders 與 paper_account 和 ops_log 一樣，build_local_db.py 重建時不會清掉。
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


def connect(db: str | Path = DB) -> sqlite3.Connection:
    con = sqlite3.connect(db, check_same_thread=False)
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
    # 後來加的條件賣單欄位；舊表補上
    have = {r[1] for r in con.execute("PRAGMA table_info(paper_orders)")}
    for col, typ in (("stop", "REAL"), ("target", "REAL"), ("expire_date", "TEXT"), ("trigger", "TEXT"),
                     ("batch", "TEXT"), ("parent", "INTEGER")):
        if col not in have:
            con.execute(f"ALTER TABLE paper_orders ADD COLUMN {col} {typ}")
    con.execute("CREATE TABLE IF NOT EXISTS paper_account (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR IGNORE INTO paper_account VALUES ('initial_cash', ?)", [str(INITIAL_CASH)])
    con.commit()


def setting(con, key: str, default: str = "") -> str:
    r = con.execute("SELECT value FROM paper_account WHERE key = ?", [key]).fetchone()
    return r[0] if r else default


def set_setting(con, key: str, value) -> None:
    con.execute("INSERT OR REPLACE INTO paper_account VALUES (?, ?)", [key, str(value)])
    con.commit()


def sweep_on(con) -> bool:
    return setting(con, "sweep", "1") == "1"


def _table(code: str) -> str:
    return "etf_prices" if code == ETF else "prices"


def initial_cash(con) -> float:
    return float(con.execute("SELECT value FROM paper_account WHERE key = 'initial_cash'").fetchone()[0])


def place(con, order_date: str, code: str, side: str, shares: int, strategy: str | None = None,
          note: str | None = None, stop: float | None = None, target: float | None = None,
          expire_date: str | None = None, batch: str | None = None, settle_now: bool = True,
          parent: int | None = None) -> int:
    """stop／target／expire_date 只給賣單：條件賣出，見檔頭。parent 是和這張賣單一起下的買單，
    那張買單沒成交（退單或取消），這張賣單也跟著退，不會去賣到別筆的持股。"""
    if expire_date and side != "sell":
        raise ValueError("條件單只能是賣單")
    cur = con.execute("INSERT INTO paper_orders (order_date, code, side, shares, strategy, note, stop, target, "
                      "expire_date, batch, parent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      [order_date, code, side, int(shares), strategy, note, stop, target, expire_date, batch,
                       parent])
    con.commit()
    if settle_now:
        settle(con)
    return cur.lastrowid


def cancel(con, order_id: int) -> bool:
    n = con.execute("UPDATE paper_orders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                    [order_id]).rowcount
    con.commit()
    return n > 0


def run_strategy(con, strategy: str, start: str, end: str, mode: str = "fixed", codes: list | None = None,
                 top_n: int | None = None, amount: float = 100_000, max_pos: int = 10, opt: dict | None = None,
                 progress=None) -> dict:
    """策略自動下單。codes 給個股清單；top_n 改成每天取模型排名前 N 名（只限模型策略）。
    progress(已完成比例, 訊息) 給前端畫進度條。回傳這一批的摘要。"""
    import time
    import forecast as F

    kind = F.STRATEGIES[strategy]["kind"]
    if top_n and kind != "model":
        raise ValueError("每天取前 N 名只適用模型策略")
    days = pd.DatetimeIndex(pd.read_sql("SELECT DISTINCT date FROM market WHERE symbol = 'TWII' AND date >= ? "
                                        "AND date <= ? ORDER BY date", con, params=[start, end],
                                        parse_dates=["date"])["date"])
    batch = (f"{F.STRATEGIES[strategy]['name']}｜{F.EXIT_MODES[mode]}｜{start}~{end}｜"
             f"{time.strftime('%m%d-%H%M%S')}{int(time.time() * 1000) % 1000:03d}")
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
    if len(days) and not setting(con, "sweep_start"):
        set_setting(con, "sweep_start", days[0].strftime("%Y-%m-%d"))

    def px_of(code):
        if code not in pxs:
            pxs[code] = F.load_px(con, code)
        return pxs[code]

    for i, d in enumerate(days):
        if progress and i % 10 == 0:
            progress(i / len(days), f"{d:%Y-%m-%d}　已進場 {n_buy} 筆")
        held = sum(1 for v in busy.values() if v > d)
        cands = top.get(d, []) if top is not None else [(c, None) for c in (codes or [])]
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
            bid = place(con, d.strftime("%Y-%m-%d"), code, "buy", shares, tag, note, batch=batch, settle_now=False)
            stop, tgt = F.exit_levels(plan, mode)
            ex = plan["exit_date"]
            if mode == "fixed":
                # 出場日前一個交易日收盤後送出，出場日開盤成交
                place(con, plan["pre_exit"].strftime("%Y-%m-%d"), code, "sell", shares, tag, note,
                      batch=batch, settle_now=False, parent=bid)
                free = ex
            else:
                place(con, d.strftime("%Y-%m-%d"), code, "sell", shares, tag, note, stop=stop, target=tgt,
                      expire_date=ex.strftime("%Y-%m-%d"), batch=batch, settle_now=False, parent=bid)
                hit = F.simulate_exit(px, plan["entry_date"], ex, stop, tgt)
                free = hit[0] if hit else ex
            busy[code] = free
            held += 1
            n_buy += 1
    settle(con)
    if progress:
        progress(1.0, f"完成：{n_buy} 筆進場")
    sm = batch_summary(con)
    sm = sm[sm["批次"] == batch]
    return sm.iloc[0].to_dict() if len(sm) else {"批次": batch, "進場筆數": 0}


def batch_summary(con) -> pd.DataFrame:
    """每一批自動下單：筆數、已平倉損益（扣成本）、勝率、還沒出場的筆數。"""
    od = pd.read_sql("SELECT * FROM paper_orders WHERE batch IS NOT NULL AND batch != ? ORDER BY id", con,
                     params=[SWEEP])
    cols = ["批次", "期間", "進場筆數", "已平倉", "持有中", "未成交或退單", "已實現損益", "勝率", "平均每筆"]
    rows = []
    for b, g in od.groupby("batch", sort=False):
        buys, sells = g[g["side"] == "buy"], g[g["side"] == "sell"]
        pnl, open_n = [], 0
        # 自動下單一定是一買一賣成對、依序寫入
        for bu, se in zip(buys.itertuples(), sells.itertuples()):
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


def delete_batch(con, batch: str) -> int:
    n = con.execute("DELETE FROM paper_orders WHERE batch = ?", [batch]).rowcount
    con.commit()
    settle(con)
    return n


def reset(con, cash: float = INITIAL_CASH, sweep: bool | None = None) -> None:
    con.execute("DELETE FROM paper_orders")
    con.execute("DELETE FROM paper_account WHERE key = 'sweep_start'")
    con.execute("INSERT OR REPLACE INTO paper_account VALUES ('initial_cash', ?)", [str(float(cash))])
    if sweep is not None:
        con.execute("INSERT OR REPLACE INTO paper_account VALUES ('sweep', ?)", ["1" if sweep else "0"])
    con.commit()


def _next_open(con, code: str, after: str):
    return con.execute(f"SELECT date, open, close FROM {_table(code)} WHERE code = ? AND date > ? "
                       "ORDER BY date LIMIT 1", [code, after]).fetchone()


def settle(con) -> int:
    """把待成交的單依成交日排進帳本，從頭重放現金與持股；錢不夠或股數不夠的退單。回傳新成交筆數。"""
    con.execute("DELETE FROM paper_orders WHERE batch = ?", [SWEEP])
    od = pd.read_sql("SELECT * FROM paper_orders WHERE status IN ('pending', 'filled')", con)
    if od.empty:
        con.commit()
        return 0
    for i, r in od[od["status"] == "pending"].iterrows():
        if pd.notna(r.get("expire_date")):
            hit = _conditional(con, r)
            if hit:
                od.loc[i, ["fill_date", "fill_price", "trigger"]] = [hit[0].strftime("%Y-%m-%d"), hit[1], hit[2]]
            continue
        nx = _next_open(con, r["code"], r["order_date"])
        if nx:
            px = nx[1] if nx[1] and nx[1] > 0 else nx[2]
            od.loc[i, ["fill_date", "fill_price"]] = [nx[0], float(px)]
    # 同一天先賣後買：賣出的錢當天開盤就能拿去買
    od = od[od["fill_date"].notna()].assign(_b=lambda x: x["side"] == "buy") \
        .sort_values(["fill_date", "_b", "id"])
    cash, pos, n_new, upd = initial_cash(con), {}, 0, []
    state = dict(con.execute("SELECT id, status FROM paper_orders"))
    sweep = sweep_on(con)
    etf = dict(con.execute("SELECT date, open FROM etf_prices WHERE code = ?", [ETF])) if sweep else {}
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
                    fee, tax, SWEEP))

    def park(day: str) -> None:
        if sweep and day in etf:
            n = int(cash // (etf[day] * (1 + FEE)))
            if n > 0:
                etf_trade(day, n)

    days = sorted(od["fill_date"].unique())
    start = setting(con, "sweep_start")
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
                    "fill_price, fee, tax, batch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", swp)
    con.executemany("UPDATE paper_orders SET status = ?, fill_date = ?, fill_price = ?, fee = ?, tax = ?, "
                    "reason = ?, trigger = ? WHERE id = ?", upd)
    con.commit()
    return n_new


def _conditional(con, r):
    """條件賣單：下單日的下一個交易日起照停損／目標檢查，回傳（成交日, 價, 原因）或 None（還沒觸發）。"""
    import forecast
    nx = _next_open(con, r["code"], r["order_date"])
    if not nx:
        return None
    px = pd.read_sql("SELECT date, open, high, low, close FROM prices WHERE code = ? AND date >= ? AND date <= ? "
                     "ORDER BY date", con, params=[r["code"], nx[0], r["expire_date"]],
                     parse_dates=["date"]).set_index("date")
    stop = r["stop"] if pd.notna(r["stop"]) else None
    tgt = r["target"] if pd.notna(r["target"]) else None
    return forecast.simulate_exit(px, nx[0], r["expire_date"], stop, tgt)


def orders(con) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM paper_orders ORDER BY id DESC", con)


def portfolio(con) -> dict:
    """現金、持股（平均成本法）、已實現損益、逐日淨值。持股用 DB 裡最新收盤評價。"""
    od = pd.read_sql("SELECT * FROM paper_orders WHERE status = 'filled' ORDER BY fill_date, id", con)
    cash0 = initial_cash(con)
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
        mv = sh * last[1]
        # 未實現損益扣掉假設現在賣出的手續費與證交稅，和已實現損益同一個口徑
        rows.append(dict(code=code, shares=sh, avg_cost=cost / sh, last_date=last[0], last=last[1],
                         value=mv, unrealized=mv * (1 - FEE - TAX) - cost))
    pos = pd.DataFrame(rows, columns=["code", "shares", "avg_cost", "last_date", "last", "value", "unrealized"])
    equity = cash + pos["value"].sum()
    return dict(initial=cash0, cash=cash, positions=pos, realized=realized,
                unrealized=float(pos["unrealized"].sum()), equity=equity, curve=_curve(con, od, cash0))


def _curve(con, od: pd.DataFrame, cash0: float) -> pd.Series:
    """逐日淨值：每天收盤時的現金＋持股市值。"""
    if od.empty:
        return pd.Series(dtype=float)
    codes = sorted(od["code"].unique())
    q = ",".join("?" * len(codes))
    px = pd.read_sql(f"SELECT date, code, close FROM prices WHERE code IN ({q}) AND date >= ? UNION ALL "
                     f"SELECT date, code, close FROM etf_prices WHERE code IN ({q}) AND date >= ?", con,
                     params=codes + [od["fill_date"].min()] + codes + [od["fill_date"].min()],
                     parse_dates=["date"]) \
        .pivot(index="date", columns="code", values="close").sort_index().ffill()
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
    print(f"新成交 {settle(con)} 筆")
    p = portfolio(con)
    print(f"淨值 {p['equity']:,.0f}　現金 {p['cash']:,.0f}　已實現 {p['realized']:+,.0f}　未實現 {p['unrealized']:+,.0f}")
    print(p["positions"].to_string(index=False))
