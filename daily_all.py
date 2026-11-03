#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日更新總控：所有資料來源＋衍生資料，一次跑完。

由 run_daily_update.bat（Windows 工作排程器）每天收盤後呼叫。每一步各自
try/except：某一步失敗會記錄在報告裡，不會擋住後面的步驟。

步驟
----
    1  股價、指數、ETF、油價、鉅亨個股新聞   daily_update.py（另補齊各指數檔的年度缺口）
    2  三大法人買賣超 T86                     t86_crawler.py → davis_tw.prepare_chips
    3  本益比／殖利率／股價淨值比             tw_fundamentals_crawler.py
    4  月營收、季報                           financials_crawler.py
    5  鉅亨國際新聞                           intl_news_crawler.py
    6  PTT Stock 板                           ptt_crawler.py
    7  英文國際新聞（GDELT）                  gdelt_news_crawler.py --daily
    8  台指期                                 taifex_crawler.py
    9  除權息、減資                           corp_actions.py
    10 下市清單與下市股價                     delisted_crawler.py → delisted_prices.py → adjust_delisted.py
    11 股價假跳動檢查                         tool/price_jump_scan.py → 新的假跳動加進 data/price_fixes.csv
    12 衍生資料重建                           成交金額母體快取、特徵庫（前 250、全母體、含下市股）、
                                              新聞與 PTT 的分析快取

「增量」怎麼做到
----------------
各爬蟲都有自己的斷點或原始檔快取，最近幾天的資料可能在還沒公布時就被標成「完成」。
所以每一步執行前先清掉最近一段的斷點或快取（見各 _reset_*），讓它重抓：
    本益比：最近 5 天　月營收：最近 2 個月　季報：最近 2 季　鉅亨國際：最近 2 週
    台指期：上個月起　除權息與減資：今年
資料若停在更早（例如 T86、本益比、台指期原本只到 2024），會從最後一天接著補到今天。

用法
----
    python daily_all.py            # 全部
    python daily_all.py --only 2 3 # 只跑指定步驟（除錯用）
    python daily_all.py --skip 12  # 跳過衍生資料重建
報告：data/daily/all_YYYYMMDD.txt
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PY = sys.executable
TODAY = date.today()


def run(*args: str, timeout: int = 4 * 3600) -> tuple[bool, str]:
    """以子行程執行（各爬蟲各自的全域狀態不互相干擾），回傳（成功, 最後幾行輸出）。"""
    p = subprocess.run([PY, "-u", "-W", "ignore", *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = "\n".join((p.stdout or "").strip().splitlines()[-3:])
    if p.returncode != 0:
        tail += "\n" + "\n".join((p.stderr or "").strip().splitlines()[-3:])
    return p.returncode == 0, tail


# ---------------- 斷點與快取重置 ----------------

def _json_set(path: Path) -> set:
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def _reset_fundamentals() -> date:
    ck = DATA / "fundamentals" / "checkpoint.json"
    done = _json_set(ck)
    cut = (TODAY - timedelta(days=5)).strftime("%Y%m%d")      # 斷點格式是 YYYYMMDD
    done = {d for d in done if d < cut}
    ck.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")
    last = max(done) if done else "20150101"
    last = date(int(last[:4]), int(last[4:6]), int(last[6:8]))
    return min(last + timedelta(days=1), TODAY - timedelta(days=5))


def _reset_financials() -> tuple[str, str]:
    raw = DATA / "financials" / "raw"
    m0 = pd.Period(TODAY, "M") - 2
    for f in (raw / "revenue").glob("*.html"):
        ym = f.stem.split("_")[1]
        if pd.Period(f"{ym[:4]}-{ym[4:6]}", "M") >= m0:
            f.unlink()
    q0 = pd.Period(TODAY, "Q") - 2
    for f in raw.glob("*/*Q*.html"):
        y, q = f.stem.split("_")[-1].split("Q")
        if pd.Period(f"{y}Q{q}", "Q") >= q0:
            f.unlink()
    return str(m0), f"{q0.year}Q{q0.quarter}"


def _reset_intl_news() -> date:
    ck = DATA / "intl_news" / "checkpoint.json"
    done = _json_set(ck)
    cut = (TODAY - timedelta(days=14)).isoformat()
    keep = {k for k in done if k.split("|")[1] < cut}
    ck.write_text(json.dumps(sorted(keep), ensure_ascii=False), encoding="utf-8")
    return TODAY - timedelta(days=14)


def _reset_taifex() -> int:
    ck = DATA / "taifex" / "checkpoint.json"
    done = _json_set(ck)
    cut = str(pd.Period(TODAY, "M") - 1)
    keep = {k for k in done if k < cut}
    ck.write_text(json.dumps(sorted(keep), ensure_ascii=False), encoding="utf-8")
    last = max(keep) if keep else "2011-01"
    return int(last[:4])


def _reset_corp_actions() -> None:
    raw = DATA / "corp_actions" / "raw"
    for f in list(raw.glob(f"TWT49U_{TODAY.year}*.json")) + list(raw.glob(f"TWTAUU_{TODAY.year}.json")):
        f.unlink()


def _fill_index_gaps() -> str:
    """fetch_index_targets 抓的 ETF／海外指數停在 2024；daily_update 只抓今年，中間的年度補上。"""
    sys.path.insert(0, str(ROOT))
    import daily_update as D
    msgs = []
    for sym, name in D.INDEX_JOBS:
        p = D.MARKET / f"{name}.csv"
        if not p.exists():
            continue
        last = pd.to_datetime(pd.read_csv(p, encoding="utf-8-sig")["date"]).max()
        for y in range(last.year, TODAY.year):
            df = D.fetch_year(sym, y)
            if df is not None and not df.empty:
                D.merge_only_grow(p, df)
                msgs.append(f"{name}:{y}")
            time.sleep(1.0)
    return "補年度缺口 " + ("、".join(msgs) if msgs else "無")


def _append_price_fixes() -> str:
    j = pd.read_csv(ROOT / "results" / "data_fix_2011" / "price_jumps.csv", dtype={"code": str})
    keep = j["why"].str.startswith("上市前") | (j["why"].str.startswith("事件日") & (j["ret"] < 0))
    fx = j[~keep].copy()
    path = DATA / "price_fixes.csv"
    old = pd.read_csv(path, dtype={"code": str})
    fx = fx[~(fx["code"] + fx["date"].astype(str)).isin(set(old["code"] + old["date"].astype(str)))]
    if fx.empty:
        return "沒有新的假跳動"
    fx["factor"] = 1 + fx["ret"]
    pd.concat([old, fx[["code", "date", "factor", "ret", "why"]]], ignore_index=True) \
        .sort_values(["code", "date"]).to_csv(path, index=False, encoding="utf-8-sig")
    return f"新增 {len(fx)} 筆假跳動修正：" + "、".join(f"{c} {d}" for c, d in zip(fx['code'], fx['date']))


# ---------------- 步驟 ----------------

def step1():
    ok, msg = run("daily_update.py", "--quiet")
    return ok, _fill_index_gaps() + "\n" + msg


def step2():
    ok, msg = run("t86_crawler.py")
    if not ok:
        return ok, msg
    ok2, msg2 = run("-m", "davis_tw.prepare_chips")
    return ok2, msg + "\n" + msg2


def step3():
    s = _reset_fundamentals()
    return run("tw_fundamentals_crawler.py", "--start", s.isoformat(), "--end", TODAY.isoformat())


def step4():
    m0, q0 = _reset_financials()
    return run("financials_crawler.py", "--start-rev", m0, "--start-q", q0, "--end", TODAY.isoformat())


def step5():
    s = _reset_intl_news()
    return run("intl_news_crawler.py", "--start", s.isoformat(), "--end", TODAY.isoformat())


def step6():
    return run("ptt_crawler.py", "--days", "3")


def step7():
    return run("gdelt_news_crawler.py", "--daily", timeout=3600)


def step8():
    y0 = _reset_taifex()
    return run("taifex_crawler.py", "--start-year", str(y0), "--end-year", str(TODAY.year))


def step9():
    _reset_corp_actions()
    return run("corp_actions.py", "--start", "2010", "--end", str(TODAY.year))


def step10():
    ok, m1 = run("delisted_crawler.py")
    ok2, m2 = run("delisted_prices.py", "--mode", "full")
    ok3, m3 = run("adjust_delisted.py")
    return ok and ok2 and ok3, "\n".join([m1, m2, m3])


def step11():
    ok, msg = run("tool/price_jump_scan.py")
    return ok, (_append_price_fixes() if ok else msg)


def step12():
    msgs = []
    ok_all = True
    for args in (("-c", "import sys; sys.path.insert(0,'.'); import universe_pit as U; "
                        f"U.get_mask('2003-01-01','{TODAY.year}-12-31',250,refresh=True)"),
                 ("feature_engineering.py", "--quiet"),
                 ("feature_engineering.py", "--uni", "400", "--tag", "_full", "--quiet"),
                 ("tool/run_surv.py", "feature_engineering", "--tag", "_surv", "--quiet")):
        ok, m = run(*args)
        ok_all &= ok
        msgs.append(m.splitlines()[-1] if m else "")
    # 新聞與 PTT 的分析快取：資料更新了就刪掉，下次分析時重建
    for f in (ROOT / "results" / "volatility_news" / "_cache_ptt_board.pkl",
              ROOT / "results" / "theme" / "theme_daily.csv"):
        if f.exists():
            f.unlink()
    return ok_all, "\n".join(msgs)


def step13():
    """開盤前要用的東西：模型預測接到最新一天、寫進本機 DB、每日自動跟單的模擬帳戶。
    要在第 12 步（特徵庫）之後跑；預測沿用 data/models/ 存好的每一折模型，約一年才重訓一次。"""
    msgs, ok_all = [], True
    for title, args in (("預測", ("tool/run_surv.py", "predict_daily", "--tag", "_surv")),
                        ("本機 DB", ("build_local_db.py", "--tables", "prices", "market", "chips", "valuation",
                                     "financials", "news", "research", "predictions", "etf")),
                        ("模擬帳戶", ("paper_daily.py",))):
        ok, m = run(*args)
        ok_all &= ok
        msgs.append(f"{title}：{'成功' if ok else '失敗'}\n  " + (m or "").replace("\n", "\n  "))
    return ok_all, "\n".join(msgs)


STEPS = {1: ("股價、指數、ETF、油價、鉅亨個股新聞", step1),
         2: ("三大法人 T86", step2),
         3: ("本益比、殖利率、股價淨值比", step3),
         4: ("月營收、季報", step4),
         5: ("鉅亨國際新聞", step5),
         6: ("PTT Stock 板", step6),
         7: ("英文國際新聞 GDELT", step7),
         8: ("台指期", step8),
         9: ("除權息、減資", step9),
         10: ("下市清單與下市股價", step10),
         11: ("股價假跳動檢查", step11),
         12: ("衍生資料重建（母體、特徵庫、快取）", step12),
         13: ("模型預測、本機資料庫、模擬帳戶（開盤前用）", step13)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="每日更新總控")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--skip", type=int, nargs="*", default=[])
    a = ap.parse_args(argv)
    todo = [k for k in STEPS if (a.only is None or k in a.only) and k not in a.skip]
    t0 = datetime.now()
    report = [f"=== 每日更新總控　{t0:%Y-%m-%d %H:%M} ==="]
    print(report[0], flush=True)
    fails = []
    for k in todo:
        name, fn = STEPS[k]
        s = time.time()
        print(f"\n[{k}/{len(STEPS)}] {name}", flush=True)
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        line = f"[{k:>2}] {'成功' if ok else '失敗'}　{name}　{(time.time() - s) / 60:.1f} 分"
        print(line + "\n      " + msg.replace("\n", "\n      "), flush=True)
        report.append(line + "\n      " + msg.replace("\n", "\n      "))
        if not ok:
            fails.append(k)
    mins = (datetime.now() - t0).total_seconds() / 60
    report.append(f"\n共 {len(todo)} 步，失敗 {len(fails)} 步{('：' + str(fails)) if fails else ''}，耗時 {mins:.0f} 分")
    print(report[-1])
    (DATA / "daily").mkdir(parents=True, exist_ok=True)
    (DATA / "daily" / f"all_{TODAY:%Y%m%d}.txt").write_text("\n".join(report), encoding="utf-8")
    return 0 if not fails else 3


if __name__ == "__main__":
    raise SystemExit(main())
