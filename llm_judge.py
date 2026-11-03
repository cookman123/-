#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
讓地端 LLM（Ollama gemma3:12b）參與決策：選股審核、換策略時的出場節奏、策略診斷。

分工原則：數字規則先決定「範圍」，LLM 只在規則給的選項裡挑，輸出固定格式（JSON），
每一筆判斷都寫進 db/local.db 的 llm_decisions 表（同一個題目第二次問直接用快取）。

    LLM 選股 select()      流程圖「實際操作」那一格：規則只給候選池（5 日模型前 N 名），
                           每檔附上「RAG 行為模式」（歷史上最像的情境後來怎麼走）與「新資訊評分」
                           （模型排名、近期漲跌、最新新聞），由 LLM 自己挑最多 K 檔、給權重與理由
    LLM 打分 score()       第二版：同樣的候選池，另外給模型沒有的資訊（月營收年增、季報毛利率與 EPS、
                           新聞內文），要 LLM 對每一檔打 0～100 分，取前 K 名。能算 IC，檢定力比挑選高
    選股審核 review()      規則挑出的候選股 → buy（照買）／half（買一半）／skip（這次不買）。
                           不能自己加股票；沒有明確負面理由時預設照規則買
    換策略   switch_plan()  策略 A 換 B 時，A 的持股依 B 的看法分成續抱／緩衝／該賣，
                           緩衝與該賣的由 LLM 挑出場節奏：now（明天開盤賣）、wait（先不賣，
                           等 B 轉空再說）、stagger3／stagger5（分 3／5 天賣完）
    策略診斷 diagnose()     邏輯檢驗亮燈時，讀檢驗結果與研究文件，寫出可能原因與候選調整，
                           只產生報告，不會自動換策略
    自主操盤 manage()       LLM 自己選股、自己組投資組合：讀 RAG 檢索到的研究結論（判斷邏輯）、
                           DB 裡各策略當天的輸出（5／20 日模型、營收創新高事件、高波動排名）組成的候選池、
                           大盤狀態與目前持股，直接給出目標持股與權重（佔總資產），沒列到的持股就賣掉，
                           剩下的錢放 0050。規則只管硬限制：處置／注意股不能新買、單檔上限、最多幾檔

偷看未來
--------
gemma3 的訓練資料大約到 2024-08（Gemma 3 model card），它可能記得那之前的行情。
所以「LLM 有沒有用」只在 EVAL_START（2024-10-01，多留一個月緩衝）以後的期間評估（llm_eval.py）。
另外研究結論與 results/*.md 是用整段期間（含未來）算出來的，回測模式（pit=True）不給 LLM 看，
只給當天以前就知道的資料：價格、新聞標題、已揭曉的歷史事件、模型預測與它在這檔的過去紀錄。
拿來判斷還沒發生的事（每日實際使用）時沒有這個問題，可以 pit=False 多給研究結論。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / "db" / "local.db"
OLLAMA = "http://localhost:11434"
MODEL = "gemma3:12b"
NUM_CTX = 8192
CUTOFF = "2024-08-31"          # Gemma 3 的訓練資料截止（model card 寫 2024 年 8 月）
EVAL_START = "2024-10-01"      # 評估從這天開始，多留一個月緩衝
PROMPT_VER = "v1"              # 提示詞改了就換版本，舊快取不會被誤用

REVIEW_SYSTEM = """你是台股量化研究的審核員，使用繁體中文。數字規則已經挑出一檔候選股，你要決定：
buy（照規則買）、half（買一半）、skip（這次不買）。
原則：
1. 只能根據「資料」區塊；不要用你自己記得的行情或新聞。
2. 規則有回測、你的判斷沒有，所以沒有明確的負面理由時選 buy。
3. 明確的負面理由例如：當天前後有重大利空新聞（掏空、下市、停工、重大訴訟、財報造假）、
   模型在這檔的過去紀錄顯示偏多時反而常跌、相似歷史事件之後多半大跌。
4. reason 用一兩句話說明，引用資料編號，例如「[3] 提到…」。
只輸出 JSON。"""

SWITCH_SYSTEM = """你是台股量化研究的交易助理，使用繁體中文。投資人要從舊策略換到新策略，
這檔持股在新策略下沒有買進訊號，你要決定出場節奏（只能從選項挑一個）：
now（下一個交易日開盤全部賣出）、wait（先不賣，之後新策略轉空再賣）、
stagger3（分 3 個交易日賣完）、stagger5（分 5 個交易日賣完）。
考量：交易成本（來回約 0.39%）、新策略對這檔的看法有多差、目前損益、新聞、流動性。
新策略明確偏空或有重大利空時選 now；只是中性、沒有理由急著賣時可以 wait；
部位大或想分散賣出時點的選 stagger。只能根據資料，reason 引用資料編號。只輸出 JSON。"""

DIAG_SYSTEM = """你是台股量化研究的助理，使用繁體中文。請根據資料寫一份高波動策略的診斷：
一、檢驗結果摘要（引用數字）。先講最近一次檢驗的結論是「正常」「注意」還是「顯著」。
二、可能原因（市場環境、成本、因子失效等，每點引用資料編號）。結論是「正常」時，簡短說明近期表現的來源即可。
三、候選調整方案（2～3 個）。每個方案的參數要事先寫死一組，不可以「試很多組挑回測最好的」——
    那是過度配適；要寫出怎麼回測（walk-forward、和原策略一起做多重比較檢定）、什麼結果才採用。
四、建議目前先怎麼做。規則是：結論「正常」→ 照原策略繼續，不要建議暫停或換策略；
    「注意」→ 繼續執行、下一期再看；「顯著」→ 先查原因，不自動換股。
不要自己編數字；資料沒有的就說需要補什麼資料。最後提醒：方案要回測通過才能上線，不是投資建議。"""

REVIEW_SCHEMA = {"type": "object", "required": ["decision", "confidence", "reason", "cites"], "properties": {
    "decision": {"type": "string", "enum": ["buy", "half", "skip"]},
    "confidence": {"type": "number"}, "reason": {"type": "string"},
    "cites": {"type": "array", "items": {"type": "integer"}}}}
SWITCH_SCHEMA = {"type": "object", "required": ["pace", "reason", "cites"], "properties": {
    "pace": {"type": "string", "enum": ["now", "wait", "stagger3", "stagger5"]},
    "reason": {"type": "string"}, "cites": {"type": "array", "items": {"type": "integer"}}}}
SELECT_SYSTEM = """你是台股的選股經理，使用繁體中文。資料裡每個 [編號] 是一檔候選股，包含：
新資訊評分（模型對之後 5 日、20 日報酬的全市場排名、近期漲跌、最新新聞標題）與
RAG 行為模式（歷史上最像現在的情境，以及它們後來的實際報酬）。
請從候選股裡挑出接下來 5 個交易日最可能贏過其他候選的股票，最多 {k} 檔，也可以少挑。
原則：
1. 只能根據資料；不要用你自己記得的行情。只能挑資料裡有的代號。
2. 綜合判斷：模型排名高、相似情境後來多半上漲、沒有重大利空新聞的優先；
   相似情境後來多半下跌、或有重大利空（掏空、下市、停工、訴訟、財報問題）的不要挑。
3. weight 是相對權重（0～1），越有把握給越高；reason 一兩句，引用資料編號。
只輸出 JSON。"""
SELECT_SCHEMA = {"type": "object", "required": ["picks", "summary"], "properties": {
    "picks": {"type": "array", "items": {"type": "object", "required": ["code", "weight", "reason", "cites"],
                                         "properties": {"code": {"type": "string"}, "weight": {"type": "number"},
                                                        "reason": {"type": "string"},
                                                        "cites": {"type": "array", "items": {"type": "integer"}}}}},
    "summary": {"type": "string"}}}
SCORE_SYSTEM = """你是台股的選股分析師，使用繁體中文。資料裡每個 [編號] 是一檔候選股，包含：
模型評分（5 日、20 日報酬的全市場排名）、基本面（月營收年增率、營收是否創新高、最近一季毛利率與 EPS 年增）、
最新新聞（標題與內文開頭）、RAG 行為模式（歷史上最像的情境後來怎麼走）。
請為「每一檔」打 0～100 分，代表接下來 5 個交易日的報酬會比其他候選好的程度（50 分＝和其他候選差不多）。
原則：
1. 只能根據資料；不要用你自己記得的行情。每個候選代號都要打分，不能漏。
2. 重點看模型沒有的資訊：營收年增率高且創新高、毛利率或 EPS 年增改善的加分；營收衰退、利空新聞內容的扣分。
3. 分數要拉開，不要全部給差不多；reason 一句話（30 字內）、引用資料編號。
只輸出 JSON。"""
SCORE_SCHEMA = {"type": "object", "required": ["scores"], "properties": {
    "scores": {"type": "array", "items": {"type": "object", "required": ["code", "score", "reason"],
                                          "properties": {"code": {"type": "string"}, "score": {"type": "number"},
                                                         "reason": {"type": "string"}}}}}}
SIZE = {"buy": 1.0, "half": 0.5, "skip": 0.0}
PACE = {"now": "明天開盤全部賣", "wait": "先不賣", "stagger3": "分 3 天賣完", "stagger5": "分 5 天賣完"}


# ---------------------------------------------------------------- 基本

def ensure(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS llm_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        kind TEXT, account INTEGER, batch TEXT, code TEXT, asof TEXT, strategy TEXT, model TEXT,
        decision TEXT, confidence REAL, reason TEXT, cites TEXT, items TEXT, raw TEXT, secs REAL, error TEXT)""")
    con.commit()


def available(model: str = MODEL) -> str | None:
    """Ollama 與模型都在就回傳 None，不然回傳原因。"""
    import requests
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=3).json().get("models", [])
    except Exception:
        return "Ollama 沒有開（http://localhost:11434 連不到）"
    if not any(m["name"] == model or m["name"] == f"{model}:latest" for m in tags):
        return f"還沒下載模型 {model}（ollama pull {model}）"
    return None


def chat_json(system: str, prompt: str, schema: dict, model: str = MODEL, num_ctx: int = NUM_CTX) -> tuple:
    """回傳（解析後的 dict, 原始字串）。溫度 0、固定種子，同一題每次答案一樣，才能重現。"""
    import requests
    r = requests.post(f"{OLLAMA}/api/chat", timeout=900, json={
        "model": model, "stream": False, "format": schema,
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    return json.loads(raw), raw


def chat_text(system: str, prompt: str, model: str = MODEL) -> str:
    import requests
    r = requests.post(f"{OLLAMA}/api/chat", timeout=900, json={
        "model": model, "stream": False, "options": {"temperature": 0.2, "seed": 42, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def _cached(con, key: str):
    r = con.execute("SELECT decision, confidence, reason, cites, items, error FROM llm_decisions WHERE key = ?",
                    [key]).fetchone()
    if not r:
        return None
    return dict(decision=r[0], confidence=r[1], reason=r[2], cites=json.loads(r[3] or "[]"),
                items=json.loads(r[4] or "[]"), error=r[5], cached=True)


def _save(con, key, kind, code, asof, strategy, model, out, raw, secs, account=None, batch=None) -> None:
    con.execute("INSERT OR REPLACE INTO llm_decisions (key, kind, account, batch, code, asof, strategy, model, "
                "decision, confidence, reason, cites, items, raw, secs, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [key, kind, account, batch, code, asof, strategy, model, out.get("decision"), out.get("confidence"),
                 out.get("reason"), json.dumps(out.get("cites", [])), json.dumps(out.get("items", []), ensure_ascii=False),
                 raw, secs, out.get("error")])
    con.commit()


# ---------------------------------------------------------------- 資料（只用 asof 當天以前已知的）

def context(con, code: str, asof: str, pit: bool = True, n_news: int = 8, n_events: int = 4) -> list:
    """每一項是一段可引用的文字。pit=True 時不放用整段期間算出來的研究結論。"""
    import forecast as F

    out = []
    s = con.execute("SELECT name, category FROM stocks WHERE code = ?", [code]).fetchone()
    name = f"{code} {s[0]}（{s[1]}）" if s else code
    px = pd.read_sql("SELECT date, close, volume FROM prices WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 61",
                     con, params=[code, asof])
    if len(px) >= 21:
        c = px["close"].to_numpy()
        tv = (px["close"] * px["volume"]).iloc[:20].mean() / 1e8
        out.append(f"{name} 截至 {px['date'].iloc[0]}：收盤 {c[0]:.2f}，近 5 日 {c[0] / c[5] - 1:+.1%}，"
                   f"近 20 日 {c[0] / c[20] - 1:+.1%}，距近 60 日高點 {c[0] / c.max() - 1:+.1%}，"
                   f"20 日均成交值 {tv:.1f} 億。")
    news = con.execute("SELECT n.ts, n.title FROM news n JOIN news_stock s USING (news_id) "
                       "WHERE s.code = ? AND n.ts < ? ORDER BY n.ts DESC LIMIT ?",
                       [code, f"{asof} 23:59:59", n_news]).fetchall()
    out += [f"{name} 的新聞（{t[:10]}）：{h}" for t, h in news]
    if not news:
        out.append(f"{name} 在 {asof} 以前的資料庫裡沒有新聞。")
    pxs = F.load_px(con, code)
    for key in ("h5", "h20", "dip"):
        try:
            r = F.forecast(con, code, asof, key, px=pxs)
        except Exception:
            continue
        out.append(f"【{r['name']}】" + "".join(r["facts"]) + ("" if pit else f"研究結論：{r['evidence']}"))
    # 這檔過去已揭曉的跌深事件（出場日在 asof 以前）
    ev = con.execute("SELECT text FROM kb_chunks WHERE kind = 'event' AND code = ? AND known_at < ? "
                     "ORDER BY known_at DESC LIMIT ?", [code, asof, n_events]).fetchall()
    out += [f"這檔過去的事件：{t[0]}" for t in ev]
    if not pit:
        pk = con.execute("SELECT asof, rank, action FROM picks WHERE code = ?", [code]).fetchone()
        if pk:
            out.append(f"{name} 在高波動策略最新清單（{pk[0]}）排第 {pk[1]}，動作：{pk[2]}。")
    return out


def _numbered(items: list) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(items, 1))


def _clean_cites(cites, n: int) -> list:
    return sorted({int(c) for c in (cites or []) if isinstance(c, (int, float)) and 1 <= int(c) <= n})


# ---------------------------------------------------------------- LLM 選股（RAG 行為模式＋新資訊評分）

_EVENTS: dict = {}


def _event_vectors(con):
    """事件卡的向量只讀一次（約 1 萬張 × 768 維）。"""
    if "V" not in _EVENTS:
        ev = pd.read_sql("SELECT code, known_at, text, embedding FROM kb_chunks WHERE kind = 'event' "
                         "AND embedding IS NOT NULL", con)
        _EVENTS["meta"] = ev[["code", "known_at", "text"]].reset_index(drop=True)
        _EVENTS["V"] = np.stack([np.frombuffer(b, dtype=np.float32) for b in ev["embedding"]])
    return _EVENTS["meta"], _EVENTS["V"]


def _embed(texts: list) -> np.ndarray:
    import requests

    from build_kb import query_prefix
    r = requests.post(f"{OLLAMA}/api/embed", timeout=120, json={
        "model": "embeddinggemma", "input": [query_prefix("embeddinggemma") + t for t in texts]})
    r.raise_for_status()
    v = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


def _brief(con, code: str, asof: str, pctile: dict, closes: pd.DataFrame | None) -> tuple:
    """一檔候選股的「新資訊評分」與相似情境檢索用的描述。回傳（文字, 檢索查詢, 價格走勢相似的結果）。"""
    import forecast as F

    s = con.execute("SELECT name, category FROM stocks WHERE code = ?", [code]).fetchone()
    name = f"{code} {s[0]}（{s[1]}）" if s else code
    px = pd.read_sql("SELECT date, close FROM prices WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 61",
                     con, params=[code, asof])
    c = px["close"].to_numpy()
    mv = (f"近 5 日 {c[0] / c[5] - 1:+.1%}、近 20 日 {c[0] / c[20] - 1:+.1%}、距 60 日高點 {c[0] / c.max() - 1:+.1%}"
          if len(c) > 20 else "價格資料不足")
    news = [h for _, h in con.execute(
        "SELECT n.ts, n.title FROM news n JOIN news_stock s USING (news_id) WHERE s.code = ? AND n.ts < ? "
        "ORDER BY n.ts DESC LIMIT 3", [code, f"{asof} 23:59:59"]).fetchall()]
    rank = "、".join(f"{h} 日模型前 {1 - p:.0%}" for h, p in pctile.items() if p == p)
    # 價格走勢的相似情境：這檔自己歷史上最像近 20 天的 10 段，之後 5 天怎麼走（只用已走完的片段）
    a = F.analog_search(closes if closes is not None else pd.DataFrame(
        {code: F.load_px(con, code)["close"]}), code, pd.Timestamp(asof), 20, 5, 10, "self")
    ana = (f"價格走勢相似的 {len(a['F'])} 段歷史，之後 5 日中位數 {np.expm1(np.median(a['F'][:, -1])):+.1%}、"
           f"上漲比例 {(a['F'][:, -1] > 0).mean():.0%}" if a else "價格走勢相似的歷史不足")
    text = (f"{name}｜新資訊評分：{rank}；{mv}；新聞：{'／'.join(news) if news else '無'}｜"
            f"RAG 行為模式：{ana}")
    query = f"{name} {mv} 新聞：{'；'.join(news)}"
    return text, query


def select(con, asof: str, cands: pd.DataFrame, k: int = 5, pit: bool = True, model: str = MODEL,
           account: int | None = None, batch: str | None = None) -> dict:
    """cands：至少有 code 欄（可另有 pctile5、pctile20）。回傳 dict(picks=[{code, weight, reason, cites}],
    summary, items, codes, error, cached)。權重已標準化到合計 1；LLM 失敗時 picks 為空（那期不進場）。"""
    ensure(con)
    codes = list(cands["code"])
    key = f"select|{asof}|{','.join(codes)}|k{k}|{model}|pit{int(pit)}|{PROMPT_VER}"
    r = con.execute("SELECT raw, items, error FROM llm_decisions WHERE key = ?", [key]).fetchone()
    if r:
        out = json.loads(r[0]) if r[0] else {"picks": [], "summary": ""}
        return dict(picks=out["picks"], summary=out.get("summary", ""), items=json.loads(r[1] or "[]"),
                    codes=codes, error=r[2], cached=True)
    meta, V = _event_vectors(con)
    known = (meta["known_at"] < asof).to_numpy()
    items, queries = [], []
    for row in cands.itertuples():
        pc = {5: getattr(row, "pctile5", np.nan), 20: getattr(row, "pctile20", np.nan)}
        t, q = _brief(con, row.code, asof, pc, None)
        items.append(t)
        queries.append(q)
    # RAG：每檔用自己的現況描述去檢索所有已揭曉的歷史事件卡，取最像的 2 張（含後來的淨報酬）
    Q = _embed(queries)
    for i, q in enumerate(Q):
        sim = np.where(known, V @ q, -np.inf)
        top = np.argsort(-sim)[:2]
        sims = []
        for j in top:
            m = re.search(r"淨報酬 ([+-]?\d+\.\d+%)", meta.at[j, "text"])
            sims.append(f"{meta.at[j, 'code']} {meta.at[j, 'known_at']} 出場、淨報酬 {m.group(1) if m else '?'}")
        items[i] += f"；最像的歷史事件：{'、'.join(sims)}"
    prompt = (f"日期：{asof} 收盤後。候選股（共 {len(items)} 檔，編號 1 到 {len(items)}）：\n{_numbered(items)}\n\n"
              f"請挑出最多 {k} 檔。")
    t0 = time.time()
    try:
        out, raw = chat_json(SELECT_SYSTEM.format(k=k), prompt, SELECT_SCHEMA, model)
        picks, seen = [], set()
        for p in out.get("picks", []):
            c = str(p.get("code", "")).strip()[:6]
            c = next((x for x in codes if c.startswith(x) or x.startswith(c)), None)   # 允許「2330 台積電」寫法
            if c and c not in seen and len(picks) < k:
                seen.add(c)
                picks.append(dict(code=c, weight=float(np.clip(p.get("weight", 1), 0, 1)) or 0.1,
                                  reason=p.get("reason", ""), cites=_clean_cites(p.get("cites"), len(items))))
        tot = sum(p["weight"] for p in picks)
        for p in picks:
            p["weight"] /= tot
        out = dict(picks=picks, summary=out.get("summary", ""))
        err = None
    except Exception as e:
        out, raw, err = dict(picks=[], summary=""), None, f"{type(e).__name__}: {e}"[:300]
    con.execute("INSERT OR REPLACE INTO llm_decisions (key, kind, account, batch, code, asof, strategy, model, "
                "decision, reason, items, raw, secs, error) VALUES (?, 'select', ?, ?, ?, ?, 'llm_pick', ?, ?, ?, ?, ?, ?, ?)",
                [key, account, batch, ",".join(p["code"] for p in out["picks"]), asof, model,
                 f"{len(out['picks'])} 檔", out["summary"], json.dumps(items, ensure_ascii=False),
                 json.dumps(out, ensure_ascii=False), time.time() - t0, err])
    con.commit()
    return dict(**out, items=items, codes=codes, error=err, cached=False)


_RESTRICTED = None


def restricted_flags(con, codes: list, asof: str) -> pd.Series:
    """asof 收盤後下單、隔日開盤成交時，哪些股票是處置／注意／全額交割股（restricted.py 的規則）。
    這是交易所公告的硬條件，由規則排除，不交給 LLM 判斷。"""
    global _RESTRICTED
    import restricted as R
    if _RESTRICTED is None:
        _RESTRICTED = R.load()
    days = pd.DatetimeIndex(pd.read_sql("SELECT DISTINCT date FROM prices WHERE date >= ? ORDER BY date", con,
                                        params=[(pd.Timestamp(asof) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")])
                            ["date"].pipe(pd.to_datetime))
    keys = pd.DataFrame({"date": pd.Timestamp(asof), "code": [str(c) for c in codes]})
    return pd.Series(R.flags(keys, days, 1, _RESTRICTED).to_numpy(), index=keys["code"])


def fundamentals(con, code: str, asof: str) -> tuple:
    """asof 當天已公告的基本面（用公告日 avail_date 判斷，不是營收月份）。回傳（文字, 最新月營收年增率）。"""
    rv = pd.read_sql("SELECT ym, revenue, yoy_pct FROM revenue WHERE code = ? AND avail_date <= ? ORDER BY ym",
                     con, params=[code, asof])
    parts, yoy = [], np.nan
    if len(rv):
        last = rv.iloc[-1]
        yoy = float(last["yoy_pct"]) if pd.notna(last["yoy_pct"]) else np.nan
        t = rv.tail(3)
        prior = (t["revenue"] / (1 + t["yoy_pct"] / 100)).sum()
        y3 = t["revenue"].sum() / prior - 1 if prior and t["yoy_pct"].notna().all() else np.nan
        hi = ""
        if len(rv) >= 25:
            gap = last["revenue"] / rv["revenue"].iloc[:-1].max() - 1
            hi = "，創歷史新高" if gap >= 0 else f"，距歷史最高 {gap:+.0%}"
        parts.append(f"{last['ym']} 月營收年增 {yoy:+.1f}%、近 3 月合計年增 "
                     f"{'缺' if y3 != y3 else f'{y3:+.1%}'}{hi}")
    q = pd.read_sql("SELECT year, q, revenue, gross_profit, op_income, eps FROM quarterly WHERE code = ? "
                    "AND avail_date <= ? ORDER BY year DESC, q DESC LIMIT 5", con, params=[code, asof])
    if len(q) and q.iloc[0]["revenue"]:
        r0 = q.iloc[0]
        gm = r0["gross_profit"] / r0["revenue"] if pd.notna(r0["gross_profit"]) else np.nan
        om = r0["op_income"] / r0["revenue"] if pd.notna(r0["op_income"]) else np.nan
        ly = q[(q["year"] == r0["year"] - 1) & (q["q"] == r0["q"])]
        eps_ly = float(ly["eps"].iloc[0]) if len(ly) and pd.notna(ly["eps"].iloc[0]) else np.nan
        parts.append(f"{int(r0['year'])}Q{int(r0['q'])} 毛利率 {'缺' if gm != gm else f'{gm:.1%}'}、營益率 "
                     f"{'缺' if om != om else f'{om:.1%}'}、EPS {r0['eps']:.2f}"
                     + ("" if eps_ly != eps_ly else f"（去年同季 {eps_ly:.2f}）"))
    return ("；".join(parts) if parts else "基本面資料缺"), yoy


def score(con, asof: str, cands: pd.DataFrame, k: int = 5, model: str = MODEL, account: int | None = None,
          batch: str | None = None) -> dict:
    """第二版：每檔打分。回傳 dict(scores={code: 分數}, reasons, picks（前 k 名、等權）, rev_yoy, items, error, cached, n_missing)。
    漏打的代號給 50 分（中間值），n_missing 記下來。只給 asof 當天以前已知的資料。"""
    ensure(con)
    codes = list(cands["code"])
    key = f"score|{asof}|{','.join(codes)}|{model}|v2"
    r = con.execute("SELECT raw, items, error FROM llm_decisions WHERE key = ?", [key]).fetchone()
    if r:
        out = json.loads(r[0])
        return out | dict(items=json.loads(r[1] or "[]"), error=r[2], cached=True)
    meta, V = _event_vectors(con)
    known = (meta["known_at"] < asof).to_numpy()
    items, queries, rev = [], [], {}
    for row in cands.itertuples():
        s0 = con.execute("SELECT name, category FROM stocks WHERE code = ?", [row.code]).fetchone()
        name = f"{row.code} {s0[0]}（{s0[1]}）" if s0 else row.code
        px = pd.read_sql("SELECT close FROM prices WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 21",
                         con, params=[row.code, asof])["close"].to_numpy()
        mv = f"近 5 日 {px[0] / px[5] - 1:+.1%}、近 20 日 {px[0] / px[20] - 1:+.1%}" if len(px) > 20 else ""
        fund, rev[row.code] = fundamentals(con, row.code, asof)
        news = con.execute("SELECT n.ts, n.title, COALESCE(NULLIF(n.text, ''), n.summary, '') FROM news n "
                           "JOIN news_stock s USING (news_id) WHERE s.code = ? AND n.ts < ? ORDER BY n.ts DESC LIMIT 2",
                           [row.code, f"{asof} 23:59:59"]).fetchall()
        nw = "／".join(f"{t[:10]} {h}：{re.sub(r'\s+', ' ', b)[:90]}" for t, h, b in news) or "無"
        a = None
        try:
            import forecast as F
            a = F.analog_search(pd.DataFrame({row.code: F.load_px(con, row.code)["close"]}), row.code,
                                pd.Timestamp(asof), 20, 5, 10, "self")
        except Exception:
            pass
        ana = (f"走勢相似的歷史之後 5 日中位數 {np.expm1(np.median(a['F'][:, -1])):+.1%}、上漲 {(a['F'][:, -1] > 0).mean():.0%}"
               if a else "走勢相似的歷史不足")
        items.append(f"{name}｜模型：5 日前 {1 - row.pctile5:.0%}、20 日前 {1 - row.pctile20:.0%}；{mv}｜"
                     f"基本面：{fund}｜新聞：{nw}｜RAG：{ana}")
        queries.append(f"{name} {mv} {fund}")
    Q = _embed(queries)
    for i, q in enumerate(Q):
        j = int(np.argmax(np.where(known, V @ q, -np.inf)))
        m = re.search(r"淨報酬 ([+-]?\d+\.\d+%)", meta.at[j, "text"])
        items[i] += f"；最像的歷史事件 {meta.at[j, 'code']} {meta.at[j, 'known_at']} 出場、淨報酬 {m.group(1) if m else '?'}"
    prompt = (f"日期：{asof} 收盤後。候選股（共 {len(items)} 檔，編號 1 到 {len(items)}）：\n{_numbered(items)}\n\n"
              f"請為這 {len(items)} 檔每一檔打分：{', '.join(codes)}。")
    t0 = time.time()
    scores, reasons, err = {}, {}, None
    try:
        res, raw = chat_json(SCORE_SYSTEM, prompt, SCORE_SCHEMA, model)
        for x in res.get("scores", []):
            c = str(x.get("code", "")).strip()[:6]
            c = next((y for y in codes if c.startswith(y) or y.startswith(c)), None)
            if c and c not in scores:
                scores[c] = float(np.clip(x.get("score", 50), 0, 100))
                reasons[c] = x.get("reason", "")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
    miss = [c for c in codes if c not in scores]
    for c in miss:
        scores[c] = 50.0
    order = sorted(codes, key=lambda c: (-scores[c], codes.index(c)))     # 同分時照模型排名
    out = dict(scores=scores, reasons=reasons, picks=order[:k], rev_yoy=rev, n_missing=len(miss))
    con.execute("INSERT OR REPLACE INTO llm_decisions (key, kind, account, batch, code, asof, strategy, model, "
                "decision, reason, items, raw, secs, error) VALUES (?, 'score', ?, ?, ?, ?, 'llm_score', ?, ?, ?, ?, ?, ?, ?)",
                [key, account, batch, ",".join(out["picks"]), asof, model, f"漏打 {len(miss)} 檔",
                 json.dumps(reasons, ensure_ascii=False)[:2000], json.dumps(items, ensure_ascii=False),
                 json.dumps(out, ensure_ascii=False), time.time() - t0, err])
    con.commit()
    return out | dict(items=items, error=err, cached=False)


def candidates(con, asof: str, n: int = 12, exclude_restricted: bool = True) -> pd.DataFrame:
    """候選池：當天 5 日模型排名前 n 名（附 20 日模型排名）。當天預測不完整就用最近一個完整的日子。
    exclude_restricted：先排除隔日不能正常買的處置／注意／全額交割股，再取前 n 名（tw-stock-quant 的
    restricted.py）。llm_eval.py、llm_select_eval.py 的規格是加入這條之前寫死的，它們傳 False 照舊重現。"""
    import forecast as F
    days = F.complete_days(con, 5)
    d = days.index[days.index <= asof]
    if not len(d):
        return pd.DataFrame(columns=["code", "pred", "pctile5", "pctile20", "pred_date"])
    d = d[-1]
    p5 = pd.read_sql("SELECT code, pred, pctile AS pctile5 FROM predictions WHERE h = 5 AND date = ?", con,
                     params=[d]).sort_values("pred", ascending=False)
    if exclude_restricted:
        bad = restricted_flags(con, p5["code"].head(n * 3).tolist(), asof)
        p5 = p5[~p5["code"].astype(str).map(bad).fillna(False).astype(bool)]
    p5 = p5.head(n)
    p20 = pd.read_sql("SELECT code, pctile AS pctile20 FROM predictions WHERE h = 20 AND date = ?", con, params=[d])
    return p5.merge(p20, on="code", how="left").assign(pred_date=d)[["code", "pred", "pctile5", "pctile20",
                                                                      "pred_date"]]


# ---------------------------------------------------------------- 自主操盤（LLM 自己選股、組投資組合）

MANAGE_VER = "pm2"
MANAGE_CTX = 16384                 # 研究結論＋30 幾檔候選，8192 放不下
MANAGE_SYSTEM = """你是台股模擬帳戶的投資組合經理，使用繁體中文。你要自己決定接下來持有哪些股票、各佔總資產多少。
資料有四種，都有 [編號]：
- 研究結論：本專題回測研究的判斷邏輯（從知識庫檢索），說明哪些訊號經得起檢驗、哪些不行、成本與出場怎麼處理
- 大盤與帳戶：大盤位置、目前持股（權重、損益、持有天數）
- 候選股：資料庫裡各策略今天的輸出（5 日模型、20 日模型、營收創新高事件、高波動排名）與目前持股，
  每檔附上模型排名、近期漲跌、波動、最新月營收、相似歷史事件的結果、最新新聞
規則：
1. 只能根據資料；不要用你自己記得的行情或新聞。只能挑候選股裡的代號。
2. 依研究結論的判斷邏輯來決定：哪些訊號比較可信、要不要分散、要不要保留在 0050。
3. weight 是佔總資產的比例（0～{max_w}），最多 {k} 檔（續抱的也算），合計不超過 1；沒用到的部分自動放 0050。
   超過 {k} 檔時，權重最小的會被賣掉。
   沒有把握時可以少買，甚至全部放 0050。
4. targets 是要買進、加碼或調整權重的股票；要全部賣掉的目前持股列在 sells 並寫理由。
   目前持股兩邊都沒列的，維持原權重續抱。來回成本約 0.39%，沒有明確理由不要頻繁換股。
5. 標示「不能新買」的股票（處置／注意／全額交割股）只能續抱或賣出，不能加碼。
6. 每檔 reason 一兩句，引用資料編號；summary 說明這期的整體判斷（例如為什麼放多少在 0050）。
只輸出 JSON。"""
MANAGE_SCHEMA = {"type": "object", "required": ["targets", "sells", "summary"], "properties": {
    "targets": {"type": "array", "items": {"type": "object", "required": ["code", "weight", "reason", "cites"],
                                           "properties": {"code": {"type": "string"}, "weight": {"type": "number"},
                                                          "reason": {"type": "string"},
                                                          "cites": {"type": "array", "items": {"type": "integer"}}}}},
    "sells": {"type": "array", "items": {"type": "object", "required": ["code", "reason"],
                                         "properties": {"code": {"type": "string"}, "reason": {"type": "string"}}}},
    "summary": {"type": "string"}}}
# 檢索研究結論用的固定問題；獲利能力結論.md 已被研究結論總整理取代，不用
RESEARCH_QUERIES = ("哪些選股訊號扣成本、多重比較後仍然有效", "營收年增率與營收創新高的選股效果",
                    "高波動策略是風險溢酬還是 alpha", "停損、目標價與固定持有的出場比較",
                    "交易成本、換手率與緩衝帶", "LLM 選股的評估結果", "跌深反彈與 0050 買進持有的比較")
_RESEARCH: dict = {}


def research(con, n: int = 8, chars: int = 600) -> list:
    """RAG：用固定問題從知識庫（kb_chunks 的研究文件）檢索判斷邏輯，每個問題取最像的段落，去重後最多 n 段。
    研究結論是用整段期間算出來的，拿來回測過去會偷看未來；實際往後跑沒有這個問題。"""
    if "items" not in _RESEARCH:
        doc = pd.read_sql("SELECT source, text, embedding FROM kb_chunks WHERE kind = 'doc' AND embedding IS NOT NULL "
                          "AND source NOT LIKE '%獲利能力結論%'", con)
        D = np.stack([np.frombuffer(b, dtype=np.float32) for b in doc["embedding"]])
        picked = []
        for q in _embed(list(RESEARCH_QUERIES)):
            for j in np.argsort(-(D @ q)):
                if j not in picked:
                    picked.append(int(j))
                    break
        _RESEARCH["items"] = [f"研究結論（{doc.at[j, 'source']}）：{re.sub(r'\s+', ' ', doc.at[j, 'text'])[:chars]}"
                              for j in picked[:n]]
    return _RESEARCH["items"]


def _rev(con, code: str, asof: str) -> tuple:
    """asof 當天已公告的最新月營收：（月份, 年增率%, 是否創歷史新高）。"""
    rv = pd.read_sql("SELECT ym, revenue, yoy_pct FROM revenue WHERE code = ? AND avail_date <= ? ORDER BY ym",
                     con, params=[code, asof])
    if rv.empty:
        return None, np.nan, False
    hi = len(rv) >= 25 and rv["revenue"].iloc[-1] >= rv["revenue"].iloc[:-1].max()
    return rv["ym"].iloc[-1], rv["yoy_pct"].iloc[-1], bool(hi)


def pool(con, asof: str, n: int = 8, held: dict | None = None) -> pd.DataFrame:
    """候選池：DB 裡各策略在 asof 的輸出，每個來源取前 n 名，加上目前持股。全部只用 asof 當天以前已知的資料。
    回傳 code、tags（來自哪些策略）、pctile5、pctile20、vol60、can_buy（不是處置／注意／全額交割股）。"""
    import forecast as F
    held = held or {}
    days = F.complete_days(con, 5)
    d = days.index[days.index <= asof]
    if not len(d):
        return pd.DataFrame()
    d = d[-1]
    pr = pd.read_sql("SELECT code, h, pred, pctile FROM predictions WHERE h IN (5, 20) AND date = ?", con, params=[d])
    p5 = pr[pr["h"] == 5].set_index("code")
    p20 = pr[pr["h"] == 20].set_index("code")
    univ = list(p5.index)                      # 模型的母體：成交金額前 250 名
    tags: dict = {}

    def add(codes, tag):
        for c in codes:
            tags.setdefault(c, []).append(tag)
    add(p5["pred"].nlargest(n).index, f"5 日模型前 {n}")
    add(p20["pred"].nlargest(n).index, f"20 日模型前 {n}")
    # 營收創新高且年增 ≥ 30%（研究 15.5 節的事件），近 40 天內公告的
    lo = (pd.Timestamp(asof) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    ev = pd.read_sql("SELECT code, yoy_pct FROM revenue WHERE avail_date > ? AND avail_date <= ? AND yoy_pct >= 30",
                     con, params=[lo, asof])
    ev = ev[ev["code"].isin(univ)].sort_values("yoy_pct", ascending=False).drop_duplicates("code")
    add([c for c in ev["code"] if _rev(con, c, asof)[2]][:n], "營收創新高且年增≥30%")
    # 高波動：母體內近 60 日日報酬標準差最高的（研究第六節的波動度溢酬）
    q = ",".join("?" * len(univ))
    px = pd.read_sql(f"SELECT date, code, close FROM prices WHERE code IN ({q}) AND date <= ? AND date >= ?", con,
                     params=univ + [asof, (pd.Timestamp(asof) - pd.Timedelta(days=100)).strftime("%Y-%m-%d")]) \
        .pivot(index="date", columns="code", values="close").sort_index().tail(61)
    vol = np.log(px).diff().std() * np.sqrt(252)
    add(vol.nlargest(n).index, f"高波動前 {n}")
    add(held, "目前持股")
    codes = list(tags)
    bad = restricted_flags(con, codes, asof)
    out = pd.DataFrame({"code": codes, "tags": ["、".join(tags[c]) for c in codes]})
    out["can_buy"] = ~out["code"].map(bad).fillna(False).astype(bool)
    out = out[out["can_buy"] | out["code"].isin(held)]          # 不能買又沒持有的直接拿掉
    out["pctile5"] = out["code"].map(p5["pctile"])
    out["pctile20"] = out["code"].map(p20["pctile"])
    out["vol60"] = out["code"].map(vol)
    return out.reset_index(drop=True)


def _pm_line(con, r, asof: str, held: dict, sim: str) -> str:
    s = con.execute("SELECT name, category FROM stocks WHERE code = ?", [r.code]).fetchone()
    name = f"{r.code} {s[0]}（{s[1]}）" if s else r.code
    c = pd.read_sql("SELECT close FROM prices WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 21",
                    con, params=[r.code, asof])["close"].to_numpy()
    mv = f"近 5 日 {c[0] / c[5] - 1:+.1%}、近 20 日 {c[0] / c[20] - 1:+.1%}" if len(c) > 20 else "價格資料不足"
    rank = "、".join(f"{h} 日模型前 {1 - p:.0%}" for h, p in (("5", r.pctile5), ("20", r.pctile20)) if p == p) \
        or "不在模型母體"
    ym, yoy, hi = _rev(con, r.code, asof)
    rev = f"{ym} 月營收年增 {yoy:+.0f}%{'、創歷史新高' if hi else ''}" if ym and yoy == yoy else "營收資料缺"
    news = con.execute("SELECT n.ts, n.title FROM news n JOIN news_stock s USING (news_id) WHERE s.code = ? "
                       "AND n.ts < ? ORDER BY n.ts DESC LIMIT 2", [r.code, f"{asof} 23:59:59"]).fetchall()
    nw = "／".join(f"{t[5:10]} {h[:40]}" for t, h in news) or "無"
    h = held.get(r.code)
    hold = (f"｜持有中：權重 {h['w']:.0%}、未實現 {h['pnl']:+.1%}、已持有 {h['days']} 個交易日" if h else "")
    return (f"{name}｜來源：{r.tags}{'' if r.can_buy else '｜不能新買（處置／注意股）'}｜{rank}；{mv}；"
            f"60 日年化波動 {r.vol60:.0%}｜{rev}｜{sim}｜新聞：{nw}{hold}")


def manage(con, asof: str, held: dict, k: int = 10, max_w: float = 0.2, n: int = 8, model: str = MODEL,
           account: int | None = None, batch: str | None = None) -> dict:
    """LLM 自主操盤。held：{代號: dict(w=權重, pnl=未實現報酬, days=持有交易日數)}。
    回傳 dict(targets=[{code, weight, reason, cites}], sells=[{code, reason}], summary, items, error, cached)。
    targets 是最後的完整目標組合：LLM 沒提到的持股照原權重補進來（續抱），列在 sells 的不在裡面。
    已經過硬限制（只能是候選池的代號、不能新買的不能加碼、單檔 ≤ max_w、最多 k 檔、合計 ≤ 1）。
    LLM 失敗時 targets 照目前持股原樣回傳（不動作）。"""
    ensure(con)
    sig = ",".join(f"{c}:{v['w']:.2f}" for c, v in sorted(held.items()))
    key = f"manage|{asof}|a{account}|{sig}|k{k}|w{max_w}|n{n}|{model}|{MANAGE_VER}"
    r = con.execute("SELECT raw, items, error FROM llm_decisions WHERE key = ?", [key]).fetchone()
    if r and not r[2]:
        return json.loads(r[0]) | dict(items=json.loads(r[1] or "[]"), error=None, cached=True)
    cands = pool(con, asof, n, held)
    items = list(research(con))
    # 大盤狀態
    tw = pd.read_sql("SELECT close FROM market WHERE symbol = 'TWII' AND date <= ? ORDER BY date DESC LIMIT 120",
                     con, params=[asof])["close"].to_numpy()
    items.append(f"大盤 {asof}：加權指數 {tw[0]:,.0f}，近 20 日 {tw[0] / tw[20] - 1:+.1%}，"
                 f"{'在' if tw[0] >= tw.mean() else '跌破'} 120 日均線（{tw.mean():,.0f}）"
                 f"{'之上' if tw[0] >= tw.mean() else ''}。")
    tot = sum(v["w"] for v in held.values())
    items.append("帳戶：" + ("目前沒有個股持股，資金都在 0050／現金。" if not held else
                           f"個股合計 {tot:.0%}（{len(held)} 檔），其餘在 0050／現金。"))
    # RAG 行為模式：每檔現況去檢索已揭曉的歷史跌深事件卡，取最像的 10 張，看它們後來的淨報酬
    meta, V = _event_vectors(con)
    ret = meta["text"].str.extract(r"淨報酬 ([+-]?\d+\.\d+)%")[0].astype(float).to_numpy()
    known = (meta["known_at"] < asof).to_numpy() & ~np.isnan(ret)
    qs = []
    for c in cands.itertuples():
        s = con.execute("SELECT name, category FROM stocks WHERE code = ?", [c.code]).fetchone()
        news = [h for (h,) in con.execute("SELECT n.title FROM news n JOIN news_stock s USING (news_id) "
                                          "WHERE s.code = ? AND n.ts < ? ORDER BY n.ts DESC LIMIT 3",
                                          [c.code, f"{asof} 23:59:59"])]
        qs.append(f"{c.code} {s[0] if s else ''}（{s[1] if s else ''}） 新聞：{'；'.join(news)}")
    Q = _embed(qs) if qs else np.zeros((0, V.shape[1]))
    n0 = len(items)
    for c, q in zip(cands.itertuples(), Q):
        top = np.argsort(-np.where(known, V @ q, -np.inf))[:10]
        sim = (f"相似歷史事件 10 筆：中位淨報酬 {np.median(ret[top]):+.1f}%、賺錢 {(ret[top] > 0).mean():.0%}")
        items.append(_pm_line(con, c, asof, held, sim))
    prompt = (f"日期：{asof} 收盤後決定，下一個交易日開盤成交。\n"
              f"資料（[1]～[{n0}] 是研究結論與帳戶，[{n0 + 1}]～[{len(items)}] 是候選股，共 {len(cands)} 檔）：\n"
              f"{_numbered(items)}\n\n請給出目標投資組合：最多 {k} 檔、單檔不超過 {max_w:.0%}。")
    t0 = time.time()
    codes = list(cands["code"])
    can_buy = dict(zip(cands["code"], cands["can_buy"]))
    try:
        out, raw = chat_json(MANAGE_SYSTEM.format(k=k, max_w=max_w), prompt, MANAGE_SCHEMA, model, MANAGE_CTX)
        sells = []
        for p in out.get("sells", []):
            c = str(p.get("code", "")).strip()[:6]
            c = next((x for x in held if c.startswith(x) or x.startswith(c)), None)
            if c and c not in {s["code"] for s in sells}:
                sells.append(dict(code=c, reason=p.get("reason", "")))
        tg, seen = [], {s["code"] for s in sells}
        for p in out.get("targets", []):
            c = str(p.get("code", "")).strip()[:6]
            c = next((x for x in codes if c.startswith(x) or x.startswith(c)), None)
            if not c or c in seen:
                continue
            w = float(np.clip(p.get("weight", 0) or 0, 0, max_w))
            if not can_buy.get(c, False):
                w = min(w, held.get(c, {}).get("w", 0.0))          # 處置／注意股只能續抱或減碼
            if w > 0:
                seen.add(c)
                tg.append(dict(code=c, weight=w, reason=p.get("reason", ""),
                               cites=_clean_cites(p.get("cites"), len(items))))
        seen |= {x["code"] for x in tg}
        tg += [dict(code=c, weight=min(v["w"], max_w), reason="續抱（沒有列入要賣）", cites=[])
               for c, v in held.items() if c not in seen]
        tg = sorted(tg, key=lambda x: -x["weight"])[:k]
        s = sum(x["weight"] for x in tg)
        for x in tg:
            x["weight"] /= max(s, 1.0)
        res, err = dict(targets=tg, sells=sells, summary=out.get("summary", "")), None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
        res = dict(targets=[dict(code=c, weight=v["w"], reason="LLM 失敗，維持原持股", cites=[])
                            for c, v in held.items()], sells=[], summary="")
    con.execute("INSERT OR REPLACE INTO llm_decisions (key, kind, account, batch, code, asof, strategy, model, "
                "decision, reason, items, raw, secs, error) VALUES (?, 'manage', ?, ?, ?, ?, 'llm_pm', ?, ?, ?, ?, ?, ?, ?)",
                [key, account, batch, ",".join(x["code"] for x in res["targets"]), asof, model,
                 f"{len(res['targets'])} 檔、個股 {sum(x['weight'] for x in res['targets']):.0%}", res["summary"],
                 json.dumps(items, ensure_ascii=False), json.dumps(res, ensure_ascii=False), time.time() - t0, err])
    con.commit()
    return res | dict(items=items, error=err, cached=False)


# ---------------------------------------------------------------- 選股審核

def review(con, code: str, asof: str, strategy: str, plan: dict | None = None, pit: bool = True,
           model: str = MODEL, account: int | None = None, batch: str | None = None) -> dict:
    """回傳 dict(decision, size, confidence, reason, cites, items, cached, error)。
    LLM 出錯時照規則買（decision='buy'，error 寫原因），不會因為模型掛掉就整批不下單。"""
    import forecast as F

    ensure(con)
    key = f"review|{code}|{asof}|{strategy}|{model}|pit{int(pit)}|{PROMPT_VER}"
    hit = _cached(con, key)
    if hit:
        return hit | {"size": SIZE.get(hit["decision"], 1.0)}
    items = context(con, code, asof, pit)
    rule = f"規則：{F.STRATEGIES[strategy]['name']} 在 {asof} 收盤後挑出這檔，"
    if plan and plan.get("enter"):
        rule += f"計畫 {plan['entry_date']:%Y-%m-%d} 開盤買、{plan['exit_date']:%Y-%m-%d} 開盤賣（固定持有）。"
    items.insert(0, rule)
    prompt = (f"資料（共 {len(items)} 項，編號只能是 1 到 {len(items)}）：\n{_numbered(items)}\n\n"
              f"問題：{code} 這筆照規則該買嗎？回答 buy／half／skip。")
    t0 = time.time()
    try:
        out, raw = chat_json(REVIEW_SYSTEM, prompt, REVIEW_SCHEMA, model)
        out["cites"] = _clean_cites(out.get("cites"), len(items))
        out["confidence"] = float(np.clip(out.get("confidence", 0.5), 0, 1))
        if out.get("decision") not in SIZE:
            raise ValueError(f"decision 不合法：{out.get('decision')}")
        out["error"] = None
    except Exception as e:                       # 模型掛掉、輸出不是 JSON 等：照規則
        out, raw = dict(decision="buy", confidence=None, reason="（LLM 失敗，照規則買）", cites=[],
                        error=f"{type(e).__name__}: {e}"[:300]), None
    out["items"] = items
    _save(con, key, "review", code, asof, strategy, model, out, raw, time.time() - t0, account, batch)
    return out | {"size": SIZE[out["decision"]], "cached": False}


# ---------------------------------------------------------------- 換策略

def switch_plan(con, acct: int, new_strategy: str, asof: str, use_llm: bool = True, pit: bool = False,
                model: str = MODEL, progress=None) -> pd.DataFrame:
    """A 換 B：目前持股（不含閒置資金的 0050）依 B 的看法分類，緩衝與該賣的決定出場節奏。
    回傳每檔一列：code, shares, view（B 的判斷）, group（續抱／緩衝／該賣）, pace, reason, cites, items。"""
    import forecast as F
    import paper_trade as P

    ensure(con)
    pos = P.portfolio(con, acct)["positions"]
    pos = pos[pos["code"] != P.ETF]
    rows = []
    for i, r in enumerate(pos.itertuples()):
        if progress:
            progress(i / max(len(pos), 1), f"{r.code}（{i + 1}/{len(pos)}）")
        try:
            f = F.forecast(con, r.code, asof, new_strategy, calib=False)
            view, tone = f["direction"], f["tone"]
        except Exception as e:
            f, view, tone = None, f"無法判斷（{e}）", "muted"
        group = "續抱" if tone == "bull" else "緩衝" if tone == "neutral" else "該賣"
        pace, reason, cites, items = ("hold", "新策略也看多，繼續抱", [], []) if group == "續抱" else \
            (("wait" if group == "緩衝" else "now"), "規則預設：緩衝帶先不賣、該賣的明天賣", [], [])
        if use_llm and group != "續抱":
            key = f"switch|{acct}|{r.code}|{asof}|{new_strategy}|{model}|pit{int(pit)}|{PROMPT_VER}"
            hit = _cached(con, key)
            if hit:
                pace, reason, cites, items = hit["decision"], hit["reason"], hit["cites"], hit["items"]
            else:
                items = context(con, r.code, asof, pit)
                items.insert(0, f"持股：{r.shares:,} 股，平均成本 {r.avg_cost:.2f}，最新收盤 {r.last:.2f}，"
                                f"未實現損益 {r.unrealized:+,.0f}（{r.unrealized / (r.avg_cost * r.shares):+.1%}）。")
                items.insert(1, f"新策略「{F.STRATEGIES[new_strategy]['name']}」對這檔的判斷：{view}。"
                                + ("".join(f["facts"]) if f else ""))
                prompt = (f"資料（共 {len(items)} 項，編號只能是 1 到 {len(items)}）：\n{_numbered(items)}\n\n"
                          f"問題：{r.code} 要用哪種節奏出場？從 now／wait／stagger3／stagger5 挑一個。")
                t0 = time.time()
                try:
                    out, raw = chat_json(SWITCH_SYSTEM, prompt, SWITCH_SCHEMA, model)
                    pace = out["pace"] if out.get("pace") in PACE else pace
                    reason, cites = out.get("reason", ""), _clean_cites(out.get("cites"), len(items))
                    err = None
                except Exception as e:
                    raw, err = None, f"{type(e).__name__}: {e}"[:300]
                    reason = f"（LLM 失敗，照規則預設）{reason}"
                _save(con, key, "switch", r.code, asof, new_strategy, model,
                      dict(decision=pace, reason=reason, cites=cites, items=items, error=err), raw,
                      time.time() - t0, acct)
        rows.append(dict(code=r.code, shares=int(r.shares), view=view, group=group, pace=pace,
                         reason=reason, cites=cites, items=items))
    if progress:
        progress(1.0, "完成")
    return pd.DataFrame(rows, columns=["code", "shares", "view", "group", "pace", "reason", "cites", "items"])


def switch_execute(con, acct: int, plan: pd.DataFrame, asof: str, new_strategy: str) -> list:
    """照 switch_plan 的節奏下賣單：now 在 asof 收盤後送出；staggerN 從 asof 起連續 N 個交易日各賣一份。"""
    import paper_trade as P

    days = [d for (d,) in con.execute("SELECT date FROM market WHERE symbol = 'TWII' AND date >= ? ORDER BY date "
                                      "LIMIT 5", [asof])]
    # 還沒有的未來交易日用平日補（下單日會停在「待成交」，價格更新後才成交）
    while len(days) < 5:
        days.append((pd.Timestamp(days[-1] if days else asof) + pd.offsets.BDay(1)).strftime("%Y-%m-%d"))
    ids, tag = [], f"switch:{new_strategy}"
    for r in plan.itertuples():
        if r.pace == "now":
            parts = [r.shares]
        elif r.pace.startswith("stagger"):
            n = int(r.pace[-1])
            parts = [r.shares // n + (1 if i < r.shares % n else 0) for i in range(n)]
        else:
            continue                   # 續抱、先不賣：舊策略已經排好的賣單照舊
        # 舊策略已經排好的賣單先取消，不然會賣兩次（後面那張變成持股不足退單）
        con.execute("UPDATE paper_orders SET status = 'cancelled', reason = '換策略，改照新的出場節奏' "
                    "WHERE account = ? AND code = ? AND side = 'sell' AND status = 'pending'", [acct, r.code])
        for d, sh in zip(days, parts):
            if sh > 0:
                ids.append(P.place(con, d, r.code, "sell", sh, tag, f"換策略：{PACE[r.pace]}｜{r.reason[:60]}",
                                   settle_now=False, acct=acct))
    P.settle(con, acct)
    return ids


# ---------------------------------------------------------------- 策略診斷

def diagnose(con, asof: str, model: str = MODEL) -> dict:
    """讀邏輯檢驗、策略近況與研究文件，寫診斷報告到 results/llm_diagnosis/。"""
    import local_ask as LA

    ensure(con)
    items = []
    sg = pd.read_sql("SELECT * FROM significance WHERE check_date <= ? ORDER BY check_date DESC LIMIT 4", con,
                     params=[asof])
    for r in sg.itertuples():
        items.append(f"邏輯檢驗 {r.check_date}：42 天超額 {r.excess:+.2%}，歷史第 {r.pctile:.0%} 百分位"
                     f"（第 5 百分位 {r.p05:+.2%}、第 10 百分位 {r.p10:+.2%}），近一年日超額漂移 t {r.drift_t:.2f}，"
                     f"結論「{r.level}」。")
    sd = pd.read_sql("SELECT date, strategy, market, in_market FROM strategy_daily WHERE date <= ? "
                     "ORDER BY date DESC LIMIT 252", con, params=[asof])
    if len(sd):
        for n in (21, 63, 252):
            x = sd.head(n)
            items.append(f"高波動策略近 {n} 個交易日：策略 {(1 + x['strategy']).prod() - 1:+.1%}、"
                         f"大盤 {(1 + x['market']).prod() - 1:+.1%}，在場比例 {x['in_market'].mean():.0%}。")
    hits = LA.retrieve(con, "策略失效 超額報酬變差 原因 調整 波動度溢酬 趨勢過濾 成本", asof, 8, None,
                       "embeddinggemma")
    items += [f"（{h.source}）{h.text}" for h in hits[hits["kind"] == "doc"].itertuples()]
    prompt = (f"資料（共 {len(items)} 項，編號只能是 1 到 {len(items)}）：\n{_numbered(items)}\n\n"
              f"請寫 {asof} 的策略診斷。")
    t0 = time.time()
    text = chat_text(DIAG_SYSTEM, prompt, model)
    bad = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text) if not 1 <= int(x) <= len(items)})
    out_dir = ROOT / "results" / "llm_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{asof}.md"
    path.write_text(f"# 策略診斷 {asof}\n\n由 {model} 根據下列資料產生，只是候選方向，要回測通過才能採用。\n\n"
                    f"{text}\n\n## 資料\n\n" + "\n\n".join(f"[{i}] {t}" for i, t in enumerate(items, 1)),
                    encoding="utf-8")
    _save(con, f"diagnose|{asof}|{model}|{hashlib.md5(prompt.encode()).hexdigest()[:8]}", "diagnose", None, asof,
          "highvol", model, dict(decision="report", reason=text[:2000], items=items,
                                 error=f"出處不存在：{bad}" if bad else None), text, time.time() - t0)
    return dict(text=text, items=items, bad=bad, path=str(path), secs=time.time() - t0)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    con = sqlite3.connect(DB)
    code, asof = (sys.argv[1:3] + ["2330", "2025-06-02"])[:2] if len(sys.argv) < 3 else sys.argv[1:3]
    r = review(con, code, asof, "h5")
    print(json.dumps({k: v for k, v in r.items() if k != "items"}, ensure_ascii=False, indent=1))
