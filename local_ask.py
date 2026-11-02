#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地端問答：用 Ollama 的本機模型，根據 db/local.db 的資料回答問題。

流程
----
  1. 檢索：問題轉向量，在 kb_chunks 找最相近的段落（研究文件、過去的跌深事件）
     只撈 known_at 早於 --asof 的事件 —— 查 2023 年的事，就看不到 2023 年之後才揭曉的結果
  2. 結構化資料：有給 --code 時，直接用 SQL 撈該股截至 --asof 的價格、RAG 規則分數、
     最近新聞標題，以及當時的策略檢驗狀態
  3. 把以上內容和問題一起送給本機模型，要求只根據提供的資料回答並標出處

模型只負責「解釋與整理」，買不買由數字規則決定（rag_rule.py、strategy_spec.py），
不讓語言模型自己下交易決策 —— 規則有回測、有檢定，語言模型的判斷沒有。

用法
----
  python local_ask.py "跌深反彈策略有效嗎？"
  python local_ask.py "這檔現在適合做反彈嗎？" --code 2330
  python local_ask.py "當時該不該進場？" --code 2330 --asof 2022-10-25
  python local_ask.py "..." --dry-run        只印出送給模型的內容，不呼叫模型
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DB = ROOT / "db" / "local.db"
OLLAMA = "http://localhost:11434"
# Google Gemma：繁中可用、12 GB 顯示卡放得下，且沒有中國模型的審查對齊（兩岸、晶片禁令新聞會被改寫）
CHAT_MODEL = "gemma3:12b"
# Ollama 預設上下文太短，一次送三到五千字的資料會被截掉後半段
NUM_CTX = 8192

SYSTEM = """你是台股量化研究的助理，使用繁體中文回答。
規則：
1. 只能根據「資料」區塊的內容回答；資料沒有的就說不知道，不要自己補數字。
2. 引用資料時在句尾標出處編號，例如 [3]。
3. 買賣與否以研究裡的數字規則為準（RAG 規則分數與門檻、策略規格、邏輯檢驗），
   你只負責說明規則怎麼判斷、理由是什麼，不要自己給出買賣建議。
4. 研究結論若說某個效應不顯著或只是候選，要照實講，不要說得比資料更樂觀。
   模型預測、相似走勢與建議進出場點都要連同它的研究結論與回測紀錄一起講。
5. 結尾提醒：這是研究規則的輸出，不是投資建議。"""

# 每次都附上的規則定義。實測時模型把「出場日早於今天 − 30 天」讀成「最近 30 天內」，
# 所以把容易誤讀的地方用白話寫死，放在資料最前面
RULE_FACTS = [
    "規則定義：RAG 相似事件排除規則（候選，未採用）。檢索庫是從 2011 年起的全部歷史跌深事件，"
    "只排除出場日在最近 30 天內、答案還沒揭曉的事件；範圍是整段歷史，不是只看最近 30 天。"
    "在檢索庫裡找特徵最像的 20 個事件，分數＝它們淨報酬的距離加權平均；"
    "分數低於過去 365 天所有事件分數的第 20 百分位就不做。",
    "規則定義：週期性邏輯檢驗。每 42 個交易日檢查一次策略減大盤的 42 天累積超額；"
    "低於歷史第 5 百分位為「注意」，連續兩期低於第 10 百分位或近一年超額顯著變差為「顯著」，"
    "顯著時先查原因，不自動換股。",
]


def retrieve(con, question: str, asof: str, k: int, code: str | None,
             embed_model: str) -> pd.DataFrame:
    kb = pd.read_sql("SELECT id, kind, source, code, known_at, text, embedding FROM kb_chunks "
                     "WHERE known_at IS NULL OR known_at < ?", con, params=[asof])
    if kb.empty:
        return kb
    if kb["embedding"].notna().all():
        import requests

        from build_kb import query_prefix

        r = requests.post(f"{OLLAMA}/api/embed", timeout=120, json={
            "model": embed_model, "input": [query_prefix(embed_model) + question]})
        r.raise_for_status()
        q = np.asarray(r.json()["embeddings"][0], dtype=np.float32)
        q /= max(np.linalg.norm(q), 1e-12)
        V = np.stack([np.frombuffer(b, dtype=np.float32) for b in kb["embedding"]])
        kb["sim"] = V @ q
    else:
        # 還沒算向量時退回字元重疊（中文沒有空白分詞，用二字組比對）
        grams = {question[i:i + 2] for i in range(len(question) - 1)}
        kb["sim"] = kb["text"].map(lambda t: sum(g in t for g in grams) / max(len(grams), 1))
    if code:
        kb.loc[kb["code"] == code, "sim"] += 0.05      # 同一檔的過去事件稍微優先
    # 文件與事件各取一半，避免一萬多張事件卡把規則文件擠掉
    # 每份文件最多 2 段：問題同時問兩件事時，不會讓同一份文件佔滿名額（實測發生過）
    docs = (kb[kb["kind"] == "doc"].sort_values("sim", ascending=False)
            .groupby("source").head(2).head(k - k // 2))
    evs = kb[kb["kind"] == "event"].nlargest(k // 2, "sim")
    return pd.concat([docs, evs]).sort_values("sim", ascending=False)


def stock_context(con, code: str, asof: str) -> list:
    """該股截至 asof 已知的結構化資料，每一項都是一段可引用的文字。"""
    out = []
    s = con.execute("SELECT name, category FROM stocks WHERE code = ?", [code]).fetchone()
    name = f"{code} {s[0]}（{s[1]}）" if s else code
    px = pd.read_sql("SELECT date, close, volume FROM prices WHERE code = ? AND date <= ? "
                     "ORDER BY date DESC LIMIT 61", con, params=[code, asof])
    if len(px) >= 6:
        c = px["close"].to_numpy()
        r5, r20 = c[0] / c[5] - 1, (c[0] / c[20] - 1) if len(c) > 20 else np.nan
        hi60 = c[0] / c.max() - 1
        out.append(f"{name} 截至 {px['date'].iloc[0]}：收盤 {c[0]:.2f}，近 5 日 {r5:+.1%}，"
                   f"近 20 日 {r20:+.1%}，距近 60 日高點 {hi60:+.1%}。"
                   f"跌深事件的條件是 5 日內跌 8%，目前{'已' if r5 <= -0.08 else '未'}達標。")
    rd = con.execute("SELECT entry, score, threshold, excluded, net, exit FROM rag_decisions "
                     "WHERE code = ? AND entry <= ? ORDER BY entry DESC LIMIT 3",
                     [code, asof]).fetchall()
    for e in rd:
        known = e[5] < asof
        out.append(f"{name} 在 {e[0]} 的跌深事件：RAG 規則分數 {e[1]:+.2%}，"
                   f"門檻 {'無（歷史不足）' if e[2] is None else f'{e[2]:+.2%}'}，"
                   f"規則判定{'排除' if e[3] else '保留'}；"
                   + (f"結果淨報酬 {e[4]:+.2%}（{e[5]} 出場）。" if known else "結果尚未揭曉。"))
    news = con.execute(
        "SELECT n.ts, n.title FROM news n JOIN news_stock s USING (news_id) "
        "WHERE s.code = ? AND n.ts < ? ORDER BY n.ts DESC LIMIT 8", [code, asof]).fetchall()
    # 每則新聞各自一個出處編號；併成一項時模型會編出不存在的編號
    out += [f"{name} 的新聞（{t[:10]}）：{h}" for t, h in news]
    # 各策略的預測、方向與建議進出場點（forecast.py，和前端「個股預測」頁同一套）
    try:
        import forecast
        out += forecast.summary_lines(con, code, asof)
    except Exception as e:                               # 舊 DB 沒有 predictions 表等
        out.append(f"{name} 的模型預測無法取得（{type(e).__name__}）。")
    pk = con.execute("SELECT asof, rank, action FROM picks WHERE code = ?", [code]).fetchone()
    if pk:
        out.append(f"{name} 在高波動策略最新清單（{pk[0]}）排第 {pk[1]}，動作：{pk[2]}。")
    return out


def strategy_context(con, asof: str) -> list:
    sg = con.execute("SELECT check_date, excess, pctile, level FROM significance "
                     "WHERE check_date <= ? ORDER BY check_date DESC LIMIT 1", [asof]).fetchone()
    if not sg:
        return []
    return [f"高波動策略最近一次邏輯檢驗（{sg[0]}）：42 天超額 {sg[1]:+.2%}，"
            f"歷史第 {sg[2]:.0%} 百分位，結論「{sg[3]}」。"]


def build_prompt(question: str, ctx: list, hits: pd.DataFrame) -> tuple:
    """回傳（prompt, 資料項清單），清單的順序就是出處編號，用來檢查與顯示出處。"""
    items = RULE_FACTS + ctx + [f"（{h.kind}｜{h.source}"
                                + (f"｜{h.known_at} 揭曉" if isinstance(h.known_at, str) else "")
                                + f"）{h.text}" for h in hits.itertuples()]
    data = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(items, 1))
    return (f"資料（共 {len(items)} 項，出處編號只能是 1 到 {len(items)}）：\n{data}"
            f"\n\n問題：{question}"), items


def bad_citations(answer: str, n: int) -> list:
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer) if not 1 <= int(x) <= n})


def chat(model: str, messages: list) -> str:
    import requests

    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": model, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": NUM_CTX}, "messages": messages})
    if r.status_code == 404:
        raise SystemExit(f"找不到模型 {model}，請先執行：ollama pull {model}")
    r.raise_for_status()
    return r.json()["message"]["content"]


def health(db: str | Path = DB, model: str = CHAT_MODEL) -> list:
    """前端顯示用：缺什麼就回傳一行說明，全部就緒回傳空清單。"""
    import requests

    problems = []
    if not Path(db).exists():
        problems.append("找不到 db/local.db，請先執行 python build_local_db.py")
    else:
        con = sqlite3.connect(db)
        if not con.execute("SELECT name FROM sqlite_master WHERE name = 'kb_chunks'").fetchone():
            problems.append("知識庫還沒建立，請執行 python build_kb.py")
        con.close()
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=3).json()
        have = {m["name"] for m in tags.get("models", [])}
        for m in (model, "embeddinggemma:latest"):
            if m not in have:
                problems.append(f"Ollama 沒有模型 {m}，請執行 ollama pull {m.split(':latest')[0]}")
    except Exception:
        problems.append("連不到 Ollama（http://localhost:11434），請先開啟 Ollama")
    return problems


def ask(question: str, code: str | None = None, asof: str | None = None, k: int = 8,
        model: str = CHAT_MODEL, embed_model: str | None = None, db: str | Path = DB,
        dry_run: bool = False) -> dict:
    """
    命令列與前端共用的問答流程。回傳：
      answer  回答（dry_run 時是送給模型的完整內容）
      items   資料項清單，第 i 項就是出處 [i+1]
      bad     回答裡仍不存在的出處編號
      n_rule / n_ctx / n_hits  各類資料的項數
    """
    asof = asof or pd.Timestamp.today().strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    em = embed_model or (con.execute(
        "SELECT value FROM meta WHERE key = 'kb.embed_model'").fetchone() or ["embeddinggemma"])[0]
    hits = retrieve(con, question, asof, k, code, em or "embeddinggemma")
    ctx = strategy_context(con, asof) + (stock_context(con, code, asof) if code else [])
    prompt, items = build_prompt(question, ctx, hits)
    con.close()
    out = {"items": items, "bad": [], "asof": asof,
           "n_rule": len(RULE_FACTS), "n_ctx": len(ctx), "n_hits": len(hits)}
    if dry_run:
        return out | {"answer": f"[system]\n{SYSTEM}\n\n[user]\n{prompt}"}

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    answer = chat(model, msgs)
    bad = bad_citations(answer, len(items))
    if bad:
        # 編出不存在的出處：指出錯誤要它重寫一次；還是錯就照實標示，不默默刪掉
        msgs += [{"role": "assistant", "content": answer},
                 {"role": "user", "content":
                  f"你引用了不存在的出處 {', '.join(f'[{b}]' for b in bad)}，資料只有 1 到 "
                  f"{len(items)} 項。請找出這些內容實際對應的編號後重新回答；找不到出處的句子請刪除。"}]
        answer = chat(model, msgs)
        bad = bad_citations(answer, len(items))
    return out | {"answer": answer, "bad": bad}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="地端問答（Ollama＋db/local.db）")
    ap.add_argument("question")
    ap.add_argument("--code", help="股票代號；給了就附上該股的結構化資料")
    ap.add_argument("--asof", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                    help="以哪一天的視角回答（只用當天以前已知的資料）")
    ap.add_argument("--k", type=int, default=8, help="檢索段落數")
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--embed-model", default=None, help="預設用建庫時的向量模型")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--dry-run", action="store_true", help="只印出送給模型的內容")
    a = ap.parse_args(argv)

    r = ask(a.question, a.code, a.asof, a.k, a.model, a.embed_model, a.db, a.dry_run)
    print(r["answer"])
    if a.dry_run:
        return 0
    if r["bad"]:
        print(f"\n⚠ 回答仍引用不存在的出處 {', '.join(f'[{b}]' for b in r['bad'])}，這些句子請勿採信。")
    print(f"\n—— 出處：{r['n_rule']} 項規則定義＋{r['n_ctx']} 項結構化資料＋"
          f"{r['n_hits']} 段檢索，共 {len(r['items'])} 項（視角 {r['asof']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
