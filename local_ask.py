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
5. 結尾提醒：這是研究規則的輸出，不是投資建議。"""


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
    docs = kb[kb["kind"] == "doc"].nlargest(k - k // 2, "sim")
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
    if news:
        out.append(f"{name} 最近的新聞標題：" + "；".join(f"{t[:10]} {h}" for t, h in news))
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


def build_prompt(question: str, ctx: list, hits: pd.DataFrame) -> str:
    items = ctx + [f"（{h.kind}｜{h.source}"
                   + (f"｜{h.known_at} 揭曉" if isinstance(h.known_at, str) else "")
                   + f"）{h.text}" for h in hits.itertuples()]
    data = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(items, 1))
    return f"資料：\n{data}\n\n問題：{question}"


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

    con = sqlite3.connect(a.db)
    em = a.embed_model or (con.execute(
        "SELECT value FROM meta WHERE key = 'kb.embed_model'").fetchone() or ["embeddinggemma"])[0]
    hits = retrieve(con, a.question, a.asof, a.k, a.code, em or "embeddinggemma")
    ctx = strategy_context(con, a.asof) + (stock_context(con, a.code, a.asof) if a.code else [])
    prompt = build_prompt(a.question, ctx, hits)
    con.close()

    if a.dry_run:
        print(f"[system]\n{SYSTEM}\n\n[user]\n{prompt}")
        return 0

    import requests

    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": a.model, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}]})
    if r.status_code == 404:
        print(f"找不到模型 {a.model}，請先執行：ollama pull {a.model}")
        return 1
    r.raise_for_status()
    print(r.json()["message"]["content"])
    print(f"\n—— 出處：{len(ctx)} 項結構化資料＋{len(hits)} 段檢索（視角 {a.asof}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
