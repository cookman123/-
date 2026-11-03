#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地端模型（Ollama）的 RAG 知識庫：把研究文件與跌深事件切成段落，算向量，存進 db/local.db。

兩種段落
--------
  doc    README 與 results/*.md 依標題切段（規則、結論、方法論）
         known_at 留空：任何時點都能引用
  event  每個跌深事件寫成一段文字卡（跌幅、技術面、籌碼、新聞、結果）
         known_at＝出場日：答案在出場那天才揭曉，查詢時只撈 known_at 早於查詢日的，
         和 rag_rule.py 的檢索庫同一個原則，避免地端模型「看到未來」

新聞不做向量：幾十萬則在筆電上算向量要好幾個小時，而且研究結論是新聞沒有可交易的訊號。
local_ask.py 會直接用 SQL 撈該股最近的新聞標題當上下文，不需要向量。

輸出
----
  db/local.db 的 kb_chunks 表（id、kind、source、code、known_at、text、embedding）
  kb/chunks.jsonl（同內容、不含向量；換別的工具時可以直接匯入）

向量模型預設 embeddinggemma（Google，多語）；不用中國機構的模型，
整套地端模型都沒有中國審查對齊的疑慮。--no-embed 只切段不算向量，
local_ask.py 會退回關鍵字比對。
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
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
KB_DIR = ROOT / "kb"
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
MAX_CHARS = 900          # embeddinggemma 上限 2K token，中文約 1 字 1 token，留足空間


def doc_prefix(model: str) -> str:
    """embeddinggemma 要求段落與查詢加上不同的任務前綴，檢索品質才會對；其他模型不加。"""
    return "title: none | text: " if model.startswith("embeddinggemma") else ""


def query_prefix(model: str) -> str:
    return "task: search result | query: " if model.startswith("embeddinggemma") else ""


def split_markdown(path: Path) -> list:
    """依 ## / ### 標題切段；過長的段落再依空行切，每段帶上所屬標題當上下文。"""
    text = path.read_text(encoding="utf-8")
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
    out, head, buf = [], "", []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        paras, cur = re.split(r"\n\s*\n", body), ""
        # 段首的引用區塊（> 開頭）是整節的但書，例如 14.5「本節的數字是稽核前的」。
        # 節被切成好幾段時，原本只有第一段帶到它，模型就會把稽核前的數字當結論引用（2026-11 實測），
        # 所以切出來的每一段都補上
        note = paras[0] if paras and paras[0].lstrip().startswith(">") else ""
        for p in paras:
            if cur and len(cur) + len(p) > MAX_CHARS:
                out.append((head, cur.strip()))
                cur = note + "\n\n" if note else ""
            cur += p + "\n\n"
        if cur.strip():
            out.append((head, cur.strip()))

    for line in text.splitlines():
        if re.match(r"^#{2,3} ", line):
            flush()
            head, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    flush()
    rel = path.relative_to(ROOT).as_posix()
    return [{"id": f"doc:{rel}#{i}", "kind": "doc", "source": rel, "code": None,
             "known_at": None, "text": f"【{title}｜{h or '前言'}】\n{b}"}
            for i, (h, b) in enumerate(out)]


def _pct(x) -> str:
    return "缺" if pd.isna(x) else f"{x:+.1%}"


def _num(x, fmt=".2f") -> str:
    return "缺" if pd.isna(x) else format(x, fmt)


def event_cards(con: sqlite3.Connection) -> list:
    """事件文字卡；名稱與產業從 stocks 表補上，RAG 規則分數從 rag_decisions 補上。"""
    ev = pd.read_sql("SELECT e.*, s.name, s.category FROM events e "
                     "LEFT JOIN stocks s ON s.code = e.code", con)
    try:
        rd = pd.read_sql("SELECT code, entry, score, threshold, excluded FROM rag_decisions", con)
        ev = ev.merge(rd, on=["code", "entry"], how="left")
    except Exception:
        ev["score"] = ev["threshold"] = ev["excluded"] = np.nan
    cards = []
    for r in ev.itertuples(index=False):
        g = r._asdict()
        rag = ("" if pd.isna(g.get("score")) else
               f"RAG 規則分數 {g['score']:+.2%}（門檻 {_pct(g.get('threshold'))}，"
               f"{'排除' if g.get('excluded') == 1 else '保留'}）。")
        text = (
            f"跌深事件：{g['code']} {g.get('name') or ''}（{g.get('category') or '產業不明'}）。"
            f"訊號日 {g['signal']}，5 日跌幅 {_pct(g.get('drop'))}，等了 {g.get('wait')} 天止跌後"
            f"於 {g['entry']} 進場。"
            f"訊號日技術面：K 值 {_num(g.get('px_k'))}、RSI14 {_num(g.get('px_rsi14'))}、"
            f"布林 z {_num(g.get('px_bb_z'))}、距 60 日高點 {_pct(g.get('dist_hi60'))}、"
            f"成交值 z {_num(g.get('turnover_z'))}。"
            f"籌碼：外資 {_num(g.get('chip_fi'), '+.3f')}、投信 {_num(g.get('chip_it'), '+.3f')}"
            f"（5 日買賣超／成交量）。新聞：{_num(g.get('news_n'), '.0f')} 則，"
            f"負面標題 {_num(g.get('news_neg'), '.0f')} 則。{rag}"
            f"結果：{g['exit']} 出場（{g.get('reason') or '—'}），持有 {g.get('hold')} 天，"
            f"淨報酬 {_pct(g.get('net'))}。"
        )
        cards.append({"id": f"event:{g['event_id']}", "kind": "event", "source": "events",
                      "code": g["code"], "known_at": g["exit"], "text": text})
    return cards


def embed(texts: list, model: str, batch: int = 32) -> np.ndarray:
    import requests

    vecs = []
    for i in range(0, len(texts), batch):
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": model, "input": texts[i:i + batch]}, timeout=600)
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
        if (i // batch) % 20 == 0:
            print(f"    向量 {min(i + batch, len(texts)):,}/{len(texts):,}", flush=True)
    v = np.asarray(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="建立地端 RAG 知識庫（kb_chunks）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--no-embed", action="store_true", help="只切段、不算向量")
    a = ap.parse_args(argv)

    con = sqlite3.connect(a.db)
    docs = [ROOT / "README.md"] + sorted((ROOT / "results").glob("*.md"))
    chunks = [c for p in docs for c in split_markdown(p)]
    n_doc = len(chunks)
    chunks += event_cards(con)
    print(f"  文件段落 {n_doc:,}　事件卡 {len(chunks) - n_doc:,}")

    KB_DIR.mkdir(exist_ok=True)
    with open(KB_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    df = pd.DataFrame(chunks)
    df["embedding"] = None
    if not a.no_embed:
        # 同一個向量模型、同一段文字就沿用舊向量，只重算新增或改過的段落
        old = {}
        prev = con.execute("SELECT value FROM meta WHERE key = 'kb.embed_model'").fetchone() \
            if con.execute("SELECT name FROM sqlite_master WHERE name = 'meta'").fetchone() else None
        if prev and prev[0] == a.embed_model and con.execute(
                "SELECT name FROM sqlite_master WHERE name = 'kb_chunks'").fetchone():
            old = {(i, t): e for i, t, e in con.execute(
                "SELECT id, text, embedding FROM kb_chunks WHERE embedding IS NOT NULL")}
        df["embedding"] = [old.get((i, t)) for i, t in zip(df["id"], df["text"])]
        todo = df["embedding"].isna()
        print(f"  沿用舊向量 {int((~todo).sum()):,} 段，重算 {int(todo.sum()):,} 段")
        if todo.any():
            t0 = time.time()
            V = embed([doc_prefix(a.embed_model) + t for t in df.loc[todo, "text"]], a.embed_model)
            df.loc[todo, "embedding"] = pd.Series([v.tobytes() for v in V], index=df.index[todo])
            print(f"  向量 {V.shape[1]} 維，{time.time() - t0:.0f} 秒")

    con.execute("DROP TABLE IF EXISTS kb_chunks")
    con.execute("""CREATE TABLE kb_chunks (id TEXT PRIMARY KEY, kind TEXT, source TEXT,
                   code TEXT, known_at TEXT, text TEXT, embedding BLOB)""")
    con.executemany("INSERT INTO kb_chunks VALUES (?,?,?,?,?,?,?)",
                    df[["id", "kind", "source", "code", "known_at", "text", "embedding"]]
                    .itertuples(index=False, name=None))
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [
        ("kb.built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("kb.embed_model", "" if a.no_embed else a.embed_model),
        ("kb.rows", str(len(df)))])
    con.commit()
    con.close()
    print(f"  完成：kb_chunks {len(df):,} 段、{KB_DIR / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
