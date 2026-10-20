#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 PTT Stock 板的逐年 JSONL 轉成「逐檔股票一個 CSV」，供訓練程式當第三個文本軸使用。

為什麼需要這一步
----------------
鉅亨新聞與同學會的資料本來就掛在個股底下，PTT 不是——它是整個看板的貼文，
沒有標註在講哪一檔股票。所以必須自己做「文章 → 股票」的歸屬。
這也是先前 E1／E3 被整批跳過的原因：程式找 data/ptt/{代號}.csv，那個目錄不存在。

歸屬規則（寧可漏抓，不要錯抓）
------------------------------
1. 代號比對（主要，精確度高）
   - 四位數字，前後不可接英數
   - 後面緊接「年月日點元塊億萬%倍股張人次」等單位就不算（排除「2023 年」）
   - 落在日期格式裡（前後是 / - .）不算
   - 代號本身落在 1990~2100 的 16 檔（2002 中鋼、2023 燁輝…）另外要求
     公司名也出現，否則「2008 年金融危機」會被歸給高興昌

2. 公司名比對（次要）
   - 長名優先遮蔽：先比對長的公司名並把命中處遮掉，再比對短的，
     否則「中鋼構」會同時命中中鋼(2002)、「聯華電子」會命中聯華(1229)
   - 常用詞名單（三星＝Samsung、巨大＝huge、大量、創意、新興…）不單獨採信，
     必須同時出現代號才歸屬

輸出
----
data/ptt/{代號}.csv，欄位 date,title,content,source,n_push,n_boo,n_comments,category
date 一律寫成帶 +08:00 的 ISO 字串 —— 下游 _read_text_csv 會做
to_datetime(utc=True) 再 tz_convert('Asia/Taipei')，若不帶時區，台北時間會被
當成 UTC 而整批平移 +8 小時，早上 9:20 的文會變成 17:20、被誤判為收盤後順延一天。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "NEW_pttcc(19-24)" / "pttcc_clean"
OUT_DIR = ROOT / "data" / "ptt"
MAP_FILE = ROOT / "data" / "industry_stock_map.json"

# 這些名稱本身是常用詞，單獨出現不足以認定在談該檔股票，需代號同時出現
COMMON_WORD_NAMES = {
    "三星", "巨大", "大量", "創意", "新興", "傳奇", "冠軍", "全國", "幸福",
    "至上", "正道", "中華", "大成", "大山", "可成", "欣欣", "統一", "華夏",
    "精誠", "資通", "南亞", "第一", "上銀", "特力", "力山",
}

# 公司名的常見別稱／全稱，用來提高召回並協助長名遮蔽
EXTRA_ALIASES = {
    "2330": ["台積電", "台積"],
    "2317": ["鴻海"],
    "2454": ["聯發科"],
    "2303": ["聯電", "聯華電子"],
    "2308": ["台達電"],
    "2412": ["中華電", "中華電信"],
    "2002": ["中鋼"],
    "2603": ["長榮海運"],
    "3711": ["日月光", "日月光投控"],
    "1301": ["台塑"],
    "1326": ["台化"],
    "6505": ["台塑化"],
    "2891": ["中信金", "中國信託"],
    "2882": ["國泰金"],
    "2881": ["富邦金"],
}

# 代號落在這個區間時容易被當成年份
YEAR_LIKE = range(1990, 2101)

UNIT_AFTER = re.compile(r"^\s*(年|月|日|點|元|塊|億|萬|%|％|倍|股|張|人|次|樓|坪)")
DATE_CHARS = set("/-.")
CODE_RE = re.compile(r"(?<![0-9A-Za-z])(\d{4})(?![0-9A-Za-z])")


def load_map():
    rows = json.load(open(MAP_FILE, encoding="utf-8"))
    code2name = {r["code"]: r["name"] for r in rows}
    # 名稱 → 代號；含別稱。同名衝突時保留第一個並記錄
    name2code, dup = {}, []
    for code, name in code2name.items():
        for nm in [name] + EXTRA_ALIASES.get(code, []):
            if nm in name2code and name2code[nm] != code:
                dup.append((nm, name2code[nm], code))
                continue
            name2code[nm] = code
    return code2name, name2code, dup


_JIEBA = None


MAX_NAME_TOKENS = 3      # 公司名最長不超過連續 3 個詞


def init_tokenizer(name2code: dict) -> None:
    """
    只用 jieba 的預設詞典，**不要** add_word 把公司名塞進去。

    第一版曾用 add_word(freq=1_000_000) 強迫斷出公司名，結果完全反效果：
    jieba 為了切出「信大」而把「電信大廠」硬切成 電/信大/廠，誤判反而被放大。
    預設詞典本來就認得「電信」「貿易」「意味」「朋友」，讓它自然斷詞才對。
    """
    global _JIEBA
    import jieba

    jieba.setLogLevel(60)
    jieba.initialize()
    _JIEBA = jieba


def _aligned_names(text: str, name2code: dict) -> set:
    """
    取出「與詞邊界對齊」的公司名。

    做法是把連續 1~3 個詞接起來，只有這樣組出來的字串才算命中，
    因此橫跨詞邊界的子字串一律被排除：
        電信/大廠   → 「信大」跨越邊界 → 不算
        和/大陸     → 「和大」跨越邊界 → 不算
        有關/貿易   → 「關貿」跨越邊界 → 不算
        意味/全     → 「味全」跨越邊界 → 不算
        朋友/達成   → 「友達」跨越邊界 → 不算
    而真正的提及會對齊：台積電/大漲 → 「台積電」命中。
    """
    toks = _JIEBA.lcut(text) if _JIEBA is not None else []
    hits = set()
    n = len(toks)
    for i in range(n):
        joined = ""
        for k in range(MAX_NAME_TOKENS):
            if i + k >= n:
                break
            joined += toks[i + k]
            if len(joined) > 8:
                break
            if joined in name2code:
                hits.add(joined)
    return hits


def find_codes(text: str, codes: set, code2name: dict, name2code: dict,
               names_desc: list) -> tuple:
    """回傳 (歸屬到的代號集合, 以代號命中的集合, 以名稱命中的集合)。"""
    by_code, by_name = set(), set()

    for m in CODE_RE.finditer(text):
        c = m.group(1)
        if c not in codes:
            continue
        after = text[m.end():m.end() + 8]
        if UNIT_AFTER.match(after):
            continue
        prev = text[m.start() - 1] if m.start() > 0 else ""
        nxt = text[m.end()] if m.end() < len(text) else ""
        if prev in DATE_CHARS or nxt in DATE_CHARS:      # 像日期就不算
            continue
        by_code.add(c)

    # 名稱比對一律走 jieba 斷詞，只認「被切成完整一個詞」的公司名。
    # 純子字串比對在中文會大量誤判：電「信大」廠→信大(1109)、「和大」陸→和大(1536)、
    # 有「關貿」易→關貿(6183)、意「味全」→味全(1201)、朋「友達」→友達(2409)。
    # 公司名已用 add_word 加進詞典且給高詞頻，jieba 會優先切出完整公司名，
    # 也因此「中鋼構」不會再切出「中鋼」、「聯華電子」不會切出「聯華」。
    for nm in _aligned_names(text, name2code):
        by_name.add(name2code[nm])

    # 常用詞名稱要有代號佐證才算
    confirmed_name = {c for c in by_name
                      if code2name.get(c) not in COMMON_WORD_NAMES or c in by_code}
    # 年份型代號要有名稱佐證才算
    confirmed_code = {c for c in by_code
                      if int(c) not in YEAR_LIKE or c in by_name}
    return confirmed_code | confirmed_name, confirmed_code, confirmed_name


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    code2name, name2code, dup = load_map()
    codes = set(code2name)
    names_desc = sorted(name2code, key=len, reverse=True)

    print(f"股票 {len(codes)} 檔、比對名稱 {len(name2code)} 個"
          f"（含別稱；常用詞名單 {len(COMMON_WORD_NAMES)} 個需代號佐證）")
    if dup:
        print(f"  名稱衝突 {len(dup)} 組（保留先出現者）：{dup[:5]}")

    init_tokenizer(name2code)
    print("  jieba 預設詞典已載入；名稱只認與詞邊界對齊者（避免中文子字串誤判）")

    files = sorted(SRC_DIR.glob("*_clean.jsonl"))
    if not files:
        print(f"找不到來源檔：{SRC_DIR}")
        return 1

    writers, handles = {}, {}
    header = ["date", "title", "content", "source",
              "n_push", "n_boo", "n_comments", "category"]

    stats = Counter()
    per_code = Counter()
    only_code = only_name = both = 0
    samples = []

    for f in files:
        year_n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    stats["壞行"] += 1
                    continue
                stats["讀入"] += 1
                year_n += 1
                title = (r.get("title") or "").strip()
                body = (r.get("content_clean") or r.get("content") or "").strip()
                ts = r.get("date") or r.get("date_ts")
                if not ts or (not title and not body):
                    stats["無時戳或空白"] += 1
                    continue

                matched, bc, bn = find_codes(title + " \n " + body, codes,
                                             code2name, name2code, names_desc)
                if not matched:
                    stats["未歸屬"] += 1
                    continue
                stats["已歸屬"] += 1
                if bc and bn:
                    both += 1
                elif bc:
                    only_code += 1
                else:
                    only_name += 1

                if len(samples) < 25 and len(matched) <= 3:
                    samples.append((title[:46], sorted(matched),
                                    [code2name[c] for c in sorted(matched)]))

                row = [
                    f"{ts}+08:00" if "+" not in str(ts) else ts,
                    title, body, "ptt",
                    r.get("n_push", 0), r.get("n_boo", 0),
                    r.get("n_comments", 0), r.get("category") or "",
                ]
                for c in matched:
                    if c not in writers:
                        h = open(OUT_DIR / f"{c}.csv", "w", encoding="utf-8-sig",
                                 newline="")
                        w = csv.writer(h)
                        w.writerow(header)
                        handles[c], writers[c] = h, w
                    writers[c].writerow(row)
                    per_code[c] += 1
                    stats["文章-股票對"] += 1
        print(f"  {f.name}：{year_n} 篇")

    for h in handles.values():
        h.close()

    print(f"\n{'=' * 70}\n歸屬結果\n{'=' * 70}")
    for k in ("讀入", "無時戳或空白", "未歸屬", "已歸屬", "文章-股票對", "壞行"):
        if stats[k]:
            print(f"  {k:<12} {stats[k]:>9,}")
    if stats["讀入"]:
        print(f"  歸屬率 {stats['已歸屬'] / stats['讀入']:.1%}"
              f"；平均每篇歸屬 {stats['文章-股票對'] / max(stats['已歸屬'], 1):.2f} 檔")
    print(f"  僅代號命中 {only_code:,}　僅名稱命中 {only_name:,}　兩者皆中 {both:,}")
    print(f"  有資料的股票 {len(per_code)} / {len(codes)} 檔")

    print(f"\n文章數最多的前 20 檔：")
    for c, n in per_code.most_common(20):
        print(f"  {c} {code2name[c]:<8} {n:>7,}")
    print(f"\n文章數最少的 10 檔（檢查是否為誤判殘留）：")
    for c, n in per_code.most_common()[-10:]:
        print(f"  {c} {code2name[c]:<8} {n:>7,}")

    print(f"\n抽樣檢查（標題 → 歸屬）：")
    for t, cs, ns in samples:
        print(f"  {t:<48} → {' '.join(f'{a}({b})' for a, b in zip(cs, ns))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
