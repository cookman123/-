"""地端 LLM 回答的數字核對：回答裡的數字，提供給它的資料裡有沒有。

數字抽取與正負號判斷取自 jameschen108/tw-stock-quant 的 evals/analyst/numcheck.py：
中文常寫「下跌 5%」「賣超 7 萬張」而不寫負號，所以方向看兩處：數字前的負號，以及同一句裡緊接在數字前的方向詞。

原版拿來和評測題的標準答案比；這裡改成和「送給模型的資料」比，用在每一次問答：
  - 回答裡的數字，在資料裡找得到同一個值（照回答寫的小數位數四捨五入後相等）才算有根據
  - 兩邊都看得出方向而方向相反（資料 −5%、回答「上漲 5%」）也算沒根據
  - 出處編號 [3]、問題裡本來就有的數字、10 以下的整數（「3 檔」「第 2 點」）不檢查
gemma3:12b 這類小模型常把資料裡的數字改寫、算錯或自己補，引用編號對了不代表數字對。
"""
import re

NUM_RE = re.compile(r"(?<![\d.])([+\-−–]?)(\d[\d,]*\.?\d*)")
CITE_RE = re.compile(r"\[\d+(?:\s*[,，、]\s*\d+)*\]")
NEG_WORDS = ("跌", "減", "衰退", "下滑", "下降", "虧", "賣超", "淨賣", "流出", "負")
POS_WORDS = ("漲", "增", "成長", "上升", "買超", "淨買", "流入")
LEVEL_WORDS = ("跌破", "跌至", "跌到", "漲至", "漲到", "站上", "下降至", "下降到", "下滑至", "下滑到", "降至", "降到",
               "減少至", "減少到", "減至", "減到", "上升至", "上升到", "回升至", "升至", "升到", "增加至", "增加到",
               "增至", "增到", "成長至", "成長到")             # 後面接的是水準，不是變動
NEUTRAL = tuple(sorted(("漲跌", "增減", "買賣超", "正負", "升降") + LEVEL_WORDS, key=len, reverse=True))
WINDOW = 8                                                # 往前看幾個字
STOP = "，。；：,;:、\n()（）|"


def _words(text: str) -> int:
    """文字裡的方向詞 → 1 / −1 / 0（沒有或兩個方向都有）。"""
    before = text.replace("負成長", "衰退")
    for w in NEUTRAL:
        before = before.replace(w, "")
    neg = any(w in before for w in NEG_WORDS)
    pos = any(w in before for w in POS_WORDS)
    return -1 if neg and not pos else 1 if pos and not neg else 0


def _direction(before: str) -> int:
    """數字前同一句的文字 → 1 / −1 / 0（看不出來）。"""
    for ch in STOP:
        before = before.rsplit(ch, 1)[-1]
    return _words(before)


def numbers_in(text: str) -> list[tuple[float, int, str]]:
    """文字裡的數字 → [(絕對值, 方向, 原字串)]，方向 1 / −1 / 0。日期裡的「-」（2026-10-27）不當負號。"""
    out, prev_end = [], 0
    for m in NUM_RE.finditer(text):
        sign, digits = m.groups()
        try:
            v = float(digits.replace(",", ""))
        except ValueError:
            continue
        if sign and sign in "-−–" and not (m.start() > 0 and text[m.start() - 1].isdigit()):
            d = -1
        elif sign == "+":
            d = 1
        else:
            d = _direction(text[max(prev_end, m.start() - WINDOW):m.start()])
        out.append((v, d, digits))
        prev_end = m.end()
    return out


def _decimals(s: str) -> int:
    s = s.rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def unverified(answer: str, items: list[str], question: str = "") -> list[str]:
    """回答裡在資料中找不到根據的數字（原字串，去重、照出現順序）。"""
    have = numbers_in("\n".join(items))
    asked = {v for v, _, _ in numbers_in(question)}
    out = []
    for v, d, raw in numbers_in(CITE_RE.sub(" ", answer)):
        if v in asked or (v < 10 and "." not in raw):
            continue
        nd = _decimals(raw)
        ok = any(abs(round(v2, nd) - v) < 0.5 * 10 ** -nd + 1e-9 and not (d and d2 and d != d2)
                 for v2, d2, _ in have)
        # 資料寫 0.0523、回答寫 5.23%：比例換成百分比也算
        ok = ok or any(abs(round(v2 * 100, nd) - v) < 0.5 * 10 ** -nd + 1e-9 for v2, _, _ in have if v2 < 1)
        tag = ("−" if d < 0 else "") + raw
        if not ok and tag not in out:
            out.append(tag)
    return out
