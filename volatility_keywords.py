#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大波動日 × 新聞關鍵字。volatility_news.py 看的是新聞「量」和整體負面比例，
這支往下一層，看「哪些字」在大波動日前特別多、能不能拿來事前預示。

時間窗沿用 volatility_news.py（前後相接、不重疊）：
    當日盤前 = t-1 13:30 ~ t 09:00　／　前一天 = t-2 13:30 ~ t-1 13:30
t 日 09:00 以後的新聞完全不用。盤勢報導、制式訊息先剔除。

兩種做法
--------
A. 資料驅動掃描（會有挑例子的風險，所以一定要樣本外驗證）
   1. 每則標題用 jieba 斷詞，算每天每個詞「出現在幾成標題」（用比例，不用則數，
      否則新聞多的日子每個詞都會變多）。
   2. 探索期（~2021）逐詞比較大波動日 vs 一般日，月份群集 t 值 + BH 多重檢定校正。
   3. 探索期挑出來的詞，拿到驗證期（2022~2024）看：方向還對嗎？還顯著嗎？
      再把它們合成一個分數，用探索期擬合的模型在驗證期算 AUC（真正的樣本外）。
B. 事先定好的詞表（不從資料挑，沒有挑例子的問題）
   恐慌類（崩盤、恐慌、黑天鵝、熔斷、開戰、制裁、封城…）跟利多類，看前一天出現
   這些字，在控制價格波動群聚之後還有沒有預測力（逐年留一樣本外 AUC），以及方向。

輸出：results/volatility_news/keywords/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volatility_news as V  # noqa: E402

OUT = V.OUT / "keywords"
SOURCES = ["intl", "cnyes", "ptt"]           # Yahoo 只有 19 個大波動日，掃詞沒有意義
DISC_END, VAL_END = "2021-12-31", "2024-12-31"
MIN_DAYS = 30                                 # 探索期至少要在幾天出現過才掃
BASE = ["absz_lag1", "absz_lag5max", "absz_ma20", "rv20_ratio", "sigma"]

PANIC = ["崩盤", "暴跌", "重挫", "大跌", "狂跌", "崩跌", "跳水", "血洗", "股災", "恐慌", "VIX",
         "黑天鵝", "黑色星期", "熔斷", "拋售", "逃命", "斷頭", "追繳", "利空", "危機", "衰退",
         "倒閉", "破產", "違約", "戰爭", "開戰", "空襲", "飛彈", "制裁", "封城", "停擺", "崩"]
BULL = ["大漲", "暴漲", "狂飆", "噴出", "飆漲", "創高", "新高", "軋空", "紓困", "救市", "降息",
        "利多", "反彈", "強彈", "大反攻", "寬鬆"]
LEX = {"恐慌詞": PANIC, "利多詞": BULL}


# ------------------------------------------------------------ 資料

def load_docs(m: pd.DataFrame) -> dict:
    cal = V._read_px("TWII").index
    stock = V.load_stock_news()
    raw = {"intl": V.load_intl(), "cnyes": stock[stock["src"] == "cnyes"], "ptt": V.load_ptt_board()}
    out = {}
    for s in SOURCES:
        r0 = raw[s]
        if s != "ptt":
            r0 = r0[~r0["title"].str.contains(V.BOILER_PAT, na=False)]
        d = V.assign_day(r0, cal)
        d = d[(d["day"] >= m.index.min()) & (d["day"] <= m.index.max())]
        cov = V.coverage_months(d)
        d = d[d["day"].dt.to_period("M").isin(cov.index) & ~d["recap"]].copy()
        out[s] = (d, cov.index)
    return out


def tokenize(d: pd.DataFrame, s: str) -> list:
    cache = OUT / f"_tokens_{s}.pkl"
    if cache.exists():
        tok = pd.read_pickle(cache)
        if len(tok) == len(d) and (tok.index == d.index).all():
            return tok.tolist()
    print(f"  斷詞 {s}（{len(d)} 則）…", flush=True)
    tok = pd.Series([V.tokens(t) for t in d["title"]], index=d.index)
    tok.to_pickle(cache)
    return tok.tolist()


def day_share_matrix(d: pd.DataFrame, toks: list, days: pd.DatetimeIndex, vocab: dict):
    """回傳 (days × vocab) 的「該詞出現在當天幾成標題」稠密矩陣，以及每天標題數。"""
    rows, cols = [], []
    for i, ts in enumerate(toks):
        for w in ts:
            j = vocab.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
    D = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(toks), len(vocab)))
    pos = days.get_indexer(d["day"])
    ok = pos >= 0
    G = sparse.csr_matrix((np.ones(ok.sum()), (pos[ok], np.flatnonzero(ok))), shape=(len(days), len(toks)))
    cnt = np.asarray(G.sum(axis=1)).ravel()
    S = (G @ D).toarray()
    with np.errstate(invalid="ignore", divide="ignore"):
        S = S / cnt[:, None]
    S[cnt < V.MIN_DOCS] = np.nan
    return S, cnt


def screen(S: np.ndarray, big: np.ndarray, normal: np.ndarray, months: np.ndarray):
    """逐詞：大波動 vs 一般日的比例差，月份群集 t（向量化版的 V.compare）。"""
    sel = (big | normal) & ~np.isnan(S).all(axis=1)
    Y = np.nan_to_num(S[sel])                       # 少於 MIN_DOCS 的天已是整列 NaN → 已排除
    d = big[sel].astype(float)
    g = months[sel]
    n1, n0 = d.sum(), (1 - d).sum()
    m1 = (Y * d[:, None]).sum(0) / n1
    m0 = (Y * (1 - d)[:, None]).sum(0) / n0
    U = Y - np.where(d[:, None] == 1, m1, m0)
    # y = a + b·d 的 (X'X)^-1 第二列
    N = len(d)
    XtX_inv = np.linalg.inv(np.array([[N, n1], [n1, n1]]))
    A0, A1 = XtX_inv[1]
    codes, gi = np.unique(g, return_inverse=True)
    Su = np.zeros((len(codes), Y.shape[1]))
    Sdu = np.zeros_like(Su)
    np.add.at(Su, gi, U)
    np.add.at(Sdu, gi, U * d[:, None])
    G = len(codes)
    var = ((A0 * Su + A1 * Sdu) ** 2).sum(0) * (G / (G - 1)) * ((N - 1) / (N - 2))
    se = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (m1 - m0) / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    return m1, m0, t, p, int(n1), int(n0)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pv)
    out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


# ------------------------------------------------------------ 主流程

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = V.market()
    m["sox_z_abs"] = m["sox_z"].abs()
    big_all, normal_all = m["big"].to_numpy(), m["normal"].to_numpy()
    months = m.index.to_period("M").astype(str).to_numpy()
    disc = (m.index <= DISC_END)
    val = (m.index > DISC_END) & (m.index <= VAL_END)
    val2 = m.index > VAL_END                  # 第二段驗證：2025 之後（含關稅崩盤），PTT 沒有資料
    print(f"大波動 {int(big_all.sum())}　探索期 ~{DISC_END}　驗證期 ~{VAL_END}　驗證期2 {VAL_END}~")

    docs = load_docs(m)
    screen_rows, val_rows, score_rows, lex_rows, lex_word_rows = [], [], [], [], []
    panel = m.copy()

    for s in SOURCES:
        d, cov = docs[s]
        covered = m.index.to_period("M").isin(cov)
        toks = tokenize(d, s)
        # 字彙：探索期內至少在 MIN_DAYS 天出現
        dd = pd.DataFrame({"day": d["day"].to_numpy(), "tok": toks})
        dd = dd[dd["day"] <= DISC_END].explode("tok").dropna().drop_duplicates()
        df_days = dd.groupby("tok")["day"].nunique()
        words = df_days[df_days >= MIN_DAYS].index.tolist()
        vocab = {w: i for i, w in enumerate(words)}
        print(f"  {s}: {len(d)} 則，掃 {len(vocab)} 個詞", flush=True)

        pre_mask = d["pre_open"].to_numpy()
        S_pre, _ = day_share_matrix(d[pre_mask], [t for t, k in zip(toks, pre_mask) if k], m.index, vocab)
        S_day, _ = day_share_matrix(d, toks, m.index, vocab)
        S_prev = np.vstack([np.full((1, len(vocab)), np.nan), S_day[:-1]])   # 前一天 = 上一個交易日的歸屬窗
        S_pre[~covered] = np.nan
        S_prev[~covered] = np.nan

        for wname, S in (("當日盤前", S_pre), ("前一天", S_prev)):
            res = {}
            for per, msk in (("探索", disc), ("驗證", val), ("驗證2", val2)):
                if (big_all & msk & covered).sum() < 5:
                    nanv = np.full(len(words), np.nan)
                    res[per] = (nanv, nanv, nanv, nanv, 0, 0)
                    continue
                res[per] = screen(S, big_all & msk, normal_all & msk, months)
            m1, m0, t, p, n1, n0 = res["探索"]
            q = bh(p)
            v1, v0, vt, vp, vn1, vn0 = res["驗證"]
            w1, w0, wt, wp, wn1, _ = res["驗證2"]
            tab = pd.DataFrame({"source": s, "window": wname, "word": words,
                                "disc_big": m1, "disc_normal": m0, "disc_lift": m1 / np.where(m0 > 0, m0, np.nan),
                                "disc_t": t, "disc_p": p, "disc_q": q,
                                "val_big": v1, "val_normal": v0, "val_t": vt, "val_p": vp,
                                "val2_big": w1, "val2_normal": w0, "val2_t": wt, "val2_p": wp,
                                "disc_n_big": n1, "val_n_big": vn1, "val2_n_big": wn1})
            screen_rows.append(tab)

            # 探索期挑詞：BH q<0.1 且比例較高；太少就退而求其次取 t 最大的 20 個
            pick = tab[(tab["disc_q"] < 0.10) & (tab["disc_t"] > 0)]
            rule = "q<0.10"
            if len(pick) < 5:
                pick = tab[tab["disc_t"] > 0].nlargest(20, "disc_t")
                rule = "t 前 20（無詞過 q<0.10）"
            ok = pick["val_t"].notna()
            same = int((pick.loc[ok, "val_t"] > 0).sum())
            sig = int(((pick["val_t"] > 0) & (pick["val_p"] < 0.05)).sum())
            binom_p = stats.binomtest(same, int(ok.sum()), 0.5, alternative="greater").pvalue if ok.sum() else np.nan
            val_rows.append({"source": s, "window": wname, "rule": rule, "n_words": len(pick),
                             "val_same_sign": same, "val_same_sign_sig": sig, "binom_p_same_sign": binom_p,
                             "val2_same_sign": int((pick["val2_t"] > 0).sum()),
                             "val2_n": int(pick["val2_t"].notna().sum()),
                             "val2_same_sign_sig": int(((pick["val2_t"] > 0) & (pick["val2_p"] < 0.05)).sum()),
                             "n_up_disc_q10": int(((tab["disc_q"] < 0.10) & (tab["disc_t"] > 0)).sum()),
                             "words": " ".join(pick.sort_values("disc_t", ascending=False)["word"].head(20))})

            # 合成分數：挑到的詞的比例（各自用探索期標準差標準化）加總
            idx = [vocab[w] for w in pick["word"]]
            X = S[:, idx]
            sd = np.nanstd(X[disc & covered], axis=0)
            sd[sd == 0] = 1
            score = np.nansum(X / sd, axis=1)
            score[np.isnan(X).all(axis=1)] = np.nan
            col = f"{s}_kwscore_{'pre' if wname == '當日盤前' else 'prev'}"
            panel[col] = score
            # 樣本外：探索期擬合（基準 vs 基準+分數），驗證期算 AUC
            base = BASE + (["sox_z_abs"] if wname == "當日盤前" else [])
            dsub = panel[base + [col, "big"]].dropna()
            tr = dsub.index <= DISC_END
            mu, sdv = dsub[tr][base + [col]].mean(), dsub[tr][base + [col]].std()
            Z = ((dsub[base + [col]] - mu) / sdv).to_numpy()
            y = dsub["big"].astype(int).to_numpy()
            m0_, _ = V.logit_fit(Z[tr][:, :len(base)], y[tr])
            m1_, _ = V.logit_fit(Z[tr], y[tr])
            for tname, te in ((f"{int(DISC_END[:4]) + 1}~{VAL_END[:4]}",
                               (dsub.index > DISC_END) & (dsub.index <= VAL_END)),
                              (f"{int(VAL_END[:4]) + 1}~", dsub.index > VAL_END)):
                if y[te].sum() < 5:
                    continue
                score_rows.append({"source": s, "window": wname, "n_words": len(pick),
                                   "train": f"~{DISC_END[:4]}", "test": tname,
                                   "test_days": int(te.sum()), "test_big": int(y[te].sum()),
                                   "AUC_test_base": V.auc(y[te], m0_.predict_proba(Z[te][:, :len(base)])[:, 1]),
                                   "AUC_test_base+kw": V.auc(y[te], m1_.predict_proba(Z[te])[:, 1]),
                                   "AUC_test_kw_alone": V.auc(y[te], dsub.loc[te, col].to_numpy()),
                                   "coef_kw_train": m1_.coef_[0][-1]})

        # ---------------- B. 事先定好的詞表（子字串比對，不靠斷詞）
        for lname, words_l in LEX.items():
            pat = "|".join(words_l)
            d[f"lex_{lname}"] = d["title"].str.contains(pat, na=False)
            for w in words_l:
                d[f"w_{w}"] = d["title"].str.contains(w, regex=False, na=False)
        cols_lex = [c for c in d.columns if c.startswith("lex_") or c.startswith("w_")]
        days_cov = m.index[covered]

        def win_share(sub):
            g = sub.groupby("day")
            n = g.size().reindex(days_cov)
            sh = g[cols_lex].sum().reindex(days_cov).div(n, axis=0)
            return sh.where(n >= V.MIN_DOCS)

        sh_pre = win_share(d[d["pre_open"]]).reindex(m.index)
        sh_prev = win_share(d).reindex(m.index).shift(1)
        for lname in LEX:
            panel[f"{s}_{lname}_pre"] = sh_pre[f"lex_{lname}"]
            panel[f"{s}_{lname}_prev"] = sh_prev[f"lex_{lname}"]
            up, down = panel["big"] & (panel["r"] > 0), panel["big"] & (panel["r"] < 0)
            for wname, col in (("當日盤前", f"{s}_{lname}_pre"), ("前一天", f"{s}_{lname}_prev")):
                r = V.compare(panel, col, panel["big"], panel["normal"])
                rd = V.compare(panel, col, down, up)
                rec = {"source": s, "lexicon": lname, "window": wname,
                       "big": r["mean_a"], "normal": r["mean_b"], "t_big_vs_normal": r["t_cluster"],
                       "p_big_vs_normal": r["p_cluster"],
                       "bigdown": rd["mean_a"] if rd else np.nan, "bigup": rd["mean_b"] if rd else np.nan,
                       "t_down_vs_up": rd["t_cluster"] if rd else np.nan,
                       "p_down_vs_up": rd["p_cluster"] if rd else np.nan}
                if wname == "前一天":
                    pr = V.predict_test(panel, "big", BASE, [col], "")
                    rec.update({"LR_p": pr["p_LR"], "AUC_oos_base": pr["AUC_oos_base"],
                                "AUC_oos_+lex": pr["AUC_oos_news"]})
                    # 跟 volatility_news 的整體負面比例比：詞表有沒有比較好
                else:
                    pr = V.predict_test(panel, "big", BASE + ["sox_z_abs"], [col], "")
                    rec.update({"LR_p": pr["p_LR"], "AUC_oos_base": pr["AUC_oos_base"],
                                "AUC_oos_+lex": pr["AUC_oos_news"]})
                lex_rows.append(rec)
            # 個別詞：前一天出現（≥1 則）時，隔天大波動的機率
            for w in LEX[lname]:
                for wname, sh in (("當日盤前", sh_pre), ("前一天", sh_prev)):
                    x = sh[f"w_{w}"]
                    okm = x.notna()
                    pres = (x > 0) & okm
                    absn = (x == 0) & okm
                    if pres.sum() < 5:
                        continue
                    a = int((pres & m["big"]).sum())
                    b = int((absn & m["big"]).sum())
                    _, pf = stats.fisher_exact([[a, int(pres.sum()) - a], [b, int(absn.sum()) - b]])
                    rec = {"source": s, "lexicon": lname, "word": w, "window": wname,
                           "days_present": int(pres.sum()), "P(big|有)": a / pres.sum(),
                           "P(big|無)": b / max(absn.sum(), 1), "big_days_with_word": a,
                           # 命中的大波動日分散在幾個月：全擠在同一波崩盤的話，其實只算一次
                           "months_of_hits": int(m.index[pres & m["big"]].to_period("M").nunique()),
                           "p_fisher": pf}
                    # 「平靜時」：前一天自己不是大波動（|z|<2），排掉「已經在崩」的情況
                    calm = (m["absz_lag1"] < 2) & okm
                    pc, ac = pres & calm, absn & calm
                    rec["calm_days_present"] = int(pc.sum())
                    rec["P(big|有,平靜)"] = (pc & m["big"]).sum() / pc.sum() if pc.sum() else np.nan
                    rec["P(big|無,平靜)"] = (ac & m["big"]).sum() / ac.sum() if ac.sum() else np.nan
                    # 2025 之後（主分析做完才補抓的資料，等於全新的樣本外）
                    late = m.index > VAL_END
                    pl, al = pres & late, absn & late
                    rec["days_present_2025~"] = int(pl.sum())
                    rec["P(big|有,2025~)"] = (pl & m["big"]).sum() / pl.sum() if pl.sum() else np.nan
                    rec["P(big|無,2025~)"] = (al & m["big"]).sum() / al.sum() if al.sum() else np.nan
                    # 控制價格波動群聚後，「有出現」這個 0/1 還有沒有額外資訊
                    if wname == "前一天" and pres.sum() >= 20:
                        tmp = panel.assign(_w=np.where(okm, pres.astype(float), np.nan))
                        pr = V.predict_test(tmp, "big", BASE, ["_w"], "")
                        if pr:
                            rec.update({"LR_p_vs_price": pr["p_LR"], "AUC_oos_base": pr["AUC_oos_base"],
                                        "AUC_oos_+word": pr["AUC_oos_news"]})
                    lex_word_rows.append(rec)

    pd.concat(screen_rows).sort_values(["source", "window", "disc_t"], ascending=[True, True, False]) \
        .to_csv(OUT / "keyword_screen.csv", index=False, encoding="utf-8-sig", float_format="%.5g")
    vt = pd.DataFrame(val_rows)
    vt.to_csv(OUT / "keyword_validation.csv", index=False, encoding="utf-8-sig", float_format="%.4g")
    st = pd.DataFrame(score_rows)
    st.to_csv(OUT / "keyword_score_oos.csv", index=False, encoding="utf-8-sig", float_format="%.4g")
    lx = pd.DataFrame(lex_rows)
    lx.to_csv(OUT / "lexicon_tests.csv", index=False, encoding="utf-8-sig", float_format="%.4g")
    lw = pd.DataFrame(lex_word_rows)
    lw.to_csv(OUT / "lexicon_words.csv", index=False, encoding="utf-8-sig", float_format="%.4g")

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 120)
    sc = pd.concat(screen_rows)
    for (s, w), g in sc.groupby(["source", "window"]):
        top = g.nlargest(15, "disc_t")[["word", "disc_big", "disc_normal", "disc_t", "disc_q", "val_big",
                                        "val_normal", "val_t", "val_p"]]
        print(f"\n== {s}｜{w}：探索期最強的 15 個詞 ==\n", top.round(4).to_string(index=False))
    print("\n挑詞的驗證：\n", vt.round(4).to_string(index=False))
    print("\n合成分數樣本外：\n", st.round(4).to_string(index=False))
    print("\n事先詞表：\n", lx.round(4).to_string(index=False))
    print("\n個別詞（前一天，出現 ≥ 20 天）：\n",
          lw[(lw["window"] == "前一天") & (lw["days_present"] >= 20)]
          .sort_values("P(big|有)", ascending=False).round(4).head(30).to_string(index=False))
    print(f"輸出 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
