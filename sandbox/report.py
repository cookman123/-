# -*- coding: utf-8 -*-
"""
績效指標與報表（計劃 M4）。

報酬之外一定要放 beta 與風險調整後的比較：本 repo 唯一存活的高波動溢酬 beta 大於 1，
只看報酬會把承擔市場風險換來的補償誤讀成 alpha（見 results/研究結論總整理.md）。
無風險利率設為 0。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Result

TRADING_DAYS = 252

LIMITATIONS = """\
- 日資料撮合：漲跌停鎖死以「開盤（或收盤）= 最高（最低）= 漲跌停價」近似，盤中是否打開無法得知。
- 未計流動性衝擊：成交量再小都能以參考價加滑價成交。
- 股利在除權息日入帳，實際發放日通常晚幾週。
- 退還股款的現金減資拆不出現金部分，以價值保持處理；現金增資認購權不處理。
- 下市股以最後收盤價轉成現金。
- 沒有處置股與上櫃股資料。
- 無風險利率設為 0，Sharpe 與 alpha 都未扣無風險利率。"""


def metrics(res: Result, bench: pd.Series | None = None) -> dict:
    """
    res：engine.run 的結果。bench：基準的每日淨值（例如買進持有 0050），索引須涵蓋 res.nav。
    """
    nav = res.nav["nav"]
    init = res.config.initial_cash
    r = pd.concat([pd.Series([nav.iloc[0] / init - 1], index=nav.index[:1]),
                   nav.pct_change().iloc[1:]])
    years = len(nav) / TRADING_DAYS
    led = res.ledger
    trades = led[led["type"].isin(["buy", "sell"])]
    slip = (trades["price"] - trades["ref"]).abs() * trades["shares"]
    fees, taxes, slippage = trades["fee"].sum(), trades["tax"].sum(), slip.sum()
    avg_nav = nav.mean()
    sd = r.std()
    out = {
        "起日": nav.index[0].date().isoformat(),
        "迄日": nav.index[-1].date().isoformat(),
        "交易日數": len(nav),
        "初始資金": init,
        "期末淨值": float(nav.iloc[-1]),
        "總報酬": float(nav.iloc[-1] / init - 1),
        "年化報酬": _cagr(nav.iloc[-1] / init, years),
        "年化波動": float(sd * np.sqrt(TRADING_DAYS)),
        "夏普": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "最大回撤": float((nav / nav.cummax().clip(lower=init) - 1).min()),
        "年化換手率": float(trades["gross"].sum() / 2 / avg_nav / years) if years > 0 else np.nan,
        "成交筆數": int(len(trades)),
        "手續費": float(fees),
        "證交稅": float(taxes),
        "滑價": float(slippage),
        "成本合計": float(fees + taxes + slippage),
        "年化成本拖累": float((fees + taxes + slippage) / avg_nav / years) if years > 0 else np.nan,
        "現金股利": float(led.loc[led["type"] == "cash_dividend", "cash"].sum()),
        "拒單": led.loc[led["type"] == "reject", "reason"].str.split(":").str[0]
                  .value_counts().to_dict(),
        "風控介入": led.loc[led["type"] == "risk", "reason"].str.split(":").str[0]
                    .value_counts().to_dict(),
    }
    if bench is not None:
        out |= _versus(r, nav, bench.reindex(nav.index), init, years)
    return out


def _versus(r: pd.Series, nav: pd.Series, bench: pd.Series, init: float, years: float) -> dict:
    """相對基準：超額報酬、beta、alpha、追蹤誤差、資訊比率。"""
    rb = pd.concat([pd.Series([bench.iloc[0] / init - 1], index=bench.index[:1]),
                    bench.pct_change().iloc[1:]])
    var = rb.var()
    beta = float(r.cov(rb) / var) if var > 0 else np.nan
    active = r - rb
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    return {
        "基準年化報酬": _cagr(bench.iloc[-1] / init, years),
        "超額年化報酬": _cagr(nav.iloc[-1] / init, years) - _cagr(bench.iloc[-1] / init, years),
        "beta": beta,
        "年化 alpha": float((r.mean() - beta * rb.mean()) * TRADING_DAYS) if var > 0 else np.nan,
        "追蹤誤差": te,
        "資訊比率": float(active.mean() * TRADING_DAYS / te) if te > 0 else np.nan,
    }


def _cagr(growth: float, years: float) -> float:
    return float(growth ** (1 / years) - 1) if years > 0 and growth > 0 else np.nan


# ============================== 輸出 ==============================


def write(res: Result, out_dir: Path | str, name: str, bench: pd.Series | None = None,
          bench_name: str = "", notes: list[str] | None = None) -> dict:
    """
    寫出一次執行的產物：
        report.md、config.json、summary.csv   上傳（.gitignore 例外）
        nav.csv、positions.csv、ledger.csv     只留本機
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = metrics(res, bench)
    res.nav.to_csv(out / "nav.csv", encoding="utf-8-sig")
    res.positions.loc[:, (res.positions != 0).any()].to_csv(out / "positions.csv", encoding="utf-8-sig")
    res.ledger.to_csv(out / "ledger.csv", index=False, encoding="utf-8-sig")
    flat = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k, v in m.items()}
    pd.DataFrame([flat]).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "config.json").write_text(
        json.dumps(asdict(res.config), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(m, name, bench_name, notes), encoding="utf-8")
    return m


def markdown(m: dict, name: str, bench_name: str = "", notes: list[str] | None = None) -> str:
    pct = lambda x: "—" if pd.isna(x) else f"{x:+.2%}"   # noqa: E731
    num = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"   # noqa: E731
    lines = [
        f"# 沙盒執行報告：{name}",
        "",
        f"期間 {m['起日']} ～ {m['迄日']}（{m['交易日數']} 個交易日），初始資金 {num(m['初始資金'])} 元。",
        "",
        "> 沙盒驗證（計劃 §7）全部通過之前，本報告的報酬不得被引用。",
        "",
        "## 績效",
        "",
        "| 指標 | 數值 |",
        "|---|---:|",
        f"| 期末淨值 | {num(m['期末淨值'])} |",
        f"| 總報酬 | {pct(m['總報酬'])} |",
        f"| 年化報酬 | {pct(m['年化報酬'])} |",
        f"| 年化波動 | {m['年化波動']:.2%} |",
        f"| 夏普 | {m['夏普']:.2f} |",
        f"| 最大回撤 | {pct(m['最大回撤'])} |",
    ]
    if "beta" in m:
        lines += [
            "",
            f"## 相對基準：{bench_name}",
            "",
            "報酬高於基準不等於有 alpha：beta 大於 1 的組合在多頭本來就會贏，要看 alpha 與資訊比率。",
            "",
            "| 指標 | 數值 |",
            "|---|---:|",
            f"| 基準年化報酬 | {pct(m['基準年化報酬'])} |",
            f"| 超額年化報酬 | {pct(m['超額年化報酬'])} |",
            f"| beta | {m['beta']:.2f} |",
            f"| 年化 alpha | {pct(m['年化 alpha'])} |",
            f"| 追蹤誤差 | {m['追蹤誤差']:.2%} |",
            f"| 資訊比率 | {m['資訊比率']:.2f} |",
        ]
    lines += [
        "",
        "## 交易與成本",
        "",
        "| 指標 | 數值 |",
        "|---|---:|",
        f"| 成交筆數 | {m['成交筆數']:,} |",
        f"| 年化換手率 | {m['年化換手率']:.2f} 倍 |",
        f"| 手續費 | {num(m['手續費'])} |",
        f"| 證交稅 | {num(m['證交稅'])} |",
        f"| 滑價 | {num(m['滑價'])} |",
        f"| 成本合計 | {num(m['成本合計'])} |",
        f"| 年化成本拖累 | {m['年化成本拖累']:.2%} |",
        f"| 現金股利 | {num(m['現金股利'])} |",
        "",
        "拒單：" + (_counts(m["拒單"]) or "無"),
        "",
        "風控介入：" + (_counts(m["風控介入"]) or "無"),
    ]
    if notes:
        lines += ["", "## 附註", ""] + [f"- {n}" for n in notes]
    lines += ["", "## 限制", "", LIMITATIONS, ""]
    return "\n".join(lines)


def _counts(d: dict) -> str:
    return "、".join(f"`{k}` {v} 筆" for k, v in d.items())
