# -*- coding: utf-8 -*-
"""
實驗主程式：消融矩陣 × 修正後的評估協定（簡報 §4 EXPERIMENT）。

用法
----
    # 先把證交所 T86 轉成籌碼面 CSV（只需做一次）
    python -m davis_tw.prepare_chips

    # 跑基線與籌碼面實驗，10 檔股票
    python -m davis_tw.run_experiment --experiments E0 E2 --max-stocks 10

    # 指定股票與全矩陣
    python -m davis_tw.run_experiment --stocks 2330 2317 1101 --experiments E0 E1 E2 E3

輸出（results/{run_id}/）
    predictions.csv     逐日預測（含實際報酬率、所屬折）
    per_stock.csv       單股 × 實驗的指標
    summary.csv         實驗層級彙總（跨股票平均與勝率）
    dm_tests.csv        Diebold–Mariano 檢定結果
    backtest.csv        含成本回測績效
    meta_weights.csv    Huber 堆疊權重（可解釋性產出）
    run_config.json     本次執行的完整設定，供複現
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from . import config as C
from . import metrics as M
from .backtest import backtest_suite
from .data_io import stock_to_sector
from .dataset import load_sector_universe, load_universe
from .features_text import Tokenizer
from .walkforward import common_evaluation_index, run_stock_experiment

from tw_market_index import benchmark_stats, load_twii

logger = logging.getLogger(__name__)


# ============================== 評估 ==============================


def experiment_runnable(exp: str, avail: set) -> bool:
    """
    這個實驗的資料齊不齊。

    文本來源（新聞／PTT／論壇）採「至少有一個就能跑」，而不是全部到齊 ——
    E1 原本定義成 price+news+ptt+forum，但 PTT 從未蒐集，結果整批 E1／E3
    在所有標的上都被靜默跳過，消融矩陣等於只比了 E0 vs E2。
    少一個來源只是少幾欄特徵（select_columns 會自動略過），不該讓整個實驗消失；
    實際用到哪些來源會記在結果的 text_sources 欄，判讀時要一併看。
    """
    need = set(C.EXPERIMENTS[exp])
    text_need = need & set(C.TEXT_GROUPS)
    if not (need - set(C.TEXT_GROUPS)) <= avail:      # 價量／籌碼這類必要軸
        return False
    if text_need and not (text_need & avail):          # 文本至少要有一個來源
        return False
    return True


def evaluate_run(res, threshold: float = 0.0, horizon: int = 1,
                 bench=None) -> dict:
    """單股單實驗的指標 + 對天真基準的 DM 檢定 + 含成本回測。"""
    pred = res.predictions
    row = {"code": res.code, "experiment": res.experiment, "horizon": horizon,
           "n_features": res.n_features}
    row.update(M.evaluate_predictions(pred))

    bp = res.baseline_predictions.reindex(pred.index)
    for name in ("naive", "always_long", "ma3", "ma5", "drift"):
        if name not in bp.columns:
            continue
        row[f"DA_{name}"] = M.directional_accuracy(pred["ret_next"], bp[name])
        row[f"MAPE_{name}"] = M.price_mape_from_returns(pred["close_t"], pred["ret_next"], bp[name])

    # 主張「模型優於天真基準」必須通過 DM 檢定（斷點 2 + 3）
    # 誤差比 naive、方向比 always_long —— 各自對上正確的對照組
    dm_mse = M.diebold_mariano(pred["ret_next"], pred["pred"], bp["naive"],
                               loss="mse", h=horizon)
    dm_sign = M.diebold_mariano(pred["ret_next"], pred["pred"], bp["always_long"],
                                loss="sign", h=horizon)
    row["DM_vs_naive_mse_p"] = dm_mse.p_value
    row["DM_vs_naive_mse_stat"] = dm_mse.stat
    row["DM_dir_vs_long_p"] = dm_sign.p_value
    row["DM_dir_vs_long_stat"] = dm_sign.stat

    bt = backtest_suite(pred, pred["pred"].to_numpy(), threshold=threshold, horizon=horizon)
    model_bt = bt[bt["name"] == "model"].iloc[0]
    bh_bt = bt[bt["name"] == "buy_hold"].iloc[0]
    row["sharpe"] = model_bt["sharpe"]
    row["total_return"] = model_bt["total_return"]
    row["ann_return"] = model_bt["ann_return"]
    row["max_drawdown"] = model_bt["max_drawdown"]
    row["n_trades"] = model_bt["n_trades"]
    row["sharpe_buyhold"] = bh_bt["sharpe"]
    row["ann_return_buyhold"] = bh_bt["ann_return"]
    # 大盤對照：投資人真正的替代方案是買指數，不是抱這一檔
    row.update(benchmark_stats(bench, pred.index, C.TRADING_DAYS_PER_YEAR))
    row["excess_vs_bench"] = row["total_return"] - row["bench_return"]
    return row


def dm_between_experiments(results: dict, code: str, exp1: str, exp2: str,
                           horizon: int = 1) -> list[dict]:
    """同一檔股票、兩個實驗之間的 DM 檢定（消融比較的統計依據）。"""
    r1, r2 = results.get((code, exp1, horizon)), results.get((code, exp2, horizon))
    if r1 is None or r2 is None:
        return []
    p1, p2 = r1.predictions, r2.predictions
    idx = p1.index.intersection(p2.index)
    if len(idx) < 30:
        return []
    out = []
    for loss in ("mse", "sign"):
        dm = M.diebold_mariano(p1.loc[idx, "ret_next"], p1.loc[idx, "pred"],
                               p2.loc[idx, "pred"], loss=loss, h=horizon)
        out.append({"code": code, "horizon": horizon,
                    "comparison": f"{exp1} vs {exp2}", **dm.as_dict()})
    return out


def dm_ensemble_vs_base(res, horizon: int = 1) -> list[dict]:
    """集成 vs 各單一基礎模型（原論文斷點 3 的直接回應）。"""
    pred = res.predictions
    bm = res.base_model_predictions.reindex(pred.index)
    rows = []
    for col in bm.columns:
        if col == "fold":
            continue
        dm = M.diebold_mariano(pred["ret_next"], pred["pred"], bm[col], loss="sign", h=horizon)
        rows.append({"code": res.code, "horizon": horizon,
                     "comparison": f"{res.experiment}:ensemble vs {col}", **dm.as_dict()})
    return rows


# ============================== 主流程 ==============================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="台股 DAViS 延伸研究：消融實驗")
    ap.add_argument("--experiments", nargs="+", default=["E0"],
                    choices=list(C.EXPERIMENTS), help="要跑的實驗代號")
    ap.add_argument("--level", default="stock", choices=["stock", "sector", "both"],
                    help="分析層級：stock=個股（預設）、sector=產業類股指數、both=兩者都跑")
    ap.add_argument("--stocks", nargs="*", default=None, help="指定股票代號（預設自動挑選）")
    ap.add_argument("--sectors", nargs="*", default=None,
                    help="指定產業（預設全部 28 個），例如 --sectors 半導體 金融 航運")
    ap.add_argument("--max-stocks", type=int, default=C.SCOPE.max_stocks)
    ap.add_argument("--per-industry", action="store_true",
                    help="改用提案 §4 的挑股規則：每產業取成交金額最高的幾檔"
                         "（預設是全市場前 max-stocks 檔，產業涵蓋會偏電子）")
    ap.add_argument("--per-industry-top", type=int, default=1,
                    help="搭配 --per-industry：每產業取幾檔（預設 1）")
    ap.add_argument("--start", default=C.SCOPE.start_date)
    ap.add_argument("--end", default=C.SCOPE.end_date)
    ap.add_argument("--horizon", type=int, nargs="+", default=C.HORIZONS,
                    help="預測期間（交易日），可給多個一起跑。預設 1 5"
                         "（1=隔日沖、5=週波段）")
    ap.add_argument("--target", default=C.TARGET, choices=["return", "log_return", "price"])
    ap.add_argument("--threshold", type=float, default=0.0, help="回測進場門檻（預測報酬率）")
    ap.add_argument("--train-days", type=int, default=C.WALK_FORWARD.train_days)
    ap.add_argument("--val-days", type=int, default=C.WALK_FORWARD.val_days)
    ap.add_argument("--test-days", type=int, default=C.WALK_FORWARD.test_days)
    ap.add_argument("--step-days", type=int, default=C.WALK_FORWARD.step_days)
    ap.add_argument("--rolling", action="store_true", help="改用固定長度滾動窗（預設為擴張窗）")
    ap.add_argument("--no-align", action="store_true",
                    help="不對齊各實驗的日期交集（不建議：實驗間將不可比）")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")

    wf = C.WalkForwardConfig(train_days=args.train_days, val_days=args.val_days,
                             test_days=args.test_days, step_days=args.step_days,
                             expanding=not args.rolling)
    scope = C.ScopeConfig(start_date=args.start, end_date=args.end,
                          max_stocks=args.max_stocks,
                          stocks=tuple(args.stocks) if args.stocks else (),
                          per_industry=args.per_industry,
                          per_industry_top=args.per_industry_top)

    horizons = sorted(set(args.horizon))
    bench = load_twii()            # 大盤加權指數，所有績效的最終對照組
    logger.info("預測期間：%s", "、".join("隔日沖" if h == 1 else f"持有 {h} 日" for h in horizons))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = C.RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 依層級決定要分析的標的：個股、產業類股指數，或兩者
    stocks = []
    if args.level in ("stock", "both"):
        stocks += load_universe(scope=scope)
    if args.level in ("sector", "both"):
        stocks += load_sector_universe(args.sectors, scope=scope)
    if not stocks:
        logger.error("研究範圍內沒有可用標的；請確認 data/stocks 是否已建立")
        return 1

    sec_of = {} if args.level == "sector" else stock_to_sector()

    # 文本斷詞器共用一個實例（CKIP 模型載入成本高）
    tokenizer = Tokenizer(C.TEXT.tokenizer) if any(
        g in C.TEXT_GROUPS for e in args.experiments for g in C.EXPERIMENTS[e]) else None

    results, eval_rows, dm_rows, bt_rows, weight_rows = {}, [], [], [], []
    skipped = []

    for sd in stocks:
        avail = sd.available_groups()
        runnable = [e for e in args.experiments if experiment_runnable(e, avail)]
        missing = [e for e in args.experiments if e not in runnable]
        if missing:
            need = sorted({g for e in missing for g in C.EXPERIMENTS[e]} - avail)
            logger.info("[%s] 可用資料軸：%s", sd.code, "、".join(sorted(avail)))
            logger.warning("[%s] 略過 %s（缺資料軸：%s）", sd.code, ",".join(missing), ",".join(need))
            skipped.extend({"code": sd.code, "experiment": e, "reason": f"缺 {need}"} for e in missing)
        if not runnable:
            continue

        for hz in horizons:
            # 對齊：所有可跑實驗共用同一段有效日期，確保消融比較的差異只來自特徵
            restrict = (None if args.no_align else
                        common_evaluation_index(sd, runnable, args.target, tokenizer, hz))
            if restrict is not None and len(restrict) < wf.train_days + wf.val_days + wf.test_days:
                logger.warning("[%s] h=%d 對齊後僅 %d 個交易日，不足一折，略過",
                               sd.code, hz, len(restrict))
                continue

            for exp in runnable:
                logger.info("[%s] h=%d 執行 %s ...", sd.code, hz, exp)
                res = run_stock_experiment(sd, exp, wf=wf, target=args.target,
                                           restrict_index=restrict, tokenizer=tokenizer,
                                           horizon=hz)
                if res is None:
                    continue
                results[(sd.code, exp, hz)] = res
                row = evaluate_run(res, args.threshold, hz, bench=bench)
                used_text = sorted(set(C.EXPERIMENTS[exp]) & set(C.TEXT_GROUPS) & avail)
                row["text_sources"] = ",".join(used_text) if used_text else ""
                row["level"] = "sector" if sd.members else "stock"
                row["sector"] = sd.code if sd.members else sec_of.get(sd.code, "")
                if sd.members:
                    row["n_members"] = len(sd.members)
                eval_rows.append(row)
                dm_rows.extend(dm_ensemble_vs_base(res, hz))

                bt = backtest_suite(res.predictions, res.predictions["pred"].to_numpy(),
                                    threshold=args.threshold, horizon=hz)
                bt.insert(0, "experiment", exp)
                bt.insert(0, "code", sd.code)
                bt_rows.append(bt)

                if not res.meta_weights.empty:
                    w = res.meta_weights.copy()
                    w.insert(0, "horizon", hz)
                    w.insert(0, "experiment", exp)
                    w.insert(0, "code", sd.code)
                    weight_rows.append(w)

    if not eval_rows:
        logger.error("沒有任何實驗成功完成")
        return 1

    # ---------- 消融比較的 DM 檢定 ----------
    codes = sorted({c for c, _, _ in results})
    pairs = [("E1", "E0"), ("E2", "E0"), ("E3", "E0"), ("E3", "E1"), ("E3", "E2"),
             ("E1a", "E0"), ("E1b", "E0"), ("E1c", "E0")]
    for code in codes:
        for hz in horizons:
            for a, b in pairs:
                dm_rows.extend(dm_between_experiments(results, code, a, b, hz))

    # ---------- 落地 ----------
    per_stock = pd.DataFrame(eval_rows)
    per_stock.to_csv(out_dir / "per_stock.csv", index=False, encoding="utf-8-sig")

    all_preds = pd.concat([r.predictions.assign(experiment=e, horizon=h)
                           for (c, e, h), r in results.items()])
    all_preds.to_csv(out_dir / "predictions.csv", encoding="utf-8-sig")

    agg = (per_stock.groupby(["experiment", "horizon"])
           .agg(n_stocks=("code", "nunique"),
                DA=("DA", "mean"),
                DA_std=("DA", "std"),
                DA_always_long=("DA_always_long", "mean"),
                DA_ma5=("DA_ma5", "mean"),
                MAPE_price=("MAPE_price", "mean"),
                MAPE_naive=("MAPE_naive", "mean"),
                RMSE_ret=("RMSE_ret", "mean"),
                sharpe=("sharpe", "mean"),
                sharpe_buyhold=("sharpe_buyhold", "mean"),
                ann_return=("ann_return", "mean"),
                max_drawdown=("max_drawdown", "mean"),
                n_trades=("n_trades", "mean"),
                bench_return=("bench_return", "mean"),
                excess_vs_bench=("excess_vs_bench", "mean"))
           .reset_index())
    # 跨股票勝率：DA > 0.5 的比例，以及方向 DM 顯著優於 naive 的比例
    agg["pct_DA_above_50"] = (per_stock.assign(w=per_stock["DA"] > 0.5)
                              .groupby(["experiment", "horizon"])["w"].mean().to_numpy())
    # 方向顯著優於 always_long 的股票比例（H1/H2 的主要判準）
    agg["pct_dir_beat_long_sig"] = (per_stock.assign(
        w=(per_stock["DM_dir_vs_long_p"] < 0.05) & (per_stock["DM_dir_vs_long_stat"] < 0))
        .groupby(["experiment", "horizon"])["w"].mean().to_numpy())
    # 誤差顯著低於 naive 的股票比例
    agg["pct_mse_beat_naive_sig"] = (per_stock.assign(
        w=(per_stock["DM_vs_naive_mse_p"] < 0.05) & (per_stock["DM_vs_naive_mse_stat"] < 0))
        .groupby(["experiment", "horizon"])["w"].mean().to_numpy())
    agg.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    # 個股結果依產業分組：呼應原論文「來源效益依產業而異」的發現
    if "level" in per_stock.columns:
        sub = per_stock[(per_stock["level"] == "stock") & (per_stock["sector"] != "")]
        if len(sub) > 3 and sub["sector"].nunique() > 1:
            by_sec = (sub.groupby(["sector", "experiment"])
                      .agg(n=("code", "nunique"), DA=("DA", "mean"),
                           DA_long=("DA_always_long", "mean"), sharpe=("sharpe", "mean"))
                      .reset_index())
            by_sec["beat_long"] = by_sec["DA"] > by_sec["DA_long"]
            by_sec.to_csv(out_dir / "by_sector.csv", index=False, encoding="utf-8-sig")

    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out_dir / "dm_tests.csv", index=False, encoding="utf-8-sig")
    if bt_rows:
        pd.concat(bt_rows).to_csv(out_dir / "backtest.csv", index=False, encoding="utf-8-sig")
    if weight_rows:
        pd.concat(weight_rows).to_csv(out_dir / "meta_weights.csv", index=False, encoding="utf-8-sig")
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False, encoding="utf-8-sig")

    (out_dir / "run_config.json").write_text(json.dumps({
        "run_id": run_id,
        "experiments": args.experiments,
        "target": args.target,
        "scope": asdict(scope),
        "walk_forward": asdict(wf),
        "model": asdict(C.MODEL),
        "text": asdict(C.TEXT),
        "cost": asdict(C.COST),
        "threshold": args.threshold,
        "aligned": not args.no_align,
        "stocks": [s.code for s in stocks],
        "horizons": horizons,
        "folds_example": next(iter(results.values())).folds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 主控台摘要 ----------
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + "=" * 78)
    print(f"實驗彙總（run_id={run_id}，{per_stock['code'].nunique()} 檔股票）")
    print("=" * 78)
    cols = ["experiment", "horizon", "n_stocks", "DA", "DA_always_long", "pct_DA_above_50",
            "pct_dir_beat_long_sig", "MAPE_price", "MAPE_naive",
            "pct_mse_beat_naive_sig", "sharpe", "sharpe_buyhold", "excess_vs_bench"]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 兩個層級都跑時，分開再列一次，否則平均會把個股與類股混在一起
    if "level" in per_stock.columns and per_stock["level"].nunique() > 1:
        print("\n依層級分開：")
        lv = (per_stock.groupby(["level", "experiment"])
              .agg(標的數=("code", "nunique"), DA=("DA", "mean"),
                   DA_always_long=("DA_always_long", "mean"),
                   sharpe=("sharpe", "mean"), sharpe_buyhold=("sharpe_buyhold", "mean"))
              .reset_index())
        print(lv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n輸出目錄：{out_dir}")
    print("提醒：MAPE 需與 MAPE_naive 對照才可判讀；DA 的優勢需由 dm_tests.csv 支持。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
