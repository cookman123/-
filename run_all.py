#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一鍵跑完全部：在 VS Code 按執行（F5）就好，不必在終端機打任何指令。

依序執行兩件事：

  第 1 步　訓練流程（davis_tw_train.py）
          383 檔個股 ＋ 28 個產業類股 × 隔日沖與週波段 × 價量＋籌碼＋文本
          產出 results/full/report.md（產業在前、個股在後）
          ※ 有斷點續跑：中途關掉再按一次執行，會從 progress.jsonl 接續

  第 2 步　消融矩陣（davis_tw_all.py run）
          E0–E3 四組特徵各自訓練並互相比較，附 Diebold–Mariano 檢定
          標的為全部個股（約 344 檔）＋ 28 個產業類股
          產出 results/ablation/summary.csv、dm_tests.csv、by_sector.csv

四種預測期間（1／5／20／60 日）的理由：交易成本拖累 = 252/h × 0.585%，
h=1 每年 −147%、h=5 −29%、h=20 −7.4%、h=60 −2.5%。實測模型選股每 5 日只比
全市場等權多約 0.24 個百分點，蓋不住短週期的成本，所以必須把月／季波段一起測，
才問得出「拉長持有期能不能讓這點優勢變成正報酬」。

預估時間：第 1 步約 9–10 小時、第 2 步約 13–14 小時（372 標的 × 4 實驗 × 4 週期）。
斷詞已加快取（同一批文件只斷一次，跨折／跨實驗／跨週期共用），否則會再多好幾倍。
全程的畫面輸出會同時寫進 results/run_all.log，關掉視窗也查得到。
"""

import logging
import sys
import time

# Windows 主控台預設是 cp950，印到非中日韓的符號（▶、✓）會直接 UnicodeEncodeError
# 把整支程式打掉。先把輸出改成 UTF-8，編不出來的字元退而求其次替換掉，不要中斷。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---- 兩個步驟的參數；想改跑法就改這裡，不必動下面的程式 ----
TRAIN_ARGS = [
    # 換新目錄：舊的 results/full 是「斷詞器只有字元 bigram、內文還帶 HTML 標籤」
    # 時期的結果，續跑會沿用那些過時的列，必須分開。
    "--out", str(RESULTS / "full_v2"),
    # 其餘全用預設：全部股票＋28 產業、四種預測期間、價量＋籌碼＋文本
]
ABLATION_ARGS = [
    "run",
    "--level", "both",                 # 個股＋產業類股
    "--experiments", "E0", "E1", "E2", "E3",
    "--max-stocks", "400",             # 不篩選，清單裡符合條件的全部納入（約 344 檔）
    "--run-id", "ablation_v2",
    "--log-level", "WARNING",          # 只留警告與最後的彙總表
]


def setup_logging() -> Path:
    """畫面與檔案同時輸出，跑幾小時也不怕關掉視窗就什麼都沒了。"""
    log_path = RESULTS / "run_all.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):        # 避免重複執行時處理器疊加
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


def run_step(title: str, func, argv: list) -> tuple:
    """跑一個步驟並計時；失敗不中斷整批，讓後面的步驟還能跑完。"""
    banner(f">> {title}")
    print(f"  參數：{' '.join(argv) if argv else '（全用預設）'}", flush=True)
    t0 = time.time()
    try:
        code = func(argv)
        ok = (code == 0)
    except KeyboardInterrupt:
        print(f"\n  使用者中斷。{title} 的進度已保留，再按一次執行會接續。", flush=True)
        raise
    except Exception as e:                                  # noqa: BLE001
        logging.exception(f"  {title} 發生錯誤：{type(e).__name__}: {e}")
        ok = False
    spent = timedelta(seconds=int(time.time() - t0))
    print(f"\n  {title} {'完成' if ok else '未成功'}，耗時 {spent}", flush=True)
    return ok, spent


def main() -> int:
    log_path = setup_logging()
    started = datetime.now()

    banner("台股 DAViS：一鍵全跑")
    print(f"  開始時間：{started:%Y-%m-%d %H:%M}")
    print(f"  紀錄檔　：{log_path}")
    print("  預估總時長約 23 小時（全部標的 × 四種預測期間 × 含文本）。中途可以關掉，"
          "第 1 步會從 progress.jsonl 接續。", flush=True)

    # 這兩個匯入放在函式裡：匯入時會載入 sklearn 等重量級套件，
    # 先把上面的說明印出來，使用者才不會以為程式卡住了。
    import davis_tw_all
    import davis_tw_train

    results = []
    results.append(("訓練流程（個股＋產業類股）",
                    *run_step("第 1 步　訓練流程", davis_tw_train.main, TRAIN_ARGS)))
    results.append(("消融矩陣 E0–E3",
                    *run_step("第 2 步　消融矩陣", davis_tw_all.main, ABLATION_ARGS)))

    banner("全部完成")
    for name, ok, spent in results:
        print(f"  [{'OK' if ok else '失敗'}] {name}　耗時 {spent}")
    print(f"\n  總耗時：{timedelta(seconds=int((datetime.now() - started).total_seconds()))}")
    print(f"\n  主要產出：")
    print(f"    {RESULTS / 'full_v2' / 'report.md'}        訓練結果報告（產業在前、個股在後）")
    print(f"    {RESULTS / 'full_v2' / 'summary.csv'}      逐標的數值")
    print(f"    {RESULTS / 'ablation_v2' / 'summary.csv'}  消融矩陣彙總")
    print(f"    {RESULTS / 'ablation_v2' / 'dm_tests.csv'} 顯著性檢定（判斷特徵有沒有效的依據）")
    print(f"    {RESULTS / 'ablation_v2' / 'by_sector.csv'} 個股結果依產業分組")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中斷。")
        raise SystemExit(130)
