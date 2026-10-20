#!/usr/bin/env bash
# 稽核修正後（股價修正表、漲跌幅門檻、ts_logvol、季頻出場價）重跑所有存活策略。
# 輸出紀錄在 results/audit/，修正前的備份在 results/pre_audit/。
cd "$(dirname "$0")/.."
L=results/audit; mkdir -p $L
step() { echo "[$(date +%H:%M)] $1 開始"; shift; "$@"; echo "[$(date +%H:%M)] 結束 exit $?"; }

step 成交金額快取 python -c "import sys; sys.path.insert(0,'.'); import universe_pit as U; U.get_mask('2003-01-01','2026-12-31',250,refresh=True)" > $L/turnover.log 2>&1
step 特徵庫 python feature_engineering.py --quiet > $L/build.log 2>&1
step 特徵庫_full python feature_engineering.py --uni 400 --tag _full --quiet > $L/build_full.log 2>&1
step 特徵庫_surv python tool/run_surv.py feature_engineering --tag _surv --quiet > $L/build_surv.log 2>&1

# 5 日標籤（耗時最久）兩個版本並行
( python nextday_horizon.py > $L/horizon.log 2>&1; echo "[$(date +%H:%M)] horizon exit $?"
  python horizon_verify.py > $L/horizon_verify.log 2>&1; echo "[$(date +%H:%M)] verify exit $?"
  python cumulative_returns.py > $L/cumulative.log 2>&1 ) &
( python tool/run_surv.py nextday_horizon --tag _surv > $L/horizon_surv.log 2>&1; echo "[$(date +%H:%M)] horizon_surv exit $?"
  python tool/run_surv.py cumulative_returns --tag _surv > $L/cumulative_surv.log 2>&1 ) &

# 季頻與長期版（同時跑，CPU 用量小）
for s in "quarterly_factors.py" "quarterly_factors.py --tag _full" "quarterly_verify.py" \
         "quarterly_verify.py --tag _full" "quarterly_alpha.py" "survivorship_fix.py" "dividend_check.py"; do
  n=$(echo $s | tr ' ' '_' | sed 's/.py//'); step "$s" python $s > $L/$n.log 2>&1
done
for m in highvol_controls highvol_diagnose strategy_spec; do
  step $m python tool/ab_lookahead.py $m > $L/ab_$m.log 2>&1
done
step 跌深反彈 python run_long.py > $L/run_long.log 2>&1
step 兩位投資人 python reversal_two_investors.py > $L/two_investors.log 2>&1
wait
echo "ALLDONE"
