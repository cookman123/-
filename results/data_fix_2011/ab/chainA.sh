#!/bin/bash
cd "."
AB=results/data_fix_2011/ab; export PYTHONIOENCODING=utf-8
run() { # name script outfiles...
  name=$1; script=$2; shift 2
  echo "=== $name old $(date +%T)"
  [ "$name" = event_model ] && rm -f results/event_model/events.csv
  python tool/ab2011.py old $script > $AB/${name}_old.log 2>&1
  mkdir -p $AB/old/$name; for f in "$@"; do cp -r $f $AB/old/$name/; done
  echo "=== $name new $(date +%T)"
  [ "$name" = event_model ] && rm -f results/event_model/events.csv
  python tool/ab2011.py new $script > $AB/${name}_new.log 2>&1
}
run highvol_controls highvol_controls.py results/highvol/controls.csv
run highvol_diagnose highvol_diagnose.py results/highvol/sector_mix.csv results/highvol/trend_filter.csv
run strategy_spec strategy_spec.py results/spec/summary.csv results/spec/yearly.csv results/spec/current_picks.csv
run run_long run_long.py results/long_rule
run factor_test factor_test.py results/factor/factors.csv
run event_model event_model.py results/event_model/scored.csv results/event_model/events.csv
run solve_sweep solve_sweep.py results/solve/sweep.csv
echo "=== overnight new"; python tool/ab2011.py new overnight_lead.py > $AB/overnight_new.log 2>&1
echo "=== adaptive new"; python tool/ab2011.py new adaptive_strategy.py > $AB/adaptive_new.log 2>&1
echo "=== DONE A $(date +%T)"
