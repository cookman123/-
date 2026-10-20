#!/bin/bash
cd "."
AB=results/data_fix_2011/ab; export PYTHONIOENCODING=utf-8
N=results/nextday
for m in old new; do
  for s in next_day_stocks nextday_diagnose nextday_tradable; do
    echo "=== $s $m $(date +%T)"; python tool/ab2011.py $m $s.py > $AB/${s}_$m.log 2>&1
  done
  if [ $m = old ]; then mkdir -p $AB/old/nextday; mv $N/pred_stocks.csv $N/nextday_stocks.csv $N/diagnose.csv $N/tradable.csv $AB/old/nextday/; fi
done
echo "=== DONE B $(date +%T)"
