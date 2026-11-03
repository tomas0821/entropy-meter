#!/bin/bash
# Overnight queue, 2026-11-01. Starts after the dβ = 0.1 AFM production batch finishes.
cd "$(dirname "$0")/../.."          # -> code/
while pgrep -f "run_seeds.py --jobs 10 --outdir results/production$" > /dev/null; do sleep 30; done
echo "[$(date)] batch A done; starting B (AFM dβ=0.05)"
python3 run_seeds.py --jobs 10 --dbeta 0.05 --outdir results/production > results/logs/afm_db0.05.log 2>&1
echo "[$(date)] B done; starting C (FM dβ=0.0125)"
python3 run_seeds.py --jobs 10 --J -1 --dbeta 0.0125 --nmeas 2000 --neq 500 --outdir results/production > results/logs/fm_db0.0125.log 2>&1
echo "[$(date)] C done; starting D (field, L=48)"
for h in 0.3 1.0; do
  python3 run_seeds.py --jobs 10 --L 48 --h $h --dbeta 0.05 --bmax 15 --nmeas 5000 --neq 1000 --outdir results/field > results/logs/field_h$h.log 2>&1
done
echo "[$(date)] queue done"
