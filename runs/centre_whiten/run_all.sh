#!/usr/bin/env bash
# Re-run every paper experiment with conditioning='centre_whiten', obj_scale=1.0, gauge_fix='R'.
# Outputs land in this directory; results/ (the paper's shipped JSON) is left untouched.
#   stream A: Fig. 3 ablation (840 synthetic solves)
#   stream B: Table II suite (stages, real, multibias) then the progressive-tightening supplement
#   then, alone so its timings are uncontended: Table III (matched + divergence) and Fig. 1
set -u
cd "$(dirname "$0")"
R=../..
PY=$R/.venv/bin/python
export EUROC_MH01=${EUROC_MH01:-$R/data/euroc/MH_01_easy/mav0}
export URBANNAV_ROOT=${URBANNAV_ROOT:-$R/data/urbannav}
export MPLBACKEND=Agg
ts() { date '+%F %T'; }
( echo "[$(ts)] A start"; $PY $R/cayley_ablation_plots.py > ablation.log 2>&1; echo "[$(ts)] A ablation rc=$?"
  $PY $R/ablation_figs.py >> ablation.log 2>&1; echo "[$(ts)] A figs rc=$?" ) &
( echo "[$(ts)] B start"; $PY $R/cayley_uniform_experiments.py stages real multibias > uniform.log 2>&1; echo "[$(ts)] B uniform rc=$?"
  $PY $R/cayley_progressive_tightening.py > progressive.log 2>&1; echo "[$(ts)] B progressive rc=$?" ) &
wait
echo "[$(ts)] C start"
$PY $R/compare_gtsam_cayley.py both > compare.log 2>&1; echo "[$(ts)] C compare rc=$?"
$PY $R/traj_178.py > traj.log 2>&1; echo "[$(ts)] C traj rc=$?"
$PY $R/teaser_3dinset.py >> traj.log 2>&1; echo "[$(ts)] C teaser rc=$?"
echo "[$(ts)] all done"
