#!/usr/bin/env bash
# Fig. 1 and Fig. 3 in the paper's gauge (gauge_fix=False), with conditioning='centre_whiten',
# obj_scale=1.0; R_0 for the centring comes from wahba_R0 (GNSS priors + IMU), not an anchor.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export EUROC_MH01=${EUROC_MH01:-$R/data/euroc/MH_01_easy/mav0}
export URBANNAV_ROOT=${URBANNAV_ROOT:-$R/data/urbannav}
export MPLBACKEND=Agg
ts() { date '+%F %T'; }
( $PY $R/traj_178.py --gauge free > traj.log 2>&1; echo "[$(ts)] traj rc=$?"
  $PY $R/teaser_3dinset.py >> traj.log 2>&1; echo "[$(ts)] teaser rc=$?" ) &
( $PY $R/cayley_ablation_plots.py --gauge free > ablation.log 2>&1; echo "[$(ts)] ablation rc=$?"
  $PY $R/ablation_figs.py >> ablation.log 2>&1; echo "[$(ts)] figs rc=$?" ) &
wait; echo "[$(ts)] all done"
