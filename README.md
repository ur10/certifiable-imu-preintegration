# A QCQP-Representable IMU Pre-Integration Factor for Certifiable State Estimation

Reference implementation and replication code for the paper.

> Utkarsh Rai, Zhexin Xu, Bang-Shien Chen, and David Rosen.
> *A QCQP-Representable IMU Pre-Integration Factor for Certifiable State Estimation.*
> arXiv:XXXX.XXXXX

Standard IMU pre-integration builds its bias correction out of the exponential map, which has no
exact polynomial form, so a pre-integrated inertial factor cannot be written as a QCQP and is
therefore out of reach of certifiable estimation. This code replaces `Exp` with the **Cayley
map**, which is a genuine retraction onto SO(3) *and* a ratio of polynomials: clearing its
denominator leaves a residual of degree two. A catalog of closed-form **redundant constraints**
on the resulting lifting variables then tightens the Shor relaxation to rank one, so the
estimate comes with a certificate of global optimality.

The relaxation is evaluated on certifiable GNSS–IMU smoothing, on synthetic trajectories and on
EuRoC and UrbanNav.

---

## Requirements

```bash
pip install numpy scipy matplotlib gtsam mosek
```

Tested with Python 3.9, NumPy 1.23, SciPy 1.13, Matplotlib 3.9, MOSEK 11.2.

**MOSEK needs a licence.** It is the solver behind every result here. Academic licences are free
(<https://www.mosek.com/license/request/>); place the licence file at `~/mosek/mosek.lic`. Nothing
in this repository will run without it.

## Datasets

EuRoC and UrbanNav are not vendored. Download them and point two environment variables at your
copies:

```bash
export EUROC_MH01=/path/to/MH_01_easy/mav0      # the directory containing imu0/ and state_groundtruth_estimate0/
export URBANNAV_ROOT=/path/to/urbannav
```

- **EuRoC MH_01** — <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
  (IMU at 200 Hz, ground truth at 200 Hz)
- **UrbanNav**, medium-urban and deep-urban — <https://github.com/IPNL-POLYU/UrbanNavDataset>
  (IMU resampled to ~400 Hz; see `urbannav_data.py` for the frame conventions, which were
  established empirically rather than from the dataset documentation)

Paths resolve through `datasets.py`, which fails with an actionable message rather than a stack
trace if a dataset is missing. The synthetic experiments need neither dataset.

---

## Reproducing the paper

| Paper item | Command | Needs | Runtime |
|---|---|---|---|
| **Fig. 1** — local vs. certifiable solution | `python3 traj_178.py` then `python3 teaser_3dinset.py` | EuRoC | ~1 min |
| **Fig. 3** — what each redundant subset buys | `python3 cayley_ablation_plots.py` then `python3 ablation_figs.py` | — (synthetic) | ~30 min |
| **Table II** — real-world benchmarks | `python3 cayley_uniform_experiments.py multibias` | EuRoC + UrbanNav | ~30 min |
| **Table III** — vs. standard pre-integration | `python3 compare_gtsam_cayley.py matched` | EuRoC + UrbanNav | ~15 min |

Fig. 2 (the factor graph) and Table I (the constraint catalog) are drawn in LaTeX and have no
code behind them.

Each script writes its JSON next to itself and the plotting scripts read it from there. Our own
outputs are kept in `results/` for comparison — see **Reference results** below.

### Start here if you have no datasets or no patience

```bash
cp results/ablation_svr.json . && python3 ablation_figs.py
```

This redraws Fig. 3 from the shipped data in a few seconds, and is the quickest check that the
environment is set up. The regenerated `ablation.png` is byte-identical to the figure in the
paper.

### A single window, end to end

```python
import sdp_preintegration_cayley as C, datasets
r = C.run_euroc(datasets.euroc(), T=16.0, kf_interval=1.0, t_start=6.0,
                weight_mode='dataset_cov', iso_reduce='mean', obj_scale='auto',
                prior_sigmas=(1e-2, 1e-2, 1e-3, 1e-3), gauge_fix=False,
                add_redundant=True, kf_per_bias=8, accept_any=True)
print(r['rank'], r['eta'], r['rot_err'].mean())   # rank 1 => certified globally optimal
```

Those keyword arguments are the paper's nominal configuration. `run_synthetic` and
`run_urbannav` take the same ones.

---

## What the code does

| File | Role |
|---|---|
| `sdp_preintegration_cayley.py` | The implementation. Cayley pre-integration, the lifted QCQP, the redundant-constraint catalog, the SDP, and the `run_synthetic` / `run_euroc` / `run_urbannav` entry points. |
| `sdp_preintegration.py` | Synthetic scenarios (`sick`, `loop`) and pre-integration helpers, shared with the GTSAM baseline. |
| `urbannav_data.py` | UrbanNav loader: resampling, geodetic→ENU, frame conventions, static bias initialisation. |
| `gtsam_baseline.py` | The comparison point: the standard exp-map factor graph solved by Levenberg–Marquardt. |
| `datasets.py` | Dataset path resolution. |

Experiments:

| File | Produces |
|---|---|
| `cayley_ablation_plots.py` | `ablation_svr.json` — 840 synthetic solves (5 constraint stages × 7 noise levels × 12 seeds × 2 trajectories) |
| `ablation_figs.py` | `ablation.png` (Fig. 3), plus `ablation_rate.png` and `ablation_svr.png` |
| `cayley_uniform_experiments.py` | `uniform_*.json` — real-data tightness under one fixed configuration (Table II) |
| `compare_gtsam_cayley.py` | `matched.json`, `divergence.json` — head-to-head against LM (Table III) |
| `traj_178.py` / `teaser_3dinset.py` | `traj_teaser.json` → Fig. 1 |
| `cayley_progressive_tightening.py` | `progressive_tightening.json` — rank collapse as constraint families are added. Supplementary; needs no dataset. |
| `cayley_dual_certificate.py` | Explicit dual certificate, retaining the constraint multipliers |

### Problem size

The SDP is `min ⟨Q,Z⟩  s.t. ⟨Aᵢ,Z⟩ = bᵢ, Z ⪰ 0`, with the PSD block of order

```
n = 15N + 6M + 24E + 1          N keyframes, M bias blocks, E = N−1 inertial edges
```

| keyframes | n | constraints | unknowns in Z | MOSEK |
|---:|---:|---:|---:|---:|
| 9 | 334 | 614 | 55 945 | 2.9 s |
| 17 | 652 | 1 206 | 212 878 | 6.7 s |
| 25 | 970 | 1 798 | 470 935 | 18.7 s |
| 50 | 1 969 | 3 648 | 1 939 465 | 169 s (1.7 GB) |

(EuRoC MH_01, `t_start = 6 s`, `kf_per_bias = 8`; rank 1 — certified — at every size.) The
lifting variables are the majority of the cone: 384 of 652 at 17 keyframes.

---

## Reference results

`results/` holds the JSON our runs produced, so the numbers can be inspected without re-running
anything:

| File | Contents |
|---|---|
| `ablation_svr.json` | the 840-solve constraint ablation behind Fig. 3 |
| `uniform_multibias.json`, `uniform_real.json` | real-data tightness sweeps |
| `matched.json`, `divergence.json` | the GTSAM/LM comparison |
| `traj_teaser.json` | the EuRoC window in Fig. 1 |

---

## Known issues

- **`results/uniform_multibias.json` does not reproduce Table II exactly.** For EuRoC
  K = 16, M = 2 it gives a median η of 1.5e-6 where the paper reports 2.9e-6, and no other
  stored run matches the table either. The qualitative result is unaffected (4/4 windows
  certified at rank one, η of order 1e-6), but the table should be regenerated from a single
  run before anyone relies on the exact figures.
- **`weight_mode`.** Use `'dataset_cov'`, the paper's setting. `'mle'` is retained for
  comparison but is rank-tight on only 6 of 24 real windows.
- **`obj_scale='auto'`** rescales `Q` so that `max|Q| ≈ 1e4`. The cone is scale-invariant, so
  the minimiser and `Z*` are unchanged in exact arithmetic, but the rescaling is load-bearing
  numerically: the raw inverse-covariance weights push `max|Q|` to ~6e8, where MOSEK returns
  `ProblemStatus.Unknown`. The default in the library is still `1.0`; every experiment script
  passes `'auto'` explicitly.
- **A certificate is not a guarantee of accuracy.** The relaxation certifies global optimality
  of the problem as posed. With a weak gauge, rank-one solutions with 8–20° rotation error are
  reachable — tightness and accuracy are separate properties.
- **Observability.** One window in 36 fails to certify; it is near-stationary, where the
  accelerometer bias and the specific force are not separately observable.
- `ablation_figs.py` had its `__main__` guard above `fig_ablation`, so running the script never
  drew the paper's figure. Fixed here.

## Citation

```bibtex
@article{rai2026cayley,
  title   = {A {QCQP}-Representable {IMU} Pre-Integration Factor for Certifiable State Estimation},
  author  = {Rai, Utkarsh and Xu, Zhexin and Chen, Bang-Shien and Rosen, David},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Not yet chosen — add one before making the repository public.

Robust Autonomy Lab, Institute for Experiential Robotics, Northeastern University.
