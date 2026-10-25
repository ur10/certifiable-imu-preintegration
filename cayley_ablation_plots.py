"""Progressive-tightening ablation with the leading eigenvalue ratio as the tightness measure.

Produces the two figures of Sec. VII:

  ablation_rate.png  -- certified rate vs injected noise, one curve per cumulative constraint set
  ablation_svr.png   -- log-SVR boxplots vs noise, one panel per cumulative constraint set

log SVR = log10(lam_1/lam_2) of the SDP solution. The integer rank counts eigenvalues above
1e-6 lam_max, so rank Z-hat = 1 corresponds to log SVR > 6; that is the certified threshold
used throughout and drawn as the shaded band.
"""
import argparse, json, time
import numpy as np
import sdp_preintegration_cayley as C
import cayley_uniform_experiments as U

OUT_JSON = 'ablation_svr.json'
THRESH = 6.0                      # log10(lam_1/lam_2) above which rank Z-hat = 1

# cumulative constraint sets, matching Table II's derivation order
# Display labels follow the paper's catalog (A1, A2, B1, B2, C1, contiguous). The tuples are
# the solver's internal family keys in sdp_preintegration_cayley, which are unchanged.
# A2 (the cross-product/handedness identity) is counted as PRIMARY in the paper's presentation:
# R^T R = I alone only pins O(3), so A2 is what makes the feasible set SO(3). The second stage is
# therefore LABELLED '+A1' even though its solver tuple still carries A2 -- the data in
# ablation_svr.json is the original run, whose stage 0 has no A2. A run with A2 actually moved
# into stage 0 is in ablation_svr_a2primary_partial.json (stages 0-3; stage 0 drops to rank 98).
STAGES = [('primary only', ()),
          ('$+$A1',          ('A1', 'A2')),
          ('$+$B1',          ('A1', 'A2', 'B2')),
          ('$+$B2',          ('A1', 'A2', 'B2', 'B3')),
          ('$+$C1 (full)',   ('A1', 'A2', 'B2', 'B3', 'C2'))]

# the identities each stage adds, for the panel annotations
ADDED = [r'',
         r'$R_kR_k^{\top}=I_3$',
         '\n'.join((r'$Y_{ij}w_{ij}=0$', r"$Y'_{ij}w_{ij}=0$")),
         r"$Y_{ij}^{\top}Y_{ij}=Y_{ij}'^{\top}Y'_{ij}$",
         r'$\|W^{v}_{ij}\|^2=\|u^{v}_{ij}\|^2$']

SIGMAS = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1e0, 2e0]
SEEDS = list(range(12))
SCEN = ('sick', 'loop')
NOMINAL = 1e-3                    # modelled density held fixed; only injected noise varies


def run(out=OUT_JSON, T=8.0, kf_per_bias=4):
    rows = []
    for si, (lab, fams) in enumerate(STAGES):
        red = True if len(fams) == len(C.ALL_FAMILIES) else (fams if fams else False)
        for s in SIGMAS:
            for sc in SCEN:
                for sd in SEEDS:
                    t0 = time.time()
                    try:
                        r = U._solve(sc, 'dataset_cov', T=T, dt=0.01, kf_per_bias=kf_per_bias,
                                     gyro_sigma=NOMINAL, accel_sigma=NOMINAL, noise_sigma=s,
                                     add_redundant=red, seed=sd)
                    except Exception as exc:
                        r = None
                        print('  EXC %s %s %g s%d: %s' % (lab, sc, s, sd, exc), flush=True)
                    ok = r is not None and r.get('rank') is not None
                    rows.append(dict(stage=si, stage_lab=lab, scen=sc, sigma=s, seed=sd,
                                     rank=int(r['rank']) if ok else None,
                                     log_svr=float(r['log_svr']) if ok else None,
                                     eta=float(r['eta']) if ok else None,
                                     rel=float(r['rel_gap']) if ok else None,
                                     n_var=int(r.get('n_var', 0)) if ok else None,
                                     M=int(r.get('num_bias', 0)) if ok else None,
                                     secs=time.time() - t0))
            g = [q for q in rows if q['stage'] == si and q['sigma'] == s and q['log_svr'] is not None]
            if g:
                sv = np.array([q['log_svr'] for q in g])
                print('%-14s sig=%-7.0e n=%2d logSVR med %6.2f  certified %2d/%2d'
                      % (lab, s, len(g), np.median(sv), int((sv > THRESH).sum()), len(g)), flush=True)
        json.dump(rows, open(out, 'w'), indent=1)
    print('wrote', out)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=8.0)
    ap.add_argument('--kfpb', type=int, default=4)
    a = ap.parse_args()
    run(T=a.T, kf_per_bias=a.kfpb)
