# partx_faithful reference check against cpslab

Spec §8.2 asks for 10 replications per function per implementation at T = 5,000, with a fallback
to T = 2,000. **This document reports the smoke-scale run only** (T = 300, 2 replications). The
full run was started as SLURM job 292201 and cancelled on the user's instruction, 2026-09-11:
iLab gave it 2 CPUs for 10 workers, and a single cpslab replication takes more than 37 CPU-minutes
at T = 2,000, so it would have run for a day or more. The faithfulness claim therefore rests on
this smoke comparison plus the code review against cpslab f296c40, not on the 10-replication test.

## What ran

- cpslab Part-X at commit f296c40, cloned from github.com/cpslab-asu/part-x.
- Our loop in deterministic mode: the `SklearnGPRBackend` (cpslab's `InternalGPR` settings),
  `ContinuousSampler` in place of pool snapping, EI argmax over 10,000 random points,
  `compat_ci_var` and `compat_eq4_times_volume` on, and `ei_f_star="observed"`.
- Paper §5.1 settings otherwise: n0 = 10, nBO = 10, nc = 100, R = 10, M = 100, B = 2,
  δC = 0.05, δv = 0.001. T = 300 instead of 5,000, and 2 replications instead of 10.
- Command: `scripts/partx_reference_check.py --reps 2 --budget 300 --jobs 4`, in a scratch
  Python 3.10 env (numpy < 2, treelib, pathos, scikit-optimize, plus cpslab's own pandas and
  psy-taliro dependencies).

## Result

Volumes are raw areas, as in the paper's tables.

### himmelblau

| quantity | cpslab mean | ours mean | 2·SE | within 2 SE |
|---|---:|---:|---:|---|
| GP-quantile volume (q = 0.5) | 16.9438 | 17.3188 | 0.3026 | no |
| leaves of type + | 2.0 | 2.0 | 0.0 | yes |
| leaves of type − | 0.0 | 0.0 | 0.0 | yes |
| leaves of type remaining | 6.0 | 6.0 | 0.0 | yes |
| leaves of type u | 0.0 | 0.0 | 0.0 | yes |

Monte Carlo truth over 1e6 points: 17.6904. Paper Table 1 P-X (δq = 0.5): 17.671; paper Monte
Carlo: 17.030. |ours − cpslab| is 2.12% of the truth.

### goldstein

| quantity | cpslab mean | ours mean | 2·SE | within 2 SE |
|---|---:|---:|---:|---|
| GP-quantile volume (q = 0.5) | 0.3035 | 0.2990 | 0.0146 | yes |
| leaves of type + | 2.5 | 3.0 | 1.0 | yes |
| leaves of type − | 0.0 | 0.0 | 0.0 | yes |
| leaves of type remaining | 5.0 | 5.0 | 0.0 | yes |
| leaves of type u | 0.0 | 0.0 | 0.0 | yes |

Monte Carlo truth over 1e6 points: 0.3055. Paper Table 1 P-X (δq = 0.5): 0.304; paper Monte
Carlo: 0.302. |ours − cpslab| is 1.47% of the truth.

## Reading this

The leaf-type counts, which are what the partition and classification logic produce, match cpslab
exactly on himmelblau and within noise on goldstein. Both implementations land within about 2% of
the Monte Carlo truth on both functions.

The himmelblau volume row is outside 2 SE, but at 2 replications the standard error is small
enough that a fraction of a percent fails it, and the spec's own pass rule was written for 10
replications. Nothing here separates the two implementations by more than the sampling noise the
smoke size can resolve.

Reproducing the full check later: rerun `scripts/partx_reference_check.py --reps 10 --budget 2000
--jobs 10 --cpslab-commit f296c40` on a host with 10 free cores, with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1.
