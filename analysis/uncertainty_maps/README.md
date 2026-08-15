# Uncertainty decomposition maps — ensemble_epistemic (noisy pendulum)

State-space maps of what each acquisition arm believes, where it collected, and
how its predictive uncertainty splits into aleatoric and epistemic parts.

Outputs live in
`/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/`:

```
members/   per-member p(success) over the eval grid, one npz per (arm, epoch)
acquired/  the states each arm added to training, one npz per arm
figures/   uncertainty_maps_<pred>_<level>.pdf  — one page per adaptive epoch
code/      a copy of this directory, so SLURM jobs do not depend on a worktree
```

## Reading a page

One page = one (predictor, noise level, adaptive epoch). One row per arm.
Columns, left to right:

| column | quantity |
|---|---|
| ground truth | oracle p(success), 90 rollouts per grid cell |
| acquired this epoch | the 1,000 states this arm just added, coloured by outcome |
| predicted p(success) | the ensemble marginal `p̄` |
| p(invalid) | flow matching only — endpoints in neither attractor |
| H total | `H(p̄)` |
| H aleatoric | `E_m[H(p_m)]` |
| H epistemic | BALD, `H(p̄) − E_m[H(p_m)]` |
| Var total | `p̄(1−p̄)` |
| Var aleatoric | `E_m[p_m(1−p_m)]` |
| Var epistemic | `Var_m(p_m)` |
| Var epistemic, MC-debiased | flow matching only — the score `epi_var` ranked on |

Both identities hold exactly on the rendered arrays:
`H total = H aleatoric + H epistemic` and `Var total = Var aleatoric + Var epistemic`.

**Every colour scale is fixed** across arms, epochs, levels and predictors — p in
[0,1], H in [0, ln 2], Var in [0, 0.25] — so any two panels anywhere in the
campaign are directly comparable. The epistemic columns are the only fitted
scales, because they are ~30x smaller than the total; their ceiling is printed
on the colorbar and is shared across every page of that predictor x level.

## What is and is not recoverable

`full_roa_per_point.npz` stores only the ensemble MARGINAL. Total uncertainty is
an exact function of it, but the aleatoric/epistemic split is not — that needs
the per-member probabilities, which were never serialised and can only be
recomputed by re-running the members.

Per-epoch member checkpoints survived for **`clf_high`, `fm_high` and
`fm_xhigh`** and were deleted everywhere else. So:

| cell | totals + acquisition | aleatoric/epistemic split |
|---|---|---|
| clf_high, fm_high, fm_xhigh | yes | yes |
| every other predictor x level | yes | **permanently unrecoverable** |

The split columns render an explicit "not recoverable" tile in those cells
rather than being dropped, so the gap is visible rather than silent.

## The epoch-sweep view

`make_sweep.py` renders one quantity for every arm across the epoch ladder on a
single fixed scale — the spatial replacement for the campaign's original line
figure, which plotted pool means on per-panel y-axes and stopped each curve
wherever preemption landed.

```bash
python analysis/uncertainty_maps/make_sweep.py --pred fm --level high \
    --quantity h_epistemic --every 2 --out .../figures
```

Arms that never reached an epoch get an explicit empty tile; the ladder is never
renumbered to hide a short run.

## Verification

Checked on real outputs, not asserted by construction:

- Both identities hold to float precision — max deviation `8.3e-17` on the
  variance decomposition, exactly `0` on entropy — and BALD is nonnegative at
  every evaluated cell.
- The classifier recompute reproduces the stored `p_success` to `0.0000`.
- The flow-matching recompute matches the stored eval marginal with mean
  difference `-8e-5` and median `|diff| / expected_sd` of 0.55–0.79 against the
  0.674 a normal predicts: pure sampling noise, no definitional drift.
- The reconstructed `med` pool reproduces the base rate where it must —
  `dir00` draws true-p mean 0.386 against the pool's 0.390 — while `total`
  draws 0.694, so the index space is right and the arms genuinely differ.

### How much of the epistemic variance is sampling noise

Compare like with like: both terms at `ddof=1`, so only the MC correction
`mean_m[p_m(1−p_m)]/(K−1)` differs. On `fm_high`, epochs 0–3:

| K | correction as a fraction of the observed between-member variance |
|---|---|
| 100 (these maps) | 0.11 – 0.65 |
| **20 (what acquisition actually used)** | **0.55 – 3.40** |

At K=20 the sampling-noise term is comparable to, and at some epochs **several
times larger than**, the entire measured between-member variance — so the
`epi_var` arm was ranking largely, at points entirely, on Monte-Carlo noise.
That is the mechanism behind the campaign's null, now visible per state rather
than as a pool average.

**Do not read the two variance-epistemic columns as before/after correction.**
`Var epistemic` uses `ddof=0` so the law-of-total-variance identity holds
exactly against the panels beside it; the debiased column uses `ddof=1` minus
the correction because that is precisely the acquisition score. With M=5 the
`ddof` change alone scales by 5/4, so at K=100 the debiased column is sometimes
the *larger* of the two. They answer different questions.

## Reproducing

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/code

# 1. per-member probabilities  (classifier: CPU, minutes; flow matching: GPU)
python analysis/uncertainty_maps/compute_members.py --pred clf --level high --out .../members --device cpu
bash   analysis/uncertainty_maps/launch_fm_members.sh          # 10 SLURM jobs, ~1 h each

# 2. acquired states (CPU, seconds)
bash analysis/uncertainty_maps/run_acquired_all.sh

# 3. figures
bash analysis/uncertainty_maps/run_all_maps.sh
```

`compute_members.py` records `marginal_drift_max` in every npz: the largest gap
between the recomputed ensemble mean and the stored `p_success`. For the
classifier this is a deterministic forward pass and the drift is **0.0000** at
every arm and epoch — a direct check that the checkpoints reload as the model
the evaluation actually used. For flow matching the drift is nonzero by
construction: the recompute draws fresh Monte-Carlo endpoint samples.

## Conventions worth knowing

- **Binary, not ternary.** Flow-matching eval assigns success / failure /
  invalid, but the acquisition backend scores success-vs-not-success
  (`EnsembleEndpointMCProbabilityBackend.estimate_members` counts only
  `classify_attractor == 1`). Every uncertainty quantity here follows that
  binary convention so the maps show what acquisition actually ranked on.
  `p_invalid` is plotted as its own column rather than renormalised away.
- **K = 100** for the recompute, matching `num_mc_samples_eval`. Acquisition
  itself ran at K = 20, where the MC floor is 5x higher — the reason the
  campaign's epistemic signal sat at or below its own noise floor.
- Acquired states come from `acquisition.d2_indices` read back through the run's
  own data source, in `start` candidate mode (no run overrides it), so the index
  space cannot drift from what the run wrote.
