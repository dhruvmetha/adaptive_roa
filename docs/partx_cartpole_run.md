# Part-X cartpole: first real run

First non-smoke run of the Part-X GP + level-set BO pipeline on the real
`cartpole_pybullet` dataset (4D state: `x, theta, x_dot, theta_dot`; upright
attractor `[0,0,0,0]`, interior to `theta in [-pi, pi]` — no circular seam
issue for this system). Composition mirrors `scripts/run_adaptive.py`
(`config_name="default"` plus Hydra group overrides — there is no
`configs/adaptive_v2/experiment/` group in this repo):

```
system=cartpole_pybullet predictor=gp acquisition=partx eval=partx
n_epochs=6 samples_per_epoch=50 initial_train_size=150 device=cuda:0
predictor.gp.n_iters=300 acquisition.n_candidates=2000 acquisition.bounds.R=200
output_dir=outputs/partx_cartpole_dev
```

Run wall-time: ~77 s on one GPU (RTX 2080 Ti, 11 GB) — no need to tune the
budget down further. All 6 epochs completed; each
`outputs/partx_cartpole_dev/epoch_00N/artifacts_v2.json` carries
`acquisition.diagnostics.roa_volume` + `roa_volume_ci` and a
`d1_eval_metrics.coverage`.

## Per-epoch results

| epoch | roa_volume | 90% CI            | leaves (total / remaining) | test_coverage | full-test-set F1 / acc |
|:-----:|:----------:|:------------------|:---------------------------:|:-------------:|:-----------------------:|
|   0   |   0.2824   | [0.2422, 0.3207]  |           2 / 2              |    0.9429     |     0.7914 / 0.9243     |
|   1   |   0.2431   | [0.2107, 0.2773]  |           4 / 4              |    0.9333     |     0.8549 / 0.9487     |
|   2   |   0.2397   | [0.2187, 0.2579]  |           8 / 8              |    0.9818     |     0.9031 / 0.9662     |
|   3   |   0.2154   | [0.2031, 0.2285]  |          16 / 16              |    1.0000     |     0.9196 / 0.9724     |
|   4   |   0.1947   | [0.1850, 0.2041]  |          32 / 32              |    0.9867     |     0.9169 / 0.9717     |
|   5   |   0.1996   | [0.1934, 0.2061]  |          64 / 64              |    0.9647     |     0.9350 / 0.9776     |

`test_coverage` is `d1_eval_metrics.coverage` (conformal q_hat calibration,
α=0.1, target ≥0.9). `full-test-set F1/acc` is the GP classifier's `λ±δ`
metrics against the full 115,242-row `cartpole_pybullet/test_set.txt` — a
separate, non-tree diagnostic reported by `evaluate_full_roa_classifier`, not
`roa_volume`.

No region-tree PNG: `PartXEvaluator` only renders `partx_region_tree.png`
when `system.state_dim == 2` (see `adaptive_roa/partx/eval.py`); cartpole is
4D so this plot is correctly skipped.

## Sanity notes

- **roa_volume trends down and the CI tightens monotonically.** Volume falls
  from 0.28 (2-leaf tree, epoch 0) to ~0.20 by epoch 4–5, and CI width shrinks
  from 0.0785 to 0.0127 as the tree subdivides. Epoch 4 (0.1947) and epoch 5
  (0.1996) overlap within each other's CI, suggesting the estimate is
  beginning to stabilize around ~0.19–0.20, but 6 epochs is not enough to
  call this converged (contrast with pendulum, which needed all 6 epochs to
  go from ~0.34 to ~0.39–0.40 and was still moving at epoch 5).
- **Conformal coverage holds the ≥1−α=0.9 guarantee every epoch** (range
  0.933–1.0), same qualitative behavior as pendulum. Unlike pendulum's run
  (which saturated at exactly 1.0 from epoch 2 on), cartpole's coverage
  fluctuates around 0.93–1.0 without pinning — i.e. less conservative /
  closer to nominal, though the calibration sets are small (`n=10` d1 cal +
  25 d1 test-ish rows growing to 85 by epoch 5) so these are noisy estimates,
  not a tight verification of the guarantee.
- **4D needs far more leaves than pendulum's 35, and none of them resolve
  within this budget.** Leaf count exactly doubles every epoch (2, 4, 8, 16,
  32, 64) because `n_remaining_leaves == n_leaves` at every epoch — i.e. every
  single leaf is still classified `'r'` (unresolved/straddling) by
  `classify_region`, so all of them get split again next epoch. This is a
  real qualitative difference from pendulum, where by epoch 5 the tree had
  resolved 11 of 35 leaves to `+`/`-` (24 remaining). The likely cause is
  dimensionality: `classify_region` requires the GP's alpha-quantile lower/
  upper confidence bound over MC points *within a region* to cross zero
  robustly, and a 4D region at this leaf count (64 leaves covering a 4D box)
  is far coarser per-unit-volume than a 2D region at 35 leaves covering a 2D
  box, so posterior variance within each region stays too high to classify
  confidently at only ~150–360 training points. This should ease with more
  epochs (more tree refinement + a genuinely growing training set — 13,280 to
  40,440 classification rows over the 6 epochs), but 6 epochs was not enough
  to see the first `+`/`-` leaf appear.
- **The GP classifier itself is learning fine, independent of the tree.**
  Full-test-set F1 rises from 0.79 to 0.94 and accuracy from 0.92 to 0.98
  across the 6 epochs (evaluated directly against the 115,242-row test set,
  not tree-mediated). This decouples two things that could otherwise be
  conflated: the *point-prediction* classifier is accurate and improving as
  expected; it's specifically the *tree's conservative region-level
  confidence test* that hasn't cleared its bar yet in 4D at this budget. That
  is expected/honest behavior for the (alpha=0.05, m_class=64) settings
  inherited unchanged from the pendulum config — not a 4D-specific bug.

## Honest caveats

- This was a deliberately modest, time-bounded budget (6 epochs, 50
  samples/epoch, 2000 candidates, R=200 for the volume MC) — the same budget
  used for the pendulum run, on a system with 2x the state dimensionality.
  The `roa_volume` numbers here are a first-run qualitative sanity artifact,
  not a converged/production estimate; a longer run (more epochs and/or a
  larger initial train size) would very likely resolve some leaves to
  `+`/`-` and materially change both the volume estimate and its CI.
- Since **all** leaves are unresolved at every epoch, the leaf-count column
  should be read as "tree refinement progress" only — it does not yet tell
  us anything about basin shape (no `+`/`-` regions to visualize or discuss),
  unlike the pendulum run's tree, which had begun carving out in-basin vs.
  out-of-basin leaves by epoch 2.
- `eval.base.decision_rule` resolves to `two_sided` for cartpole (system
  default) vs. `one_sided` for pendulum, because `configs/adaptive_v2/eval/
  partx.yaml` reads the top-level `${decision_rule}`, and `predictor/gp.yaml`
  only overrides `threshold.decision_rule` / `calibration.decision_rule` (not
  the top-level key). This is a latent config inconsistency, not a bug in
  practice: `evaluate_full_roa_classifier` always sets
  `p_failure = 1 - p_success` for a binary GP classifier, and with that
  identity `_predict_lambda_delta`'s `two_sided` and `one_sided` branches are
  provably equivalent (the "both flagged" ambiguous case requires
  `p_success > lambda*+delta AND p_success < lambda*-delta`, which is empty
  since `delta >= 0`). Confirmed by inspection, not just by these run
  numbers looking sane — flagging here in case a future non-GP cartpole
  predictor path relies on a genuinely independent `p_failure`.
- No region-tree PNG (4D, correctly skipped — see above).

Raw artifacts (git-ignored):
`outputs/partx_cartpole_dev/epoch_00N/artifacts_v2.json`,
`final_results.json`.
