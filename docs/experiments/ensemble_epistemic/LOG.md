# Ensemble Epistemic Acquisition — Experiment Log

Design: `docs/superpowers/specs/2026-08-04-ensemble-epistemic-acquisition-design.md`
Machine-readable run records: `runs.jsonl` (see `scripts/exp_log.py`).

Append newest entries at the top. Record what was launched, what broke, what was
decided, and why — the "why" is the part that is impossible to reconstruct later.

## 2026-08-04 — plan approved, implementation started

## 2026-08-04 03:00-04:00 — FM arms relocated to iLab; numpy shadow found and fixed

The five FM arms never started on Amarel. Their estimated start slipped from
03:19 to 11:21 the following day: five-GPU jobs on a contended cluster under the
preemptible `general` account simply do not schedule.

Freed capacity on both clusters by cancelling the previous stochastic-pendulum
campaign's still-running arms, after re-running `scripts/refresh_stoch_compare.sh`
so its deliverable was captured at maximum depth first. Kept the arms backing its
one unsettled question, xhigh-FM: `sc_fm_xhigh_dir00` + `sc_fm_xhigh_tb10` on
Amarel and `sc_fm_xhigh_ent10` on iLab. Cancelled 9 on Amarel and 7 on iLab, all
backing verdicts already computed (med-FM, high-FM, med/high/xhigh-CLF).

A canary arm on 4x a100 then exposed a bug that would have wasted the whole
allocation. Two numpy 2.2.6 installs coexist on iLab — the env's own and one in
`~/.local` — and the user-site copy shadows the env's. Every `mp.spawn` child
re-imports the main module, numpy's C extension initializes twice in one process,
and the child dies with "CPU dispatcher tracer already initlized". The failure is
silent: members created their checkpoint dirs and printed epoch-0 validation, then
died, leaving the GPUs at 0% and the parent blocked in `mp.spawn(join=True)` with
`squeue` still reporting R. Fixed with `PYTHONNOUSERSITE=1` in both job templates
(commit bd6a80f), and verified by relaunch: 0 dispatcher errors, 5 GPU processes
per arm, checkpoints written.

Relaunched at **2 GPUs per arm, not 5**. Members train concurrently and a pendulum
flow matcher only drives an a4500 to ~25%, so 2-3 members share a card cheaply.
All five arms then fit in 10 of the 12-GPU quota and advance together — which is
what the matched-epoch comparison needs, since its common depth is bounded by the
slowest arm. a4500 is requested explicitly so the arms do not scatter across
iLab's mixed pool at different speeds.

Running (iLab, seed 42, high): epi_var 199892, total 199894, dir00 199895,
epi_bald 199896, aleat 199897. The 25 CLF arms continue on Amarel and arrakis.

## 2026-08-04 05:00-06:00 — final review, two guard fixes, FM restart

Whole-branch review: **0 Critical, 9 Important, 9 Minor**, 199 tests passing. It
confirmed directly that the estimators, mode dispatch, eval member-weighting and arm
configs are correct, and that members are never sampled anywhere.

Fixed and committed (ee40097), then restarted all five FM arms on the fixed code —
they were still inside adaptive epoch 0, so nothing was lost:

- The domain guard used `np.nanmin`/`np.nanmax`, which ignore NaN, so a NaN
  probability returned NaN instead of raising. `select_greedy` skips non-finite
  scores, so one diverged member could have silently emptied an epoch's adaptive
  batch while the run reported success.
- `epistemic_variance` never calls `binary_entropy`, so `epi_var` — the flagship
  arm — was the one unguarded mode. Validation moved into `_check`, covering all four.
- OMP/MKL thread pools capped in the iLab template. Each torch process opened a pool
  sized to the node's core count; 5 members x several arms per node exhausted the
  thread limit and killed the dir00 control arm at 15 minutes with "can't start new
  thread", surfaced as "DataLoader worker exited unexpectedly". Verified fixed:
  dir00 v2 passed 16:51 clean.

Also fixed: `predictor.lightning_trainer.enable_progress_bar` is not in the classifier
config struct, so Hydra rejected it outright. It killed the seed replicate and would
have killed all four CLF resubmissions. `BayesianMLPTrainer` already hardcodes
`enable_progress_bar=False`, so the override was redundant as well as invalid.

### Open caveats the analysis must account for (deliberately NOT changed mid-campaign)

- **FM ensemble hardcodes `p_invalid = 0`.** `estimate_members` folds `classify_attractor`'s
  0 (outside every attractor) into `1 - p`, so `ConfidencePairFilter` classifies high-invalid
  states as FAILURE and drops them, where the single-model FM kept them UNKNOWN. All five FM
  arms share this, so **within-campaign comparisons are unaffected**; it makes the FM ensemble
  non-comparable to the previous single-model FM campaign.
- **`fm_ensemble.yaml` uses max_epochs 200 vs generative.yaml's 1000** (and adds EarlyStopping
  min_delta). Again shared by all five arms. It does confound the design's secondary
  "ensemble-total vs single-total measures pure ensembling" comparison — raising it to 1000
  would roughly 5x the campaign's runtime, so that comparison is abandoned rather than bought.
- **`epi_var` is unbiased in the mean but not under argmax.** Its sampling variance is
  proportional to p(1-p)^2 — largest exactly where aleatoric uncertainty is — so with
  n_candidates=50000 the selected tail sits ~4 sigma out and, once genuine disagreement decays
  below that scale, `epi_var` preferentially selects high-p(1-p) states. Check
  `epistemic_mean_selected` / `aleatoric_mean_selected` (already recorded per epoch) before
  concluding that `epi_var` ~ `epi_bald` means the BALD bias did not matter.
- **The FM acquisition path has still never executed.** The smoke test covers CLF only; the
  arms are inside adaptive epoch 0's member training, so `estimate_members`, `_load_member`
  and the FM `ConfidencePairFilter` first run on real models in ~a day, on all five arms at
  once. Highest-value remaining pre-emptive check.
- Latent, not live: FM warm start is a silent no-op (path mismatch) and would collapse the
  ensemble if naively fixed, since all members would resume from one checkpoint — guarded only
  by `warm_start: false`. `K % M == 0` is load-bearing for threshold/calibration and asserted
  nowhere (currently 10 % 5 and 100 % 5, both fine).
