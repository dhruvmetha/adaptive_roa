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

## 2026-08-04 06:00-06:30 — thread-limit fix, FM smoke test, old campaign closed out

**FM smoke test passes for all five arms.** `tests/adaptive_v2/test_fm_decomposition_smoke.py`
runs two adaptive epochs per arm on a real flow-matching ensemble, so `estimate_members`,
`_load_member` and the decomposition strategy are now exercised against real models rather
than fakes. This was the largest remaining unknown: the live arms first reach that code about
a day into training, on all five arms at once.

**The thread-limit diagnosis was wrong the first time.** Capping OMP/MKL did not fix it; three
arms died again at 25-35 minutes with "can't start new thread" and
"BlockingIOError: [Errno 11] Resource temporarily unavailable". The real limit is
`ulimit -u` = 2000, which on Linux counts THREADS per user per node across all jobs. One arm is
~570 threads (5 members x train+val DataLoader workers); SLURM packs four 2-GPU arms onto one
8-GPU node. OMP caps shrink each process's pool, not the process count. `num_workers` 4 -> 1
cut it to ~316/arm; measured 1278/2000 with four arms co-located. All five arms relaunched
together (v3) to keep them config-identical.

**Old campaign closed.** Its last three arms were being kept alive for the one unsettled
question, xhigh-FM. Refreshing the deliverable showed why that was futile: verdicts exist for
med-FM, med-CLF, high-FM, high-CLF and xhigh-CLF, but **no xhigh seed replicates were ever
run**, so xhigh-FM has no run-to-run floor and the verdict is uncomputable regardless of how
long those arms train. Cancelled all three and redirected the capacity to seed replicates for
*this* campaign, which had exactly one.

Replicates now running for the run-to-run floor: `clf_high_dir00_s43` (199906),
`clf_high_dir00_s44` (199914), `clf_xhigh_dir00_s43` (60082690), `clf_xhigh_dir00_s44`
(60082691). Without these no verdict in this campaign is declarable -- the same rule that
reversed three conclusions last time.

## 2026-08-04 06:45 — first readout: the deterministic validation case does NOT behave as designed

The deterministic-pendulum CLF arms are nearly complete (`det_epi_bald` at 19/19). The design
set this up as the validation case: "on the deterministic pendulum aleatoric ~ 0, so `epi_*` and
`total` should select nearly the same points... Divergence there means the decomposition measures
something other than what it claims."

Per-epoch acquisition diagnostics, deterministic pendulum, seed 42 (enrich = selected / pool):

| arm | ep | epi_pool | ale_pool | epi_sel | ale_sel | enrich_epi | enrich_ale |
|---|---|---|---|---|---|---|---|
| total    | 5  | 0.00419 | 0.00518 | 0.10459 | 0.15958 | 25.0 | 30.8 |
| total    | 10 | 0.00114 | 0.00247 | 0.03422 | 0.09047 | 30.0 | 36.6 |
| epi_var  | 5  | 0.00293 | 0.00553 | 0.08175 | 0.16575 | 27.9 | 30.0 |
| epi_var  | 10 | 0.00139 | 0.00245 | 0.04251 | 0.07409 | 30.7 | 30.3 |
| epi_bald | 5  | 0.00616 | 0.00664 | 0.16176 | 0.20422 | 26.3 | 30.8 |
| epi_bald | 10 | 0.00158 | 0.00209 | 0.04448 | 0.05731 | 28.1 | 27.4 |
| aleat    | 5  | 0.00698 | 0.00916 | 0.12400 | 0.26956 | 17.8 | 29.4 |
| aleat    | 10 | 0.00107 | 0.00194 | 0.04719 | 0.06220 | 43.9 | 32.1 |

Two things, consistent across all four arms and both epochs:

1. **Aleatoric is not ~0 on a deterministic system — it consistently EXCEEDS epistemic**
   (e.g. `total` at epoch 10: 0.00247 vs 0.00114, a factor of 2.2). The design's premise for this
   validation case is false as stated.
2. **No arm separates the two components in what it actually selects.** Every arm enriches BOTH
   terms by ~25-35x, including the ones optimising the opposite quantity: `aleat` enriches
   epistemic 17.8-43.9x, and `epi_var`/`epi_bald` enrich aleatoric 27-32x. The enrichment ratios
   are near-identical regardless of which term the arm maximises.

The likely mechanism is that for a classifier ensemble, "aleatoric" = mean per-member binary
entropy, and near a decision boundary a well-trained member outputs p ~ 0.5. That registers as
aleatoric even when the underlying dynamics are perfectly deterministic. So on a deterministic
system this term measures **boundary/representational** uncertainty, not irreducible process
noise — the two are not distinguishable by this decomposition at a separatrix.

This is the same coupling the final review raised independently (finding 7: `epi_var`'s selection
noise is proportional to p(1-p)^2, i.e. largest exactly where aleatoric is largest).

**Status: strong preliminary signal, not a verdict.** It is one seed, classifier only, and the
run-to-run floor (`clf_det_dir00_s43/s44`, jobs 60089992/60089993) is still training. The
direction is consistent across 8 arm-epoch observations, but the campaign rule stands: no verdict
without the floor and two consecutive epochs. The stochastic arms are where the decomposition is
actually supposed to earn its keep, and they are still running.

## 2026-08-04 06:50 — the high-noise readout, and a correction to the entry above

Same diagnostics at **stochastic high noise**, epoch 5, the regime the decomposition was
designed for:

| arm | epi_pool | ale_pool | epi_sel | ale_sel | enrich_epi | enrich_ale |
|---|---|---|---|---|---|---|
| total    | 0.00834 | 0.16879 | 0.03068 | 0.60896 | 3.7 | 3.6 |
| epi_var  | 0.00498 | 0.18416 | 0.02270 | 0.55910 | 4.6 | 3.0 |
| epi_bald | 0.00510 | 0.21428 | 0.02865 | 0.44787 | 5.6 | 2.1 |
| aleat    | 0.00823 | 0.20056 | 0.01805 | 0.63568 | 2.2 | 3.2 |

**The decomposition works here, and the arms order exactly as designed.** Ranked by epistemic
enrichment: `epi_bald` 5.6 > `epi_var` 4.6 > `total` 3.7 > `aleat` 2.2. Ranked by aleatoric
enrichment the order inverts: `aleat` 3.2 > `total` 3.6/`epi_var` 3.0 > `epi_bald` 2.1. The
epistemic arms buy epistemic enrichment while *suppressing* aleatoric relative to `total`, and
the negative control does the opposite. That is the intended behaviour, demonstrated causally.

It also confirms the motivating diagnosis: at high noise the pool is overwhelmingly aleatoric —
`ale_pool` ~ 0.17-0.21 against `epi_pool` ~ 0.005-0.008, a ratio of 20-40x. Total entropy at
this noise level is almost entirely irreducible, which is why acquiring on it degenerates.

### Correction to the deterministic entry

The entry above overstated the deterministic result by calling the design's premise "false as
stated". That framing was wrong on the absolute scale. Deterministic `ale_pool` is 0.002-0.009
against 0.17-0.21 at high noise — 20-100x smaller, so aleatoric IS nearly zero there in absolute
terms, as the design predicted. And the design's actual operational prediction for that case,
that all arms select nearly the same points, HOLDS: every arm enriches both terms ~25-35x.

What the deterministic numbers genuinely show is narrower: the small residual uncertainty on a
deterministic system is boundary-driven, and at a separatrix the per-member entropy term and the
between-member disagreement term move together, so the decomposition cannot separate them there.
That is a caveat about what "aleatoric" means near a decision boundary, not a refutation of the
method — and it is consistent with the high-noise result, where genuine process noise creates
aleatoric mass that IS separable.

Both readouts remain preliminary: one seed, classifier only, floors still training.
