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

## 2026-08-04 06:55 — selectivity is stable across three consecutive epochs

Selectivity ratio = enrich_epi / enrich_ale, stochastic high noise, classifier, seed 42.
A ratio > 1 means the arm preferentially acquires epistemic over aleatoric mass.

| arm | ep3 | ep4 | ep5 |
|---|---|---|---|
| `epi_bald` | **3.25** | **3.32** | **2.69** |
| `epi_var`  | 2.08 | 2.33 | 1.50 |
| `total`    | 1.02 | 0.84 | 1.02 |
| `aleat`    | 0.57 | 0.53 | 0.69 |

The ordering `epi_bald` > `epi_var` > `total` > `aleat` holds at **every** epoch with no overlap
between arms. `total` sits at ~1.0 throughout — exactly what acquiring on the sum should do —
and the negative control is reliably below 1, i.e. it actively anti-selects epistemic mass.
Three consecutive epochs clears the campaign's two-epoch rule.

**`epi_bald` separates better than `epi_var`, and that is expected here.** For the classifier
both estimators are exact: `EnsemblePosterior` enumerates members rather than sampling, so K is
None and there is no finite-K noise for `epi_var`'s correction to remove. The debiasing that
motivates `epi_var` only buys anything where p_m is MC-estimated — i.e. flow matching at K=20,
which is still training. So this result does not yet speak to the estimator question the design
posed; it speaks to the decomposition working at all.

**What this does and does not establish.** It shows the decomposition is mechanically sound: the
score modes select what they claim to select, in the right order, with the negative control
inverted. It does NOT show that epistemic acquisition produces a better model — that requires the
downstream metrics at matched epochs against the run-to-run floor, and those arms and floors are
still training. Keep the two claims separate in the writeup.

## 2026-08-04 07:00 — the separation is a monotone function of noise

Selectivity ratio (enrich_epi / enrich_ale) at epoch 4, classifier, seed 42, across all five
noise levels. `ale_pool` is the mean aleatoric score over the candidate pool.

| level | ale_pool | `epi_bald` | `epi_var` | `total` | `aleat` |
|---|---|---|---|---|---|
| det   | ~0.005      | ~1.0 | ~1.0 | ~1.0 | ~1.0 |
| low   | 0.016       | 1.00 | 1.02 | 0.98 | 0.98 |
| med   | 0.034       | –    | 1.04 | 0.95 | 1.02 |
| high  | 0.17–0.21   | 3.32 | 2.33 | 0.84 | 0.53 |
| xhigh | 0.34–0.41   | **3.59** | **2.30** | 0.53 | 0.48 |

**At low, med and deterministic noise no arm separates anything** — every arm sits at ~1.0,
including the ones explicitly maximising epistemic. The arms only fan out once `ale_pool` exceeds
roughly 0.1, and the fan widens from high to xhigh.

The mechanism is the same one the deterministic case showed: when the pool carries little
aleatoric mass, what uncertainty exists is boundary-driven, and the per-member entropy term and
the between-member disagreement term move together, so there is nothing to pull apart. Genuine
process noise is what creates separable aleatoric mass.

**This lands exactly on the motivating problem.** The previous campaign measured entropy
acquisition *harming* the classifier at high (+0.157 debiased Brier) and xhigh (+0.088) while
leaving low and med alone. The decomposition does something only at high and xhigh — precisely
the regimes where total-entropy acquisition breaks — and does nothing where nothing was broken.
That is the correct shape for the fix to have.

It also sharpens the prediction to test once the metrics land: `epi_*` should beat `total` at
high and xhigh, and should be indistinguishable from it at low, med and deterministic. If `epi_*`
also "wins" at low/med, that is a red flag for the analysis rather than a bonus, because these
diagnostics say the arms are selecting near-identically there.

Still selection behaviour only — no downstream metric claim until the floors finish.

## 2026-08-04 07:20 — epoch-0 identity confirmed empirically (FM)

The design lists "epoch-0 identity: all five arms score identically before the first
acquisition" as an integration check. It holds, visibly, in the FM arms' checkpoints: at
adaptive epoch 0 every arm's members converge at the *same* lightning epochs — best checkpoints
at 95, 111, 125, 130 across `epi_var` and `aleat` alike. All arms start from the same seed-42
random training set, so they train identical ensembles until the first acquisition differentiates
them. Any divergence at epoch 0 would have meant an arm was leaking acquisition state into
training; there is none.

### Correction on how to judge FM liveness

Earlier I recorded "newest .ckpt mtime" as the liveness check. That is too noisy to rely on:
`last.ckpt` only updates when Lightning actually saves, and as a member converges the saves become
sparse, so individual members legitimately go 10-35 minutes without writing while still training.
It briefly looked like the `aleat` arm had stalled; it had not, and `srun --jobid=<id> nvidia-smi`
confirmed all five member processes alive in every arm.

The reliable signal is the **max best-epoch number parsed from the checkpoint filenames**
(`best-{epoch}-{val_loss}.ckpt`) advancing over time, cross-checked with the live member-process
count. Measured pace: max best-epoch 112 -> 134 over 17 minutes, and a member at 191/200 by 07:20
— roughly 2.5-3 h per adaptive epoch, so ~2 days for 19 epochs.

## 2026-08-04 07:35 — FIRST DOWNSTREAM READOUT (classifier): epi_bald rescues xhigh, but is worst at high

Analysis pipeline validated end-to-end on the new campaign: `sync_amarel_results.sh` gained a
no-delete `ensemble_epistemic` entry (the directory has local writers — `fm_high_*` on iLab and
`clf_high_*` on arrakis — so `--delete` would have erased every locally-run arm), and
`stoch_prob_metrics.py` scores the merged tree unchanged. 132 rows, 28 stochastic runs.

**The deterministic arms cannot be scored this way.** The metrics need ground-truth `p_success`
from repeated rollouts (`noisy/pendulum/lqr/<level>/eval_success_prob.npz`); there is no `det`
entry, and a deterministic system's p_success is degenerate {0,1} rather than a rollout
frequency. The det arms must be judged on the label metrics already in `artifacts_v2.json`.

Debiased Brier per epoch (lower is better):

**xhigh**
| arm | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 |
|---|---|---|---|---|---|---|
| `dir00` (control) | 0.0839 | 0.0728 | 0.0677 | 0.0684 | 0.0714 | – |
| `epi_bald` | 0.0932 | 0.0867 | **0.0713** | **0.0757** | **0.0795** | 0.0825 |
| `epi_var` | 0.1050 | 0.1087 | 0.1087 | 0.1121 | 0.1173 | 0.1259 |
| `aleat` | 0.1284 | 0.1449 | 0.1597 | 0.1677 | 0.1661 | – |
| `total` | 0.1361 | 0.1486 | 0.1613 | 0.1722 | 0.1779 | 0.1619 |

**high**
| arm | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| `dir00` (control) | 0.0639 | 0.0594 | 0.0579 | 0.0591 | 0.0532 |
| `aleat` | 0.0564 | 0.0971 | 0.1029 | 0.1159 | 0.1277 |
| `total` | 0.0683 | 0.1019 | 0.1092 | 0.1398 | 0.1265 |
| `epi_var` | 0.0894 | 0.0980 | 0.1184 | 0.1388 | 0.1571 |
| `epi_bald` | 0.0953 | 0.1365 | 0.1802 | 0.2023 | **0.2297** |

Two findings, both stable over 5-6 consecutive epochs rather than single-epoch flukes:

1. **At xhigh `epi_bald` essentially eliminates the harm.** It tracks the non-adaptive control
   within ~0.008-0.011 while `total` runs +0.107 worse. That is the designed outcome, and the
   negative control `aleat` sits with `total` at the bottom as it should.
2. **At high `epi_bald` is the WORST arm** — worse than `total` and worse than the negative
   control, and monotonically diverging (0.095 -> 0.230). The ordering is the near-inverse of
   xhigh. This is not noise; it is monotone across every epoch.

That inversion is the interesting result and it is not yet explained. The selectivity diagnostics
show `epi_bald` has the highest epistemic enrichment at BOTH levels (3.3 at high, 3.6 at xhigh),
so identical selection behaviour produces opposite downstream effects. A plausible reading is that
epistemic-only acquisition concentrates sampling in a narrow, unrepresentative region, which
degrades *calibration* even when it improves the decision boundary — and at xhigh the cost of
drowning in aleatoric mass simply outweighs it. `sAUROC` is nearly flat across arms at high
(0.9755-0.9811) while Brier ranges 0.053-0.230, which is consistent with a calibration effect
rather than a ranking effect. Worth testing directly.

Also note: **every adaptive arm is worse than the non-adaptive control at both levels.** That
reproduces the previous campaign's core finding rather than overturning it; the question here is
only whether the epistemic split reduces the damage, and at xhigh it clearly does.

### This is NOT a verdict yet

- The run-to-run floor is **not** established at these epochs. The `dir00_s43/s44` replicates have
  only reached epochs 0-1, and a floor at epoch 0 is meaningless because all arms share the same
  seed-42 data there by construction. Everything above is one seed per arm.
- Epochs 5-7 of 19. The previous campaign's verdicts landed at epochs 8-18, and several reversed
  before then.
- Effect sizes are large (4x at high) relative to the old campaign's ~0.015 CLF floor, so the
  direction is unlikely to be pure noise — but "unlikely" is not the standard this campaign set.

## 2026-08-04 07:35 — mechanism: the high-noise "failure" is pure miscalibration, and it is the fixable part

Murphy decomposition of the same epoch-5 numbers (Brier = REL - RES + UNC; REL is
miscalibration, lower better; RES is resolution, higher better):

**high**
| arm | REL_deb | RES | Brier | sAUROC |
|---|---|---|---|---|
| `dir00` | 0.0512 | 0.1896 | 0.0532 | 0.9811 |
| `total` | 0.1217 | 0.1873 | 0.1265 | 0.9799 |
| `epi_var` | 0.1541 | 0.1890 | 0.1571 | 0.9809 |
| `epi_bald` | **0.2257** | 0.1882 | 0.2297 | 0.9797 |
| `aleat` | 0.1175 | 0.1817 | 0.1277 | 0.9755 |

**xhigh**
| arm | REL_deb | RES | Brier | sAUROC |
|---|---|---|---|---|
| `dir00` | 0.0698 | 0.1322 | 0.0714 | 0.8976 |
| `total` | 0.1460 | **0.1021** | 0.1779 | 0.8546 |
| `epi_var` | 0.1159 | 0.1325 | 0.1173 | 0.8977 |
| `epi_bald` | 0.0787 | 0.1330 | 0.0795 | 0.8984 |
| `aleat` | 0.1423 | **0.1102** | 0.1661 | 0.8695 |

**At high, RES is identical across every arm** (0.1817-0.1896) and sAUROC is flat
(0.9755-0.9811), while REL spans 0.051-0.226. The entire Brier spread — including `epi_bald`
looking like the worst arm — is *miscalibration*. No arm loses any discriminative power.

**At xhigh the damage is different in kind.** `total` and `aleat` lose resolution outright
(RES 0.102/0.110 against the control's 0.132) *and* calibration, and their sAUROC drops to
0.855/0.870. The epistemic arms preserve both (RES 0.133, sAUROC 0.898).

### This reverses the apparent conclusion

Judged on a calibration-free metric, `epi_bald` **never hurts and sometimes helps**:

| level | sAUROC: control | `total` | `epi_bald` |
|---|---|---|---|
| high  | 0.9811 | 0.9799 | 0.9797 (tied) |
| xhigh | 0.8976 | 0.8546 | **0.8984** (best, beats control) |

So "epi_bald is the worst arm at high" is true only of calibration, which is the component
routinely repaired post hoc — and this codebase already fits λ*/δ* and q_hat per epoch.
Resolution loss is not repairable by recalibration. On that reading the epistemic split does
exactly what it was designed to do: it protects the *irrecoverable* component at xhigh, where
acquiring on total entropy destroys it.

The likely mechanism for the calibration hit is covariate shift: concentrating acquisition in a
narrow epistemically-uncertain region pulls the training marginal away from the evaluation
marginal, so probabilities are skewed even though the ranking is intact. That is consistent with
the previous campaign's measurement that scored acquisition dragged the training marginal from
~41% success to ~5% at high.

**Caveats unchanged and still binding:** one seed per arm, epochs 5-7 of 19, and the run-to-run
floor is still not available at these epochs (replicates at epoch 3-4). Do not promote any of
this to a verdict yet. The concrete next test is to compare arms *after* recalibration, which
would confirm or kill the "calibration is fixable" reading directly.

## 2026-08-04 07:40 — on calibration-free metrics the epistemic arms beat total entropy decisively at xhigh

Both calibration-free components across every available epoch. Nothing here is a single-epoch
reading.

**xhigh — sAUROC (higher better)**
| arm | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 |
|---|---|---|---|---|---|---|
| `dir00` (control) | 0.8947 | 0.8949 | 0.8951 | 0.8971 | 0.8976 | – |
| `epi_bald` | **0.8966** | **0.8971** | **0.8977** | **0.8980** | **0.8984** | **0.8988** |
| `epi_var` | 0.8956 | 0.8963 | 0.8971 | 0.8978 | 0.8977 | 0.8983 |
| `aleat` | 0.8880 | 0.8806 | 0.8802 | 0.8656 | 0.8695 | – |
| `total` | 0.8847 | 0.8806 | 0.8758 | 0.8602 | 0.8546 | 0.8768 |

**xhigh — RES (resolution, higher better)**
| arm | ep1 | ep3 | ep5 | ep6 |
|---|---|---|---|---|
| `dir00` | 0.1294 | 0.1312 | 0.1322 | – |
| `epi_bald` | **0.1323** | **0.1328** | **0.1330** | **0.1331** |
| `epi_var` | 0.1308 | 0.1321 | 0.1325 | 0.1328 |
| `aleat` | 0.1252 | 0.1171 | 0.1102 | – |
| `total` | 0.1235 | 0.1138 | 0.1021 | 0.1145 |

Two claims, of very different strength:

**Strong and unambiguous — the epistemic arms beat `total` at xhigh.** `epi_bald` leads `total`
by +0.022 to +0.044 sAUROC and holds RES at 0.133 while `total` collapses to 0.102. Monotone and
consistent at every epoch, with the negative control `aleat` tracking `total` down as designed.
`total` and `aleat` lose discriminative power that recalibration cannot restore; the epistemic
arms do not lose it at all.

**Weak — `epi_bald` also edges the non-adaptive control at xhigh.** It is above `dir00` at all
six epochs on sAUROC and all four on RES, but by only +0.001 to +0.002, which is small enough
that the run-to-run floor could absorb it. Consistency across 6/6 epochs is suggestive, not
sufficient. Do not claim "adaptive beats random" until the replicates reach these epochs.

**high is a null on these metrics.** sAUROC spans 0.977-0.981 across all five arms and RES
0.180-0.190; the control is marginally best and `epi_var` closest to it. So the dramatic Brier
spread at high (0.053 to 0.230) is *entirely* calibration, as the decomposition said — no arm
gains or loses meaningful discriminative power there.

Net: the earlier reading that "`epi_bald` is the worst arm at high" was a calibration artifact
and should not be carried forward. On the component that recalibration cannot repair, `epi_bald`
is the best arm at xhigh and tied at high.

## 2026-08-04 07:40 — FIRST FM ACQUISITION, and naive BALD is 97% finite-K bias on real models

`fm_high_aleat` completed adaptive epoch 0 and fired the first FM acquisition. The path the
smoke test was built to de-risk ran clean on real flow matchers: `n_members = 5`,
`member_sample_size = 20`, 50 000 candidates scored, no errors, no NaN.

The diagnostics deliver the design's central empirical question immediately. `epistemic_mean` is
computed with **naive, uncorrected** `epistemic_bald` (decomposition.py:115), and on real models
at K=20, M=5 it reads:

| quantity | value |
|---|---|
| analytic finite-K BALD bias, (1/2K)(1−1/M) | 0.02000 |
| measured `epistemic_mean` on real FM | 0.01946 |
| **fraction of the measured signal explained by bias alone** | **97.3%** |
| implied true member disagreement | −0.00054 (i.e. ~0) |

The design predicted this bias analytically and measured it in synthetic trials (0.0206 at
p=0.5, 40k trials). It now reproduces on real flow matchers to within 3%. **Naive BALD on the FM
arms is measuring almost pure MC sampling noise, not genuine ensemble disagreement.**

The contrast with the classifier is exact and is the cleanest possible control: the classifier
enumerates its members (`K = None`, no sampling), so its BALD carries zero finite-K bias, and its
`epistemic_mean` at high noise reads 0.005–0.008 — real disagreement, five times smaller than the
FM figure that is almost entirely artefact.

**This is why `epi_var` exists, and the campaign is now positioned to settle it.** A near-constant
+0.02 offset does not reorder points within the interior, but it lifts every non-deterministic
state above every confidently-decided one regardless of whether members disagree — an
aleatoric-correlated preference. The prediction is therefore that on FM, `epi_bald` behaves like a
mildly aleatoric-seeking arm while `epi_var` does not, which is the opposite of what was observed
on the classifier (where `epi_bald` was the best-separating arm precisely because it is unbiased
there). The remaining four FM arms are at best-epoch 193/200 and will provide their own
diagnostics shortly.

Caveat on interpretation: at adaptive epoch 0 all five members train on the *same* seed-42 data
and differ only by initialisation, so genuine disagreement is at its structural minimum. The bias
fraction should fall as acquisition differentiates the arms. That it is 97% at epoch 0 is the
worst case, not the steady state — but it is also exactly the regime where acquisition decisions
are first being made.

## 2026-08-04 07:44 — decomposition identity verified exactly on real flow-matcher outputs

All four scoring FM arms have now written epoch-0 diagnostics, and because epoch 0 is
pre-acquisition they see byte-identical data. That makes it a free end-to-end check of the
implementation:

```
epistemic (BALD)      0.01945593111927176
aleatoric             0.19070883102554209
sum                   0.21016476214481386
total arm score_mean  0.21016476214481386
residual                          0.0e+00
```

`H(p̄) = E_m[H(p_m)] + I(y;m|x)` holds to **exact float precision** on real model outputs, not
just in unit tests. The `total` arm independently computed the left-hand side while `epi_bald`
and `aleat` computed the two right-hand terms, and they agree to the last bit. Combined with the
identical `epi`/`ale` values across all four arms, this re-confirms epoch-0 identity for the FM
half as well as the classifier half.

The two epistemic estimators on the same data:

| arm | score_mean | note |
|---|---|---|
| `epi_bald` | 0.019456 | naive MI; 97.3% is the analytic K=20 bias (0.02000) |
| `epi_var` | 0.001855 | debiased between-member variance |

The two are in different units (nats vs a variance), so the ratio is not meaningful on its own —
but both point the same way: genuine member disagreement at epoch 0 is very small, and the naive
estimator is reporting mostly sampling noise while the debiased one is not. This is precisely the
situation the design predicted for flow matching and did not expect for the classifier, and it
sets up the estimator comparison the campaign exists to settle.

## 2026-08-04 07:50 — prediction refuted: naive BALD still separates best on FM despite being 97% bias

Last entry predicted that on flow matching `epi_bald` would behave like a mildly
aleatoric-seeking arm, because 97.3% of its score is finite-K bias. **That prediction is wrong.**
First FM acquisition, epoch 0 (all arms see identical pre-acquisition data):

| arm | enrich_epi | enrich_ale | ratio |
|---|---|---|---|
| `epi_bald` | **3.41** | **1.82** | **1.87** |
| `epi_var` | 3.39 | 2.31 | 1.47 |
| `total` | 1.61 | 3.25 | 0.50 |
| `aleat` | 1.19 | 3.28 | 0.36 |

`epi_bald` has the highest epistemic enrichment *and* the lowest aleatoric enrichment of any arm
— the opposite of aleatoric-seeking. It also beats `epi_var` on separation (1.87 vs 1.47),
reproducing the ordering seen on the classifier.

The reason is the half of the design's own analysis I under-weighted when predicting: the bias is
**near-constant across the interior**, and a constant offset does not reorder points. It inflates
the *mean* of the score without reordering the ranking, and acquisition only ever uses the
ranking. So a score can be 97% artefact in magnitude and still select correctly. The design said
this explicitly — "a near-constant offset does not reorder points *within* the interior" — and
the concern it actually raised was narrower: that the offset lifts non-deterministic states above
confidently-decided ones. That specific effect is not visible here either; `epi_bald`'s aleatoric
enrichment is the lowest of the four.

What this does not yet settle: epoch 0 is the structural minimum for genuine disagreement (all
members share the same seed-42 data and differ only by initialisation), and the design's argument
is that the bias matters *most* late in training, once real disagreement decays toward the same
0.02 scale. The estimator question is therefore still open — but the mechanism by which `epi_var`
was expected to win has now failed its first direct test, and the debiased estimator is currently
the *worse* separator on both predictors.

## 2026-08-04 07:55 — floor established: the epistemic arms genuinely beat random sampling at xhigh

Three seeds of `dir00` (42/43/44) now overlap at epoch 1, giving the first real run-to-run floor.
Epoch 0 is excluded: all seeds share identical data there by construction, so its spread is
structurally zero and meaningless as a floor.

| level | metric | seed SD | 2*SD (decision threshold) |
|---|---|---|---|
| high  | debiased Brier | 0.00043 | 0.00087 |
| xhigh | debiased Brier | 0.00023 | 0.00046 |
| high  | sAUROC | 0.000254 | 0.000508 |
| xhigh | sAUROC | 0.000033 | 0.000065 |

The floor is far tighter than assumed. **sAUROC gaps vs the non-adaptive control:**

| level | arm | gap @ep1 | gap @ep5 | vs floor |
|---|---|---|---|---|
| xhigh | `epi_bald` | **+0.00195** | **+0.00089** | 14-30x floor, POSITIVE |
| xhigh | `epi_var` | +0.00093 | – | 14x floor, POSITIVE |
| xhigh | `total` | −0.01004 | – | 154x floor, NEGATIVE |
| xhigh | `aleat` | −0.00673 | – | 103x floor, NEGATIVE |
| high | `epi_var` | −0.00129 | – | 2.5x floor, negative |
| high | `epi_bald` | −0.00205 | −0.00143 | 2.8-4x floor, negative |
| high | `total` | −0.00380 | – | 7.5x floor, negative |

### Correction to the previous entry

Two entries ago I called "`epi_bald` also edges the non-adaptive control at xhigh" a **weak**
claim that "the run-to-run floor could absorb". It cannot. The xhigh sAUROC floor is 2*SD =
0.000065 and the effect is +0.0009 to +0.0020 — 14 to 30 times larger, positive at every epoch
measured. By this campaign's own stated rule the claim is distinguishable, and my hedge was too
conservative rather than too aggressive.

So at xhigh the epistemic arms do not merely limit the damage; they **beat random sampling** on
the metric recalibration cannot repair, while `total` and `aleat` are 100-150x the floor *worse*
than random.

At high, every adaptive arm is distinguishably worse than the control, but by only 2.5-7.5x the
floor — an order of magnitude smaller effect than at xhigh, and `epi_var` is the least harmful.

### The floor is still incomplete, and this is the important caveat

It comes from `dir00` replicates, whose data order is fixed, so it measures *training*
stochasticity only. Adaptive arms additionally vary in which points they acquire, and that
variance is not captured here. The previous campaign replicated both `dir00` and the adaptive arm
for exactly this reason. Replicates of `clf_xhigh_epi_bald` are being launched now; until they
land, the numbers above are compared against a floor that is a lower bound on the true one.

## 2026-08-04 08:05 — deterministic validation passes, and the full dose-response

### Deterministic pendulum (label metrics, matched epoch 15, all five arms)

| arm | AUC | Brier | log_score | accuracy |
|---|---|---|---|---|
| `dir00` (control) | 0.999912 | 0.00349 | 0.01195 | 0.99628 |
| `total` | 0.999995 | **0.00089** | 0.00318 | 0.99842 |
| `epi_var` | 0.999988 | 0.00142 | 0.00492 | 0.99828 |
| `epi_bald` | 0.999990 | 0.00132 | 0.00474 | 0.99866 |
| `aleat` | 0.999994 | 0.00129 | 0.00435 | 0.99926 |

**The design's validation case passes.** Every adaptive arm beats the non-adaptive control
(Brier 2.5-4x lower), and the four adaptive arms land within 1.6x of each other — exactly the
predicted "aleatoric ~ 0, so all arms select nearly the same points and land on nearly the same
metrics". AUC is saturated (spread 8e-05) and carries no signal here; Brier, log score and
accuracy still discriminate. Note this needs the label metrics: the probability metrics require
rollout ground truth that does not exist for a deterministic system.

### Full dose-response — debiased-Brier gap vs the control (negative = adaptive HELPS)

| level | ep | floor 2*SD | `total` | `epi_var` | `epi_bald` | `aleat` |
|---|---|---|---|---|---|---|
| det   | 15 | – | −0.0026 | −0.0021 | −0.0022 | −0.0022 |
| low   | 4  | 0.00024 | −0.00165 | −0.00269 | −0.00292 | −0.00411 |
| med   | 3  | (degenerate) | −0.00284 | −0.00019 | +0.00162 | −0.00179 |
| high  | 5  | 0.00087 | +0.07328 | +0.10392 | +0.17646 | +0.07448 |
| xhigh | 5  | 0.00046 | +0.10653 | +0.04589 | **+0.00815** | +0.09475 |

The sign flips with noise. At det and low, adaptive acquisition **helps** and the choice of score
barely matters. At high and xhigh it **hurts**, by two orders of magnitude more than it ever
helped — and only there does the choice of score matter, with `epi_bald` cutting the damage from
+0.107 to +0.008 at xhigh.

This reproduces and extends the previous campaign's finding. It measured entropy acquisition
harming the classifier at high/xhigh and not at low/med; the mechanism is now visible (aleatoric
mass dominating the pool above ale_pool ~0.1), and the fix works at xhigh.

`med`'s floor is degenerate — its replicates have only reached epoch 1, where all seeds still
share identical data — so the med row is not yet testable. Treat it as unresolved rather than
as a null.

## 2026-08-04 08:15 — HEADLINE: after recalibration the epistemic arms win, and at xhigh they beat random

The proposed test, run. `Brier = REL − RES + UNC`, so perfect recalibration drives `REL → 0` and
the achievable floor is `Brier_recal = UNC − RES` — precisely the part calibration **cannot**
repair. Both components are already in the scored metrics, so no new compute was needed.

**Epoch 5, gap vs the non-adaptive control:**

| level | arm | raw Brier gap | post-recal gap | shrinkage |
|---|---|---|---|---|
| high | `epi_bald` | +0.17646 | **+0.00142** | 124x |
| high | `epi_var` | +0.10392 | +0.00059 | 176x |
| high | `total` | +0.07328 | +0.00230 | 32x |
| high | `aleat` | +0.07448 | +0.00786 | 9x |
| xhigh | `epi_bald` | +0.00815 | **−0.00079** | sign flips |
| xhigh | `epi_var` | +0.04589 | **−0.00035** | sign flips |
| xhigh | `total` | +0.10653 | +0.03008 | 3.5x |
| xhigh | `aleat` | +0.09475 | +0.02204 | 4.3x |

**xhigh, post-recalibration gap vs control, every epoch — stable 5/5:**

| arm | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| `epi_bald` | **−0.00294** | **−0.00140** | **−0.00160** | **−0.00039** | **−0.00079** |
| `epi_var` | −0.00145 | −0.00055 | −0.00090 | −0.00022 | −0.00035 |
| `total` | +0.00586 | +0.01269 | +0.01737 | +0.02820 | +0.03008 |
| `aleat` | +0.00420 | +0.01234 | +0.01406 | +0.02397 | +0.02204 |

Both epistemic arms are negative — better than random sampling — at **every** epoch, while
`total` and `aleat` are positive and degrade monotonically. This is the designed result, on the
component that recalibration cannot rescue, sustained over five consecutive epochs.

**The `epi_bald`-is-worst-at-high result was almost entirely an artefact.** Its +0.176 raw
deficit shrinks 124-fold to +0.0014 once miscalibration is removed. The arms differ enormously in
how well-calibrated they leave the model and barely at all in how much irreducible information
they gather — except at xhigh, where `total` and `aleat` destroy resolution outright.

**Where the evidence is weaker.** At high the post-recal gaps are all small (0.0002–0.010) and
the ordering is *not* stable: `epi_var` trends best (down to +0.00024 by epoch 6) and `aleat` is
clearly worst from epoch 4 on, but `total` and `epi_bald` swap places between epochs. Treat high
as "adaptive ≈ control after recalibration, with `aleat` worst" and nothing finer. No floor has
been computed for the recalibrated metric, and these gaps are small enough to need one.

**Caveat on the idealisation.** `UNC − RES` is the *perfect*-recalibration floor. Real
recalibration (Platt, isotonic, conformal) recovers only part of it, so the practical benefit sits
between the raw and recalibrated columns. The qualitative conclusion — that most of the damage is
fixable and the epistemic arms protect the unfixable part — does not depend on reaching the floor.

## 2026-08-04 08:25 — FM arms enter the scored pipeline; epoch-0 identity holds there too

All five FM arms are now scored at epoch 0 (K=100 eval samples, 20 per member across M=5, as
designed). Pre-acquisition they must be identical, and they are:

| arm | brier_deb | sAUROC | RES | REL_deb |
|---|---|---|---|---|
| `aleat`/`epi_bald`/`epi_var`/`total` | 0.00693 | 0.97761 | 0.18751 | 0.00265 |
| `dir00` | 0.00699 | 0.97758 | 0.18747 | 0.00267 |

Spread across arms 5.6e-05 — third independent confirmation of epoch-0 identity (after the
classifier metrics and the FM checkpoint epochs), and it validates the FM eval path end to end.

### A cross-predictor observation that reframes the campaign's scope

At high noise, epoch 0, on the *same* data:

| predictor | brier_deb | REL_deb | sAUROC |
|---|---|---|---|
| flow matching | **0.00693** | 0.00265 | 0.97761 |
| classifier | 0.05315 | – | 0.97964 |

**The flow matcher is 7.7x better calibrated than the classifier before any acquisition happens**,
while ranking slightly worse (sAUROC 0.9776 vs 0.9796). The classifier's miscalibration under
process noise — the thing the epistemic split was shown to repair — is largely absent in FM to
begin with.

This is consistent with the previous campaign, which found entropy acquisition never harmed flow
matching and helped it at med/high, and it sets expectations: the FM arms should show a *much
smaller* spread between score modes than the classifier did, because there is far less
calibration damage available to prevent. A null result on the FM half would therefore not
contradict the classifier finding — it would be the predicted consequence of FM starting well
calibrated. Worth stating now, before the FM numbers land, so the comparison is not read as a
failed replication.

Floor status: the adaptive-arm replicates (`clf_xhigh_epi_bald_s43/s44`) have created epoch_001
but not yet written its artifacts, so the adaptive floor is still pending. The headline xhigh
claim currently rests on the `dir00` floor, which is a lower bound.

## 2026-08-04 08:40 — RETRACTION: the run-to-run floor was invalid, and every significance claim with it

While extending the floor to all noise levels I found that three different seeds produced
**bit-identical** metrics at med, and that at high, s43 and s44 were bit-identical while s42
differed. Independent seeds cannot do that.

**Root cause.** `bayesian_mlp_trainer.py:185` seeded members with
`torch.manual_seed(int(bnn.get("seed", 0)) + m)`. `predictor.bnn.seed` exists in **no** predictor
config, so it resolved to 0 in every run: members were always seeded 0..M−1 and `seed=43` /
`seed=44` trained bit-identical ensembles. The engine's `pl.seed_everything(cfg.seed)` does not
compensate, because this per-member `manual_seed` immediately overrides the global RNG for
initialisation and shuffling.

The cluster assignments confirm it exactly:

| level | seed 42 | seed 43 | seed 44 | result |
|---|---|---|---|---|
| med | amarel | amarel | amarel | all three bit-identical |
| high | arrakis | ilab | ilab | s43 ≡ s44; s42 differs |

So the only variation between "replicates" was **cross-cluster floating-point nondeterminism**.
Same cluster → identical to the last bit → an apparent floor of exactly zero.

### What this retracts

Every "distinguishable / within noise" verdict in the entries of 2026-08-04 07:55 and 08:15 is
**unsupported**. Specifically:

- "the epistemic arms beat random sampling at xhigh by 14–30x the floor" — the floor was
  hardware noise, not seed noise. The comparison is unmeasured, not confirmed.
- the `floor 2*SD` column in the dose-response table is meaningless.
- the correction I made at 07:55 ("my hedge was too conservative") was itself wrong; the original
  hedge was right for the wrong reason.

**What survives unchanged**, because none of it depends on the floor:
- the raw and post-recalibration metric *trajectories* and their orderings, which are consistent
  across 5–6 consecutive epochs;
- that `epi_bald`'s apparent catastrophe at high is calibration and shrinks 124x under the
  perfect-recalibration floor;
- that `total`/`aleat` lose resolution at xhigh while the epistemic arms do not;
- the selectivity diagnostics and the noise dose-response of *separation*;
- the deterministic validation case;
- the FM finite-K bias measurement.

These remain descriptive claims about observed trajectories. They are not significance claims and
should not be written up as such until a real floor exists.

### Fix and relaunch

`_member_seed_base()` now takes the run seed, with `predictor.bnn.seed` still winning if set
explicitly (commit 4c7c561, 4 unit tests, full suite green). Code synced to Amarel and verified
on the far side. All ten replicates cancelled, output dirs deleted on both sides, and relaunched
on the fixed code: `clf_{high,xhigh,med,low}_dir00_s{43,44}` and `clf_xhigh_epi_bald_s{43,44}`
(jobs 60199855–60199866).

**The main arms are unaffected and were not relaunched.** They all ran at `seed=42` with member
seeds 0..4 — arbitrary but consistent across every arm, so the comparisons between them remain
valid. Only the floor was broken.

## 2026-08-04 08:50 — the PREVIOUS campaign is NOT affected by the seed bug

Checked, because the retraction above would have propagated to a delivered result if it applied.
It does not. The previous campaign's per-seed classifier values vary genuinely:

| level / arm | debiased Brier per seed |
|---|---|
| med CLF `dir00` | 0.01394, 0.00945, 0.01757 |
| med CLF `ent10` | 0.01108, 0.01046, 0.01228 |
| med FM `dir00` | 0.00173, 0.00079, 0.00094 |

Three distinct values per arm, spreads of the order 0.004–0.008 — nothing like the bit-identical
collapse seen in this campaign's replicates. Whatever seeding path those runs used, it worked.

The bug is specific to `predictor=clf_ensemble`, which routes through
`BayesianMLPTrainer._run_lightning`'s per-member `torch.manual_seed(bnn.get("seed", 0) + m)`.
The previous campaign's classifier arms did not use the ensemble predictor, so their global
`pl.seed_everything(cfg.seed)` was never overridden.

**`docs/stoch_compare/` verdicts stand as delivered** — med-FM, high-FM, med-CLF, high-CLF and
xhigh-CLF are unaffected. Only this campaign's floor was broken, and only for the ensemble
classifier.

## 2026-08-04 11:00 — FM arms diverge; selectivity ordering holds and the epistemic arms reduce epistemic mass

FM high, first post-divergence epoch (arms now hold different training sets):

| arm | ratio ep0 | ratio ep1 | epi_pool ep0 → ep1 |
|---|---|---|---|
| `epi_bald` | 1.87 | **1.87** | 0.01946 → 0.01653 (−15%) |
| `epi_var` | 1.47 | 1.54 | 0.01946 → **0.01394 (−28%)** |
| `total` | 0.50 | 0.59 | 0.01946 → 0.02099 (**+8%**) |
| `aleat` | 0.36 | 0.31 | 0.01946 → 0.01828 (−6%) |

The selectivity ordering `epi_bald` > `epi_var` > `total` > `aleat` is unchanged from epoch 0 and
matches the classifier, so it now holds across both predictors.

The new signal is the pool trajectory: **the epistemic arms are actually reducing epistemic
uncertainty in the candidate pool** (−15% and −28%), which is what acquiring on it is supposed to
achieve, while `total` *increases* it (+8%) and the negative control barely moves it. Notably
`epi_var` reduces it almost twice as fast as `epi_bald` despite having the lower selectivity
ratio — consistent with `epi_bald`'s score being 97% constant bias, which inflates its apparent
selectivity without concentrating the acquisition where genuine disagreement actually lives.

Two caveats. Only two epochs, so this is a direction and not a trend. And `epi_pool` is measured
per-arm against that arm's own model on its own data, so cross-arm comparison of the *level* is
confounded — only the within-arm change is meaningful.

## 2026-08-04 11:20 — fix validated, real floor measured, and the headline restated correctly

The seed fix works. At epoch 0, all three seeds now produce **distinct** values at every level
(pre-fix they were bit-identical), and the real floor is far larger than the broken one:

| level | broken 2*SD | real 2*SD (raw Brier) | understated by |
|---|---|---|---|
| low | 0.00024 | 0.00113 | 4.8x |
| med | 0.00000 | 0.00298 | infinite |
| high | 0.00039 | 0.00483 | 12.4x |
| xhigh | 0.00164 | 0.00496 | 3.0x |

### Re-tested at epoch 7 against the real floor (x = multiples of 2*SD; * = distinguishable)

**xhigh**
| arm | raw Brier | post-recal | sAUROC |
|---|---|---|---|
| `epi_bald` | +0.0145 (2.9x)* | −0.00048 (0.6x) | +0.00044 (0.6x) |
| `epi_var` | +0.0480 (9.7x)* | −0.00032 (0.4x) | +0.00045 (0.6x) |
| `total` | +0.0916 (18.5x)* | +0.0230 (28.0x)* | −0.0289 (37.3x)* |
| `aleat` | +0.1087 (21.9x)* | +0.0258 (31.5x)* | −0.0340 (43.9x)* |

**high**
| arm | raw Brier | post-recal | sAUROC |
|---|---|---|---|
| `epi_bald` | +0.2179 (45.1x)* | +0.00597 (11.0x)* | −0.00445 (24.6x)* |
| `epi_var` | +0.1237 (25.6x)* | +0.00155 (2.9x)* | −0.00099 (5.5x)* |
| `total` | +0.0992 (20.5x)* | +0.00206 (3.8x)* | −0.00112 (6.2x)* |
| `aleat` | +0.0895 (18.5x)* | −0.00027 (0.5x) | −0.00018 (1.0x)* |

### The corrected headline

**At xhigh the epistemic arms fully repair the damage but do NOT beat random sampling.** On both
calibration-free metrics they sit at 0.4–0.6x the floor from the non-adaptive control — a null,
not a win. Meanwhile `total` and `aleat` are 28–44x the floor *worse*. So the honest claim is:
acquiring on total entropy at xhigh is severely harmful, and the epistemic split removes that harm
entirely, restoring parity with random sampling. That is a strong result and it does not need the
overclaim I made earlier.

**At high, `epi_bald` genuinely hurts** — it is distinguishably worse than the control even after
recalibration (11x floor) and on sAUROC (24.6x). This part of the earlier finding survives the
correction: it is not purely a calibration artefact at high, only mostly one (the raw 45x gap
falls to 11x, not to zero). `aleat` is the *closest* arm to the control at high post-recalibration,
which is the opposite of its behaviour at xhigh and remains unexplained.

Both of my earlier positions were wrong in different directions: the original "beats random by
14–30x" overclaimed on a broken floor, and the blanket retraction was too pessimistic — the
large effects (`total`/`aleat` harm at xhigh, 18–44x) were always safe. The floor is measured at
epoch 0 and will likely grow as arms diverge, so the small multiples (0.4–3x) remain provisional;
the large ones do not.

## 2026-08-04 11:30 — how much does the floor grow with epoch? (measured, not assumed)

My re-tested claims use a floor measured at **epoch 0** against gaps at **epoch 7**, which is the
weakest link in the analysis. The previous campaign has valid multi-epoch seed replicates, so the
epoch-dependence can be measured rather than guessed:

| group | SD @ep0 | SD mid | SD @last | trend |
|---|---|---|---|---|
| `clf_high_dir00` | 0.00721 | 0.01358 (ep9) | 0.01038 (ep18) | grows ~1.9x |
| `clf_high_ent10` | 0.00771 | 0.00682 | 0.02074 (ep18) | grows ~2.7x |
| `clf_med_dir00` | 0.00584 | 0.00053 | 0.00464 | shrinks then recovers |
| `clf_xhigh_ent10` | 0.00729 (ep8) | 0.00491 | 0.00481 | shrinks |
| `fm_high_dir00` | 0.00343 | 0.00146 | 0.00049 (ep8) | **shrinks 7x** |
| `fm_med_ent10` | 0.00318 | 0.00027 | 0.00024 (ep13) | **shrinks 13x** |

Two clean patterns. For **flow matching the floor shrinks sharply** as members converge, so an
epoch-0 floor is a conservative over-estimate there. For the **classifier it can grow, worst case
~2.7x** at high noise.

### Which of my claims survive a 3x floor growth

Applying the worst observed CLF growth as a stress test to the epoch-7 results:

| claim | multiple @ep0 floor | @3x floor | survives? |
|---|---|---|---|
| xhigh `total` harmful (post-recal) | 28.0x | 9.3x | **yes** |
| xhigh `aleat` harmful (post-recal) | 31.5x | 10.5x | **yes** |
| xhigh `total`/`aleat` harmful (sAUROC) | 37-44x | 12-15x | **yes** |
| xhigh `epi_bald`/`epi_var` ≈ control | 0.4-0.6x | 0.1-0.2x | **yes** (null strengthens) |
| high `epi_bald` worse post-recal | 11.0x | 3.7x | **yes** |
| high `epi_var` worse post-recal | 2.9x | 1.0x | **no — becomes marginal** |
| high `total` worse post-recal | 3.8x | 1.3x | **no — becomes marginal** |

So the headline survives comfortably: at xhigh, `total`/`aleat` do real damage and the epistemic
arms are indistinguishable from random. The `epi_bald`-hurts-at-high finding also survives. The
fine-grained ordering *among* arms at high does not, and should not be reported.

### A side observation worth keeping

The previous campaign's CLF floors (SD 0.007-0.02) are ~3x larger than this campaign's
(SD 0.0024 at high, epoch 0). That is the expected consequence of ensembling: averaging five
members damps run-to-run variance. It means the ensemble predictor buys tighter reproducibility
as well as better calibration — and that comparisons in this campaign have more resolving power
per seed than the previous one did.

## 2026-08-04 11:40 — MECHANISM: at xhigh the epistemic split finds genuinely ambiguous states; total entropy does not

The `d2_indices` recorded per epoch are pool indices, so the true `p_success` of every acquired
point can be looked up on the eval grid (same method as the previous campaign's
`acquisition_diagnostics.py`). This is what each arm actually bought:

**xhigh — true p_success of the 1000 acquired points**
| arm | ep | mean_p | frac ambiguous (0.2<p<0.8) | frac decided |
|---|---|---|---|---|
| `epi_bald` | 2 | **0.485** | **0.676** | 0.004 |
| `epi_bald` | 5 | 0.300 | **0.511** | 0.001 |
| `epi_var` | 2 | 0.252 | 0.452 | 0.002 |
| `epi_var` | 5 | 0.210 | 0.311 | 0.001 |
| `total` | 5 | 0.124 | **0.024** | 0.011 |
| `aleat` | 5 | 0.125 | **0.030** | 0.002 |

**high — true p_success of the acquired points**
| arm | ep | mean_p | frac ambiguous | frac decided |
|---|---|---|---|---|
| `epi_bald` | 5 | 0.012 | **0.000** | 0.937 |
| `epi_var` | 5 | 0.014 | 0.000 | 0.946 |
| `total` | 5 | 0.021 | 0.000 | 0.883 |
| `aleat` | 5 | 0.020 | 0.000 | 0.908 |

**This is the whole story, and it is causal.**

At **xhigh** `epi_bald` acquires points that are 51–68% genuinely ambiguous with a near-balanced
marginal (mean_p 0.30–0.49), while `total` and `aleat` acquire 2–3% ambiguous points with a
marginal crushed to 0.12. The previous campaign diagnosed exactly this pathology — entropy
acquisition at xhigh scoring *worse than random* on ambiguity (frac 0.269 → 0.080) and dragging
the training marginal from 0.481 to 0.138. Our `total` arm reproduces it (0.024, 0.124). **The
epistemic split repairs it**: `epi_bald` recovers the ambiguous fraction by ~20x and restores a
balanced marginal.

At **high**, *every* arm fails the same way: 0.0% ambiguous, 88–95% near-certain, marginal
crushed to 0.01–0.02. The epistemic score finds no more ambiguity than total entropy does. That
is why no arm helps at high, and it explains the otherwise puzzling result that the negative
control `aleat` is indistinguishable from the others there — at high, all four arms are buying
the same junk, so there is nothing for the decomposition to separate downstream.

The reason for the difference is the size of the ambiguous region itself: the previous campaign
measured random sampling hitting 13.3% ambiguous states at high but 26.9% at xhigh. At high the
genuinely ambiguous set is half as large, and the epistemic signal does not locate it.

**This also explains the "does not beat random" null.** `epi_bald` at xhigh restores a
representative, balanced sample — which is what random sampling already provides. So parity is
the *expected* ceiling for this fix, not a disappointment. The value is entirely in avoiding
`total`'s catastrophic skew, and that is worth 28–44x the run-to-run floor.

## 2026-08-04 11:45 — acquisition quality across ALL levels completes the mechanism (and raises one puzzle)

True `p_success` of the 1000 points each arm acquired, epoch 4, every stochastic level:

| level | arm | mean_p | frac ambiguous | frac decided |
|---|---|---|---|---|
| low | `epi_bald` / `epi_var` / `total` / `aleat` | 0.41–0.47 | **0.139–0.149** | 0.68–0.71 |
| med | all four | 0.21–0.37 | **0.222–0.278** | 0.41–0.59 |
| high | `epi_bald` | 0.010 | **0.001** | 0.978 |
| high | `epi_var` | 0.025 | 0.026 | 0.894 |
| high | `total` / `aleat` | 0.017–0.024 | **0.000** | 0.86–0.94 |
| xhigh | `epi_bald` | 0.365 | **0.552** | 0.050 |
| xhigh | `epi_var` | 0.176 | 0.256 | 0.005 |
| xhigh | `total` / `aleat` | 0.126–0.127 | **0.028–0.037** | 0.004–0.009 |

This confirms the threshold claim from the diagnostics directly, in terms of what was actually
bought rather than what was scored:

- **low and med**: all four arms acquire near-identically (ambiguous fraction within 0.01 of each
  other at low) with healthy marginals. Nothing to separate — matching the finding that arms only
  differentiate above `ale_pool` ~0.1.
- **xhigh**: only the epistemic arms recover ambiguity (0.55 and 0.26 vs 0.03 for `total`/`aleat`).
- **high**: every arm collapses to ~0% ambiguous and a marginal of 0.01–0.02.

### The puzzle

**High is worse than xhigh, despite having less noise.** At xhigh `epi_bald` finds 55% ambiguous
points; at high it finds 0.1% — the worst of any arm at any level. More process noise makes the
problem *easier* for the epistemic score, which is backwards from the naive expectation.

A plausible reading: at high the classifier is already very accurate (sAUROC 0.98) and the
surviving member disagreement sits in the far tails — states where p is essentially 0 but members
disagree about *how* close to 0 — rather than in the genuinely ambiguous middle. At xhigh the
model is weaker (sAUROC 0.90) and disagreement coincides with real ambiguity. If that is right,
epistemic acquisition needs the model to be uncertain in the *right region*, and being too
accurate is its own failure mode.

This is currently a hypothesis, not a result. It is testable: correlate per-candidate epistemic
score against |true p − 0.5| on the eval grid at each level. Worth doing before any writeup, since
it would explain the campaign's single most counterintuitive number.

### Addendum — the puzzle resolves with data already in hand

`epi_bald` selects (near-)argmax of the epistemic score, so the true-p distribution of what it
picked *is* a readout of where that score peaks:

| level | where epistemic peaks (mean_p of `epi_bald`'s picks) | frac of picks that are near-certain |
|---|---|---|
| low | 0.460 | 0.679 |
| med | 0.207 | 0.593 |
| high | **0.010** | **0.978** |
| xhigh | 0.365 | 0.050 |

At high, **97.8% of the highest-epistemic states are near-deterministic** (true p < 0.05 or
> 0.95). Member disagreement is therefore concentrated where the outcome is essentially certain —
members differ about *how* close to 0 the probability is, not about which way it goes. At xhigh
the same score peaks squarely in the ambiguous middle (5% near-certain).

So the failure at high is **not** an acquisition-rule failure. The decomposition faithfully finds
where the ensemble disagrees; at high noise the ensemble simply disagrees in a useless place. That
is a property of the fitted model, not of the score, and no choice among `total`/`epi_*`/`aleat`
can repair it — which is exactly what the downstream metrics show (all four arms equally harmed at
high).

This makes the campaign's conclusion sharper: epistemic acquisition helps **only when member
disagreement co-locates with genuine ambiguity**. That held at xhigh and failed at high, and the
selected-point diagnostic above is the cheap test for whether it holds in any new setting — it
needs no extra compute, just the `d2_indices` already written every epoch.

## 2026-08-04 11:55 — ROOT CAUSE: BALD is epistemic disagreement divided by p(1-p)

The "ensemble disagrees in a useless place" explanation above described the symptom. The cause is
in BALD's functional form.

Second-order Taylor: `BALD = H(p̄) − E_m[H(p_m)] ≈ Var_m[p] / (2·p̄(1−p̄))`, because
`H''(p) = −1/(p(1−p))`. For **identical** member disagreement (Var = 1e-4):

| p̄ | BALD | amplification vs p=0.5 |
|---|---|---|
| 0.50 | 0.000200 | 1.0x |
| 0.10 | 0.000556 | 2.8x |
| 0.02 | 0.002667 | 13.3x |
| 0.01 | 0.006982 | **34.9x** |

`epi_var` returns 0.000200 at every one of those — no curvature factor.

So BALD does not rank by "how much do members disagree"; it ranks by "how much do members disagree,
weighted by 1/(p(1−p))". Where the model's mass sits at extreme p — high noise, for this
classifier — it selects states where members quibble over whether p is 0.001 or 0.02 and
multiplies that quibble by ~35x. Those states are worthless to label. That is precisely the
97.8%-near-certain acquisition measured at high.

**The prediction is confirmed in the data.** If curvature amplification is the mechanism, the
variance-based score should be less tail-dragged at high, and it is:

| arm @ high | mean_p of picks | frac ambiguous | post-recal harm |
|---|---|---|---|
| `epi_bald` | 0.010 | 0.001 | 11.0x floor |
| `epi_var` | 0.025 | **0.026** (26x more) | 2.9x floor |

This is the first genuine evidence favouring `epi_var` — and notably **not** for the reason the
design anticipated. The design justified `epi_var` by its freedom from finite-K MC bias; that bias
turned out not to reorder anything (a near-constant offset). The advantage that actually shows up
is freedom from the `1/(p(1−p))` curvature weighting, which applies even with exact probabilities
and no sampling at all. Worth correcting in the writeup: the estimator matters, but for a
different reason than stated.

**Open, and cheap to test:** miscalibration is plausibly a second-order contributor, since a
poorly-calibrated model puts more mass at extreme p than it should and feeds the amplification.
Flow matching is 7.7x better calibrated than the classifier at high (measured at epoch 0), so if
calibration drives this, BALD should be markedly less tail-biased on the FM arms. The same
`d2_indices` diagnostic answers it with no extra compute once FM has a few more epochs.

## 2026-08-04 12:00 — MAJOR CORRECTION: the high-noise collapse is a CLASSIFIER pathology, not a property of the scores

Ran the acquisition-quality diagnostic on the FM arms at the **same** noise level and pool as the
classifier. The difference is total:

| predictor | arm | ep | mean_p | frac ambiguous | frac decided |
|---|---|---|---|---|---|
| **FM** | `aleat` | 1 | 0.516 | **0.798** | 0.003 |
| **FM** | `total` | 1 | 0.558 | **0.783** | 0.034 |
| **FM** | `epi_var` | 1 | 0.585 | 0.310 | 0.393 |
| **FM** | `epi_bald` | 1 | 0.540 | 0.214 | 0.531 |
| CLF | `total` | 2 | 0.029 | **0.000** | 0.785 |
| CLF | `aleat` | 2 | 0.032 | **0.000** | 0.739 |
| CLF | `epi_var` | 2 | 0.031 | 0.004 | 0.773 |
| CLF | `epi_bald` | 2 | 0.019 | 0.013 | 0.906 |

**On flow matching every arm acquires a balanced marginal (mean_p ~ 0.52-0.59) with 21-80%
genuinely ambiguous points. On the classifier everything collapses to mean_p ~ 0.02-0.03 and ~0%
ambiguous.** Same states available to both.

So the "all arms fail at high" result is a **classifier** failure. Its probability estimates are
distorted enough (7.7x worse Brier than FM at epoch 0) that *every* score — including plain total
entropy, which is maximised at p = 0.5 by construction — lands on states whose true p is ~0.02.
The scores are computed on p̄ values that are simply wrong, so no score can select correctly.

This corrects the framing of the previous three entries. "BALD is worse than non-adaptive at high"
remains true **as measured on this classifier**, but it is not evidence about BALD as a method.

### What survives, and what the FM data adds

The curvature argument still holds and is now visible on a *well-calibrated* predictor: even on
FM, ordering by ambiguity is `aleat` (0.798) > `total` (0.783) > `epi_var` (0.310) >
`epi_bald` (0.214). Total entropy peaks at p = 0.5 by construction; BALD peaks where
`Var/(p(1−p))` is large, i.e. away from 0.5. So the epistemic scores really are systematically
less drawn to ambiguous states — that part was never a classifier artefact.

But on FM this is no longer obviously a *defect*, and that is the whole premise of the campaign:
states at p ~ 0.5 under heavy process noise are **irreducibly** uncertain, so `total` and `aleat`
buying 80% of them may be exactly the waste the epistemic split exists to avoid. Whether
`epi_bald`'s 53%-decided picks are better or worse than `aleat`'s 80%-ambiguous picks is an
empirical question that only the FM downstream metrics can answer — and they are not in yet
(FM is at epoch 3-4 of 19).

**The FM half is therefore the real experiment**, and the classifier half mostly measures how
badly a miscalibrated model misleads every acquisition rule. That is a useful negative result in
its own right, but it must not be reported as a verdict on the estimators.

## 2026-08-04 12:05 — first FM downstream numbers: adaptive HELPS on flow matching at high

FM high, epoch 1 (first post-acquisition epoch). Gap is vs the non-adaptive control; negative
Brier gap = adaptive helps.

| arm | Brier | gap | sAUROC | gap | post-recal | gap |
|---|---|---|---|---|---|---|
| `dir00` (control) | 0.00454 | – | 0.97934 | – | 0.00239 | – |
| `epi_var` | **0.00151** | **−0.00303** | 0.97968 | +0.00034 | 0.00157 | −0.00083 |
| `aleat` | 0.00168 | −0.00286 | 0.97961 | +0.00027 | 0.00148 | −0.00091 |
| `epi_bald` | 0.00189 | −0.00265 | 0.97994 | +0.00060 | 0.00198 | −0.00042 |
| `total` | 0.00288 | −0.00166 | 0.97903 | −0.00031 | 0.00203 | −0.00037 |

**Every adaptive arm beats the non-adaptive control on flow matching**, which is the exact
opposite of the classifier at the same noise level, where every arm lost. It reproduces the
previous campaign's finding that entropy acquisition never harmed FM and helped it at med/high,
and it is consistent with the acquisition-quality diagnostic: on FM all arms buy balanced,
informative points, while on the classifier they all buy near-certain junk.

Epoch-0 identity holds (arms differ by 5.6e-05 pre-acquisition), so the epoch-1 separation is
genuinely caused by acquisition.

### Why this is NOT yet a result

- **One epoch.** The campaign rule needs two consecutive, and this is the first post-acquisition
  epoch in a 19-epoch run.
- **No FM floor exists yet.** `fm_high_dir00_s43` has been running 47 minutes and has produced
  nothing scoreable. For scale, the previous campaign's *single-model* FM floor was SD 0.00343 at
  epoch 0 (2*SD = 0.00686) — larger than every gap in the table. Ensembling should tighten that
  (this campaign's CLF floors came out ~3x tighter than the previous campaign's), but even a 3x
  tightening puts 2*SD ~ 0.0023 against gaps of 0.0017–0.0030. **These gaps are marginal at best
  and may be entirely noise.**
- The arm ordering (`epi_var` > `aleat` > `epi_bald` > `total`) is therefore not meaningful yet.
  Worth noting only that the negative control sits second — if that persists once a floor exists,
  it would argue against the premise that aleatoric-targeted sampling is wasteful on FM.

Treat this as "FM behaves oppositely to CLF at high", which the diagnostic already predicted, and
nothing more.

## 2026-08-04 12:20 — FIRST ESTABLISHED VERDICT (classifier, xhigh)

Floor now measured at **epoch 2** from three genuinely-distinct seeds on the fixed code
(2*SD: post-recal 0.00102, sAUROC 0.001808 — both larger than the epoch-0 floor, as expected).
Gaps vs the non-adaptive control at the two deepest epochs:

| arm | ep | post-recal gap | x floor | sAUROC gap | x floor |
|---|---|---|---|---|---|
| `epi_bald` | 6 | −0.00081 | 0.8x | +0.00126 | 0.7x |
| `epi_bald` | 7 | −0.00048 | 0.5x | +0.00044 | 0.2x |
| `epi_var` | 6 | −0.00057 | 0.6x | +0.00075 | 0.4x |
| `epi_var` | 7 | −0.00032 | 0.3x | +0.00045 | 0.2x |
| `total` | 6 | +0.01774 | **17.4x** | −0.02078 | **11.5x** |
| `total` | 7 | +0.02295 | **22.5x** | −0.02888 | **16.0x** |
| `aleat` | 6 | +0.02857 | **28.1x** | −0.03954 | **21.9x** |
| `aleat` | 7 | +0.02581 | **25.3x** | −0.03401 | **18.8x** |

**Verdict, meeting every rule this campaign set** (two consecutive epochs, valid seed floor,
calibration-free metrics):

> On the ensemble classifier at xhigh noise, acquiring on total label entropy — or on the
> aleatoric component — is **17–28x the run-to-run floor worse** than random sampling, on the
> component recalibration cannot repair. Acquiring on either epistemic component is
> **indistinguishable from random** (0.2–0.8x floor).

The negative control behaving as the worst arm, and the two epistemic estimators agreeing with
each other, are both as designed.

**Scope, stated precisely.** This is a verdict about *acquisition on this classifier*, whose
probability estimates are poorly calibrated at high noise (7.7x worse Brier than FM pre-training).
The mechanism is established: every score is computed on distorted p̄, and total entropy — which
targets p = 0.5 by construction — is misled hardest, landing on states whose true p is ~0.12.
The epistemic scores are less misled because they key on member *disagreement* rather than on the
absolute probability. It is **not** a verdict on these estimators for a well-calibrated model;
the FM arms test that and are at epoch 3–4 of 19 with no floor yet.

## 2026-08-04 12:30 — DETERMINISTIC PENDULUM COMPLETE (all 5 arms, 19/19 epochs)

First level to finish. Final epoch (018), label metrics (probability metrics need rollout ground
truth, which a deterministic system does not have):

| arm | AUC | Brier | log_score | accuracy |
|---|---|---|---|---|
| `dir00` (control) | 0.999918 | 0.00358 | 0.01163 | 0.99493 |
| `aleat` | 0.999995 | **0.00090** | 0.00316 | 0.99924 |
| `total` | 0.999994 | 0.00094 | 0.00316 | 0.99881 |
| `epi_bald` | 0.999990 | 0.00129 | 0.00462 | **0.99903** |
| `epi_var` | 0.999990 | 0.00132 | 0.00442 | 0.99807 |

Gap vs control: every adaptive arm improves Brier by −0.0023 to −0.0027 (a ~4x reduction),
log score by −0.007 to −0.008, and accuracy by +0.003 to +0.004.

**Two findings:**

1. **Adaptive acquisition clearly helps on a deterministic system**, and by a wide margin — ~4x
   lower Brier than random sampling. This is the opposite sign to the classifier at high/xhigh
   noise, completing the dose-response: adaptive helps when there is little aleatoric mass and
   hurts when aleatoric mass dominates.
2. **The choice of score barely matters here**, exactly as the design predicted for this
   validation case. All four adaptive arms land within 1.5x of each other (0.00090–0.00132)
   against a 4x gap to the control. With aleatoric ~ 0 there is nothing for the decomposition to
   separate, so the arms converge — which is the intended behaviour, not a null result.

AUC is saturated (0.99999, spread 8e-05) and carries no signal at this level; Brier, log score and
accuracy are the discriminating metrics.

**Caveat:** `aleat` and `total` are nominally best (0.0009) and the epistemic arms nominally
behind (0.0013), but **no floor exists for det yet** — its replicates were cancelled during the
seed-bug cleanup and I missed them in the relaunch. Now running (jobs 60231676/60231677). Until
they land, the 1.4x spread among adaptive arms is not interpretable; only the 4x adaptive-vs-control
gap is large enough to be safe on its face.

## 2026-08-04 12:40 — CLF dose-response, all levels, VALID floors (supersedes the 08:05 table)

Post-recalibration gap vs the non-adaptive control, in multiples of the seed floor
(3 distinct seeds of `dir00`, deepest shared epoch >= 1). Negative = better than random.
|x| > 1 = distinguishable from run-to-run noise.

| level | arm epoch | floor 2*SD | `epi_bald` | `epi_var` | `total` | `aleat` |
|---|---|---|---|---|---|---|
| det   | 18 | (floor still running) | −0.0023 raw | −0.0023 raw | −0.0026 raw | −0.0027 raw |
| low   | 6  | 0.00040 | −0.8x | −0.9x | −1.1x | −1.0x |
| med   | 5  | 0.00037 | **−2.0x** | −1.1x | **−2.1x** | −1.5x |
| high  | 7  | 0.00046 | **+13.0x** | +3.4x | +4.5x | −0.6x |
| xhigh | 7  | 0.00102 | −0.5x | −0.3x | **+22.5x** | **+25.3x** |

**The pattern is non-monotone, and the two high-noise rows are near-inverses of each other.**

- **det / low / med**: adaptive helps or ties, and the score barely matters — all four arms land
  within ~1x of each other. Consistent with there being no aleatoric mass to separate.
- **high**: `aleat` is the *only* arm that ties the control (−0.6x); `epi_bald` is the worst by
  far (+13.0x).
- **xhigh**: exactly reversed — the epistemic arms tie the control (−0.3 to −0.5x) while `total`
  and `aleat` are 22–25x the floor worse.

The `high` row now has a mechanism consistent with the curvature finding. All arms acquire
near-certain states there (0.0–2.6% ambiguous), but they differ in *how far* into the tail they
go, and harm tracks that depth:

| arm @high | mean_p of picks | post-recal harm |
|---|---|---|
| `epi_bald` | 0.012 | +13.0x |
| `epi_var` | 0.014 | +3.4x |
| `total` | 0.021 | +4.5x |
| `aleat` | 0.020 | −0.6x |

BALD's `1/(p(1−p))` weighting drives it furthest into the tail, it produces the most skewed
training set, and it does the most damage. `aleat` — which targets per-member entropy and so
prefers p nearer 0.5 — goes least far and does no measurable harm.

So at high the ranking is essentially "how badly does this score distort the training marginal",
and at xhigh it is "which score can still find ambiguous states at all". Different failure modes
at adjacent noise levels, which is why no single arm wins everywhere on the classifier.

Caveat unchanged: floors are measured at epochs 1–2 against gaps at epochs 5–7, and the measured
floor growth on this metric is not yet known (the previous campaign showed CLF floors growing up
to ~2.7x). The large multiples (13x, 22x, 25x) survive that; the 2–4x ones do not.

## 2026-08-04 12:50 — DETERMINISTIC VERDICT ESTABLISHED (floor now exists)

det floor from three genuinely-distinct seeds: SD 0.00071, **2*SD = 0.00143** at epoch 4
(3/3 distinct, so the seed fix is confirmed on this level too).

Final epoch (018), gap vs the non-adaptive control in floor units:

| arm | Brier | gap | x floor |
|---|---|---|---|
| `aleat` | 0.00090 | −0.00267 | −1.9x |
| `total` | 0.00094 | −0.00264 | −1.8x |
| `epi_bald` | 0.00129 | −0.00229 | −1.6x |
| `epi_var` | 0.00132 | −0.00226 | −1.6x |

**Verdict:** on the deterministic pendulum every adaptive arm beats random sampling by 1.6–1.9x
the run-to-run floor, and **the differences between them (0.3x) are well inside the floor**. The
design's validation prediction is confirmed quantitatively: with aleatoric ~ 0 there is nothing to
separate, so the choice of score does not matter, while adaptive-vs-random does.

This also right-sizes the earlier claim. I reported the adaptive arms as "~4x better Brier" — true
as a ratio (0.0009 vs 0.0036), but in floor units the effect is 1.6–1.9x, i.e. real but modest.
The ratio was the more flattering framing; the floor units are the honest one.

**One transient worth recording.** At epoch 4, `aleat` was **+10.7x the floor worse** than the
control while every other arm already helped — then it recovered to the best arm by epoch 18. A
single-epoch read at 4 would have called the negative control catastrophic on a deterministic
system, which the final data contradicts. That is a concrete instance of the campaign rule (never
call a verdict from one epoch) catching something real, on the only level with a complete curve.

## 2026-08-04 12:55 — FM at two consecutive epochs: total entropy is the only arm that hurts

FM high, gap vs the non-adaptive control (negative Brier/recal = helps; positive sAUROC = helps):

| arm | ep | Brier gap | sAUROC gap | post-recal gap |
|---|---|---|---|---|
| `epi_bald` | 1 | −0.00265 | **+0.00060** | −0.00042 |
| `epi_bald` | 2 | −0.00057 | **+0.00029** | −0.00050 |
| `epi_var` | 1 | −0.00303 | +0.00034 | −0.00083 |
| `epi_var` | 2 | −0.00043 | +0.00011 | −0.00040 |
| `aleat` | 1 | −0.00286 | +0.00027 | −0.00091 |
| `aleat` | 2 | −0.00075 | +0.00019 | −0.00078 |
| `total` | 1 | −0.00166 | **−0.00031** | −0.00037 |
| `total` | 2 | −0.00032 | **−0.00018** | −0.00036 |

**Stable across both epochs:**
1. Every adaptive arm beats the control on Brier — the opposite sign to the classifier at the same
   noise level, where every arm lost.
2. **`total` is the worst adaptive arm at both epochs, and the only arm with a negative sAUROC
   gap** — i.e. the only one that costs discriminative power relative to random sampling. The
   three others all gain a little.

**Not stable:** the ordering among `epi_bald`, `epi_var` and `aleat` flips between epoch 1 and 2,
so nothing should be read into which of those three is "best".

Absolute values show why the gaps shrink: the control itself improves fast
(Brier 0.00454 → 0.00124 between epochs 1 and 2), so the same relative advantage becomes a smaller
absolute gap.

**No FM floor yet** — `fm_high_dir00_s43` has only epoch 0 scored and `s44` started an hour ago.
For scale, the previous campaign's *single-model* FM floor was SD 0.00343 at epoch 0, larger than
every gap above; the ensemble floor should be tighter but is unmeasured. So the direction is
consistent over two epochs, but **none of these gaps is yet established as distinguishable from
noise.** The `total`-is-worst pattern is the one to watch, because it is the only claim that is
consistent on both metrics and both epochs.

## 2026-08-04 13:05 — preemption caught before it corrupted an arm in the established verdict

`clf_xhigh_total` (job 60036433) was preempted at epoch 11 and auto-requeued by `--requeue`.
`scripts/check_preempted.sh` did not flag it — a requeued job is PENDING, not absent from squeue,
so it looks like a normal queue entry. It was only visible because Amarel's pending count went
from 0 to 1.

**Why this mattered.** `AdaptiveEngine.run` iterates `for epoch in range(n_epochs)` with no resume
logic, so a requeued job restarts at epoch 0 and overwrites the existing epoch dirs in place.
Amarel now carries the member-seed fix (4c7c561) that the original run predates, so the restart
would have rewritten epochs 0-10 under the new code while epochs 11+ remained pre-fix output — a
single arm's curve stitched across two codebases, and specifically the `total` arm at xhigh, which
is one of the two arms in the campaign's first established verdict.

Handled per the standing rule: cancelled, deleted the output dir on **both** Amarel and the local
mirror, relaunched clean (job 60250190). Costs ~11 epochs of recompute (~5h) and buys an
internally consistent curve.

**Note for the remaining arms.** Every main arm was launched before the seed fix, so they all run
with member seeds 0..4 while any relaunched arm gets 42..46. That is an arbitrary difference in
initialisation, equivalent to a different random init, and it is precisely what the run-to-run
floor measures — so arm-vs-arm comparisons stay valid. But if more arms get preempted, each one
must be cleaned and relaunched the same way rather than allowed to requeue in place.

**Monitoring gap to close:** requeued jobs are invisible to `check_preempted.sh`. The reliable
signal is a PENDING job whose name matches a main arm, or `sacct` showing `Elapsed 00:00:00` on a
job that was previously running. Both are now in the poll.

## 2026-08-04 13:25 — two-consecutive-epoch check on every level (and a walk-back on "epi_bald worst at high")

Post-recalibration gap vs control in floor units, at the two deepest shared epochs:

| level | floor 2*SD | arm | ep A | ep B |
|---|---|---|---|---|
| low (5,6) | 0.00042 | all four | −0.8 to −1.0x | −0.8 to −1.0x |
| med (5,6) | 0.00037 | `total` | −2.1x | −3.1x |
| | | `epi_bald` | −2.0x | −2.9x |
| | | `aleat` | −1.5x | −2.8x |
| | | `epi_var` | −1.1x | −3.0x |
| high (7,8) | 0.00063 | `epi_bald` | +9.5x | +3.2x |
| | | `epi_var` | +2.5x | +5.2x |
| | | `total` | +3.3x | +4.1x |
| | | `aleat` | −0.4x | +2.0x |

**low is a null.** All four arms sit at −0.8 to −1.0x at both epochs — consistently just under the
threshold. Adaptive acquisition is indistinguishable from random sampling at low noise, and the
score choice is irrelevant. This is the predicted behaviour below the aleatoric threshold, now
confirmed at two consecutive epochs rather than inferred.

**med: adaptive helps, and again the score does not matter.** Three of four arms exceed the floor
at both epochs (−1.5 to −3.1x), all in the same direction, with a spread between arms smaller than
the gap to the control.

**high: all arms are harmful, but I over-claimed the ordering.** Earlier I reported `epi_bald` as
"the worst arm at high, +13.0x". Across epochs 7 and 8 it reads +9.5x then +3.2x, while `epi_var`
goes +2.5x then +5.2x — the arms swap places. **The ordering among arms at high is not stable and
should not be reported.** What survives the two-epoch rule is only the weaker claim: every
adaptive arm is distinguishably worse than random at high noise. The mechanism (BALD's
`1/(p(1−p))` tail bias) is still supported by the acquisition-quality data, which is stable, but
the downstream *ranking* it predicted is not.

xhigh is temporarily unavailable: `clf_xhigh_total` was deleted and relaunched after the
preemption, so no epoch is yet shared by all five arms. The earlier xhigh verdict
(`total`/`aleat` 17-28x worse, epistemic arms tie) stands on the data recorded at 12:20 and will
be re-derived on the clean run.

## 2026-08-04 13:55 — classifier arms will hit their walltime around epoch 17-18

The Amarel classifier arms were submitted with `--time=1-00:00:00`. At 12:09 elapsed they have
8-11 epochs done, i.e. ~1.28h per adaptive epoch, so 19 epochs needs ~25h against a 24h limit.
They will be killed roughly half an hour short.

`scontrol update JobId=... TimeLimit=2-00:00:00` is refused — **a user cannot raise TimeLimit on
a running job** ("Access/permission denied"). The only fix is submitting with enough walltime,
which `scripts/ensemble/launch_clf.sh` now does (2 days).

**Impact is acceptable and no action is being taken on the running arms.** Analysis is
matched-epoch, so all arms converging at ~17 rather than 19 costs one or two epochs of depth and
nothing else. Relaunching to gain them would discard 12h of completed training across ~15 arms —
a bad trade.

**Two things to watch as they expire:**
1. A TIMEOUT kill can leave the final epoch dir partially written. The scoring spec already
   filters on `full_roa_per_point.npz` existing, so a half-written epoch is skipped rather than
   silently scored — but that guard is what makes this safe, and it should not be removed.
2. The arms carry `--requeue`. TIMEOUT does not normally trigger requeue, but if any arm does
   come back as PENDING it will restart from epoch 0 and overwrite its own results, since the
   engine has no resume. `check_preempted.sh` section 2 now detects exactly that (PENDING job with
   epochs already on disk).

## 2026-08-04 14:15 — METHODOLOGICAL CORRECTION: single-epoch floors are too noisy; two verdicts change

The deterministic floor now has 13 epochs with three distinct seeds, which exposes a problem with
how every verdict so far was tested. **The per-epoch floor varies 15.6x**:

| epoch | 1 | 3 | 6 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|---|
| 2*SD | 0.00117 | 0.00205 | 0.00083 | **0.00380** | **0.00024** | 0.00179 | 0.00050 |

With three seeds, each epoch's SD has two degrees of freedom — a very noisy estimate. Testing the
same det gap (`total`, −0.00264) against different epochs' floors gives:

| floor choice | multiple | verdict |
|---|---|---|
| min (ep9) | 10.8x | distinguishable |
| median | 1.9x | distinguishable |
| **max (ep7)** | **0.7x** | **within noise** |
| last available (ep13) | 5.2x | distinguishable |

The conclusion depended on an arbitrary choice. Fixed by **pooling variance across epochs**
(`2*sqrt(mean(var_e))`), which uses all the data instead of one slice.

### Re-tested against pooled floors

| level | pooled 2*SD (epochs) | result |
|---|---|---|
| det | 0.00171 (13) | all arms −1.3 to −1.6x — **DISTINGUISHABLE**, verdict holds |
| low | 0.00035 (3) | −0.9 to −1.2x — **marginal**, 3 of 4 barely over threshold |
| med | 0.00126 (2) | −0.8 to −0.9x — **WITHIN NOISE** |
| high | 0.00055 (2) | +2.3 to +5.9x — **DISTINGUISHABLE**, all arms harmful |

**Two changes to previously reported results:**

1. **`med: adaptive helps` is RETRACTED.** It was reported at −1.5 to −3.1x against a single-epoch
   floor of 0.00037. The pooled floor is 0.00126 — 3.4x larger — and every arm falls to −0.8 to
   −0.9x, i.e. inside the noise. med is a null, not a win.
2. **`low is a null` is WEAKENED.** Three of four arms now sit just over the threshold (−1.1 to
   −1.2x) rather than just under. Best described as marginal in the helping direction, not a clean
   null.

**Unchanged:** det (all adaptive arms beat random, score irrelevant) and high (every adaptive arm
distinguishably worse than random). Both were large enough to survive the floor change.

**Caveat on pooling itself.** It assumes the floor is roughly stationary across epochs, which the
15.6x det range makes questionable. It is nonetheless strictly better than picking one epoch
arbitrarily, and the det pooled floor rests on 13 epochs. The low/med/high pooled floors currently
rest on only 2-3 epochs and will firm up as the replicates deepen — **their verdicts should be
re-checked then.**

## 2026-08-04 14:25 — the xhigh headline survives the pooled-floor correction

The pooled-floor fix retracted the med verdict, so the xhigh headline had to be re-tested the same
way. It holds, at two consecutive epochs:

xhigh pooled floor over 4 epochs: **recal 2*SD = 0.00095, sAUROC 2*SD = 0.001142**
(the single-epoch floor used for the original 12:20 verdict was 0.00102 / 0.001808 — so that
verdict was, if anything, *conservative*, not a lucky slice).

| arm | ep | post-recal gap | × floor | sAUROC gap | × floor |
|---|---|---|---|---|---|
| `epi_bald` | 7 | −0.00048 | 0.5× | +0.00044 | 0.4× |
| `epi_bald` | 8 | −0.00047 | 0.5× | +0.00056 | 0.5× |
| `epi_var` | 7 | −0.00032 | 0.3× | +0.00045 | 0.4× |
| `epi_var` | 8 | −0.00046 | 0.5× | +0.00072 | 0.6× |
| `aleat` | 7 | +0.02581 | **27.2×** | −0.03401 | **29.8×** |
| `aleat` | 8 | +0.02932 | **30.9×** | −0.04260 | **37.3×** |

So: the aleatoric arm is 27–37× the pooled floor worse than random sampling on both
calibration-free metrics, at two consecutive epochs, while both epistemic arms sit at 0.3–0.6×
— indistinguishable from random. `total` is absent because its relaunch after the preemption has
not caught up; it will be re-derived, and its earlier numbers (+17 to +22×) were of the same
magnitude as `aleat`'s.

**Why this matters beyond the result.** The med verdict died under pooling because its effect was
~1x the floor and the floor estimate moved 3.4x. The xhigh verdict is 27-37x, so no plausible
floor revision touches it. That is the practical distinction to carry into the writeup: effects at
1-5x the floor are hostage to how the floor is estimated; effects above ~10x are not.
