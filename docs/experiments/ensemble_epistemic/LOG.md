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
