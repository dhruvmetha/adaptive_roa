# Cartpole (stochastic, sigma_020.0) — pre-registration and floor characterisation

**Written 2026-08-11, before any acquisition arm exists.** Only the three floor seeds
(`fm_cpstoch_s020_dir00`, `_s43`, `_s44`) have been run. Nothing in this file was chosen after
seeing an arm result, because there are no arm results. That is the point of writing it now.

Companion to `FINDINGS.md`, which covers the pendulum campaign.

---

## 1. What is running

Three seeds of the non-adaptive control (`acquisition=direct acquisition.d2_ratio=0`), iLab
jobs 205995/205996/205997, `gpu:a4500:2`, 100G, 19 epochs, seeds 42/43/44. All three on the
same node, which is the condition under which a broken-seeding bug would be detectable.

The four acquisition arms (`epi_bald`, `epi_var`, `total`, `aleat`) are **not yet launched** —
they are waiting on iLab quota. They will run on iLab too: the floor is all-iLab, and putting
arms on another cluster would fold cross-cluster nondeterminism into the arm-vs-control gap
without it appearing in the floor, which is an anti-conservative test. `FINDINGS.md` §4a carries
that caveat for [FM] high; cartpole starts clean.

## 2. The seeds are genuinely distinct

The campaign was previously burned by ensemble members being seeded from a config key present in
no config, so seed replicates trained bit-identical models and the floor collapsed to ~0
(fixed in `4c7c561`). Checked here, on the same node:

| epoch | AUC across the three seeds | spread |
|---|---|---|
| 0 | 0.9179 / 0.9162 / 0.9167 | 1.67e-03 |
| 1 | 0.9559 / 0.9531 / 0.9429 | 1.30e-02 |

Distinct. The floor is real.

## 3. The floor has two regimes, and they differ by ~4 orders of magnitude

Per-epoch `2×SD` across the three seeds:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.00173 | 0.01371 | 0.00313 | 0.00105 | 0.00096 | 0.00031 | 0.00099 | 0.00068 | 0.00104 | 0.00020 |
| Brier | 0.01739 | 0.06628 | 0.01616 | 0.00269 | 0.00071 | 0.00141 | 0.00110 | 0.00190 | 0.00212 | 0.00093 |
| mean Brier | 0.1580 | 0.1307 | 0.1090 | 0.0958 | 0.0914 | 0.0909 | 0.0907 | 0.0907 | 0.0900 | 0.0899 |

Epochs 1–2 are a **convergence transient**: the seeds have not yet settled, the mean Brier is
still falling steeply (0.131 → 0.109 → 0.096), and the between-seed spread is 10–90× larger than
it becomes later. From epoch 3 the mean Brier is flat to within 1% per epoch and the spread
drops to 0.0007–0.0027.

**Consequence for the pooled floor.** The campaign's rule is `2·sqrt(mean(var_e))` over all
shared epochs with epoch 0 excluded. Epoch 1's variance is roughly 9000× epoch 4's, so it
supplies ~99% of the pooled mean and the pooled floor decays only as 1/sqrt(n):

| pooling | AUC floor | Brier floor |
|---|---|---|
| all shared epochs, ep0 excluded (campaign rule) | 0.00474 | 0.02279 |
| converged regime only (ep ≥ 3) | 0.00082 | 0.00169 |

A 13× difference in the Brier floor, purely from whether the transient is included.

## 4. Pre-registered scoring decision

**The primary cartpole verdict will use the campaign's standard rule unchanged** — pooled over
all shared epochs, epoch 0 excluded — so cartpole is scored the same way as every pendulum level
and no goalpost moves. Epoch 0 is excluded for the existing structural reason: all seeds share
identical pre-acquisition data, so its spread is structurally zero, not small.

**The converged-regime floor (epoch ≥ 3) will be reported alongside it as a declared
sensitivity**, with the regime boundary fixed here, in advance, by an objective criterion:

> the first epoch from which the seed-mean Brier changes by less than 1% per epoch, and every
> epoch after it.

On the floor seeds that criterion selects **epoch 3** (0.0958 → 0.0914 is 4.6%; 0.0914 → 0.0909
is 0.5%). Recording the number now means it cannot be tuned later to move a verdict.

**Direction of the error.** Including the transient makes the floor *larger*, so the primary
verdict is conservative: it will under-report effects, not over-report them. If an arm clears
the 0.0228 floor it has cleared a demanding bar. If an arm clears only the 0.00169 floor, that
must be reported as the sensitivity result and labelled as such — never as the headline.

**Why this is written before the arms exist.** `FINDINGS.md` §4c records five passes at [FM] low
where the set of "distinguishable" arms churned none → two → two → three → none while the effect
sizes never moved, because the threshold kept being recrossed by noise. The lesson taken from it
was: quote effect sizes and floor depth, never a bare "DISTINGUISHABLE". Fixing the floor
convention before any arm result exists is the same lesson applied one step earlier.

## 5. Known weakness: the λ/δ half is unreliable here

Threshold-free metrics are healthy and converge by epoch 3 (AUC plateaus at 0.956–0.957). The
λ/δ decision metrics are not:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| invalid % | 67.9 | 28.1 | 32.9 | 15.7 | 6.8 | 7.7 | 12.6 | 12.5 |
| F1 | 0.164 | 0.169 | 0.000 | 0.306 | 0.338 | 0.207 | 0.401 | 0.342 |

F1 recovers from near-zero to 0.34–0.40 but swings between adjacent epochs (0.000 at ep2, 0.401
at ep6), and stays far below pendulum's 0.82–1.00. λ\* also varies 0.10–0.60 across seeds at
epoch 0, so the threshold optimiser is not finding a stable operating point on this system.

**Cartpole verdicts will therefore be threshold-free only** (AUC / AUPRC / Brier / `recal`).
Any λ/δ number quoted for cartpole must carry this caveat.
