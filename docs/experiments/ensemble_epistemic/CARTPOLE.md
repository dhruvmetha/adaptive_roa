# Cartpole (stochastic, sigma_020.0) — pre-registration and floor characterisation

**Sections 2-5 written 2026-08-11, before any acquisition arm existed.** Only the three floor seeds
(`fm_cpstoch_s020_dir00`, `_s43`, `_s44`) have been run. Nothing in this file was chosen after
seeing an arm result, because there are no arm results. That is the point of writing it now.

Companion to `FINDINGS.md`, which covers the pendulum campaign.

---

## 1. What is running

Three seeds of the non-adaptive control (`acquisition=direct acquisition.d2_ratio=0`), iLab
jobs 205995/205996/205997, `gpu:a4500:2`, 100G, 19 epochs, seeds 42/43/44. All three on the
same node, which is the condition under which a broken-seeding bug would be detectable.

**Arms launched 2026-08-12** — jobs 207080/207081/207082, `gpu:a4500:1`, 100G, 19 epochs,
seed 42. Queued behind the floor and start when it releases its a4500s.

They run on iLab a4500, the same type as the floor. Direct-box GPUs were free at submission time
and were deliberately not used: the floor is all-iLab-a4500, so arms on other hardware would fold
a hardware term into the arm-vs-control gap that the floor cannot capture — anti-conservative.
`FINDINGS.md` §4a carries that caveat for [FM] high; cartpole stays clean.

**Three arms, not four: `aleat` was dropped at the user's direction, and one seed only.**

| arm | status |
|---|---|
| `epi_bald` | launched (207080) |
| `epi_var` | launched (207081) |
| `total` | launched (207082) |
| `aleat` | **not run** |

**What dropping `aleat` costs, stated plainly.** On the pendulum side `aleat` is the negative
control — it acquires on the component that by construction cannot be reduced by more data — and
it is what turned the campaign's conclusion from soft to sharp: at [FM] low it ranked *first*
among the four arms, and at [FM] det it was the only arm clearing the floor with a stable sign
(`FINDINGS.md` §4c, §4d). Without it, cartpole can still answer *"does the epistemic split beat
plain total entropy?"* by comparing `epi_bald`/`epi_var` against `total`, and *"does adaptive beat
uniform?"* against the `dir00` floor. It **cannot** answer *"is any apparent ordering better than
an acquisition rule known to be useless?"* — the question that repeatedly exposed pendulum
orderings as noise. Any cartpole ranking must therefore be reported without that check.

With one seed per arm, arm-vs-arm differences are also uncontrolled for run-to-run variation; only
arm-vs-`dir00` comparisons are backed by the 3-seed floor.

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

Epochs 1–4 are a **convergence transient**: the seeds have not yet settled, the mean Brier is
still falling steeply (0.131 → 0.109 → 0.096 → 0.091), and the between-seed spread is 10–90×
larger than it becomes later. From epoch 5 the mean Brier is flat to within 1% per epoch and the
spread settles at 0.0009–0.0021.

**Consequence for the pooled floor.** The campaign's rule is `2·sqrt(mean(var_e))` over all
shared epochs with epoch 0 excluded. Epoch 1's variance is roughly 9000× epoch 4's, so it
supplies ~99% of the pooled mean and the pooled floor decays only as 1/sqrt(n):

| pooling | AUC floor | Brier floor |
|---|---|---|
| all shared epochs, ep0 excluded (campaign rule) | 0.00474 | 0.02279 |
| converged regime only (ep ≥ 5, see correction below) | 0.00082 | 0.00169 |

A 13× difference in the Brier floor, purely from whether the transient is included.

## 4. Pre-registered scoring decision

**The primary cartpole verdict will use the campaign's standard rule unchanged** — pooled over
all shared epochs, epoch 0 excluded — so cartpole is scored the same way as every pendulum level
and no goalpost moves. Epoch 0 is excluded for the existing structural reason: all seeds share
identical pre-acquisition data, so its spread is structurally zero, not small.

**The converged-regime floor (epoch ≥ 5) will be reported alongside it as a declared
sensitivity**, with the regime boundary fixed here, in advance, by an objective criterion:

> the first epoch from which the seed-mean Brier changes by less than 1% per epoch, and every
> epoch after it.

On the floor seeds that criterion selects **epoch 5** (0.0958 → 0.0914 is 4.6% — still above the
threshold; 0.0914 → 0.0909 is 0.5% — the first transition under it). Recording the number now
means it cannot be tuned later to move a verdict.

> **Correction, 2026-08-12 — read this before using the number.** The first version of this
> section stated the criterion selects **epoch 3**, while the worked example in the same sentence
> computed epoch 5. The criterion text was never ambiguous and has not changed; the stated answer
> was simply wrong. Re-evaluated on 13 epochs of floor data (three more than were available when
> it was written), the criterion still selects **epoch 5**, so this is a labelling error, not an
> unstable criterion.
>
> **The correction moves in the permissive direction and that must be stated plainly.** Starting
> the converged regime at epoch 5 rather than 3 drops epoch 3's `2×SD` of 0.00269 — the largest
> remaining per-epoch spread — from the pool, so the sensitivity floor gets *tighter*, which makes
> effects *easier* to declare. That is the self-serving direction, which is precisely why the
> correction is recorded here in the open rather than silently edited.
>
> **The primary verdict is unaffected.** It pools over all shared epochs with only epoch 0
> excluded, so it does not depend on where the converged regime is judged to begin. Only the
> secondary, explicitly-labelled sensitivity number changes.

Per-epoch change in seed-mean Brier, all 13 epochs available at the time of the correction:

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % change | 17.3 | 16.6 | 12.2 | 4.5 | **0.5** | 0.3 | 0.0 | 0.7 | 0.1 | 0.8 | 0.8 | 0.8 |

Every epoch from 5 onward stays under 1%, so the boundary is not sitting on a knife edge.

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

Threshold-free metrics are healthy and converge by epoch 5 (AUC plateaus at 0.956–0.957). The
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


---

## 6. First verdict (arms at 7/19) — all three arms are HARMFUL

Scored 2026-08-13 with the arms at 7/19 against the completed 19-epoch `dir00` floor. The
collapse screen ran first as a gate: 80 rows, 0 collapsed.

Window is **ep5–6**, the first two epochs entirely past the convergence transient (§3).

**Primary — campaign rule, floor pooled over all 18 shared epochs, epoch 0 excluded:**

```
fm sigma_020.0: pooled 2*SD = 0.00079 over 18 epochs
   epi_bald   ep5: +0.00624 (+7.9x)   ep6: +0.01038 (+13.2x)   -> DISTINGUISHABLE (sign stable 6/6)
   epi_var    ep5: +0.01711 (+21.7x)  ep6: +0.01158 (+14.7x)   -> DISTINGUISHABLE (sign stable 6/6)
   total      ep5: +0.01609 (+20.4x)  ep6: +0.01548 (+19.6x)   -> DISTINGUISHABLE (sign stable 6/6)
```

**Declared sensitivity — converged-regime floor (ep ≥ 5), 14 epochs, floor 0.00036:** the same
three arms at +17.3× to +47.5×. Larger, because the converged floor is tighter. Per §4 this is
reported as a sensitivity and is **not** the headline; the primary above is the claim.

**Positive means worse.** Raw `recal` (lower is better):

| ep | dir00 | s43 | s44 | epi_bald | epi_var | total |
|---|---|---|---|---|---|---|
| 0 | 0.05312 | 0.05347 | 0.05296 | 0.05303 | 0.05303 | 0.05303 |
| 5 | 0.04391 | 0.04427 | 0.04394 | 0.05015 | 0.06102 | 0.06001 |
| 6 | 0.04427 | 0.04417 | 0.04375 | 0.05465 | 0.05585 | 0.05975 |

Epoch 0 is identical across all six runs — the pre-acquisition identity check passing, confirming
the arms differ only through acquisition. From epoch 1 the three seeds stay clustered near 0.044
while every arm sits above them, at every shared epoch.

### What this does and does not say

**Says:** at cartpole `sigma_020.0`, acquiring on *any* of these three uncertainty scores is
substantially worse than uniform sampling — 8× to 22× the run-to-run floor on the primary rule,
sign-stable across all 6 shared epochs. This is the first level in the campaign where flow
matching shows a large, consistent acquisition effect at all; every [FM] pendulum level was null
or unresolved (`FINDINGS.md` §4).

**Does not say** which arm is worse. `epi_bald` reads least harmful and `total`/`epi_var` worst,
but with **one seed per arm** that ordering has no floor behind it — the 3-seed floor backs
arm-vs-control comparisons only. And with **`aleat` dropped** there is no negative control, so
the ordering cannot be checked against a rule known to be useless, which is exactly the check
that exposed pendulum orderings as noise (`FINDINGS.md` §4c).

**Provisional on depth.** The arms are at 7/19 and the window is the earliest admissible one.
Unlike the marginal pendulum effects that churned across passes, an 8–22× effect is far outside
threshold-crossing noise, so the *direction* is unlikely to reverse — but the magnitudes should be
re-scored as the arms deepen. The floor itself will not change: `dir00` is complete at 19/19.


---

## 7. Pass 2 (arms at 12/19) — verdict holds, and `total` alone compounds

Rescored 2026-08-13 with the arms at 12/19, a 70% depth increase over pass 1. Collapse screen
gated first: 93 rows, 0 collapsed. Window **ep10–11**.

```
fm sigma_020.0: pooled 2*SD = 0.00079 over 18 epochs
   epi_bald   ep10: +0.00729 (+9.2x)   ep11: +0.00768 (+9.7x)    -> DISTINGUISHABLE (sign stable 11/11)
   epi_var    ep10: +0.01122 (+14.2x)  ep11: +0.01041 (+13.2x)   -> DISTINGUISHABLE (sign stable 11/11)
   total      ep10: +0.01918 (+24.3x)  ep11: +0.01801 (+22.9x)   -> DISTINGUISHABLE (sign stable 11/11)
```

| arm | pass 1 (ep5–6, arms at 7) | pass 2 (ep10–11, arms at 12) |
|---|---|---|
| `epi_bald` | +7.9× / +13.2× | +9.2× / +9.7× |
| `epi_var` | +21.7× / +14.7× | +14.2× / +13.2× |
| `total` | +20.4× / +19.6× | +24.3× / +22.9× |

**The verdict survived a 70% depth increase** with the sign stable across all 11 epochs. That is
the pattern `FINDINGS.md` §4 identifies as trustworthy (med, xhigh), not the churn pattern that
made [FM] low unreportable.

### `total`'s harm grows with epochs; the epistemic arms' does not

Linear fit of the arm-minus-control gap against epoch, epochs 1–11:

| arm | slope per epoch | t (9 dof) | reading |
|---|---|---|---|
| `epi_bald` | −0.000009 | −0.03 | flat |
| `epi_var` | +0.000399 | +1.33 | flat |
| **`total`** | **+0.000923** | **+7.81** | **growing** |

The control's own level is flat over the same range (slope −0.000014, t = −0.61), so the growth
is not an artifact of `dir00` drifting.

**So the three arms are not harmful in the same way.** `epi_bald` and `epi_var` impose a fixed
penalty that does not worsen as acquisition continues. `total` degrades progressively — its gap
roughly doubles from +0.0095 at epoch 1 to +0.018 by epoch 10.

This rhymes with the classifier mechanism in `FINDINGS.md` §2: at [CLF] xhigh, `total` and `aleat`
were the arms that destroyed *resolution* while the epistemic arms cost only calibration. A
plausible common story is that total-entropy acquisition progressively distorts the training
distribution while epistemic acquisition causes a one-off shift — but that is a hypothesis these
runs cannot test, not a finding.

**Caveat.** One seed per arm, so this is a single run's trajectory. The t-statistic is large and
the control is flat, which makes a chance trend unlikely, but replication needs more seeds. The
between-arm ordering remains unbacked by any floor for the reasons in §6.
