# Yield-aware acquisition — the i100 pendulum sweep

**Status as of 2026-08-16 13:45.** Live campaign; two arms still training. This file records what
is currently defensible and flags in-flight work as in-flight. `FINDINGS.md` covers the earlier
init=1000/step=1000 sweep across five noise levels, whose headline was that **the score does not
matter** — every uncertainty arm was interchangeable with the non-adaptive control. This sweep
found the reason on one system, and a fix that beats the control on one of three metrics.

**Headline.** On `noisy/pendulum/lqr/high` the acquisition budget is denominated in
**trajectories** but the model trains on **pairs**, and the two are coupled to the outcome:
success rollouts stop at the goal (mean 136.5 steps) while failures run to the 1001-step cap. So
the pool is 40.3% success by trajectory but only **8.4% by pair**. Every uncertainty score buys
the model's uncertain band, which on this system is the true-success region, hence the *short*
rollouts — so the scored arms accrued training pairs at 0.27–0.50× the uniform control's rate and
lost at matched trajectory budget while *winning* at matched pair count. Ranking by
informativeness **per trajectory purchased** rather than per point (`yield_aware`, `score × E[L]`)
reverses that: it beats uniform sampling at the campaign's own budget on **KL divergence**, by
+1.87 and +1.68 floor units across two independent seeds. sAUROC and recal do **not** replicate.

---

## 1. Setup

| | |
|---|---|
| dataset | `noisy/pendulum/lqr/high`, 300,000 trajectories, 195,580,290 endpoint pairs |
| predictor | `fm_ensemble`, 5 members |
| budget | `initial_train_size=100`, `samples_per_epoch=100`, `n_epochs=20` |
| filter | `filter_confident_pairs=false` on every scored run |
| eval | full-ROA grid, 49,770 cells; metrics vs **continuous** ground-truth p |
| run dirs | `…/ensemble_epistemic/fm_high_i100_<arm>` |

The step size is the point: the earlier sweep used 1000/1000, which is large enough that every
arm's training set is dominated by the initial pool and differences in *what* was acquired are
diluted. At 100/100 the acquisition choice drives the training set from epoch 1, which is what
made the pair/trajectory coupling visible.

**Metrics.** sAUROC (AUROC against the continuous ground-truth p, not a 0.5-dichotomy), KL, and
`recal = UNC_debiased − RES` from the Murphy decomposition `Brier = REL − RES + UNC`. The
artifact-level `auc`/`brier` fields are **not** used: they score against `p_success >= 0.5`, and
28.6% of the grid lies in [0.05, 0.95] where that dichotomy discards the signal.

**Headroom, which determines what can possibly be shown.** The oracle ceiling on this grid is
sAUROC 0.9834 (0.9824 at the K=100 sampling limit) against a control of 0.9785 — at most ~0.005
of room, versus a 0.0015 floor. sAUROC and recal are nearly saturated and cannot separate
robustly. KL has ~15× headroom. **KL is the only metric on which a win can show here**, and this
was stated before the arms were scored, not after.

---

## 2. The central finding: the budget is trajectories, the training set is pairs

Measured on the pool:

| quantity | value |
|---|---|
| trajectories | 300,000 |
| success share **by trajectory** | 0.4026 |
| success share **by pair** | **0.0837** |
| mean length, success | 136.5 |
| mean length, failure | 1001.0 (timeout cap) |
| mean length, pool | 652.9 |

A trajectory of length L contributes L−1 endpoint pairs, so buying a success costs the same slot
in a 100-trajectory budget but delivers ~7× fewer training pairs than buying a failure. The mean
true p of points acquired by the scored arms was measured at 0.90–0.99 — they buy states that
really do succeed, i.e. the short rollouts.

Total pairs bought over the full 2,000-trajectory budget:

| arm | pairs | pairs per 100 trajectories | vs control |
|---|---|---|---|
| `dir00` / `_s43` / `_s44` (uniform) | 1,310,733 | 65,537 | 1.00× |
| `length_only` (in flight) | — | **95,813** | **1.46×** |
| `epi_var_anch` | 873,633 | 43,682 | 0.67× |
| `epi_var_yield` | 871,630 | 43,582 | 0.66× |
| `yield_s43` | 768,628 | 38,431 | 0.59× |
| `epi_var_qknn2` | 650,147 | 32,507 | 0.50× |
| `total` | 646,965 | 32,348 | 0.49× |
| `epi_var_qknn` | 456,674 | 22,834 | 0.35× |
| `epi_bald` | 361,071 | 18,054 | 0.28× |
| `epi_var` | 358,197 | **17,910** | **0.27×** |

The three uniform seeds bought *identical* trajectories (65,537 pairs per 100, to the unit) —
uniform acquisition is seeded by the data split, not the model, so the control seeds differ only
in training noise. That is what makes them a clean null for judging an acquisition arm.

**This is a property of the system, not of the method.** On stochastic cartpole the coupling
inverts — successes there are the long rollouts (~600 steps vs a pool mean of 195) — so any fix
must read the coupling from data rather than assume its sign.

---

## 3. The approaches

Every arm below shares the same predictor, budget, seed (42 unless noted), and evaluation. They
differ only in `acquisition`.

### 3.1 Baselines

**`dir00` — uniform (the control).** `d2_ratio=0`; candidates drawn uniformly from the unlabelled
pool, no scoring. Run at three seeds (42/43/44) to establish the run-to-run floor.

**`total` — total predictive entropy H(p̄).** Greedy top-N. On binary outcomes H(p̄) is monotone
in |p̄ − 0.5|, so this reduces exactly to "nearest 0.5" and saturates at ln 2 = 0.6931.

**`epi_var` — MC-debiased between-member variance Var_m[p].** Greedy top-N. The debiasing can
make the score negative when members genuinely agree, which matters for any downstream weighting.

**`epi_bald` — mutual information I(y;m|x) = H(p̄) − E_m[H(p_m)].** Greedy top-N. Note BALD is not
"epistemic focus" in a neutral sense; it is epistemic focus divided by p(1−p), which biases it
toward the decision boundary.

All three scored baselines use the decomposition `H(p̄) = E_m[H(p_m)] + I(y;m|x)`, verified
mechanically correct in the earlier campaign.

### 3.2 Arms designed against the pair/trajectory finding

**`epi_var_anch` — half uniform, half scored** (`decomp_epi_var` with `d2_ratio=0.5`). The
simplest hedge: anchor half the budget on the pool marginal so the training set cannot drift
arbitrarily far from it. Diagnostically the most informative arm in the campaign, because its two
halves can be measured separately (§6.1).

**`epi_var_qknn` / `epi_var_qknn2` — class-quota epistemic**
(`adaptive_roa/adaptive_v2/strategy/class_quota.py`). Rather than *predicting* yield, buy
*composition*: a distance-weighted kNN fitted on the true labels of everything already acquired
gates candidates into predicted-failure and predicted-success pots; the budget is split across
pots by the measured class marginal of acquired data, and each pot is filled by the unchanged
`epi_var` score. Worst case — score useless within a pot — degrades to a control-composition
batch rather than a starved one. `qknn` and `qknn2` are the same configuration; `qknn2` is the
relaunch after a label-convention bug was fixed (§8).

**`epi_var_yield` — yield-aware, the winning arm**
(`adaptive_roa/adaptive_v2/strategy/yield_aware.py`). Rank by informativeness per unit of budget:

```
score'(x) = score(x) · E[L(x)]^alpha
E[L(x)]   = p̄(x)·L_success + (1 − p̄(x))·L_failure
```

`L_success`/`L_failure` are estimated **online from trajectories already acquired**, never from
the candidate (whose length is unknown before rollout), falling back to priors until each class
has ≥5 examples. So the strategy uses only the ensemble's own p̄ plus statistics of data already
paid for. Because the coupling is read from data, the same expression steers the opposite way on
cartpole without a code change. The weight is applied to the score's *magnitude* with the sign
restored, so a strongly-negative debiased score is not promoted by a larger weight. `alpha=0`
reduces **exactly** to `epi_var`, making this a single-knob comparison.

Replicated at seeds 43 and 44 (`yield_s43`, `yield_s44`).

### 3.3 Arms isolating the mechanism (launched 2026-08-16 12:11, in flight)

**`length_only` — expected length alone** (`…/strategy/length_only.py`). The control that the
campaign was missing. Ranks by `E[L(x)]` and ignores the uncertainty score entirely for
selection, while still computing and recording it. If this reproduces `epi_var_yield`'s result,
the epistemic term contributes nothing and the honest description is "buy long trajectories".
Because L_failure ≫ L_success here, ranking by E[L] is monotone *decreasing* in p̄, so this arm
greedily buys the model's predicted-**failure** region — the opposite pole from every scored arm.

**`yield_a05` / `yield_a20` — alpha = 0.5 and 2.0.** Turns the single winning point into a
dose-response. With `alpha=0` known to lose by 3.32 floor units on KL and `alpha=1` to win by
1.87, these fill in the curve and test whether the win is a mechanism or a lucky setting.

---

## 4. Standard of evidence

A gap counts only if it exceeds the **run-to-run floor**: 2×SD over three genuinely distinct
seeds of the *same* configuration, pooled across epochs as `2·sqrt(mean_e(var_e))`, with epoch 0
excluded. Pooling matters — with three seeds each epoch's SD has two degrees of freedom.

Measured from `dir00`, `dir00_s43`, `dir00_s44`, all complete at 20/20:

| metric | full ep1–19 | converged ep12–19 |
|---|---|---|
| sAUROC | 0.00155 | **0.00151** |
| KL | 0.03045 | **0.01693** |
| recal | 0.00153 | **0.00148** |

Verdicts below use the converged floor and the pooled ep15–19 window.

An independent check on the floor: over ep15–19 the three control seeds span only **0.26–0.28
floor units** on every metric. The floor is, if anything, conservative.

---

## 5. Results

### 5.1 Matched trajectory budget — the campaign's own budget

Pooled ep15–19, gaps in converged-floor units, positive = better than `dir00`:

| arm | sAUROC | KL | recal | sA | KL | recal |
|---|---|---|---|---|---|---|
| **`epi_var_yield`** | 0.9803 | **0.0209** | 0.00126 | +1.17 | **+1.87** | +1.34 |
| **`yield_s43`** | 0.9792 | **0.0240** | 0.00264 | +0.43 | **+1.68** | +0.40 |
| `epi_var_anch` | 0.9790 | 0.0354 | 0.00212 | +0.35 | +1.01 | +0.76 |
| `epi_var_qknn2` | 0.9794 | 0.0379 | 0.00221 | +0.60 | +0.86 | +0.69 |
| `dir00_s43` *(control)* | 0.9789 | 0.0481 | 0.00282 | +0.26 | +0.26 | +0.28 |
| `dir00_s44` *(control)* | 0.9786 | 0.0485 | 0.00304 | +0.05 | +0.24 | +0.14 |
| `epi_var_qknn` | 0.9776 | 0.0482 | 0.00386 | −0.61 | +0.25 | −0.42 |
| `dir00` *(control)* | 0.9785 | 0.0525 | 0.00324 | — | — | — |
| `total` | 0.9763 | 0.0565 | 0.00507 | −1.45 | −0.23 | −1.24 |
| `epi_bald` | 0.9745 | 0.0886 | 0.00625 | −2.69 | −2.13 | −2.02 |
| `epi_var` | 0.9747 | 0.1086 | 0.00650 | −2.56 | −3.32 | −2.19 |

### 5.2 What replicates and what does not

**KL replicates.** +1.87 (seed 42) and +1.68 (seed 43), both far outside the 0.26-unit
control-seed spread. `epi_var_yield` reduces KL from the control's 0.0481–0.0525 to 0.0209–0.0240
— a **2.1–2.4× reduction at matched trajectory budget**, with 59–66% of the control's training
pairs. Final epochs: 0.0154 (s42), 0.0205 (s43), 0.0443 (control).

**sAUROC and recal do not replicate.** Seed 42 gave +1.17 and +1.34; seed 43 gave +0.43 and
+0.40, inside the control seeds' own range. An earlier version of this claim — that the arm beats
uniform "on all three metrics" — was true of seed 42 alone and is **withdrawn**. This is the
outcome the headroom analysis predicted in §1.

**The defensible claim is: yield-aware acquisition beats non-adaptive sampling on KL at matched
trajectory budget, replicated across two seeds; the sAUROC and recal gaps are within seed noise.**

Third seed (`yield_s44`) at 18/20 and will extend or qualify this.

### 5.3 Matched pair count — where selection quality shows

KL as a function of cumulative training pairs, pooled over 3 seeds per group (each row is a bucket
of epoch-points, not a single aligned epoch):

| cumulative pairs | control KL (n) | yield KL (n) | ratio |
|---|---|---|---|
| 0–150k | 0.3808 (3) | 0.4269 (17) | 0.89× |
| 150–300k | 0.2759 (6) | 0.1287 (7) | **2.14×** |
| 300–450k | 0.1727 (6) | 0.0596 (12) | **2.90×** |
| 450–600k | 0.0948 (9) | 0.0433 (8) | **2.19×** |
| 600–800k | 0.0657 (9) | 0.0226 (8) | **2.91×** |
| 800–1050k | 0.0788 (12) | 0.0169 (2) | **4.66×** |
| 1050–1400k | 0.0443 (12) | — | — |

Below 150k pairs the yield arm is *worse* than uniform — that is the pre-transition regime where
p̄ is miscalibrated and the weight is inert (§6.3). Above it the arm is 2.1–4.7× better at every
pair budget and the advantage does not decay with data. The control's non-monotonicity between
the 600–800k and 800–1050k buckets is epoch-to-epoch variation, not a trend.

This is the comparison that motivated the whole design: the selection was always more informative
per pair; it was the conversion rate from budget to pairs that lost.

---

## 6. Mechanism (measured, not inferred)

### 6.1 The anchored arm separates the two halves cleanly

`epi_var_anch` splits its budget 50/50 between uniform (`d1`) and scored (`d2`). Pooled over all
20 epochs:

| half | traj succ | mean len | pairs per 100 traj | PAIR succ |
|---|---|---|---|---|
| `d1` uniform | 0.402 | 655 | 65,366 | 0.0851 |
| `d2` scored | **0.900** | **221** | **21,998** | **0.5454** |
| pool reference | 0.403 | 653 | — | 0.0837 |

The uniform half reproduces the pool to three digits, as it must. The scored half buys
90%-success, 221-step rollouts at **one third** the pair rate. This is the central finding
isolated inside a single run, with the uniform half acting as an in-run control. It also explains
why `anch` finished mid-pack (+1.01 on KL): halving the exposure dilutes the deficit rather than
fixing it.

### 6.2 Length and epistemic uncertainty are anti-correlated on this system

`length_only` records the epistemic score of its selected set without using it for selection. Six
epochs so far:

| epoch | score, candidate pool | score, selected | p̄ selected |
|---|---|---|---|
| 0 | 3.0427e-03 | **0.0000e+00** | 0.0000 |
| 1 | 3.7233e-03 | **0.0000e+00** | 0.0000 |
| 2 | 1.1009e-02 | **0.0000e+00** | 0.0000 |
| 3 | 8.6934e-03 | **0.0000e+00** | 0.0000 |
| 4 | 8.4875e-03 | **0.0000e+00** | 0.0000 |
| 5 | 1.2733e-02 | **0.0000e+00** | 0.0000 |

Zero to machine precision, every epoch, against a nonzero pool mean — all five members return
identical probabilities on every selected point. The longest-expected rollouts are exactly where
the ensemble agrees most. So `length_only` is a genuine opposite pole (maximum pairs, zero
information) rather than a near-duplicate of `yield_aware`, and it is *not* drifting toward the
uncertain band as the model improves.

**This is the discriminating test and it is not yet decided.** Verdict at ep15–19.

### 6.3 The yield weight has three regimes, reproduced across all three seeds

`E[L]` routes through p̄, which is exactly what is miscalibrated early — at epoch 0 the ensemble
called 99%-success states likely failures. Measured expected-to-actual pair ratio:

| phase | epochs | batch success | E[pairs]/actual |
|---|---|---|---|
| weight inert (p̄ wrong) | 0–5 | 0.91–1.00 | 3.3–7.6× over |
| transition | 6–9 | 0.13–0.64 | 0.94–1.72× |
| working | 10+ | oscillates 0.31–0.93 | ~0.9–2.4× |

All three seeds pass through these at the same depths. The pre-registered falsification test —
E/act staying above 3× through epoch 6 would mean the weight is inert on this system — is
**passed** by every alpha arm run so far.

### 6.4 Alpha dose-response (in flight, 8/20 epochs)

| alpha | E/act by epoch (0→7) | cum pairs @ep6 |
|---|---|---|
| 0.5 | 8.3 · 3.1 · 2.3 · 4.2 · 3.8 · 2.8 · **2.5** · 1.4 | 134,473 |
| 1.0 | 7.2 · 4.8 · 2.0 · 4.9 · 2.1 · 2.1 · **1.3** · 1.6 | 223,489 |
| 2.0 | 8.2 · 1.9 · 4.0 · 2.4 · 2.9 · 1.6 · **1.6** · 1.1 | 214,665 |

With `alpha=0` (= plain `epi_var`) losing by 3.32 floor units, the emerging shape is a **plateau
from alpha≈1 to alpha≈2 rather than a monotone gain** — alpha=0.5 is clearly behind both, while 1
and 2 are close. If that holds at ep15–19 it says the knob is not delicate. Eight epochs in, this
is the direction of the acquisition diagnostics, **not a result**; pair volume is not the outcome,
and alpha=1 beat uniform *despite* having fewer pairs than the control.

---

## 7. In flight

| run | job | depth | what it decides |
|---|---|---|---|
| `yield_s44` | 209954 | 18/20 | third seed of the KL replication |
| `length_only` | 210229 | 6/20 | whether the win is the score or the pairs |
| `yield_a05` | 210230 | 7/20 | dose-response below alpha=1 |
| `yield_a20` | 210231 | 7/20 | dose-response above alpha=1 |

Health across the whole campaign: **258 epochs screened by magnitude
(`max(success_mae, failure_mae, overall_mae) > 1.0` or non-finite), zero flags**; endpoint MAE
band 0.225–0.343. `filter_diagnostics` null in every `results.json` on every scored run.

---

## 8. Cautions and corrections earned in this sweep

**Screen instability by magnitude, not by non-finiteness.** A non-finite-only screen cleared a
cartpole arm that was carrying `success_mae = 9.4e+09`. The screen used here flags
`> 1.0` OR non-finite.

**Acquisition indices are shuffled-order positions.** They index the shuffled order, not
`train.npz` directly: map `perm[idx]` via `train_test_splits/shuffled_indices_0.txt` first.
Verified — `shuffled_labels[i] == labels[perm[i]]` matches at 1.000000, direct indexing at 0.635.
Every pair count in this document uses the mapped form.

**Labels are {−1, +1}, not {0, 1}.** `data_source.get_label` returns −1 for failure. A
`y == 0` failure test in `class_quota.py` made its quota gate report `n_labeled_failure=0` out of
57 and rendered the gate inert; `epi_var_qknn` is that inert run and `epi_var_qknn2` the fixed
relaunch. Both are reported.

**The acquisition strategy is instantiated once, in the parent.** `engine.py` builds it at
construction, so a code fix cannot reach a running job — the arm must be relaunched. Conversely
`mp.spawn` trainer children *do* re-import strategy modules from disk each epoch, which is why
`yield_aware.py` and `length_only.py` are new modules rather than flags on an existing one.

**`filter_confident_pairs` was a real confound and its direction was not the intuitive one.** It
discarded 58–97% of new pairs. The prediction was that it penalised the control most; measurement
showed the opposite — removing it helped the epistemic arms (ΔKL −0.085, −0.099) and slightly
*hurt* the control (+0.010). Every run scored here has it off.

**Do not read single epochs.** Over this campaign a single-epoch reading was wrong repeatedly: the
yield arm looked falsified at ep0, looked like a breakthrough at ep8–9, and `qknn2` was called
falsified on its ep0 gate before finishing ahead of the control on all three pooled metrics. Use
pooled multi-epoch windows.

**Quote effect sizes and floor depth, never a bare verdict.** Carried over from `FINDINGS.md` and
re-earned here: the sAUROC/recal claim for `epi_var_yield` survived one seed and died on the
second, while the KL claim held.
