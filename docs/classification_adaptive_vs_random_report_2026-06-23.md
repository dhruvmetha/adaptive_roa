# Adaptive vs Random Sampling — Classification ROA Results

**Date:** 2026-06-23
**Source:** `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_classification`
**Analyst note:** Each (system × method) cell is a **single run** (one pool shuffle, one seed). No seed-to-seed variance is available, so confidence in each conclusion is calibrated against the run's own epoch-to-epoch noise (see caveats). CSVs accompany this report (`..._summary.csv`, `..._per_epoch.csv`).

---

## What was run

A discriminative **classifier** predicts whether an initial state lies in the region of attraction (ROA). The experiment is an **active-learning study**: starting from a shared seed dataset, each "epoch" adds a fixed budget of new labeled trajectories and retrains the classifier, for 8–10 rounds. Two acquisition strategies are compared at **identical data budgets**:

- **Adaptive** (`sampling_mode=ranked`, `d2_ratio=1.0`): each round draws a 1000-candidate window from the shuffled pool, scores each candidate's non-conformity relative to the current decision boundary λ\* (label fixed to UNKNOWN), and adds the **most uncertain** (lowest-score, nearest-boundary) candidates.
- **Random** (`sampling_mode=direct`, `d2_ratio=0.0`): adds the **next N** trajectories from the same pre-shuffled pool — a uniform random draw.

Everything else is held constant: same classifier (max 80 epochs, patience 15), same per-epoch budget, same held-out test set, same pool. **Both arms are identical at epoch 0** (verified — epoch-0 metrics match exactly in all four systems), so any divergence is attributable to the acquisition strategy alone.

**Systems & scale** (test set held fixed per system):

| System | Init traj | +per epoch | Epochs | Final budget | Test points | % in-ROA |
|---|---|---|---|---|---|---|
| Pendulum | 50 | +50 | 10 | 500 | 48,770 | 38.6% |
| CartPole | 300 | +100 | 8 | 1,000 | 115,242 | 18.0% |
| Quadrotor-2D | 3,000 | +1,000 | 10 | 12,000 | 479,789 | **8.0%** |
| Quadrotor-3D | 10,000 | +10,000 | 10 | 100,000 | 990,000 | 22.0% |

**Headline metric:** full-coverage **F1** (`lambda_only`, δ=0 — every test point gets a binary success/failure decision). This is the apples-to-apples metric because it does not depend on how many points each pipeline chooses to abstain on. F1 (not accuracy) is the metric of interest: in Quad-2D the in-ROA class is only 8% of states, so a trivial "all-failure" classifier already scores ~92% accuracy.

---

## Executive summary

Across the four systems, **uncertainty-ranked adaptive sampling beats random sampling clearly on the two lower-dimensional systems (Pendulum, CartPole), edges it consistently on Quadrotor-3D, and is a wash on Quadrotor-2D** (the hardest, most class-imbalanced case, where both methods are poor and the run is noisy). The single most striking result is **CartPole**, where random sampling is not just worse but *unstable* — its classifier's F1 collapses to 0.67 mid-run — while adaptive climbs smoothly to 0.975.

| System | Final F1 (adaptive) | Final F1 (random) | Verdict |
|---|---|---|---|
| Pendulum | **0.997** | 0.975 | Adaptive wins (robust) |
| CartPole | **0.975** | 0.900 | Adaptive wins decisively + far more stable |
| Quad-2D | 0.534 | **0.631** | No clear winner; both poor & noisy |
| Quad-3D | **0.850** | 0.834 | Adaptive consistently ahead (modest) |

---

## Per-experiment analysis

### Pendulum (adaptive vs random)

**Verdict.** Adaptive sampling produces a near-perfect ROA classifier (final F1 0.997) and gets there faster than random. Random also improves but plateaus around 0.97–0.98 and never closes the gap. This is a clean, robust win for adaptive — the margin (+0.022 final F1) comfortably exceeds the run's epoch-to-epoch wobble.

**Technical analysis.**
- *Training dynamics:* Adaptive improves nearly monotonically (full-coverage F1: 0.87 → 0.91 → 0.94 → 0.98 at 200 traj → 0.997 at 500). Random is bumpier and stalls (0.97 at 150 traj, still 0.975 at 500 traj — essentially flat for the back half).
- *Data efficiency:* Adaptive reaches F1 ≈ 0.98 by ~200 trajectories; random never reliably exceeds 0.98 even at 500.
- *Config-to-outcome:* With a 38.6%-positive, low-dimensional state space, the decision boundary is a smooth 1-D-ish curve in (θ, θ̇). Uncertainty ranking concentrates labels right on that separatrix, which is exactly where a classifier needs resolution — hence the steady climb to near-perfect.
- *Recommendations:* Pendulum is effectively solved by adaptive at this budget; if anything, reduce the budget to study how few boundary samples adaptive needs (it may saturate well before 500).

### CartPole (adaptive vs random)

**Verdict.** The strongest result in the study. Adaptive reaches final F1 0.975 and trains stably; random reaches only 0.900 **and is unstable** — its full-coverage F1 collapses from 0.91 to 0.67 around 600 trajectories before partially recovering. For a safety-relevant ROA classifier, the stability difference matters as much as the 7.5-point F1 gap.

**Technical analysis.**
- *Training dynamics:* Random's collapse coincides with early stopping firing almost immediately in its later rounds — the best checkpoint lands at training-epoch 0–4 (vs 7–71 for adaptive). The selected models barely train yet still pass a low local validation loss (~0.01). **Hypothesis (not established):** the random val split becomes uninformative as data grows, so early stopping halts before the ROA boundary is learned. An alternative explanation consistent with the same evidence is majority-class collapse on the 18%-positive split — both produce low val-loss with poor test F1. Distinguishing them requires logging val-set class balance / per-class val metrics, which aren't in these artifacts.
- *Evaluation performance:* Adaptive — precision and recall both end high (0.99 / 0.96). Random — recall is the weak axis (final 0.83, dipping to 0.62 mid-run), i.e. it misses in-ROA states.
- *Config-to-outcome:* In the 4-D (x, θ, ẋ, θ̇) space the ROA boundary is more intricate; random draws spend budget in already-resolved interior regions, while ranking keeps feeding boundary cases. This both raises F1 and stabilizes training.
- *Recommendations:* Re-run random with ≥3 seeds to confirm the collapse is systematic rather than one unlucky shuffle. If the val-set hypothesis holds, switch to a fixed held-out val split (independent of the growing pool) for early stopping.

### Quadrotor-2D (adaptive vs random)

**Verdict.** Inconclusive — no clear winner, and both methods are poor. Random has the higher *final* full-coverage F1 (0.631 vs 0.534), but adaptive has the higher *best* (0.695), higher *mean across epochs* (0.565 vs 0.471), and higher recall throughout. The series is dominated by noise, not by method: adaptive's F1 swings 0.69 → 0.58 → 0.53 over the last three rounds while the budget moves only 10k → 12k. A swing that large from 20% more data means training variance exceeds the cross-method gap, so the final-epoch ordering should not be over-read.

**Technical analysis.**
- *The hard case:* 8% positive class and a thin, high-curvature ROA. Accuracy is a misleading ~0.95 for both; the real signal is **recall**, which never exceeds ~0.54 (full coverage) for either method — both classifiers systematically miss in-ROA states.
- *Reconciling the two metric views:* On the *confident-subset* metric (`lambda_delta`), adaptive looks much better (F1 up to 0.93). That is partly an artifact: adaptive abstains on ~7–12% of points — disproportionately the hard boundary points — so its remaining confident set is easier. Under full coverage (`lambda_only`), where those hard points must be classified, the apparent advantage shrinks to noise. The honest read is the full-coverage view: roughly tied, both weak.
- *Config-to-outcome:* With only 8% positives, the 1000-candidate uncertainty window may rarely contain enough true-positive boundary cases to move recall; ranking can't fix a coverage problem it can't see.
- *Recommendations:* This system needs a different intervention than acquisition strategy: (a) class-rebalancing in the loss or sampler to attack the recall floor; (b) widen `n_ranked_candidates` well beyond 1000 so the uncertainty window actually spans the rare positive boundary; (c) run multiple seeds — at this noise level, single-run conclusions are unsafe.

### Quadrotor-3D (adaptive vs random)

**Verdict.** Adaptive wins, modestly but consistently. The final-F1 gap (0.850 vs 0.834, +0.016) is borderline relative to noise, but the stronger evidence is that **adaptive is at or above random at nearly every epoch**, with smoother monotone improvement and consistently higher recall. The consistency — not the size of the final gap — is what makes this a (cautious) win.

**Technical analysis.**
- *Training dynamics:* At this scale (10k–100k trajectories) both classifiers converge almost instantly each round (best checkpoint at training-epoch 0–2). Adaptive's full-coverage F1 rises smoothly 0.74 → 0.85; random rises 0.74 → 0.83 with more wobble.
- *Evaluation performance:* Adaptive holds a small, persistent recall edge (final 0.778 vs 0.755). Precision is comparable (~0.93–0.94) for both.
- *Config-to-outcome:* In the highest-dimensional system the boundary surface is large; even random sampling gets reasonable coverage from the sheer volume of data, which compresses the adaptive advantage — but ranking still buys a steady recall margin by prioritizing boundary states.
- *Recommendations:* Test whether adaptive matches random's final F1 at a smaller budget (e.g. 50k vs 100k) — the data-efficiency win is the likely value proposition here, more than the asymptotic F1.

---

## Cross-experiment comparison

### Axis of variation
Exactly one axis differs: **acquisition strategy** — uncertainty-ranked (`ranked`, d2_ratio=1.0) vs uniform-random (`direct`, d2_ratio=0.0). All architecture, budget, classifier, and test sets are identical, and both arms start from the identical epoch-0 model.

### Where adaptive helps, and why
- **Largest, most reliable gains on the lower-D systems** (Pendulum, CartPole) where the ROA boundary is a well-defined, learnable surface and concentrating labels on it pays off directly.
- **Gains shrink as dimension/scale grows** (Quad-3D): abundant random data already covers the boundary, leaving a small but consistent edge.
- **Gains vanish under severe class imbalance** (Quad-2D, 8% positive): uncertainty ranking can't help recall when positive boundary cases are too rare to enter the candidate window.

### Decisiveness / abstention (secondary, end-to-end pipeline outcome)
Adaptive pipelines abstain less — mean uncertain-band rate is lower in every system (Pendulum 0.8% vs 1.6%, CartPole 1.7% vs 9.3%, Quad-2D 9.2% vs 11.0%, Quad-3D 23% vs 26%). **Caveat:** the band width δ\* (and λ\*) is re-optimized per epoch *per method*, so this compares two differently-tuned decision rules, not raw classifier confidence. It is a legitimate statement about the *deployed pipeline* (adaptive yields a more decisive end-to-end classifier), not proof the underlying network is intrinsically more confident.

### Overall ranking
1. **CartPole-adaptive** — best margin over its baseline (+0.075 F1) plus a stability win.
2. **Pendulum-adaptive** — near-perfect (0.997), robust margin.
3. **Quad-3D-adaptive** — consistent small win.
4. **Quad-2D** — no winner; both methods weak.

### Recommended next experiments (prioritized)
1. **Multi-seed runs (highest priority).** Every conclusion here rests on n=1 per cell. Re-run with ≥3 shuffles/seeds, at minimum for CartPole (confirm the random collapse) and Quad-2D (where single-run noise currently dominates the method effect).
2. **Fix the CartPole instability mechanism.** Log per-class val metrics and val-set balance; test a fixed (pool-independent) validation split for early stopping. This isolates "uninformative val set" vs "majority-class collapse."
3. **Attack Quad-2D recall directly.** Class-balanced loss/sampler + larger `n_ranked_candidates` (≫1000) so the uncertainty window spans the rare positive boundary. Acquisition strategy is the wrong lever here.
4. **Data-efficiency curves.** For Pendulum/Quad-3D, find the budget at which adaptive matches random's *final* F1 — quantify the savings, which is adaptive's clearest selling point.
