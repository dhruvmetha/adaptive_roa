# Q-hat Calibration Pipeline

How the conformal prediction calibration works in `scripts/reevaluate.py`, from MC sampling through to prediction sets.

## Overview

The pipeline has 5 stages:

```
Flow Matcher + MC Sampling
    → (p_s, p_f, p_inv) per state
        → Filter invalid cal points
            → Non-conformity scores on cal set
                → q_hat (quantile of scores)
                    → Prediction sets at test time
```

---

## 1. The Three Outcomes: How classify_attractor Works

Every MC sample produces a single endpoint. The system's `classify_attractor(endpoint, radius)` maps that endpoint to one of three labels:

| Label | Meaning | When assigned |
|-------|---------|--------------|
| `1` (SUCCESS) | Endpoint is within `radius` of the goal attractor | `dist(endpoint, goal) < radius` |
| `-1` (FAILURE) | Endpoint is near a failure attractor or exceeded termination thresholds | System-specific |
| `0` (SEPARATRIX) | Endpoint is in neither region | Not close to any attractor |

### System-specific classification

**Pendulum** (state = `[θ, θ̇]`):
- SUCCESS: within `radius` of the stable bottom equilibrium `[0, 0]`
- FAILURE: within `radius` of the unstable top equilibria `[±2.1, 0]`
- SEPARATRIX: not within `radius` of any of the 3 attractors
- Distances use circular wrapping for `θ` and Euclidean for `θ̇`

**CartPole** (state = `[x, θ, ẋ, θ̇]`):
- SUCCESS: combined distance to `[0, 0, 0, 0]` < `radius` (circular wrapping on `θ`)
- FAILURE: **exceeds termination thresholds** (not proximity-based):
  - `|x| > 5.9` (cart hit wall)
  - `|ẋ| > 4.9` (cart velocity too high)
  - `|θ̇| > 4.9` (angular velocity too high)
- SEPARATRIX: neither in attractor nor failed — the trajectory "ended up somewhere in between"

**Key insight**: For Pendulum, failure is proximity to a specific attractor. For CartPole/Quadrotors, failure is hitting a termination boundary. This difference motivates the two decision rules.

---

## 2. Monte Carlo Probability Estimation

`ProbabilityEstimator.estimate()` runs K forward passes per state, each with a fresh latent `z ~ N(0, I)`:

```
For each state x_i in calibration set (N states total):
    success_count = 0, failure_count = 0, invalid_count = 0

    For k = 1 to K:    # K = num_mc_samples
        z_k ~ N(0, I)  # Fresh latent each time (internal to flow matcher)
        endpoint_k = flow_matcher.predict_endpoint(x_i)  # ODE solve with z_k
        label_k = system.classify_attractor(endpoint_k, radius)

        if label_k ==  1: success_count += 1
        if label_k == -1: failure_count += 1
        if label_k ==  0: invalid_count += 1

    p_s[i] = success_count / K
    p_f[i] = failure_count / K
    p_inv[i] = invalid_count / K
```

For each state: **p_s + p_f + p_inv = 1.0** (they're a simplex).

The stochasticity comes entirely from the latent `z`. Different `z` values produce different ODE trajectories from the same initial state, so we get a distribution over endpoints.

### What does each probability mean?

- **p_s(x)**: Fraction of MC samples that land in the success attractor. High p_s → model is confident this state converges to goal.
- **p_f(x)**: Fraction that land in a failure region. High p_f → model is confident this state fails.
- **p_inv(x)**: Fraction that don't resolve to either attractor. High p_inv → the model's trajectories "go nowhere" — the ODE doesn't converge cleanly. This is a signal of model uncertainty or states on the separatrix.

---

## 3. Filtering Invalid Calibration Points

Before calibrating q_hat, we optionally remove cal points with high p_inv. Two modes:

### Conformal filtering (default: ON)
```python
conformal_threshold = lambda_star - delta
valid_mask = p_inv < conformal_threshold
```
Uses the same decision boundary parameters. Points where too many MC samples land in "nowhere" are excluded.

### Explicit threshold filtering
```python
valid_mask = p_inv < invalid_threshold   # e.g., 0.5
```
User-specified hard cutoff.

**Why filter?** High-p_inv points are unreliable — the model can't resolve where they end up. Including them in calibration would inflate q_hat (make prediction sets larger) to cover noisy, uninformative points. Better to calibrate on points where the model at least has an opinion.

After filtering: `n_cal_used` points remain (could be less than `n_cal_total`). If 0 remain, q_hat = None and the epoch is skipped.

---

## 4. Non-Conformity Scores

The non-conformity score measures: **"How strange is it to claim label y for this point, given the model's probabilities?"**

Lower score = more conforming (the label fits the probabilities well).

### One-sided rule (Pendulum)

Uses only p_s. Decision regions defined by `[λ-δ, λ+δ]`:

| True label y | Score formula | Intuition |
|-------------|--------------|-----------|
| SUCCESS (1) | `max(0, (λ+δ) - p_s)` | Low p_s makes SUCCESS strange |
| FAILURE (-1) | `max(0, p_s - (λ-δ))` | High p_s makes FAILURE strange |
| UNKNOWN (0) | `max(0, (λ-δ) - p_s, p_s - (λ+δ))` | Being outside `[λ-δ, λ+δ]` makes UNKNOWN strange |

The "uncertain band" is `[λ-δ, λ+δ]`. If p_s falls squarely in this band, UNKNOWN has score 0 (perfectly conforming). If p_s is well above λ+δ, SUCCESS has score 0 but FAILURE has a large score.

### Two-sided rule (CartPole, Quadrotor2D, Quadrotor3D)

Uses both p_s and p_f. Two thresholds:
- `u = λ + δ` — success confidence threshold on p_s
- `v = 1 - λ + δ` — failure confidence threshold on p_f

| True label y | Score formula | Intuition |
|-------------|--------------|-----------|
| SUCCESS (1) | `max(0, u - p_s)` | Need p_s ≥ u to be "conformingly successful" |
| FAILURE (-1) | `max(0, v - p_f)` | Need p_f ≥ v to be "conformingly failed" |
| UNKNOWN (0) | `max(0, p_s - u, p_f - v)` | Neither p_s nor p_f should be confident |

**Geometric view**: Each label defines a region in (p_s, p_f) space. The score is the L∞ distance from the point to that region. Score = 0 means you're inside the region.

---

## 5. Q-hat Computation

Given non-conformity scores for all n calibration points (using their **true** labels):

```python
scores = [score(p_s[i], p_f[i], y_true[i], λ*, δ)  for i in range(n)]

quantile_level = min((1 - α) * (n + 1) / n, 1.0)
q_hat = np.quantile(scores, quantile_level)
```

The `(n+1)/n` factor is the finite-sample correction from conformal prediction theory. It inflates the quantile slightly to account for the fact that n is finite.

**Example**: n=500 cal points, α=0.1 → quantile_level = 0.9 * 501/500 = 0.9018. So q_hat is roughly the 90th percentile of the scores.

**Coverage guarantee**: For any future test point drawn from the same distribution, the probability that the true label is in the prediction set is ≥ 1 - α.

---

## 6. Prediction Sets at Test Time

For a new test point with estimated (p_s, p_f):

```
For each candidate label y ∈ {-1, 0, 1}:
    score_y = nonconformity_score(p_s, p_f, y_candidate=y, λ*, δ)
    if score_y ≤ q_hat:
        add y to prediction_set
```

The prediction set can contain 0, 1, 2, or 3 labels:
- **{1}** → confident SUCCESS
- **{-1}** → confident FAILURE
- **{0}** → confident SEPARATRIX/UNKNOWN
- **{1, -1}** or **{1, 0}** etc. → ambiguous, classified as UNCERTAIN
- **{}** → empty set (shouldn't happen with proper calibration)

In `evaluate_full_roa_fast`, the final classification logic is:

```
if 0 ∈ prediction_set:
    pred = INVALID (separatrix)
elif {1} only:
    pred = SUCCESS
elif {-1} only:
    pred = FAILURE
else:
    pred = UNCERTAIN (ambiguous)
```

---

## 7. Relationship Between λ*, δ, and q_hat

| Parameter | What it is | When it's determined | What it controls |
|-----------|-----------|---------------------|-----------------|
| **λ\*** | Decision boundary center | Optimized on **training** data (grid search) | Where the success/failure/unknown regions sit in probability space |
| **δ** | Decision boundary half-width | Optimized on **training** data (joint with λ* or fixed) | Width of the uncertain band |
| **q_hat** | Prediction set threshold | Calibrated on **calibration** data | How much slack to add — inflating prediction sets to guarantee coverage |

The flow is strictly sequential:
1. **Train** the flow matcher
2. **Optimize** λ* and δ on the training set (or val split) to minimize weighted loss
3. **Calibrate** q_hat on the held-out calibration set to achieve (1-α) coverage
4. **Evaluate** on the test set using all three parameters

λ* and δ define the "ideal" decision regions. q_hat then relaxes them to account for the gap between the model's probabilities and reality, providing a statistical coverage guarantee.

---

## 8. Does the Three-Outcome Model Make Sense?

**Yes, but with an important subtlety about what "invalid" means.**

The three outcomes from `classify_attractor` have clear physical meaning:
- **SUCCESS**: The controlled trajectory reaches the goal. The system is stabilized.
- **FAILURE**: The system hits a termination condition (pole falls, cart crashes) or reaches an unstable equilibrium.
- **SEPARATRIX**: The trajectory ends somewhere in between — it didn't converge to any recognized attractor within the time horizon.

**p_inv is not "model error"** — it's a legitimate outcome. A state near the separatrix genuinely has mixed dynamics: some perturbations (latent z values) send it to success, some to failure, and some leave it wandering. The proportion that wander (p_inv) reflects how "on the boundary" that state is.

**However**, very high p_inv (say > 0.8) can also indicate model pathology — the flow matcher may have poorly learned dynamics in that region, producing endpoints that don't converge to anything meaningful. This is why filtering high-p_inv points from calibration makes sense: they'd add noise without adding signal about the success/failure boundary.

The conformal framework handles this gracefully: the UNKNOWN/SEPARATRIX label (y=0) is a first-class citizen in the prediction sets, not just an error bucket. Points with high p_inv will naturally get UNKNOWN in their prediction set, which is the correct conservative answer.
