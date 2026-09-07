# Adaptive ROA via Conditional Flow Matching + Conformal Inference (System-Agnostic)

This document gives an **idea-level, system-agnostic, mathematical** description of the method implemented by the adaptive training scripts (e.g., `scripts/run_adaptive.py system=cartpole_pybullet`, `scripts/run_adaptive.py system=pendulum`, `scripts/run_adaptive.py system=quadrotor2d`, `scripts/run_adaptive.py system=quadrotor3d`) and the post-hoc conformal re-evaluation scripts (e.g., `scripts/reevaluate_cartpole.py`).

The core concept is:

> Learn a **conditional generative model of trajectory endpoints** (via *flow matching*) and use **Monte Carlo + conformal prediction** to (a) produce a set-valued success/failure/unknown prediction with finite-sample coverage, and (b) **adaptively select new training trajectories** concentrated near the ROA boundary (the separatrix).

---

## 1) Problem setting: ROA as a probabilistic classification task

Consider a closed-loop dynamical system with state space \(\mathcal{X}\). In this repository, \(\mathcal{X}\) is often a **product manifold**, e.g.
- \(\mathbb{R}^n \times \mathbb{S}^1 \times \mathbb{R}^m\) (angles),
- \(\mathbb{R}^3 \times \mathrm{SO}(3) \times \mathbb{R}^6\) (quaternions/orientation),
but we keep the description generic.

Let \(x_0 \in \mathcal{X}\) denote an initial state and \(x_T\) the terminal state at a fixed horizon \(T\) under the closed-loop policy/controller.

Define a **goal attractor set** \(\mathcal{A}_{\text{goal}}\subset\mathcal{X}\) (e.g., “upright and stabilized” or “hover near origin”), and optionally a failure set \(\mathcal{A}_{\text{fail}}\) and/or a constraint-violation condition. The code uses a classifier
\[
h:\mathcal{X}\to\{-1,0,+1\}
\]
that assigns an endpoint label:
- \(h(x_T)=+1\): SUCCESS (endpoint is in/near \(\mathcal{A}_{\text{goal}}\)),
- \(h(x_T)=-1\): FAILURE (endpoint in failure basin or violates a terminal/failure condition),
- \(h(x_T)=0\): INVALID / SEPARATRIX / UNKNOWN (endpoint not confidently in either basin; interpretation is system-specific).

The **region of attraction (ROA)** for the goal can be viewed as
\[
\mathcal{R} \;=\; \{x_0 \in \mathcal{X} \;:\; h(x_T(x_0))=+1\}
\]
in a deterministic setting, or in a stochastic/multi-modal setting as a **success probability level set**
\[
\mathcal{R}_\tau \;=\; \{x_0 \in \mathcal{X} \;:\; \mathbb{P}[h(X_T)\!=\!+1 \mid x_0] \ge \tau\}.
\]

This repo targets the stochastic/multi-modal view: *predict a distribution over endpoints*, then classify by probabilities, with an explicit **abstention band** near the boundary.

---

## 2) Data: offline trajectory pool and endpoint pairs

Assume an offline pool of trajectories
\[
\tau_i = (x_{i,0}, x_{i,1}, \dots, x_{i,T_i}) \quad \text{with a trajectory-level label } y_i \in \{-1,+1\}.
\]

The training scripts construct an **endpoint dataset** by turning each trajectory into many supervised pairs:
\[
\mathcal{D}_{\text{end}} = \{(x_{i,t},\, x_{i,T_i}) \;:\; i \in \mathcal{I}_{\text{train}},\; t=0,\dots,T_i-1\}.
\]
Intuitively: *every state along the trajectory is treated as a “start state” that maps to the same terminal outcome*.

Notes:
- The flow-matching model is trained on \((x, x_T)\) pairs only; **trajectory labels are not used** in the generative training loss.
- Labels \(\{-1,+1\}\) are used later for threshold tuning and conformal calibration, operating on start states \(x_0\).

---

## 3) Model: conditional generative endpoint predictor via flow matching

### 3.1 What is learned?

We fit a conditional generative model \(p_\theta(x_T \mid x)\) that approximates the distribution of terminal states given a start state.

In the implementation this is a **latent-conditional flow matching model** (a continuous-time generative model closely related to diffusion / continuous normalizing flows), with:
- a base noise \(x_{\text{noise}} \sim \pi_0\) in a normalized coordinate system,
- an optional latent \(z \sim \mathcal{N}(0,I)\),
- a time-dependent velocity field \(v_\theta(\cdot)\) parameterized by a neural network.

Sampling (inference) is done by solving a (Riemannian) ODE on the state manifold:
\[
\frac{d x(t)}{dt} = v_\theta\big(x(t),\, t,\, z;\, c\big), \qquad x(0) = x_{\text{noise}},
\]
where the condition \(c\) is derived from the start state \(x\) (e.g., an embedding of \(x\)).
The output \(x(1)\) is interpreted as an endpoint sample \(x_T \sim p_\theta(\cdot \mid x)\).

### 3.2 Training objective (flow matching on manifolds)

Flow matching trains \(v_\theta\) to match the *velocity of a known path* between noise and data. Concretely:

1. Sample a target endpoint \(x_1\) from the endpoint dataset and a noise point \(x_0 \sim \pi_0\).
2. Sample \(t \sim \mathrm{Uniform}(0,1)\).
3. Construct an interpolated point \(x_t\) along a **geodesic probability path** \(\varphi_t(x_0,x_1)\) on the manifold (implementation uses `GeodesicProbPath`).
4. The path object returns both \(x_t\) and the corresponding target velocity \(\dot{x}_t = \frac{d}{dt}\varphi_t(x_0,x_1)\).
5. Minimize the mean-squared error
\[
\min_\theta \;\mathbb{E}\Big[\big\|v_\theta(x_t,t,z;c)-\dot{x}_t\big\|^2\Big]
\]
optionally with per-component weighting to balance physical scales.

This yields a conditional generative model that can produce multi-modal endpoint samples efficiently.

---

## 4) From endpoint samples to success/failure probabilities

Given a start state \(x\), estimate the categorical probabilities induced by the learned endpoint distribution:
\[
p_{+}(x) = \mathbb{P}(h(X_T)=+1 \mid x),\quad
p_{-}(x) = \mathbb{P}(h(X_T)=-1 \mid x),\quad
p_{0}(x) = \mathbb{P}(h(X_T)=0 \mid x).
\]

The pipeline computes Monte Carlo estimates with \(K\) samples:
\[
\widehat{p}_{+}(x) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[h(x_T^{(k)})=+1],\quad
\widehat{p}_{-}(x) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[h(x_T^{(k)})=-1],\quad
\widehat{p}_{0}(x) = 1-\widehat{p}_{+}(x)-\widehat{p}_{-}(x).
\]

This “probability estimator” is exactly what drives both conformal inference and adaptive sampling.

---

## 5) A selective classifier: thresholds \((\lambda,\delta)\) and decision rules

We want a classifier that can say:
- “SUCCESS” when we are confident,
- “FAILURE” when we are confident,
- “UNKNOWN/SEPARATRIX” near the boundary.

Introduce parameters \(\lambda \in (0,1)\) and \(\delta \in [0, 0.5)\).

### 5.1 One-sided rule (binary success vs not-success)

Used when failure is essentially “not success”:
\[
\hat{y}(x) =
\begin{cases}
+1 & \text{if } \widehat{p}_{+}(x) > \lambda+\delta,\\
-1 & \text{if } \widehat{p}_{+}(x) < \lambda-\delta,\\
0  & \text{otherwise.}
\end{cases}
\]

### 5.2 Two-sided rule (separate success and failure evidence)

Used when “failure” is a distinct terminal mode:
\[
\hat{y}(x) =
\begin{cases}
+1 & \text{if } \widehat{p}_{+}(x) > \lambda+\delta,\\
-1 & \text{if } \widehat{p}_{-}(x) > 1-(\lambda-\delta),\\
0  & \text{otherwise.}
\end{cases}
\]

In practice there is often an additional “invalid” heuristic driven by \(\widehat{p}_0(x)\) (e.g., mark invalid if \(\widehat{p}_0(x)\) exceeds some threshold). The repo currently defaults this invalid threshold to \(\lambda-\delta\) in several places; treat that as an **implementation heuristic**, not a universal principle.

---

## 6) Threshold optimization: learning \(\lambda^\*\) and/or \(\delta^\*\)

The scripts choose \((\lambda,\delta)\) by grid search on labeled training points \(\{(x_i,y_i)\}\) to trade off:
- **misclassification rate** among confident predictions, and
- **unknown rate** (fraction of abstentions).

Define:
- \(U(\lambda,\delta)\): unknown/abstention rate under the chosen decision rule,
- \(M(\lambda,\delta)\): misclassification rate *restricted to confident predictions*.

Then optimize
\[
(\lambda^\*,\delta^\*) \in \arg\min_{\lambda,\delta} \;\; w\,M(\lambda,\delta) + (1-w)\,U(\lambda,\delta),
\]
where \(w \in (0,1]\) weights “being wrong” vs “abstaining”.

Two common modes:
- **Optimize \(\lambda\)** with fixed \(\delta\) (search \(\lambda\in[\delta,1-\delta]\)),
- **Optimize \(\delta\)** with fixed \(\lambda=0.5\) (search \(\delta\in[\delta_{\min},\delta_{\max}]\)).

This step is *not yet conformal*; it’s a training-time choice of a risk/coverage operating point.

---

## 7) Conformal inference: set-valued predictions with coverage

### 7.1 Nonconformity scores

Conformal prediction wraps the selective classifier with a **calibrated slack** \( \hat{q} \) so that prediction sets satisfy finite-sample coverage.

Define a nonconformity score \(s(x,y)\) that measures how inconsistent it is to claim label \(y\) at state \(x\), given the estimated probabilities and thresholds \((\lambda^\*,\delta^\*)\).

One convenient design (used in this repo) is “distance to the decision region”, e.g. for the one-sided case with \(\ell=\lambda^\*-\delta^\*, u=\lambda^\*+\delta^\*\):
- \(y=+1\): \(s(x,+1)=\max(0,\,u-\widehat{p}_{+}(x))\)
- \(y=-1\): \(s(x,-1)=\max(0,\,\widehat{p}_{+}(x)-\ell)\)
- \(y=0\): \(s(x,0)=\max(0,\,\ell-\widehat{p}_{+}(x),\,\widehat{p}_{+}(x)-u)\)

Analogous two-sided scores use both \(\widehat{p}_{+}(x)\) and \(\widehat{p}_{-}(x)\).

### 7.2 Calibrating \(\hat{q}\) (a quantile of calibration scores)

Given a calibration set \(\{(x_i,y_i)\}_{i=1}^n\) that is **exchangeable** with the test distribution, compute scores on the true labels:
\[
s_i = s(x_i,y_i).
\]

Then set
\[
\hat{q} = \mathrm{Quantile}_{\;\min\left(1,\;(1-\alpha)\frac{n+1}{n}\right)}(\{s_i\}_{i=1}^n),
\]
which is the standard finite-sample correction used in split conformal prediction.

### 7.3 Prediction sets and interpretation

The conformal prediction set is
\[
\Gamma(x) = \{y \in \{-1,0,+1\}\;:\; s(x,y)\le \hat{q}\}.
\]

Interpretation:
- \(\Gamma(x)=\{+1\}\): confident success,
- \(\Gamma(x)=\{-1\}\): confident failure,
- \(\Gamma(x)=\{0\}\): confidently “invalid/separatrix”,
- \(|\Gamma(x)|>1\) (or empty): uncertain / ambiguous.

**Coverage guarantee (distribution-free):** if calibration and test are exchangeable and the nonconformity computation is treated as part of the algorithm, then
\[
\mathbb{P}\big(Y \in \Gamma(X)\big) \ge 1-\alpha.
\]

Practical nuance: the pipeline uses Monte Carlo probability estimates and a stochastic generative model, which introduces additional algorithmic randomness. In practice, the guarantee is best viewed as holding for the *randomized algorithm*, and it becomes more stable as MC sample count increases.

---

## 8) Adaptive sampling: pool-based active learning near the separatrix

The training scripts implement an **offline, pool-based active learning loop**:

- You have a large pool of precomputed trajectories.
- You iteratively select which trajectories to include in training.
- Selection is biased toward *uncertain* initial conditions, which concentrates data near the ROA boundary.

### 8.1 D1 / D2 split per epoch

Each epoch adds `samples_per_epoch` new trajectories, split into:
- **D1 (calibration)**: uniformly sampled from the remaining pool, used to calibrate \(\hat{q}\) (and then added to training).
- **D2 (adaptive)**: selected from the remaining pool as “informative” states near the decision boundary (and then added to training).

The ratio is controlled by `d2_ratio`:
\[
|D1| \approx (1-\texttt{d2\_ratio})\cdot N,\qquad |D2| \approx \texttt{d2\_ratio}\cdot N,
\]
where \(N=\texttt{samples\_per\_epoch}\).

### 8.2 Selection criteria = uncertainty sampling / margin sampling

The adaptive scripts support multiple selection modes; conceptually:

1. **Conformal uncertainty sampling** (`sampling_mode=conformal`):
   - Calibrate \(\hat{q}\) on D1 at level \(\alpha_{\text{sampling}}\).
   - For candidate states \(x\) from the pool, compute \(\Gamma(x)\).
   - Keep states with \(|\Gamma(x)|>1\) (“ambiguous”) and often also \(\Gamma(x)=\{0\}\) (“invalid”), discard singleton confident states.

2. **Direct threshold band** (`sampling_mode=direct`):
   - Skip \(\hat{q}\).
   - Keep states that lie in the raw \([\lambda^\*-\delta^\*,\lambda^\*+\delta^\*]\) uncertainty band (or its two-sided analogue).

3. **Ranked (margin) sampling** (`sampling_mode=ranked`):
   - Skip \(\hat{q}\).
   - For many candidates, compute a nonconformity score for the “unknown” label (e.g., \(s(x,0)\)).
   - Select the lowest-scored points (closest to the boundary / most ambiguous).

These are standard active learning heuristics (“uncertainty sampling” / “margin sampling”), but driven by a learned generative endpoint model instead of a direct classifier.

---

## 9) Evaluation and re-evaluation (statistical guarantees on held-out sets)

Within the training loop, the scripts also evaluate on held-out datasets:

1. **Held-out calibration set**: used to compute an evaluation-time \(\hat{q}_{\text{eval}}\) at \(\alpha_{\text{eval}}\).
2. **Held-out test set**: used to report ROA metrics (F1/accuracy on confident predictions, separatrix rate, and conformal coverage/avg set size when \(\hat{q}\) is used).

The separate `scripts/reevaluate_*.py` utilities automate “post-hoc conformal inference”:
- load saved model weights for each epoch,
- reload \((\lambda^\*,\delta^\*)\) from training,
- recompute \(\hat{q}\) on the held-out calibration set with user-specified \(\alpha_{\text{eval}}\), MC sample count, and attractor radius,
- re-run evaluation on the test set.

These scripts also contain optional filtering rules based on \(\widehat{p}_0(x)\) (invalid/separatrix probability). The default filtering threshold is currently coupled to \(\lambda^\*-\delta^\*\); this should be treated as a configurable heuristic rather than a principled universal choice.

---

## 10) Summary in ML terms

This method can be summarized as:

- **Generative modeling:** learn a conditional endpoint distribution \(p_\theta(x_T\mid x_0)\) via manifold-aware flow matching.
- **Monte Carlo UQ:** estimate class probabilities by sampling endpoints and classifying them.
- **Selective classification:** define a success/failure classifier with abstention via thresholds \((\lambda,\delta)\).
- **Conformal prediction:** calibrate a nonconformity threshold \(\hat{q}\) to obtain **set-valued predictions with finite-sample coverage**.
- **Active learning / adaptive sampling:** iteratively expand the training subset by selecting trajectories whose initial states are most uncertain under the current model, concentrating data near the separatrix and improving ROA estimates efficiently.

