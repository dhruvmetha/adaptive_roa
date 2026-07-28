# Probabilistic Predictors for the Adaptive ROA Pipeline

**Date:** 2026-07-27
**Status:** Design approved, implementation not started

## Goal

Extend the adaptive ROA pipeline from two probabilistic predictors (MLP classifier,
flow matching) to a full benchmark across two categories — **outcome-probability
predictors** and **final-state predictors** — with Gaussian-process and Bayesian
neural network members in each.

The purpose is a **benchmark comparison study**: run every arm across all four
systems under identical conditions and measure which family produces better ROA
estimates and better-calibrated uncertainty. This prioritizes uniform interfaces,
matched training budgets, and comparable metrics over depth on any single method.

## Background: what already exists

Two model-handle contracts exist, and everything downstream is duck-typed against
them:

| Contract | Call | Backend |
|---|---|---|
| Outcome | `handle(raw_states) -> logits` | `ClassifierProbabilityBackend` |
| Final-state | `handle.predict_endpoint(states) -> [B,D]` | `EndpointMCProbabilityBackend` |

`EndpointMCProbabilityBackend` calls `predict_endpoint` K times on the same batch
and counts `system.classify_attractor` labels
(`adaptive_roa/conformal/probability_estimator.py:143`). The spread across those K
calls *is* the outcome probability.

`adaptive_roa/partx/model_handle.py` already proves the contract is duck-typed
rather than class-based: `GPModelHandle` is not an `nn.Module`; it implements
`eval()`, `to()`, and `__call__`, converting GP `p_success` back into logits.

Gaps this design closes:

- No deterministic endpoint regressor exists in `adaptive_v2`. The closest is
  `adaptive_roa/partial_trajs/model/regressor.py:23` (`DynamicsRegressor`), a T-step
  dynamics map in a separate pipeline — a template, not a drop-in.
- GP works *only* with partx acquisition. `ranked`, `conformal`, and `direct` all
  read `probability_backend.estimator` (`adaptive_v2/strategy/ranked.py:44`), which
  `GPProbabilityBackend` does not expose.
- `adaptive_v2/interfaces.py:12-20` `PredictorTrainer` is stale; the real signature
  is `fit(dataset_files: dict, output_dir, resume_checkpoint=None)`.
- Nothing in the codebase models observation noise.

## Architecture

### The unification

Every arm is a **(backbone, posterior, head)** triple. The *head* alone decides
which existing backend it plugs into. Posterior and backbone vary underneath
without downstream awareness.

| Head | Output | Probability formed by | Backend reused |
|---|---|---|---|
| Outcome | `p(success \| x)` | marginalizing the weight posterior | `ClassifierProbabilityBackend` |
| Final-state | distribution over `x_T` | sample K endpoints → `classify_attractor` → count | `EndpointMCProbabilityBackend` |

**Consequence: no new probability backends, acquisition strategies, evaluators, or
conformal code.** A new arm is a trainer, a model handle, and a config. This works
because a GP or BNN posterior draw substitutes cleanly for a flow-matching latent
draw.

### Component layout

New package `adaptive_roa/predictors/`:

| File | Contents |
|---|---|
| `heads.py` | `OutcomeHead` (1 logit); `FinalStateHead` (manifold-aware likelihood) |
| `posteriors.py` | `Deterministic`, `MFVI`, `Ensemble`, `LastLayerLaplace`, `HMC` — uniform `sample_forward(x)` / `marginal_prob(x, S)` |
| `bayesian_mlp.py` | `BayesianMLP(backbone, posterior, head)` |
| `gp_regressor.py` | Multi-output sparse variational GP (independent SVGP per output dim) |
| `hmc.py` | Hand-rolled leapfrog HMC with dual-averaging step-size adaptation |
| `handles.py` | `OutcomeModelHandle`, `FinalStateModelHandle` |

New trainers in `adaptive_roa/adaptive_v2/trainers/`:
`bayesian_mlp_trainer.py` (covers 7 arms via `posterior` × `head`) and
`gp_regressor_trainer.py`.

The backbone reuses `ClassifierMLP`'s structure
(`adaptive_roa/model/classifier_mlp.py:29-36`), a plain `nn.Sequential` of
`Linear`/`ReLU`, so swapping `Linear -> VILinear` for MFVI is a local change.

### The determinism asymmetry

The two handles have **opposite** determinism requirements. This is the single
easiest thing to get wrong, and both failure modes are silent.

- `OutcomeModelHandle.__call__` is invoked **once** per query by
  `ClassifierProbabilityEstimator`, and separately by threshold optimization,
  calibration, and evaluation — which must all agree. It therefore marginalizes
  **internally** (`S` posterior draws, default 64 → mean predictive probability →
  back to logits, as `GPModelHandle` does at `partx/model_handle.py:38-41`) using a
  `torch.Generator` seeded at construction. Repeated calls on the same states
  return identical logits.
  Do **not** copy `GPModelHandle`'s `p.clip(eps, 1 - eps)` guard: that code runs
  in float64, where `1 - 1e-12` is representable. In float32 it rounds to exactly
  `1.0`, the guard becomes a no-op, and a saturated posterior yields `+inf`
  logits. Compute the average in log space instead —
  `logit(p_bar) = logsumexp_i logsigmoid(s_i) - logsumexp_i logsigmoid(-s_i)` —
  which is exact and needs no clamp.
  Posteriors with **finite** support (the deep ensemble: M members, weight 1/M)
  must **enumerate** that support rather than sample it. Sampling M atoms with
  replacement under the handle's fixed seed produces a fixed, biased mixture
  weight vector reused for every prediction in the run.
- `FinalStateModelHandle.predict_endpoint` is invoked **K times on the same
  batch**, and the spread across calls is the probability. It must draw a
  **fresh** posterior sample every call.

Seeding the final-state handle yields `p in {0,1}` for every arm. Not seeding the
outcome handle makes calibration disagree with evaluation.

## Posteriors

All share one backbone and differ only in how weights are drawn.

**Deterministic.** Point estimate. Used for the existing MLP classifier and for the
`mlp_det` final-state baseline.

**MFVI (Bayes-by-Backprop).** `VILinear` layers carrying `(mu, rho)`, weights
`w = mu + softplus(rho) * eps`. Loss `= NLL + beta * KL(q||prior) / N_train`.
Doubles parameter count. The only genuinely new optimization loop.

**Ensemble.** M independent deterministic backbones (M=5 default), different seeds
and data shuffling. Outcome head averages member probabilities. Final-state head
picks a member uniformly at random per `predict_endpoint` call, so K calls give an
M-atom empirical posterior — this implies **K >= 2M** to resolve. Trains M times
longer; the main cost driver.

**Last-layer Laplace.** Train the backbone deterministically (reusing the existing
trainer path), then fit a Gaussian over last-layer weights via the GGN
`H = sum_n phi_n^T Lambda_n phi_n + tau*I`, with `Lambda = p(1-p)` for the outcome
head and `Lambda = 1/sigma^2` for final-state. Restricting to the last layer keeps
`H` small and PSD, needing no new dependency.

**HMC (reference only).** Hand-rolled leapfrog with dual-averaging step-size
adaptation — matching what Staber & Da Veiga and Yao et al. actually ran
(fixed-length HMC, not NUTS). Uses **tanh or GELU, not ReLU**: Dinh et al. (2024)
show ReLU non-differentiability degrades leapfrog local error to `Omega(eps)`.

## Heads

### Outcome head

One logit; `sigmoid` gives `p(success)`. Trained with
`binary_cross_entropy_with_logits` and `pos_weight` for class imbalance, matching
the existing `ClassifierModule`.

### Final-state head: a predictive likelihood, not a point

A BNN predicts by marginalizing `p(y|x,D) = integral p(y|x,w) p(w|D) dw`. A
mean-only MSE head omits the likelihood `p(y|x,w)`, which is not cosmetic:

- **It breaks the Laplace arm.** The regression GGN needs `Lambda = 1/sigma^2`.
  With mean-only MSE training there is no `sigma`, so it defaults to 1 — an
  arbitrary constant scaling the entire posterior covariance. (Same trap as
  `laplace-torch`'s `sigma_noise=1.0` default.) The arm would produce numbers, and
  they would be meaningless.
- **It makes the ensemble arm a different method than the one cited.**
  Lakshminarayanan et al. (2017) §2.2.1 trains each member on Gaussian NLL with a
  predicted variance, and combines via the law of total variance
  (`mean of aleatoric variances + variance of means`). Ensembling mean-only MSE
  members drops the aleatoric term entirely.
- **It removes the standard metrics** — NLL, CRPS, calibration coverage.
- **It is provably overconfident.** Kendall & Gal (2017): the epistemic term
  *"will vanish when we have zero parameter uncertainty"*, while true residual
  variance is bounded below by the noise floor. Asymptotic undercoverage is
  structural, not incidental.

Both data regimes make this concrete: in `deterministic/`, `sigma` learns to be
small and the arm degrades gracefully toward the deterministic map; in `noisy/`
(the 4-level stochastic pendulum) the same `x_0` gives different endpoints, which a
mean-only head structurally cannot represent.

### Manifold-aware likelihood

Built generically from `system.manifold_components`, one likelihood per component,
NLLs summed:

| Component | Systems | Parameters | Sampling |
|---|---|---|---|
| `Real` (d dims) | all | `mu`, `log sigma` | `mu + sigma*eps` |
| `SO2` (1 dim) | pendulum, cartpole, quad2d | `(sin, cos)` mean direction → `mu_theta = atan2`; `log kappa` | wrapped normal / von Mises |
| `SO3` (4 dims) | quad3d | unit quaternion `q_bar`; tangent covariance in `so(3)` | `xi ~ N(0,Sigma)`, `q = q_bar (*) exp(xi/2)` |

The `SO2` treatment is required, not stylistic: pendulum's **failure** attractors
sit at `theta = ±pi`, exactly the wrap point, so a Euclidean Gaussian on `theta`
would place mass in a false mode.

### Two head refinements

**beta-NLL (`beta = 0.5`).** Plain Gaussian NLL down-weights high-error points as
`sigma` grows there, starving them of gradient (Seitzer et al., ICLR 2022;
Detlefsen et al., NeurIPS 2019). Without this, an arm can fail for optimization
reasons we would misread as a method result.

**Optional MDN likelihood.** Near a separatrix the true endpoint distribution is
*bimodal* — the trajectory lands at attractor A or B, not between. A unimodal
Gaussian cannot represent this; flow matching, being generative, can. So
`FinalStateHead` takes `likelihood in {gaussian, mdn}` with `n_components`; the
`Real`/`SO2`/`SO3` structure is unchanged, each component gaining a mixture
dimension. Lakshminarayanan et al. (2017) explicitly sanction this: *"In cases
where the Gaussian is too-restrictive, one could use a complex distribution e.g. a
mixture density network."*

Metrics are **reported broken out by distance-to-separatrix** so this effect is
measured rather than averaged away.

## Arms and tiers

`predictor.type` keeps its current values but is understood as a **family tag**
driving the six engine branches. A new `predictor.name` uniquely identifies the arm
and keys the export registry and output paths.

| Family (`type`) | `name` | Status |
|---|---|---|
| `classifier` | `mlp` | exists |
| | `gp` | exists; needs unblocking |
| | `bnn_mfvi`, `bnn_ensemble`, `bnn_laplace` | new |
| | `hmc` | new — reference tier only |
| `generative` | `fm` | exists |
| | `mlp_det` | new — non-adaptive baseline (`d2_ratio=0`) |
| | `gp_reg` | new |
| | `bnn_mfvi_reg`, `bnn_ensemble_reg`, `bnn_laplace_reg` | new |
| | `hmc_reg` | new — reference tier only |

Backward compatible: existing configs already carry correct `type` values, and the
registry keeps `classifier`/`generative` as aliases so runs already on disk still
export. Without `name`, all five outcome arms collide on the key `"classifier"` and
the export layer silently loads a BNN checkpoint as a `ClassifierMLP`.

`mlp_det` is a fixed-dataset baseline only. Its outcome probability collapses to
`{0,1}`, so it has no ranking signal and is not run adaptively.

### Two tiers

HMC cost is dominated by parameter count, which is set by hidden widths — nearly
identical across systems:

| System | state_dim | embed | `[256,512,256]` | `[50,50]` |
|---|---|---|---|---|
| pendulum | 2 | 3 | ~264,193 * | ~2,954 * |
| cartpole | 4 | 5 | 264,705 | 3,258 |
| quad2d | 6 | 7 | 265,217 | 3,562 |
| quad3d | 13 | 13 | 266,753 | 4,576 |

`*` analytic; the pendulum bounds file was not resolvable at the configured path
when measuring. Others measured directly.

Going from the 2D pendulum to the 13D quadrotor changes parameter count by **under
1%**, so restricting HMC by system would gate on the wrong variable. The real gate
is width: Izmailov et al. ran full-batch HMC on ResNet-20-FRN (~272k parameters) on
**512 TPUv3 devices**, *"as many computations as over 60 million epochs of standard
SGD training."* Our production backbone is ~265k parameters — the same scale.
Full-width HMC is out of reach on every system, pendulum included.

HMC is also only a valid reference for the **same architecture** the approximations
use; an HMC run at `[50,50]` says nothing about an MFVI posterior at
`[256,512,256]`.

- **Reference tier — `[50,50]`, all four systems.** The four **MLP-backbone**
  posterior arms (MFVI, ensemble, Laplace, HMC) at this width, in both head
  variants. 3.0k–4.6k parameters, inside the 2,651–20,501 range Staber & Da Veiga
  verified as routine. Carries **all posterior-fidelity claims**: agreement and
  total variation vs. HMC, the HMC-vs-HMC ceiling row, and function-space R-hat.
  GP and FM arms are **excluded** from this tier — they are different model
  classes, so an HMC posterior over MLP weights is not a reference for them. They
  are compared on downstream task metrics in the production tier only.
- **Production tier — `[256,512,256]`, all four systems.** Everything except HMC.
  Carries the downstream ROA-task and adaptive-sampling results, with **no**
  posterior-fidelity claims attached.

This matches practice — Foong et al. state plainly that *"HMC is only run for 1 and
2 hidden layers"* — and yields fidelity results on quad2d and quad3d, where the
approximations are most likely to degrade.

## Fixes to existing code

1. **`GPProbabilityBackend` gains `.estimator`** (a `ClassifierProbabilityEstimator`
   over the handle), making GP usable with `ranked`/`conformal`/`direct` rather
   than partx only.
2. **`interfaces.py:12-20` `PredictorTrainer` corrected** to
   `fit(dataset_files: dict, output_dir, resume_checkpoint=None)`.
3. **Two handle Protocols added to `interfaces.py`**, documenting the determinism
   asymmetry, which currently exists only as unwritten convention.
4. **Final-state handles expose `get_manifold_component_names()`** —
   `adaptive/endpoint_evaluation.py:100` calls it *unguarded*, so a final-state arm
   missing it crashes at epoch end, after training is already paid for.

## Experimental protocol

**Random-acquisition control for every arm.** Foong et al. (NeurIPS 2020) found
MFVI-driven **active learning** performed *worse than random* selection (0.94 vs
0.15 RMSE). Adaptive sampling here is active learning, so a plausible outcome is
that MFVI arms lose to random acquisition. Without a paired random control per arm,
that effect is invisible.

**Matched training budgets, trained to convergence.** Mukhoti et al. (2018) show
weak baselines are the characteristic failure of this literature.

**Tempering.** Headline MFVI at `beta = 1` with tempered results in a separately
labelled block (Ober & Aitchison, 2021). Expect a cold-posterior effect anyway from
small N and read it as prior misspecification rather than tuning it away (Noci et
al., 2021).

**Priors.** Isotropic Gaussian, same `sigma` across all arms, reported numerically
alongside a prior-scale sensitivity sweep.

**GP fairness.** GP endpoint samples must be drawn from the **predictive**
distribution including likelihood noise, not the latent function posterior, or the
GP arm gets an unfair aleatoric-free advantage over the BNN arms.

Two objections worth pre-empting in the writeup, both in our favor: Foong's
in-between-uncertainty impossibility is a **1-hidden-layer** result (their Thm 3
shows mean-field is universal at 2+ layers) and our backbone has three; and Coker
et al.'s wide-limit "MFVI ignores the data" result requires odd Lipschitz
activations, with ReLU an explicit counterexample.

## Testing

New `tests/predictors/`, following `tests/adaptive_v2/` and `tests/partx/`
conventions:

- **Posterior units:** MFVI KL finite and > 0; ensemble spread > 0; Laplace GGN
  covariance PSD; **Laplace `sigma` is learned, not left at 1.0**.
- **Head units:** `SO2` round-trip recovers `theta` at `±pi`; `SO3` samples stay
  unit-norm; beta-NLL reduces to Gaussian NLL at `beta = 0`.
- **Contract conformance, parametrized over every arm:** outcome handles return
  identical logits on repeated calls with the same states; final-state handles
  return different endpoints. The determinism asymmetry as an executable test.
- **Degeneracy guard:** for a fitted final-state arm, `p_success` is not
  concentrated in `{0,1}`. Catches an accidentally-seeded handle, which would
  otherwise look like a plausible-but-wrong benchmark result rather than a bug.
- **Smoke e2e per family** via `+adaptive_v2.smoke_mode=true`.
- **HMC artifacts:** function-space R-hat and the HMC-vs-HMC agreement ceiling
  written per run, so the reference arm's own convergence is auditable.

## Out of scope

- Humanoid (`humanoid_standup_reach`) — impractical for GP arms at its state
  dimension and dataset size.
- MC dropout as an arm — considered by the user and not selected.
- Repulsive ensembles, SG-MCMC, SWAG — reasonable follow-ups, not in this pass.
- Migrating FM and the MLP classifier onto a new first-class `ProbabilisticPredictor`
  abstraction. Cleaner long-term but touches engine, threshold, calibration, and
  eval, risking regressions against in-flight experiment runs.

## Risks

| Risk | Mitigation |
|---|---|
| Final-state handle accidentally seeded → all arms report `p in {0,1}` | Degeneracy guard test; contract conformance test |
| Gaussian NLL optimization pathology mistaken for a method result | beta-NLL at `beta = 0.5` |
| MFVI arms lose to random acquisition | Paired random-acquisition control per arm; this is a reportable finding, not a bug |
| Ensemble arm undersampled at K < 2M | Assert `K >= 2M` at config load |
| GP arm gets aleatoric-free advantage | Sample from predictive distribution including likelihood noise |
| Export layer silently mis-loads a BNN as `ClassifierMLP` | `predictor.name` keys the registry |

## References

Cited claims are drawn from a literature review conducted for this design. Items
flagged as unverified there (Foong et al. NeurIPS 2020 page numbers; Staber & Da
Veiga journal placement; Abe et al. 2022 page range) must be confirmed before
submission.

- Nix & Weigend (1994), *Estimating the mean and variance of the target probability
  distribution*, IEEE ICNN — origin of the mean/variance head.
- MacKay (1992), *A Practical Bayesian Framework for Backpropagation Networks*,
  Neural Computation 4(3).
- Neal (1996), *Bayesian Learning for Neural Networks*, Springer LNS 118.
- Blundell et al. (2015), *Weight Uncertainty in Neural Networks*, ICML — Bayes by
  Backprop. Note: prior is a scale mixture of two Gaussians, not isotropic.
- Gal & Ghahramani (2016), *Dropout as a Bayesian Approximation*, ICML.
- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles*, NIPS.
- Kendall & Gal (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for
  Computer Vision?*, NIPS.
- Ritter et al. (2018); Daxberger et al. (2021), *Laplace Redux*, NeurIPS.
- Wenzel et al. (2020), *How Good is the Bayes Posterior in Deep Neural Networks
  Really?*, ICML.
- Wilson & Izmailov (2020), *Bayesian Deep Learning and a Probabilistic Perspective
  of Generalization*, NeurIPS.
- Foong et al. (2020), *On the Expressiveness of Approximate Inference in Bayesian
  Neural Networks*, NeurIPS.
- Izmailov et al. (2021), *What Are Bayesian Neural Network Posteriors Really Like?*,
  ICML, PMLR 139:4629-4640.
- Noci et al. (2021), *Disentangling the Roles of Curation, Data-Augmentation and the
  Prior in the Cold Posterior Effect*, NeurIPS.
- Ober & Aitchison (2021), *Global inducing point variational posteriors for BNNs and
  deep GPs*, ICML.
- D'Angelo & Fortuin (2021), *Repulsive Deep Ensembles are Bayesian*, NeurIPS.
- Seitzer et al. (2022), *On the Pitfalls of Heteroscedastic Uncertainty Estimation
  with Probabilistic Neural Networks*, ICLR.
- Coker et al. (2022), *Wide Mean-Field Bayesian Neural Networks Ignore the Data*,
  AISTATS.
- Jospin et al. (2022), *Hands-On Bayesian Neural Networks*, IEEE CIM 17(2).
- Fortuin et al. (2022), *Bayesian Neural Network Priors Revisited*, ICLR.
- Papamarkou et al. (2024), *Position: Bayesian Deep Learning is Needed in the Age of
  Large-Scale AI*, ICML.
- Dinh et al. (2024), *Hamiltonian Monte Carlo on ReLU Neural Networks is
  Inefficient*, NeurIPS.
- Mukhoti et al. (2018), *On the Importance of Strong Baselines in Bayesian Deep
  Learning*, NeurIPS BDL Workshop.
- Staber & Da Veiga, *Benchmarking Bayesian neural networks and evaluation metrics
  for regression tasks*, arXiv:2206.06779.
- Yao et al., *Quality of Uncertainty Quantification for Bayesian Neural Network
  Inference*, arXiv:1906.09686.
