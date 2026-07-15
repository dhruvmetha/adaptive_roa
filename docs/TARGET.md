---
read_when: "you need to know what object we are actually estimating"
class: living
---

# What we are estimating

Read `PROBLEM.md` first if you don't know why RoA matters.

## The predictors are four different OBJECTS

This is the most misunderstood thing here. `predictor=generative|classifier|gp` are **not four
implementations of one estimator.** They answer *different questions* about the RoA — which is why
comparing them needs machinery at all.

| Predictor | The RoA is… | Uniquely expresses | Costs |
|---|---|---|---|
| **classifier** | a set-membership function `p(success\|x)` | a calibrated scalar per point | one forward pass |
| **generative** (flow matching) | induced by an **endpoint distribution** `p(endpoint\|x)` | multimodality; a third "invalid" outcome | MC × ODE solves |
| **gp / Part-X** | a **measurable set with a volume + credible interval** | set-level bounds — what verification actually wants | GP + partition tree |
| **partial_trajs** | **reachability under learned dynamics** | trajectory structure | ~20 net calls vs ~500 sim steps |

`partial_trajs` is a **T-step forward-dynamics surrogate**, not a classifier. It rolls the learned
map forward and applies the analytic `classify_attractor` after each jump. It is a cheaper
simulator, not a discriminator — and it sits outside the shared harness, so its numbers are not
directly comparable to the other three.

## The common currency

All predictors are forced through one contract:

```
estimate(states) -> (p_success, p_failure, p_invalid)
```

`conformal/estimator_factory.py` hides "MC over generated endpoints" vs "classifier sigmoid" vs "GP
probit" behind that one signature. **This seam is the experimental design, not plumbing** — it is
what makes generative-vs-discriminative a controlled comparison rather than two incomparable
pipelines.

`p_invalid` is **not a ground-truth class**. It is a generative-inference artifact: what happens
when a generated endpoint lands between basins. A discriminative classifier can never produce one —
it emits zero identically. Deterministic label files contain only `{0,1}`; there is no "invalid" on
disk.

## Success is decided by the SYSTEM, not the model

Every predictor emits a *state*; the **system** judges it. `system.classify_attractor(state, radius)`
is the arbiter for every method. No model decides its own correctness.

That is the conceptual core: swap the predictor, keep the judge, and the comparison stays honest.

## Labels mean different things at different layers

`0` means *separatrix* to the systems and *failure* to the evaluator; `-1` means *failure* to the
systems and *uncertain* to the evaluator. The evaluator silently drops its `-1`/`-2`, so reported F1
is computed on a **retained subset** and is not comparable across methods with different abstention
rates — always read F1 together with the abstention rate.

**The codebook itself lives in `tests/test_label_codebook.py`**, not in prose. It kept going stale
here; as a test it fails instead.

## Deterministic vs stochastic — the object changes

**Deterministic:** one rollout per start, so each state has one outcome and one hard label. The RoA
**is a set**; the separatrix has measure zero. Everything above assumes this.

**Stochastic:** the classical set stops existing. Each start is rolled out repeatedly under process
noise, so the honest target is `p(success|x) ∈ [0,1]` and a large fraction of starts have genuinely
mixed outcomes. The boundary becomes an **irreducible aleatoric band** whose width is set by the
noise level.

This matters more than it sounds: **"the RoA" no longer names a unique object.** You must choose one
— an almost-sure basin, a high-probability basin `{x : p ≥ τ}`, or a conservative acceptance set
`{x : LCB_α(p) ≥ τ}` — and fix τ and α. Until that choice is made, the four predictors may be
estimating different things, and "A beats B" is not yet a claim.

It also breaks adaptive sampling's premise. Acquisition chases uncertainty because uncertainty means
*missing data*. In the mixed band the uncertainty is irreducible, so an uncertainty-driven sampler
will spend its whole budget where nothing can be learned. Separating epistemic from aleatoric
uncertainty is an open problem here, not a solved one.

## Why the pendulum's attractors look strange

- **θ = 0 is the UPRIGHT (inverted) equilibrium and it is UNSTABLE.** It is the goal the LQR
  stabilises. The EOM `theta_ddot = (g/l)·sin(theta) + u/I − (b/I)·theta_dot` gives
  `d(theta_ddot)/d(theta)|₀ = +(g/l) > 0`, hence unstable. Both dataset descriptions agree.
- **±2.1 are the TORQUE-SATURATION equilibria, not "the top".** With the controller pinned at
  `u = −u_sat`, gravity balances control where `sin(θ*) = (u_sat/I)/(g/l) = √3/2`, i.e.
  `θ* = 2.0944 rad = 120°`. The pendulum sticks there because saturated torque cannot lift it
  further. That is the failure mode, and why ±2.1 is a failure attractor rather than a magic number.

The code was always right about this; the comments in two separate files were both wrong and agreed
with each other. Consensus among comments is not evidence.
