# Audit findings — 2026-07-15

**Frozen record.** Point-in-time findings against `main` @ `d0a4406`. Do not edit; if something here
is later fixed, write a new entry rather than rewriting this one.

Every claim below was verified against the code or the data, not inferred from names or comments.
Anything that could not be verified is marked as such.

---

## Not bugs (investigated and cleared)

### 1. `conformal/calibrator.py` is correct — the "dropped constraint" is not real
The two-sided nonconformity score omits one term of the full L∞ formula for `y=±1`. **This is a
proven reduction, not an omission.** With `u = λ+δ`, `v = 1−λ+δ`, we have `u+v = 1+2δ`; the dropped
term binds only if `p_s + p_f > 1 + 2δ`, impossible since `p_s + p_f ≤ 1` and `δ ≥ 0`.

Verified numerically: **0 disagreements over 2×10⁶ random valid probability triples** (max |Δ| = 0.0).
The `y=0` branch correctly keeps both terms — there neither dominates (~15.6% of cases). The
reduction relies on `δ ≥ 0`, which the configs enforce (`delta_min` > 0).

**Three independent readers flagged this correct code as a bug.** A `NOTE` with the dominance proof
now sits in the function. Do not "restore" the terms. `CONFORMAL_ADAPTIVE_SAMPLING.md` documented the
disproved version and has been archived.

### 2. `adaptive_roa/adaptive/` is not dead code
The v1 *loop* was retired, but its **data layer is load-bearing for `adaptive_v2`** and holds the
repo's **only** DATA_DIR write guard. All three acquisition strategies import `UncertainSampler` from
it. The name misleads: `adaptive_v2` is not its successor, it is a layer that depends on it.

---

## Real, but latent

### 3. Angle wrapping is a no-op in the adaptive_v2 local path
`flow_matching_trainer.py` reads `getattr(self.system, 'angle_indices', None)` — **no system defines
`angle_indices`**; the real API is `get_circular_indices()`. So `None` is always passed and wrapping
is skipped.

Narrow scope: only `prediction_mode=local`, not the global/endpoint path used by the
classifier-vs-FM study. And the on-disk pendulum data is **already wrapped at generation** — 200
trajectories / 78,537 states checked, theta range `[-3.1416, 3.1398]`, **zero** outside `[-π, π]`. So
it is a **broken safety net**, not a live bug. It will bite the first system whose data is not
pre-wrapped. Only pendulum-deterministic was checked.

### 4. Comments that were wrong (now fixed)
`systems/pendulum.py`'s docstring and `configs/system/pendulum.yaml`'s header both described `[0,0]`
as a stable *bottom* equilibrium and `±2.1` as *top* equilibria. Both wrong; the code was right.
θ=0 is the unstable **upright** (the EOM has `+sin(θ)`), and `±2.1 = 2.0944 rad = 120°` is the
**torque-saturation** equilibrium where `sin(θ*) = (u_sat/I)/(g/l) = √3/2`. Two independent sources
agreed with each other and were both wrong. Locked by `tests/test_label_codebook.py`.

---

## Affects every comparison

### 5. F1 is computed on a retained subset
`evaluate_roa` drops its `-1` (uncertain) and `-2` (invalid) predictions before scoring. Methods with
different abstention rates therefore have **non-comparable F1**. Every cross-method comparison run so
far inherits this. F1 must always be read together with the abstention rate.

---

## Infrastructure

### 6. The evidence layer is not durable
`results/` and `outputs/` are gitignored; `results/*.md` (7 files) are untracked and invisible to
collaborators. **No `artifacts_v2.json` exists anywhere** — run artifacts are being discarded.
`job_registry.tsv` has space-vs-tab corruption in rows 16–19 and is missing jobs reported in prose.
A registry should be written by the submit path, not by hand.

Code can be re-read. A deleted overnight run cannot.

### 7. Nothing is gated
No `.github/`, no pre-commit config, no active git hooks. 194 tests run only if someone types
`pytest`.

### 8. The repo has a stale twin checkout
`../adaptive_roa` is the same GitHub repo (`adaptive_roa.git`), frozen on pre-force-push history at
`7742872` (2025-11-06), carrying its own 359-line CLAUDE.md with 24 dead `src/` references. Anyone
opening an agent there gets the poisoning removed from this checkout. Nothing in either tree mentions
the other. Same pattern for `flow_matching` / `flow_matching_wcontrastive_maybe`.
