# Research Journal — Classifier vs. Flow-Matching for ROA / Reach-Stability

**Started:** 2026-06-21 ~01:45 EDT (overnight autonomous run)
**Driver:** Claude (Opus 4.8), unattended. User AFK.
**Method:** Hypothesis → Experimental runs → Evidence → Accept/Reject.

## Central question
Is a **discriminative classifier** (state → p(success)) + **threshold optimization** (λ*/δ*) + **adaptive sampling** as powerful — or more — than learning a **generative flow-matching "reach/stability" dynamics model** (state → outcome distribution via MC rollouts) for Region-of-Attraction estimation?

Sub-questions:
- What is the *power* of flow matching here that a classifier lacks? (multimodal outcome distribution; per-step trajectory/separatrix reasoning; p_invalid as a 3rd outcome; uncertainty from generative variance.)
- Where does each win: ROA F1, coverage (conservative F1), abstention, false-positive rate, **data efficiency**, **compute efficiency**?

## Systems & schedules (matched across both model classes)
| system | data dir | init → max (incr) | n_epochs | pool avail |
|---|---|---|---|---|
| pendulum | pendulum_lqr_50k | 50 → 500 (50) | 10 | 49,770 |
| cartpole | cartpole_pybullet | 300 → 1000 (100) | 8 | 116,242 |
| quad2d | quadrotor2D_rl | 3000 → 12000 (1000) | 10 | 489,789 |
| quad3d | quadrotor3D_lqr | 10000 → 100000 (10000) | 10 | 800,000 (full pool, via all_shuffled_*) |

Each system run BOTH: **adaptive** (sampling_mode=ranked, d2_ratio=1.0 — 1000-uncertain/epoch via top-N non-conformity) and **non-adaptive** (d2_ratio=0.0 — random/epoch). Classifier uses one_sided decision rule (binary). Eval on the system's fixed held-out test grid.

## Compute-feasibility note (already a finding)
- **Classifier eval** = 1 forward pass over the grid → cheap at any grid size.
- **FM eval** = `num_mc_samples_eval` ODE-solves × grid_size → grid 48k (pend) feasible; 990k (quad3d) infeasible at default MC=100.
- Consequence: FM runs use reduced `num_mc_samples_eval`; quad3d FM at full 100k scale is NOT run fresh (uses existing Jan-2026 FM runs as reference). This compute asymmetry is **Evidence for H4** below.

---

## Hypotheses

### H1 — Parity on committed accuracy
At matched data budget, the classifier's non-abstained F1 ≥ FM's confident F1 on ROA classification.
- **Prediction:** classifier ≈ or > FM on F1(non-abstained), per system.
- **Status:** OPEN.

### H2 — Adaptive > non-adaptive, for both model classes
Uncertainty (boundary) sampling beats random at matched budget, for classifier AND FM.
- **Prediction:** adaptive conservative-F1 / abstain better than random in mid-budget epochs; converges late as pool of uncertain points depletes.
- **Status:** PARTIAL (quad2d classifier earlier: ranked 0.54 vs random 0.34 conservative F1 — supports). Need other systems + FM.

### H3 — FM's distinct power is the 3rd outcome + boundary reasoning
FM models a full outcome distribution → can express p_invalid (separatrix / non-resolving) and per-step reachability the binary classifier cannot. If the ROA boundary has genuine measure (separatrix band), FM should show lower error there / better-calibrated uncertainty.
- **Prediction:** FM has structurally richer uncertainty near the boundary; classifier collapses it to an abstain band. Whether this *helps ROA F1* is the test.
- **Status:** OPEN.

### H4 — Classifier dominates on compute/data efficiency
Classifier reaches comparable ROA quality at far lower compute (no MC, no ODE) and equal/less data.
- **Prediction:** orders-of-magnitude cheaper train+eval; FM eval intractable at large grids. Strongly expected.
- **Status:** SUPPORTED a priori by the eval-cost asymmetry above; quantify wall-clock per system.

### H5 — The classifier's ceiling is boundary coverage, not model class
Both classifier and FM plateau in conservative F1 because training under-covers the separatrix relative to the uniform eval grid (train = trajectory-states, eval = grid). Adaptive partially repairs it; neither fully.
- **Prediction:** both model classes show the same coverage ceiling; the gap is distribution, not generative-vs-discriminative.
- **Status:** OPEN (classifier quad2d showed ~0.5 conservative-F1 ceiling earlier).

---

## Experiment registry
(job name → config; filled at submit; status updated as results arrive)

_(populated below by the run log)_

---

## Run log
- **01:45** Journal created. SLURM verified (test job OK on rlab3). Classifier pathway smoke-tested OK on pendulum + cartpole; quad3d training OK (eval I/O slow). Submitting fleet.

## Experiment registry (submitted 01:55 EDT)
| jobid | name | predictor | system | mode | d2 | schedule |
|---|---|---|---|---|---|---|
| 164288 | clf_pend_a | classifier | pendulum | ranked | 1.0 | 50→500/50, 10ep |
| 164289 | clf_pend_r | classifier | pendulum | random | 0.0 | 50→500/50, 10ep |
| 164290 | clf_cp_a | classifier | cartpole | ranked | 1.0 | 300→1000/100, 8ep |
| 164291 | clf_cp_r | classifier | cartpole | random | 0.0 | 300→1000/100, 8ep |
| 164292 | clf_q2d_a | classifier | quad2d | ranked | 1.0 | 3000→12000/1000, 10ep |
| 164293 | clf_q2d_r | classifier | quad2d | random | 0.0 | 3000→12000/1000, 10ep |
| 164294 | clf_q3d_a | classifier | quad3d | ranked | 1.0 | 10000→100000/10000, 10ep (800k pool) |
| 164295 | clf_q3d_r | classifier | quad3d | random | 0.0 | 10000→100000/10000, 10ep (800k pool) |
| 164296 | fm_pend_a | generative | pendulum | ranked | 1.0 | 50→500/50, 10ep, mc_eval=20 |
| 164297 | fm_pend_r | generative | pendulum | random | 0.0 | 50→500/50, 10ep, mc_eval=20 |
| 164298 | fm_cp_a | generative | cartpole | ranked | 1.0 | 300→1000/100, 8ep, mc_eval=20 |
| 164299 | fm_cp_r | generative | cartpole | random | 0.0 | 300→1000/100, 8ep, mc_eval=20 |
| 164300 | fm_q2d_a | generative | quad2d | ranked | 1.0 | 3000→12000/1000, 10ep, mc_eval=5 (reduced) |
| 164301 | fm_q2d_r | generative | quad2d | random | 0.0 | 3000→12000/1000, 10ep, mc_eval=5 (reduced) |

quad3d FM: NOT submitted fresh (MC eval on 990k grid infeasible). Will reference existing Jan-2026 FM runs + note compute asymmetry (H4).

## Run log (cont.)
- **01:50** Fixed pre-existing FM-eval crash (refine_invalids=False -> None rstats). Committed.
- **01:55** Submitted 14 jobs (8 classifier + 6 FM). FM jobs use num_workers=0 (NFS) + reduced mc_eval.

## Run log (cont.)
- **02:25** Classifier 6/8 done (pend, cp, quad2d). quad3d classifier FAILED (Hydra override word-split artifact in submit helper) → resubmitted directly (164302/3), now RUNNING. FM all 6 mid-run (2-6 epochs).

## EVIDENCE — Classifier, best-over-epochs (band = non-abstained F1; cons = conservative/full-coverage F1)
| system | adaptive band | random band | **adaptive cons** | **random cons** |
|---|---|---|---|---|
| pendulum | 0.999 | 0.980 | **0.996** | 0.978 |
| cartpole | 0.992 | 0.987 | **0.972** | 0.907 |
| quad2d   | 0.932 | 0.909 | **0.421** | 0.344 |

### Verdicts (interim, classifier only)
- **H2 (adaptive > non-adaptive): ACCEPT (classifier).** Best conservative F1: pendulum +0.018, cartpole +0.065, quad2d +0.077 — adaptive wins on all 3; margin grows with problem difficulty. (FM side pending.)
- **H5 (coverage ceiling): REFINE → ceiling is system-dependent, not universal.** pendulum/cartpole reach cons F1 0.97–1.0 (no ceiling); only quad2d collapses (0.42). So the conservative-F1 problem is NOT intrinsic to the discriminative classifier — it appears specifically where the ROA boundary is geometrically hard (quad2d; quad3d TBD). Band (confident) F1 is high everywhere (0.91–0.999): the classifier is reliably correct on what it commits to; what varies across systems is how much of the ROA it must abstain on.
- **H1 (classifier ≥ FM committed accuracy): pending FM completion.**
