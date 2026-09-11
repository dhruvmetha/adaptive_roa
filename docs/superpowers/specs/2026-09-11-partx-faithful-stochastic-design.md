# Faithful Part-X for the stochastic RoA benchmarks

Status: design approved in brainstorming, 2026-09-11. Next step: implementation plan.

## 1. Why this exists

The paper tables report a "Part-X (GP)" arm, run as `partx_fix` from `adaptive_roa/partx/`. An
audit on 2026-09-11 found that arm is not Part-X. It is one global sparse GP classifier with
straddle level-set sampling (Bryan et al. 2005), stratified over a partition tree. Compared with
Pedrielli et al., "Part-X" (arXiv 2110.10729) and its reference code
(github.com/cpslab-asu/part-x), it has:

- no local per-region GPs,
- no per-region n0 initial design and no nBO expected-improvement samples,
- no continued sampling of classified regions,
- no budget gate on branching and no final volume-proportional phase,
- a region classifier that takes the 5% quantile of the confidence bounds over 64 points, where
  Part-X takes the max and min over R×M = 1,000 to 10,000 points.

Run diagnostics showed three more problems. On quad3D the root box is classified as all-failure
because the ROA is under 5% of it, and on three of the four levels the arm falls back to global
straddle for 6 to 12 of 20 epochs. On quad2D the tree grows to 237,840 leaves on 13,000 trajectories. And
`select_pool_indices` truncates the batch in leaf-list order once the number of leaves exceeds
the per-epoch pick count.

This design builds a Part-X arm that implements Alg. 1 to 4 of the paper as published, with only
the changes our stochastic, pool-based benchmark forces. It is scored in the same 12 paper cells.

Non-goals:

- Winning on the paper metrics. Part-X is a baseline.
- Changing `adaptive_roa/partx/`. Live jobs may import it.
- An in-loop simulator. Every arm samples from the same pre-simulated pool.
- Per-system hyperparameter tuning on our metrics.
- Manifold-aware partitioning. Part-X partitions hyperboxes, and so do we, including angles.

## 2. Decisions

| # | Decision | Choice |
|---|---|---|
| D1 | What each region's GP models | A GP classifier on the binary outcomes. The latent f gives p(x) = Φ(f), and Part-X's zero level set is p = 0.5 |
| D2 | Which outcome is the falsifier | Failure. The requirement is "the rollout succeeds", so the robustness-oriented latent is f. A config flag `falsify: success` negates the latent. It is off by default and exists only for a later sensitivity run |
| D3 | Scored readout | Part-X's own piecewise model. An eval state takes p̂ from the GP of the leaf that contains it |
| D4 | Scope | All 12 paper cells: pendulum LQR ×3, cartpole PPO ×4, quad2D corridor_sine smooth, quad3D PPO 800k ×4. The pendulum gate (§8.3) runs first |
| D5 | Integration | A standalone runner that writes engine-format epoch directories, not an engine plug-in |
| D6 | Hyperparameters | Match each system to the closest published setting by dimension (§6.5) |
| D7 | Paper vs cpslab code | Where they disagree, follow the paper, except for plain typos in the paper (§4) |
| D8 | Early termination | If Part-X stops with budget left, record the unspent budget. Do not top up |

## 3. Algorithm mapping

Notation: g is the robustness-oriented latent (g = f under `falsify: failure`). Each region's GP
gives a posterior mean m(x) and standard deviation s(x) of g. z = Φ⁻¹(1 − δC/2) throughout.

| Part-X step | This implementation | Status |
|---|---|---|
| Input space S | The box from `system.per_dim_bounds()`, padded by a relative 1e-6 as in `partx/tree.py::build_root` | As published |
| Initial samples | The shared initial set (pool indices 0..n−1, the same set every arm starts from) sits in the root as existing samples. Every system's initial set is at least n0, so Alg. 1 step 1 uses it as is | Adapted |
| Root | Like cpslab: nBO EI steps on the root, then MCstep and Classify, before the main loop | As published |
| Branching | Remaining regions (r, r+, r−) split into B = 2 equal halves. The split dimension is `perm[branch_dir mod d]`, where `perm` is one seeded permutation per run and a child's `branch_dir` is its parent's plus 1 (cpslab `singlereplication.py:64-65,109`). Children inherit the parent's samples that fall inside them | As published |
| Budget gate | Branch only if the remaining budget ≥ Σ, over the children of every remaining region in this iteration, of (max(n0 − n_child, 0) + nBO) (cpslab `singlereplication.py:118,125`). Otherwise run the final phase | As published |
| SampleBO (Alg. 1) | Top the child up to n0 with an LHS design in its box, then take nBO sequential EI steps, refitting the child's GP after each | As published |
| Pool sampling | Each LHS design point snaps to the nearest unused pool start inside the box (Euclidean distance in box-normalized coordinates, greedy in design order, no repeats). Each EI step takes the argmax over unused pool starts inside the box, a random subset of at most 20,000 | Forced |
| EI on binary data | Plug-in EI on g: f* = min over the region's sampled points of m(xᵢ), EI(x) = (f* − m)Φ(u) + sφ(u) with u = (f* − m)/s. EI hunts the minimum of g, so it hunts failures under D2 | Adapted |
| MCstep (Alg. 2) | R replicates of M LHS points in the box, with no evaluations. q̄ᵣ = maxₘ(m + zs), qᵣ = minₘ(m − zs). Q̄ and Q are the means over r, and Var(Q̄) = var(q̄ᵣ)/R, likewise Var(Q). One GP fit per region per MCstep | As published |
| Classify (Alg. 3) | 'u' if v(σ) ≤ δv^d · v(S). A '+' region becomes 'r+' if Q − z√Var(Q) ≤ 0, else stays '+'. A '−' region becomes 'r−' if Q̄ + z√Var(Q̄) ≥ 0, else stays '−'. An 'r' region becomes '−' if Q̄ + z√Var(Q̄) < 0, '+' if Q − z√Var(Q) > 0, else stays 'r' | As published |
| Continued sampling | When the gate passed and classified regions exist, nc_k = min(nc, remaining budget). It is split across '+' and '−' regions by a seeded multinomial with weights ∝ Iⱼ, where Iⱼ is the mean over R·M LHS points of P(g < 0) = Φ(−m/s) (paper eq. 4, volume-normalized). Each region draws its count by LHS, snapped to the pool, then gets refit, MCstep and Classify | As published |
| Final phase | When the gate fails, the whole remaining budget is split across r, r+, r−, + and − regions by a seeded multinomial with weights ∝ volume. Each region draws LHS points, snapped to the pool, then gets refit, MCstep and Classify, and the loop ends | As published |
| Empty regions | A region that needs a sample but has no unused pool start in its box becomes 'i'. cpslab uses 'i' for infeasible regions. If a box has some unused starts but fewer than requested, take them all and record the shortfall | Forced |
| Termination | Stop when the budget is spent or when no r, r+, r−, + or − region remains | As published |
| Output | Region types, v(Θ⁻)/v(S), and the GP-quantile falsification volume at q ∈ {0.5, 0.05, 0.01} (cpslab `fv_using_gp`) go into the diagnostics. The scored p_success field comes from the readout (§5.4) | As published |

## 4. Where the paper and the cpslab code disagree

| # | Paper | cpslab | We follow |
|---|---|---|---|
| 1 | The quantile CI uses √Var (Alg. 2 to 3) | Uses Var (`regionQuantileEstimation.py:130-131`) | Paper |
| 2 | Eq. 4 divides by region volume, so I ∈ (0, 1) | Multiplies by volume (`calculateMCIntegral.py:27`) | Paper |
| 3 | One GP per region per MCstep | Refits the same GP R times (`mc_step`) | Paper. No behavioral difference |
| 4 | Alg. 1 sets t ← n0 after the top-up, then loops `while t < nBO`, which skips BO whenever n0 ≥ nBO | Always takes nBO EI steps (`boClass.py:63`). Alg. 4's budget accounting and the paper's text agree with the code | Code, as a paper typo |
| 5 | Alg. 4's gate reads Σ max(n_jk − n0, 0) | Uses max(n0 − n_child, 0), the number of points needed for the top-up | Code, as a paper typo |
| 6 | "Randomly selects a direction" | One random permutation per run, cycled by depth | Code, as one reading of the paper |
| 7 | Not specified | `assign_budgets` uses the unseeded global `np.random` | The run's seeded generator |

The reference check (§8.2) can switch 1 and 2 back to cpslab's behavior (`compat_ci_var`,
`compat_eq4_times_volume`, both off by default), so that comparison is like for like.

## 5. The local GP classifier

### 5.1 Model

An exact GP classifier with the Laplace approximation (Rasmussen and Williams 2006, Alg. 3.1 and
3.2, with the marginal-likelihood gradient from Alg. 5.1). The runs use the probit likelihood
p(y = 1 | f) = Φ(f). A logistic link exists only for the sklearn cross-check test. The prior is
f ~ GP(μ, k), with a constant mean μ and the Laplace machinery applied to f − μ.

We write our own instead of using sklearn's `GaussianProcessClassifier` for two reasons. sklearn
raises an error on single-class data, and all-failure regions will be common. It also only
offers the logistic link.

### 5.2 Kernel and hyperparameters

- k = σ_f² · Matérn-5/2 with one isotropic lengthscale, on inputs standardized within the region
  (a `StandardScaler` fit on the region's training inputs). This matches cpslab's `InternalGPR`,
  which uses isotropic `Matern(nu=2.5)` on scaled inputs. σ_f² is added because a classifier has
  no `normalize_y`.
- σ_f² and the lengthscale are fit by type-II maximum likelihood with L-BFGS-B on log
  parameters, 5 restarts (cpslab `n_restarts_optimizer=5`), and bounds lengthscale ∈ [0.01, 100]
  and σ_f² ∈ [0.01, 25].
- The mean is μ = Φ⁻¹((k + 1)/(n + 2)), where k is the number of successes among the region's
  n training points. This is the analog of cpslab's `normalize_y`, which sets the mean from the
  data. The add-one smoothing keeps single-class regions finite: 30 failures out of 30 gives
  μ = −1.86. Fitting μ by maximum likelihood would send it to −∞ on single-class data.
- Restart seeds derive from (run seed, iteration, region id, fit counter), so every fit is
  reproducible regardless of process scheduling.

### 5.3 Size cap

Exact Laplace scales as n³. A region with more than 2,000 training points fits on a seeded random
subsample of 2,000. This only affects the root and the first few levels on quad2D (2,000 initial
points) and quad3D (10,000). The number of capped fits goes into the diagnostics.

### 5.4 Readout

p̂(x) = Φ(m(x)/√(1 + s²(x))), using the GP of the leaf containing x, where m and s² are the
latent posterior of f. This is f itself, not the falsify-oriented g. Under `falsify: success`
the readout negates the backend's mean back to f, so p̂ means the same thing under either setting
of D2. A leaf with zero samples uses its nearest ancestor that
has samples, refit on that ancestor's samples as of the snapshot.

### 5.5 Interface

`LocalModel.fit(X, y) -> self` and `LocalModel.latent(X) -> (m, s)` return the robustness-oriented
posterior. There are two implementations: `LaplaceGPClassifier`, used for the runs, and
`SklearnGPRBackend`, used only for the reference check. `SklearnGPRBackend` is configured exactly
like cpslab's `InternalGPR` and takes real-valued robustness.

## 6. The runner

### 6.1 Layout

```
adaptive_roa/partx_faithful/
  gp_laplace.py     LaplaceGPClassifier (probit; logit for tests only)
  backends.py       LocalModel protocol, SklearnGPRBackend
  region.py         Region box, type, parent, depth, branch_dir, sample indices, branch()
  mcstep.py         MCstep and Classify (Alg. 2 and 3)
  sampling.py       LHS, pool-snapping sampler, continuous sampler (reference only)
  bo.py             plug-in EI and SampleBO (Alg. 1)
  algorithm.py      Alg. 4 as a resumable loop that emits an ordered acquisition log
  volume.py         v(Θ⁻)/v(S) and the GP-quantile falsification volume
  readout.py        piecewise-model handle for evaluate_full_roa_classifier
  state.py          atomic pickle and restore
scripts/run_partx_faithful.py          Hydra entry point
scripts/partx_reference_check.py       one-off, runs in a scratch env (§8.2)
configs/adaptive_v2/partx_faithful.yaml
tests/partx_faithful/
```

### 6.2 Configuration

`configs/adaptive_v2/partx_faithful.yaml` has `defaults: [default, _self_]` and adds a
`partx_faithful:` block. The runner uses `config_path="../configs/adaptive_v2"`, the same as
`run_adaptive.py`. So the `system`, `controller`, `noise_family`, `noise_level`, `seed`,
`initial_train_size`, `samples_per_epoch`, `n_epochs` and `output_dir` overrides resolve the data
paths and eval grid identically. Launch scripts change only the entry point and the arm name.

### 6.3 Data flow

1. Build the `TrajectoryPool` as the engine does and call `initialize(initial_train_size)`. The
   initial set is pool indices 0..n−1.
2. Load every pool start state (`states[offsets[r]]`, as `NpzTrajectoryDataSource` does) and every
   label into arrays, with a used-mask. The initial set is marked used.
3. Run Alg. 4 with total budget T = (n_epochs − 1) · samples_per_epoch new evaluations. Each
   evaluation marks one pool index used, reads its label, and appends the index to the
   acquisition log.

| Cell | initial | samples/epoch | n_epochs | T | Checkpoints |
|---|---|---|---|---|---|
| pendulum LQR low/med/high | 100 | 100 | 20 | 1,900 | 100 to 2,000 |
| cartpole PPO baseline/low/med/high | 300 | 150 | 12 | 1,650 | 300 to 1,950 |
| quad2D corridor_sine smooth | 2,000 | 500 | 24 | 11,500 | 2,000 to 13,500 |
| quad3D PPO 800k f_0.00/0.12/0.20/0.40 | 10,000 | 1,500 | 20 | 28,500 | 10,000 to 38,500 |

### 6.4 Checkpoints

The acquisition log has a canonical order. Within an iteration, regions go in ascending region
id, and within a region, samples go in the order drawn: top-up, then EI steps. Continued sampling
and the final phase follow in the same region order. So the log is the same whether regions are
processed serially or in a process pool.

The snapshot for epoch e holds the first b_e − initial log entries plus the initial set, where
b_e = initial + e · samples_per_epoch. Its partition is the leaf set of the iteration that
contains sample number b_e, after that iteration's branching. Epoch 0 is the root with the initial
set only, before root BO. At each snapshot the runner fits the leaves' GPs, caching fits whose
sample sets have not changed, and writes `epoch_XXX/`:

- `full_roa_per_point.npz` and the eval metrics, from the existing
  `evaluate_full_roa_classifier` called with a handle around the piecewise model. The handle
  follows the `partx/model_handle.py` contract (`model(raw_states) -> logits`), so the file format
  and metrics match every other classifier-family arm.
- `artifacts_v2.json` with the same top-level keys as a `partx_fix` artifact:
  - `epoch`, `train_trajectories` (the actual count),
  - `sampling_mode: "partx_faithful"`,
  - `threshold_state` (λ* = 0.5, δ* = 0.1, as for `predictor=gp`),
  - `eval_metrics`,
  - `acquisition.d2_indices`, the pool indices taken since the previous checkpoint, which
    `acquired_true_p.py` reads,
  - `acquisition.diagnostics`: iteration, phase, a count of leaves by type, both falsification
    volumes, unspent budget, 'i' shortfalls, capped fits, and fits that hit a hyperparameter
    bound,
  - `endpoint_error: {"skipped": ...}`, `d1_eval_metrics: null`, `conformal_state: null`,
  - `extra` with the git commit and the full `partx_faithful` config.

Every epoch directory is written. An epoch whose budget exceeds what Part-X actually spent (D8)
carries the final state and its true `train_trajectories`. The scorer truncates a table at the
last epoch every arm reached, so a missing directory would cut every other arm short too.

### 6.5 Hyperparameters

Chosen by dimension from the paper's two published settings, with no tuning on our metrics:

| | n0 | nBO | nc | R × M | B | δC | δv |
|---|---|---|---|---|---|---|---|
| pendulum (d = 2): paper §5.1 with its Table 3 grid | 10 | 10 | 100 | 20 × 500 | 2 | 0.05 | 0.001 |
| cartpole (4), quad2D (6), quad3D (13): paper §5.2, F16 | 30 | 10 | 100 | 20 × 500 | 2 | 0.05 | 0.001 |

R × M = 20 × 500 everywhere because MCstep's max and min have to see small features. With a 1%
ROA, M = 100 leaves 0.99¹⁰⁰ ≈ 37% of replicates with no point inside it. The paper found its 2-D
results insensitive to grid size (Tables 1 to 3), so this is still a published configuration in
every cell.

### 6.6 Resume and parallelism

- State (the tree, per-leaf sample indices, used-mask, RNG state, counters, acquisition log and a
  config hash) is pickled atomically (temp file, then `os.replace`) to `output_dir/partx_state.pkl`
  after every iteration and every checkpoint.
- On start, a matching state file resumes the run. A config-hash mismatch refuses to start.
- An interrupted iteration replays from the saved RNG state. Epoch directories that already have
  both files are skipped.
- GPs are refit from their seeds, not pickled. Replay is exact on the same machine. Across machines
  BLAS differences can change choices after a resume, which is acceptable.
- Within an iteration, regions are independent, because boxes partition the space and each pool
  point lies in exactly one leaf. So regions can be mapped over a process pool, sized from the job's
  CPU allocation.

## 7. Scoring and reporting

- Run directories are `<campaign prefix>_partx_faithful` under the existing experiment roots.
- `scripts/score_stoch_incremental.py`: add `partx_faithful` to `ARMS`. This is an additive
  one-line change. `predictor_of()` already maps `partx*` to "gp".
- Paper scripts: the "Part-X" label moves to `partx_faithful`, and `partx_fix` leaves the paper
  tables. Its rows stay in the CSVs.
- Docs:
  - `METHODS.md` describes the new arm and renames `partx_fix` as a GP straddle-LSE arm.
  - Fix `pendulum/lqr/README.md:65`, which lists the predictor as `gp_reg` when the runs used `gp`.
  - Fix `quadrotor2d/rl/README.md:378-381`, where the under-spending text belongs to the pre-fix arm.

## 8. Validation

### 8.1 Unit tests (`tests/partx_faithful/`)

- **Laplace GP classifier.** With a logistic link and a fixed kernel (`optimizer=None`), predicted
  probabilities match sklearn's `GaussianProcessClassifier` to within 1e-6. Probit
  marginal-likelihood gradients match finite differences. Single-class data fits and gives μ from
  §5.2.
- **MCstep.** On a stub model with constant m and s, Q̄ = m + zs and Q = m − zs.
- **Classify.** The full Alg. 3 truth table for current types r, + and −, plus 'u' by volume.
- **Budget gate.** It chooses branching or the final phase correctly on constructed counts.
- **Branching.** Directions follow `perm[branch_dir mod d]`, and children split their parent's
  samples exactly.
- **Pool sampler.** Snapped points are unused, inside the box and never repeated, and an empty box
  gives 'i'.
- **Allocation.** Eq. 4 and the volume allocation are seeded and reproducible, and their
  proportions are right in expectation.
- **Resume.** A run killed after k iterations and resumed ends in the same state, and with
  byte-identical epoch files, as an uninterrupted run.
- **Scorer compatibility.** `stoch_prob_metrics.score_epoch_full` runs on a produced epoch
  directory.

### 8.2 Reference check against cpslab

This is a one-off script in a scratch Python 3.10 env, because cpslab needs numpy 1.x, treelib and
scikit-optimize and the project env has numpy 2.2.

- Our loop runs in a deterministic mode:
  - the `SklearnGPRBackend`,
  - continuous LHS in place of pool snapping,
  - EI argmax over 10,000 random points in the box,
  - `compat_ci_var` and `compat_eq4_times_volume` on.
- It runs on Himmelblau and Goldstein-Price as defined in paper §5.1, with the §5.1 settings (n0 = 10,
  nBO = 10, nc = 100, T = 5,000, R = 10, M = 100, B = 2, δC = 0.05, δv = 0.001), 10 replications
  per function per implementation. If cpslab's runtime makes that impractical, T drops to 2,000
  for both implementations alike.
- Pass: for each function, our mean GP-quantile falsification volume at q = 0.5 and our mean count
  of leaves by type are within 2 standard errors of cpslab's.
- The paper's Table 1 values are reported next to both implementations for information, not as a
  gate. cpslab's current code may differ from the code behind the paper.

### 8.3 Pendulum gate

Runs pendulum low, med and high at n0 = 10, and med again at n0 = 30. The criteria were fixed
before any results:

1. Every epoch directory exists, and each either reaches its budget or records why not.
2. At the last checkpoint on low, classified ('+' or '−') leaves hold at least 50% of the eval grid.
3. On every level, at most 5% of eval-grid points in classified leaves sit on the wrong side of
   p = 0.5 according to the oracle (`eval_success_prob.npz`). This is the analog of Part-X's
   misclassification guarantee.
4. On every level, the fraction of acquired points with oracle p ∈ [0.2, 0.8] exceeds the same
   fraction in the initial uniform set. This is the concentration near the level set seen in the
   paper's Fig. 4. Oracle p at pool starts is mapped as in `scripts/paper/acquired_true_p.py`.
5. The n0 = 10 and n0 = 30 med runs are reported side by side.

No criterion depends on how Part-X scores on KL, sAUROC or the level-set metrics. If the gate
fails, work stops and the failure is reported before any other cell launches. Bugs get fixed;
design changes need sign-off. Hyperparameters are never changed to pass the gate.

## 9. Rollout

1. Create a worktree on a new branch, `feat/partx-faithful`, off the current
   `feat/quadrotor-stoch` HEAD. The main tree is shared with other sessions.
2. Implement §5 and §6 with the unit tests from §8.1.
3. Run the reference check (§8.2) and write the results to `docs/partx_faithful_reference_check.md`.
4. Run the pendulum gate (§8.3), write `docs/partx_faithful_pendulum_gate.md`, and review it with
   the user.
5. Time a short quad3D probe, since the capped 2,000-point fits at the top levels dominate. Then
   launch cartpole ×4, quad2D and quad3D ×4 through the compute skill, with an hourly poller until
   the runs finish.
6. Make the scoring, paper-script and doc changes from §7.

The work is CPU-bound, from the Laplace fits and MCstep, and needs no GPU.

## 10. Risks

- **Weak local fits.** Local GP classifiers on 10 to 40 binary points are weakly identified, and
  the hyperparameter fits may often hit their bounds. The diagnostics count this.
- **Readout seams.** The piecewise readout has seams at region boundaries, and small leaves will
  calibrate poorly. That is Part-X's model (D3), and it will show in KL and sAUROC.
- **High dimensions.** In 13-D, classification will be rare, and most of the budget will go to
  top-ups and the final phase. The Part-X authors list higher dimensions as future work, and we
  report what happens.
- **Pool exhaustion.** Small boxes in high dimensions can run out of pool points. 'i' regions
  and shortfalls are counted.
- **Angles.** Circular coordinates are partitioned as plain intervals with no wraparound, as in
  Part-X.
