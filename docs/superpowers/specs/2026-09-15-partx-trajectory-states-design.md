# Part-X with trajectory states (`partx_faithful_traj`)

A second Part-X arm that gives the algorithm the same labelled states per rollout the
neural arms train on. Everything else is identical to Part-X* (`partx_faithful`,
spec `2026-09-11-partx-faithful-stochastic-design.md`), so a difference between the two
arms is attributable to the data.

## 1. Why this exists

Part-X* trains on one `(x0, outcome)` pair per evaluated rollout. The FM, CLF and BNN
arms build their rows with `TrajectoryDataSource.get_all_endpoint_pairs_from_trajectory`,
which uses `traj[0:-1]`: every state before `x_T`, each carrying the rollout's label, and
`x0` only for timeouts. At the same rollout budget the neural arms therefore train on many
labelled states per rollout and Part-X* on one.

For success and wall-failure rollouts those extra labels are valid. A rollout from `x0`
that succeeds within `T` succeeds from `x_t` within `T - t`, so within `T`; a crash from
`x_t` is a failure the same way. Timeouts are the exception, since `x_t` never got its
full horizon, which is what `timeout_intermediates=drop` removes. The label on `x_t` is a
draw of `p(success | x_t)` only if the controller acts on the current state alone; the
neural arms rely on the same assumption.

## 2. Decisions

| # | Decision | Choice |
|---|---|---|
| T1 | Role | A second arm beside Part-X*. The paper is not updated yet |
| T2 | Where the states enter | Acquisition, not only the readout: they are samples in the tree during the run |
| T3 | Model | Identical to Part-X*: exact Laplace probit GP, Matern-5/2, 5 restarts, `max_points: 2000` |
| T4 | Which states | Full parity with the neural arms: `traj[1:-1]` of every evaluated rollout, no thinning; none for timeouts |
| T5 | Runs | Handed to the `timeout-correction-comp` session once implemented and piloted |

T3 was chosen over raising the cap after costing it (section 3): the added states pile
into a few leaves far above any affordable cap, so a larger cap buys little on quadrotor3D
at cubic cost, and an unchanged cap keeps the data as the only difference.

## 3. Measured basis

From the finished seed-42 `partx_faithful` runs (every rollout they evaluated, initial
prefix included; added states are `traj[1:-1]`):

| cell | rollouts | timeouts | added states | per rollout (median, max) | outside S | data |
|---|---|---|---|---|---|---|
| cp baseline | 2,000 | 0 | 23,530 | 11.8 (1, 122) | 0.0% | x12.8 |
| q2d baseline | 8,000 | 0 | 218,897 | 27.4 (3, 658) | 0.0% | x28.4 |
| q3d f_0.00 | 15,000 | 0 | 395,304 | 26.4 (5, 314) | 0.0% | x27.4 |
| q3d f_0.40_a0.04 | 15,000 | 36 | 442,224 | 29.5 (4, 997) | 0.0% | x30.5 |

No added state falls outside the input space S, because S is the box of the pool's
recorded start states. Routed into each run's final tree, the largest leaf holds 4,662
(cp), 12,630 (q2d), 23,735 and 60,441 (q3d) points, with medians of 201 to 1,170.

GP fit wall time at 6 threads, 13-D, 5 restarts: 12.5 s at 1,000 points, 56.5 s at 2,000,
163.9 s at 3,000, 394.9 s at 4,000.

Estimated GP fit time per run at `max_points: 2000`, lower to upper bound, reconstructed
from the old trees (the new trees will differ): cartpole 0.7-5 h, quadrotor2D 3-27 h,
quadrotor3D 5-40 h. Part-X* takes about 7 min, 1 h and 2-2.5 h.

## 4. Algorithm changes

All other behaviour is Part-X* as specified.

1. **Source.** For pool index `p` with trajectory length `T`, the added states are rows
   `1 .. T-2` (that is, `traj[1:-1]`) with the label of `p`. None when
   `_drops_intermediates(p, T)` is true, using `timeout_intermediates=drop` and the
   horizon from the dataset description.
2. **When.** Immediately after every pool take, in `PartX._take`, which every acquisition
   path goes through (SampleBO top-up and EI steps, continued sampling, the final phase).
   Once for the initial prefix at construction.
3. **Where.** Each state goes to the current leaf whose box contains it, not necessarily
   the region that acquired the rollout. States are clipped into S first (measured 0%
   outside, so this is a guard). Leaves partition S; the first match in region-id order
   wins on a shared face.
4. **Store.** States are stored with `pool_idx = -1`, the existing non-pool convention.
   `attach` already skips those when rebuilding the used-mask and checking labels, and
   `PoolSampler.used` is never touched by them.
5. **Counts toward.** Region sample counts, so the n0 top-up, every SampleBO fit and the
   plug-in `f*`, MCstep and Classify, the equation-4 indicator, split inheritance (by
   position, as for any sample), and the readout fits.
6. **Does not count toward.** The budget (`used`), the acquisition log (`log`),
   `d2_indices`, the schedule and the epoch checkpoints, all of which stay in rollouts.
   Pool candidates, `snap_design` and EI candidates remain pool start states only.
7. **Snapshot.** Epoch `e`'s readout uses the initial ids, the rollouts in `log[:n_new]`,
   and the states of exactly those rollouts. A pickled map `traj_ids[sample_id] -> [state
   ids]` supplies them.
8. **Determinism.** Routing is immediate and region processing order is unchanged
   (ascending region id, seeded), so a run replays exactly. A state that lands in a region
   already processed in the current iteration is used at that region's next fit.
9. **Resume.** The trajectory provider is detached in `__getstate__` like the sampler and
   reattached by `attach`.

## 5. Configuration

- `partx_faithful.trajectory_states: false` by default, so Part-X* is unchanged.
- The key enters the config hash only when it is true, so every existing Part-X* run keeps
  its hash and still resumes.
- Run directories: `tf_<sys>_<tag>_partx_faithful_traj`, with `_sNN` for seeds other than 42.

## 6. Diagnostics

`final_results.json` and epoch diagnostics add `n_traj_states` (states added),
`n_traj_rollouts` (rollouts that contributed states) and `n_timeout_rollouts`, beside the
existing fit counters.

## 7. Validation

Unit tests (`tests/partx_faithful/`):

- a non-timeout rollout adds exactly `max(T-2, 0)` states with its label; a timeout adds none;
- each state lands in the leaf whose box contains it, including a leaf other than the
  acquiring region;
- `used`, `log` and the budget are unchanged by added states;
- the epoch-`e` snapshot includes only states of rollouts acquired by then;
- `trajectory_states: false` reproduces the current run trace exactly;
- a pickled run with states resumes exactly, and `attach` still rejects a pool whose labels
  differ.

Pilot, before any campaign: cartpole baseline, seed 42, on a direct box. Record wall time,
fit count and capped-fit count against the section 3 estimate, and whether the root still
branches. Then one quadrotor3D cell.

## 8. Rollout

1. Implement with tests on `feat/partx-faithful`.
2. Cartpole pilot, then one quadrotor3D pilot.
3. Hand the campaign to `timeout-correction-comp`: the nine paper cells (cartpole
   baseline/low/med, quadrotor2D baseline/smooth/loud, quadrotor3D f_0.00/f_0.12_a0.03/
   f_0.40_a0.04) at seeds 42, 62 and 72, with `shuffle_variant` keyed on the seed and the
   campaign's common block (`n_epochs=11 num_workers=0 adaptive_v2.filter_confident_pairs=false
   ++data_source.timeout_intermediates=drop`). CPU-only; it does not need GPUs.

## 9. Risks

- **Cost.** Section 3's quadrotor3D estimate is 5-40 fit-hours per run. The pilot replaces it.
- **Correlated states make the GP overconfident.** States along one rollout are near
  duplicates, but the Laplace likelihood treats every label as independent, so posterior
  variance shrinks faster than the information warrants. Part-X uses those intervals for
  control flow (MCstep, Classify), so regions may classify earlier than they should. The
  neural arms share the correlation but do not branch on it.
- **Root classification.** Tighter intervals could make the root classify before branching
  more often or less often than the 2-of-9 seed-62 and seed-43 cases seen in Part-X*. The
  pilots report it.
- **Heavy tails.** A few long rollouts supply most added states (medians 1-5, maxima up to
  997), so they dominate the regions they cross.
