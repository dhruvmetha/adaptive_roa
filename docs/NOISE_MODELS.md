# Noise models: pendulum, cartpole, quadrotor2D, quadrotor3D

What actually perturbs each system in the `stochastic/` datasets, where the parameters live, and
which numbers are comparable to which.

Every fact here is read from the shipped `dataset_description.json` under
`${DATA_DIR}/stochastic/<system>/<family>/<controller>/<level>/`, cross-checked against the config
headers in `configs/adaptive_v2/system/*_stoch.yaml`. Verified 2026-09-09.

## The one-line version

Two families, one per pair of systems, and they are not on a common scale.

| System | Family | Channel | Law | Matched |
|---|---|---|---|---|
| pendulum | `gaussian_signal` | shaft torque, post-saturation | `w ~ N(0, alpha + beta*abs(u))`, i.i.d. per control step | yes, enters through B |
| cartpole | `gaussian_signal` | cart force, pre-saturation | `w ~ N(0, alpha + beta*abs(u))`, i.i.d. per control step | yes |
| quadrotor2D | `corridor_sine_ambient` | external force at COM, world frame | per-rollout coherent sine + i.i.d. ambient | no |
| quadrotor3D | `corridor_sine_ambient` | external force at COM, world frame | per-rollout coherent sine + i.i.d. ambient | no |

The scale in `N(0, ...)` is a **standard deviation**, not a variance. This trips people up because
the field is named like a variance in some of the older notes.

## Family 1: signal-dependent Gaussian on the action (pendulum, cartpole)

    sigma = alpha + beta * abs(u)          # standard deviation
    w ~ N(0, sigma)                        # zero-order hold, redrawn each control step

`alpha` is the floor that survives as the command goes to zero. It is the only term still acting at
the goal, so it is what blurs the success boundary. `beta` is effort-proportional and acts only in
the transient, so it randomizes the outcome rather than the edge. Widening `alpha` widens the
uncertain band without deepening it: it turns `p=1` cells into `p=0.98`.

Delivered noise is a scale mixture, since every draw carries its own sigma. Its standard deviation
is `sqrt(E[sigma^2])`, not `E[sigma]`. On the cartpole PPO policy those two differ by 6x to 20x,
because mean `abs(u)` is 9 to 16 N while RMS `abs(u)` is 158 to 390 N.

### Parameters

| System | Controller | Level | alpha | beta |
|---|---|---|---|---|
| pendulum | lqr | low | 0.05 | 0.16 |
| pendulum | lqr | med | 0.10 | 0.64 |
| pendulum | lqr | high | 0.20 | 1.00 |
| cartpole | lqr | low | 0.405 | 1.5 |
| cartpole | lqr | med | 0.945 | 3.5 |
| cartpole | lqr | high | 1.08 | 4.0 |
| cartpole | safe_explorer_ppo | baseline | 0.0 | 0.0 |
| cartpole | safe_explorer_ppo | low | 0.3 | 0.5 |
| cartpole | safe_explorer_ppo | med | 0.75 | 0.3 |
| cartpole | safe_explorer_ppo | high | 0.5 | 0.75 |

The cartpole PPO ladder is not monotone in either parameter, and that is deliberate. Levels are
matched on the fraction of eval cells with `p_success` in `[0.2, 0.8]`, not on noise magnitude.
Never read cartpole `med` as "the same noise" as pendulum `med`, and never order the PPO levels by
alpha.

### Placement, and why it matters for the pendulum

The pendulum adds `w` **after** the clip: `sat(u) + w`, with `u_sat = 0.6372`. The physical claim is
an external shaft torque (wind, contact, friction). It is matched, but it is not bounded by the
actuator, because it does not come from the actuator. So the applied torque can exceed `u_sat`, and
the noisy region of attraction is **not** a subset of the deterministic one. Noise can rescue states
the deterministic controller loses.

The cartpole adds `w` **before** the clip, but the LQR clip is +/- 2000 N against a median demand of
0.53 N and a p99 of 137.6 N over 25,845 measured steps. `clip(u+w)` and `clip(u)+w` are the same
function on this data, so the pendulum's pre/post distinction has nothing to bite on here. It is
also why this family cannot rescue at scale on the cartpole: the noise adds no authority the
controller lacks.

On the cartpole PPO branch, `control_bound_N: 10.0` is the policy loop gain, not a saturation bound.
`normalized_rl_action_space` is true, so the policy emits `[-1, 1]` and the env multiplies by
`action_scale`. Do not read it as the LQR's 2000 N clip; they mean different things.

Measured on that PPO policy, alpha does not help. Across a 7x7 alpha-by-beta grid the count of
noiseless failures rescued into sometimes-success falls monotonically with alpha once it passes 0.3:
at `beta = 1.0` it runs 154, 150, 118, 80, 46, 26, 0 as alpha goes 0 to 3. beta rescues, alpha
suppresses.

### Verification on record

For the cartpole, `alpha = beta = 0` reproduces the deterministic labels exactly: 58 of 58 on a
contiguous shard, on both iLab and Amarel, mean p 0.1724 against the deterministic 0.1724.

## Family 2: corridor plus ambient (quadrotor2D, quadrotor3D)

    F = sigma(gate) * (0.5 + 0.5*A*sin(2*pi*t/period + phi)) + N(0, ambient_std)
    sigma(s) = f_max * exp(-0.5*((s - centre)/width)^2)

Applied with `applyExternalForce` at the COM link, world frame, **no torque**. Unmatched: it is
independent of drone velocity and applies no moment.

The part that is easy to get wrong: `A ~ U(0,1)` and `phi ~ U(-pi, pi)` are drawn **once in
reset()** and held for the whole episode. Given `(A, phi)` the corridor term is a deterministic
function of position and time. It is coherent, not white. Only the ambient term is redrawn every
control step, and unlike the corridor it acts everywhere rather than inside the gated band.

That matters beyond bookkeeping. Each rollout carries a per-episode latent, so the aleatoric
variability here has a different structure from the i.i.d. process noise on the pendulum and
cartpole. Two rollouts from the same start differ because they drew different curtains, not only
because they accumulated different per-step kicks.

### Geometry and levels

quadrotor2D, one curtain gated on altitude `z`, pushing one-sided along `+x`. Body weight 0.26487 N.

| Level | f_max | ambient_std | centre | width | period | Share of weight |
|---|---|---|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.55 | 0.12 | 2.0 s | 0.0 |
| smooth | 0.05 | 0.09 | 0.55 | 0.12 | 2.0 s | 0.189 |
| sharp | 0.08 | 0.06 | 0.55 | 0.12 | 2.0 s | 0.302 |

quadrotor3D, twin vertical curtains at `x = +0.9` and `x = -0.9`, both pushing `+y`, gated on `x`.
The drone gets shoved sideways as it crosses. Body weight 0.2646 N.

| Controller | Level | f_max | ambient_std | Share of weight |
|---|---|---|---|---|
| lqr | f_0.25 | 0.25 | 0.008 | 0.944 |
| lqr | f_0.30 | 0.30 | 0.008 | 1.133 |
| ppo | f_0.00 | 0.0 | 0.0 | 0.0 |
| ppo | f_0.12 / f_0.20 / f_0.40 | 0.12 / 0.20 / 0.40 | 0.0 | 0.45 / 0.76 / 1.51 |
| ppo_800k | f_0.12_a0.03 | 0.12 | 0.03 | 0.45 |
| ppo_800k | f_0.20_a0.04 | 0.20 | 0.04 | 0.76 |
| ppo_800k | f_0.40_a0.04 | 0.40 | 0.04 | 1.51 |

Curtain geometry is identical across every quadrotor3D level: centre `+/- 0.9`, width 0.25, period
2.0 s, mask `[0, 1, 0]`, gated on `state_index 0`.

The 3D forcing is roughly 1x body weight at the LQR levels, against a maximum of 0.302x anywhere in
the 2D set. The two systems are not on a common noise scale even though they share the family name.

The `ppo` branch is pure sine with `ambient = 0`. The `_a0.03` / `_a0.04` suffixes on `ppo_800k` add
the ungated wobble back, and that is the only difference from the bare level of the same `f_max`.

`f_0.00` and quadrotor2D `baseline` are deterministic references, not low noise levels. Both measure
`frac_mid = 0.000`, meaning no ground-truth probability mass at all. Exclude them from any
probabilistic comparison. Their KL is a clipped log-loss against binary truth.

## Plant, horizon, and success rule per dataset

| System / controller | ctrl_freq | Horizon | Success rule (as labelled) |
|---|---|---|---|
| pendulum lqr | 100 Hz | 800 steps | per-channel box, `abs(theta) < 0.05` and `abs(theta_dot) < 0.05`, hold 1, entry-cut |
| cartpole lqr | 100 Hz | 1000 steps / 10 s | L2 ball r=0.05 over the 4-D state, entry-cut, goal at origin |
| cartpole ppo | 15 Hz | 150 steps / 10 s | L2 ball r=0.05 over the 4-D state, centre `[0.7, 0, 0, 0]`, theta wrapped |
| quadrotor2D rl | 100 Hz | 1200 steps / 12 s | L2 ball r=0.2 over the 6-D state, goal `[0, 1, 0, 0, 0, 0]`, entry-cut |
| quadrotor3D lqr | 100 Hz | 2000 steps / 20 s | r=0.05 over the env's 12-D Euler state, entry-cut |
| quadrotor3D ppo | 50 Hz | 1000 steps / 20 s | r=0.05 over the stored 13-D quaternion row, entry-cut |

Two consequences worth keeping in front of you.

The quadrotor3D PPO branch runs at half the plant rate of the LQR branch, on a different start box,
with `index_aligned: false`. Never put a PPO level beside an LQR level.

Horizon is load-bearing for both quadrotors. The sharpest recorded case is the quadrotor3D
`noisy_dynamics` sweep described below: given unlimited time, success at `f_0.072` is about 0.24
against `f_0.000`'s 0.25, but at H=1000 it reads 0.058, so most of the apparent decline is the
deadline rather than loss of stability. The same reasoning applies to the corridor sets, where
rollouts start hitting the cap from low forcing upward. `p_success` is a bounded-time reach
probability everywhere in these datasets. Do not report it as an asymptotic region of attraction.

## Labelling radius vs `attractor_radius`

The datasets label success with one radius. The pipeline labels the K sampled endpoints inside
`endpoint_mc_probabilities` with `attractor_radius` from the system yaml, and that is what defines
the model's predicted `p_success`, which feeds straight into sAUROC, KL and recalibration. They do
not match anywhere.

| System | Labelling rule | `attractor_radius` | Volume ratio |
|---|---|---|---|
| pendulum | box, tol 0.05 per channel (area 0.0100) | 0.1 ball (area 0.0314) | 3.1x |
| cartpole | r=0.05 ball, 4-D | 0.2 | 256x |
| quadrotor2D | r=0.2 ball, 6-D | 0.3 | 11.4x |
| quadrotor3D | r=0.05 ball, 12-D | 0.3 | 2.2e9 |

These gaps were left in place deliberately (decision 2026-08-17 for pendulum and cartpole). Every arm
shares the bias, so rankings should survive, but magnitudes are not comparable to a run scored at the
labelling radius. Predicted `p_success` is systematically inflated and there is a floor under
achievable KL.

The quadrotor3D case is the one to watch, and it is not obviously survivable at 2.2e9. On the ground
truth it is measurably free, because `entry_cut` stops a flight the moment it enters the ball, so
every terminal row is either inside 0.05 or out at a wall 1.17 or more away. Recomputing `p_success`
at every radius from 0.05 to 0.50 reproduces the shipped labels identically on all four PPO levels.
What that does not settle is the model side: predicted endpoints are continuous and can land in the
gap. Check on a pilot run that predicted `p_success` is not saturated at 1 before launching a fleet.

## What is comparable to what

Nothing pools across the two families. Within a family:

- pendulum levels compare to pendulum levels.
- cartpole lqr levels compare to cartpole lqr levels. The PPO branch adds `baseline`, which lqr has
  no counterpart for, so do not match levels across controllers by position.
- quadrotor2D and quadrotor3D share a family name and a formula but differ by 3x to 4x in forcing
  relative to body weight. Do not put them in one table.
- quadrotor3D lqr and quadrotor3D ppo differ in ambient term, level ladder, start box, plant rate
  and eval states. Four independent reasons not to compare them.

The old `noisy/` regime is gone from `${DATA_DIR}`; only `deterministic/`, `partial_deterministic/`
and `stochastic/` remain. Anything recorded against `noisy/pendulum/lqr` predates the 2026-08-17
repoint to `stochastic/pendulum/gaussian_signal/lqr`.

## Known gap: `noisy_dynamics`

Both `configs/adaptive_v2/system/quadrotor2d_stoch.yaml` and `quadrotor3d_stoch.yaml` document a
second family and **default to it**:

```yaml
noise_family: noisy_dynamics
```

`noisy_dynamics` is described as `F ~ U(-f, +f)` applied at the COM link in the world frame,
zero-order hold redrawn every control step at 100 Hz, no torque, uniform rather than Gaussian, white
in time and explicitly `matched: false`. Levels `f_0.000 f_0.070 f_0.100 f_0.150 f_0.200` for the 2D
system and `f_0.000 f_0.032 f_0.048 f_0.060 f_0.072` for the 3D one.

As of 2026-09-09 no `noisy_dynamics` directory exists anywhere under
`/common/users/shared/pracsys/genMoPlan/data_trajectories`. Only `corridor_sine_ambient` ships for
either quadrotor. If that data lives on Amarel scratch only, the iLab defaults resolve to nothing.
Anyone running `system=quadrotor3d_stoch` without an explicit `noise_family=` on iLab will hit a
missing path.

## Where the parameters live

- Per-dataset ground truth: `${DATA_DIR}/stochastic/<system>/<family>/<controller>/<level>/dataset_description.json`,
  under `mechanism` (pendulum uses `collection.signal_noise`), `generation_parameters.noise_model`,
  and for the quadrotor3D PPO sets, `wind`.
- Config headers with the campaign-level caveats: `configs/adaptive_v2/system/pendulum_stoch.yaml`,
  `cartpole_stoch.yaml`, `quadrotor2d_stoch.yaml`, `quadrotor3d_stoch.yaml`,
  `quadrotor3d_ppo_stoch.yaml`.
- Pendulum dataset prep: `scripts/prepare_stochastic_pendulum.py`.
