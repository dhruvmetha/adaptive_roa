# Flagship Figure 1: re-prioritization (2026-11-12, from paper_writer for Dhruv)

Figure 1 has exactly 3 systems: cartpole, planar quadrotor and spatial quadrotor. Each one shows
the single best slice. Cherry-picking is allowed ("This is for flagship, so we cherry pick"), for
the controller (LQR or PPO) and for the seed. Thresholds are still the certified c6c ones of the
same run whose checkpoint you load, and only those.

## What changes for you

1. Finalists before everything else. Pause or deprioritize the stage 2 work on slices not on your
   finalist list below. They can finish later.
2. Run each finalist on all three seeds of your setting's method run (`<base>_r1`, `_r2`, `_r3`; the
   README has the base name, and `infer.run_dir(name)` / `infer.thresholds(name)` take any of
   them). Load each model once.
3. Resolution.
   - 121x81 slices: full resolution for all three seeds (cheap).
   - 400x400 slices: screen all three seeds at stride 2 (as in STAGE2.md), then run full
     resolution for the best seed by in-box rank_score. Any seed's stride-2 or full result that
     already exists counts.
4. At most 2 of your GPU processes at once on your card, each under `nice -n 5`. Never block on one
   seed: run the next one in parallel or in sequence.
5. Naming: `pred/<setting>/<id>__<rK>.npz` and `figs/<setting>/<id>__<rK>.{png,pdf}`, where `<id>` is
   as in STAGE2.md (`<plane>` or `<plane>_s2`) and `<rK>` is r1, r2 or r3. Leave existing median-seed
   files where they are, and copy or symlink them to the new name so every seed has the same
   pattern.
6. Write `index_<setting>_finalists.csv`: one row per (plane, seed, resolution), from
   `sheet.index_row(sl, gt_m, tera_m, run=<that run name>, extra={png, pdf})`.
7. Read the full-resolution best-seed figure of each finalist yourself, and judge how it looks
   shrunk to about 2 inches wide: is the boundary still clear, are the wrong-call crosses noise or
   blobs?

## Finalists

| Setting | Finalists, in priority order |
|---|---|
| q2d-ppo | k100_z_z_dot, k100_x_x_dot (both re-flown at damping 0.04: re-fly these FIRST), then viz_x_vs_xdot, viz_z_vs_zdot, viz_theta_vs_thetadot |
| cp-ppo | k100_x_theta_dot, k100_x_x_dot, k100_x_theta |
| cp-lqr | viz_x_vs_xdot_old, viz_theta_vs_thetadot, viz_x_vs_xdot |
| q3d-ppo | k100_tilt_p, k100_z_z_dot_at_x1.4, k100_x_dot_y_dot_at_x1.4 |
| q3d-lqr | viz_theta_vs_q, viz_phi_vs_p |

## Report back (under 250 words, as soon as your finalists are done)

Per finalist and seed: R_s, R_f, abstain, n_wrong, rank_score (in-box) and resolution. Your best
(plane, seed) with one line on why, including how it looks small. The paths of the best figure and
its pred npz.
