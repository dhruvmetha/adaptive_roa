# Figure 1 shortlist, per system

Generated from manifest.json (117 slice-seed pairs, all at full resolution, certified c6c thresholds of the run whose checkpoint was loaded). Metrics are on cells inside the D_test start box. rank = gt_clarity x tera_match.

## Cartpole

| # | controller | plane | seed | rank | R_s | R_f | abstain | FP | FN | figure |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LQR | viz_x_vs_xdot_old | r2 (33407) | 0.806 | 0.996 | 0.959 | 0.022 | 826 | 0 | slices/figs/cp-lqr/viz_x_vs_xdot_old__r2.png |
| 2 | LQR | viz_x_vs_xdot_old | r1 (99512) | 0.787 | 0.943 | 1.000 | 0.011 | 0 | 1263 | slices/figs/cp-lqr/viz_x_vs_xdot_old__r1.png |
| 3 | PPO | k100_x_theta_dot | r3 (24475) | 0.784 | 0.927 | 0.855 | 0.116 | 8 | 2 | slices/figs/cp-ppo/k100_x_theta_dot__r3.png |
| 4 | PPO | k100_x_theta_dot | r1 (99512) | 0.782 | 0.920 | 0.881 | 0.100 | 27 | 5 | slices/figs/cp-ppo/k100_x_theta_dot__r1.png |
| 5 | PPO | k100_x_x_dot | r3 (24475) | 0.775 | 0.919 | 0.899 | 0.093 | 10 | 3 | slices/figs/cp-ppo/k100_x_x_dot__r3.png |
| 6 | PPO | k100_x_theta_dot | r2 (33407) | 0.759 | 0.887 | 0.896 | 0.101 | 17 | 47 | slices/figs/cp-ppo/k100_x_theta_dot__r2.png |

## Planar quadrotor

| # | controller | plane | seed | rank | R_s | R_f | abstain | FP | FN | figure |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PPO | k100_x_x_dot | r1 (99512) | 0.263 | 0.993 | 0.984 | 0.005 | 1 | 15 | slices/figs/q2d-ppo/k100_x_x_dot__r1.png |
| 2 | PPO | k100_x_x_dot | r2 (33407) | 0.263 | 0.995 | 0.984 | 0.003 | 1 | 18 | slices/figs/q2d-ppo/k100_x_x_dot__r2.png |
| 3 | PPO | k100_x_x_dot | r3 (24475) | 0.261 | 0.968 | 0.992 | 0.029 | 1 | 13 | slices/figs/q2d-ppo/k100_x_x_dot__r3.png |
| 4 | PPO | viz_x_vs_xdot | r2 (33407) | 0.260 | 0.994 | 0.970 | 0.004 | 62 | 267 | slices/figs/q2d-ppo/viz_x_vs_xdot__r2.png |
| 5 | PPO | viz_x_vs_xdot | r1 (99512) | 0.259 | 0.993 | 0.966 | 0.005 | 98 | 249 | slices/figs/q2d-ppo/viz_x_vs_xdot__r1.png |
| 6 | PPO | viz_x_vs_xdot | r3 (24475) | 0.258 | 0.965 | 0.972 | 0.033 | 2 | 195 | slices/figs/q2d-ppo/viz_x_vs_xdot__r3.png |

## Spatial quadrotor

| # | controller | plane | seed | rank | R_s | R_f | abstain | FP | FN | figure |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PPO | k100_tilt_p | r3 (24475) | 0.443 | 0.993 | 0.593 | 0.069 | 4 | 0 | slices/figs/q3d-ppo/k100_tilt_p__r3.png |
| 2 | LQR | viz_phi_vs_p | r3 (24475) | 0.396 | 0.803 | 0.952 | 0.117 | 305 | 1448 | slices/figs/q3d-lqr/viz_phi_vs_p__r3.png |
| 3 | LQR | viz_phi_vs_p | r1 (99512) | 0.388 | 0.816 | 0.926 | 0.120 | 185 | 1927 | slices/figs/q3d-lqr/viz_phi_vs_p__r1.png |
| 4 | LQR | viz_theta_vs_q | r3 (24475) | 0.353 | 0.823 | 0.948 | 0.086 | 699 | 4411 | slices/figs/q3d-lqr/viz_theta_vs_q__r3.png |
| 5 | LQR | viz_theta_vs_q | r1 (99512) | 0.346 | 0.776 | 0.955 | 0.109 | 174 | 4738 | slices/figs/q3d-lqr/viz_theta_vs_q__r1.png |
| 6 | PPO | k100_x_dot_y_dot_at_x1.4 | r3 (24475) | 0.345 | 0.961 | 0.771 | 0.096 | 230 | 0 | slices/figs/q3d-ppo/k100_x_dot_y_dot_at_x1.4__r3.png |

## Candidate triplets (composites/)

- cand_T4_crop (recommended): Cartpole PPO k100_x_theta_dot r3; Planar quad PPO k100_z_z_dot r2 CROPPED to the D_test start box (composites/_crops, preview helper); Spatial quad PPO k100_tilt_p r3.
- cand_T1: as T4 but Planar quad k100_x_x_dot r1, uncropped.
- cand_T2: Cartpole LQR viz_x_vs_xdot_old r2; Planar quad k100_z_z_dot r2 uncropped; Spatial quad LQR viz_phi_vs_p r3.
- cand_T3: Cartpole PPO k100_x_x_dot r3; Planar quad viz_x_vs_xdot r2; Spatial quad PPO k100_z_z_dot_at_x1.4 r1.
Each has _2x3 (GT over TERA-FM) and _1x3 (TERA-FM only) PNG/PDF and a .txt sidecar with runs, thresholds, fixed values and metrics.
