# Experiment Directories — CLF vs FM vs part-X ROA plots

All paths are relative to `EXP = /common/users/shared/pracsys/adaptive_roa_experiments`

Plot script: `docs/plots/plot_clf_vs_fm.py`. Two plot sets:
- **`docs/plots/`** = current / new-criteria (FM retrained for quad2d/quad3d/humanoid; pendulum/cartpole unchanged; includes part-X).
- **`docs/old_plots/`** = old-criteria snapshot (all FM from the original Dhruv runs; no part-X).

CLF runs are identical in both sets (never retrained). Metrics source: FM newcrit + part-X read INLINE `epoch_XXX/results.json → full_roa.lambda_delta`; old FM (pendulum/cartpole + old_plots) read `evaluations/<eval>/epoch_XXX/artifacts_v2.json`.


## pendulum

**CLF** (same in both plot sets):
- `clf_adaptive`: `dhruv/adaptive_classification/pendulum/adaptive`
- `clf_nonadapt`: `dhruv/adaptive_classification/pendulum/random`

**part-X** (current plots only):
- `partx`: `adaptive_pendulum_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_10/2026-07-09_17-37-51`

**FM — current / new-criteria** (`docs/plots/`):
- `fm_ranked`: `dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-55-55`
- `fm_direct`: `dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07`
- `fm_nonadapt`: `dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-28_13-15-25`

## cartpole

**CLF** (same in both plot sets):
- `clf_adaptive`: `dhruv/adaptive_classification/cartpole/adaptive`
- `clf_nonadapt`: `dhruv/adaptive_classification/cartpole/random`

**part-X** (current plots only):
- `partx`: `adaptive_cartpole_pybullet/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_15/2026-07-09_17-37-51`

**FM — current / new-criteria** (`docs/plots/`):
- `fm_ranked`: `dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_19-22-14`
- `fm_direct`: `dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-18_14-26-32`
- `fm_nonadapt`: `dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_14-26-34`

## quad2d

**CLF** (same in both plot sets):
- `clf_adaptive`: `dhruv/adaptive_classification/quad2d/adaptive`
- `clf_nonadapt`: `dhruv/adaptive_classification/quad2d/random`

**part-X** (current plots only):
- `partx`: `adaptive_quadrotor2d/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_10/2026-07-09_23-13-21`

**FM — current / new-criteria** (`docs/plots/`):
- `fm_ranked`: `adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/newcrit_20260708`
- `fm_direct`: `adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/newcrit_20260708`
- `fm_nonadapt`: `adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/newcrit_20260708`

**FM — old-criteria** (`docs/old_plots/`, original Dhruv runs):
- `fm_direct`: `dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-05-32`
- `fm_nonadapt`: `dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-17-55`

## quad3d

**CLF** (same in both plot sets):
- `clf_adaptive`: `/common/home/st1122/Projects/adaptive_roa/outputs/clf_quad3d_match_fm/adaptive`
- `clf_nonadapt`: `/common/home/st1122/Projects/adaptive_roa/outputs/clf_quad3d_match_fm_slurm/random`

**part-X** (current plots only):
- `partx`: `adaptive_quadrotor3d/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_15/2026-07-10_09-10-18`

**FM — current / new-criteria** (`docs/plots/`):
- `fm_ranked`: `adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/newcrit_20260708`
- `fm_direct`: `adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/newcrit_20260708`
- `fm_nonadapt`: `adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/newcrit_20260708`

**FM — old-criteria** (`docs/old_plots/`, original Dhruv runs):
- `fm_ranked`: `dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-26_12-02-36`
- `fm_direct`: `dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-26_12-02-36`
- `fm_nonadapt`: `dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-49-54`

## humanoid

**CLF** (same in both plot sets):
- `clf_adaptive`: `adaptive_humanoid_standup_reach_classifier/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/2026-06-24_23-58-56`
- `clf_nonadapt`: `adaptive_humanoid_standup_reach_classifier/outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/2026-06-24_23-58-56`

**part-X** (current plots only):
- `partx`: `adaptive_humanoid_standup_reach/outputs/training_index_0_warm_start_False_manifold_True_adapt_iter_30/2026-07-10_09-10-18`

**FM — current / new-criteria** (`docs/plots/`):
- `fm_adapt_mfF`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/newcrit_20260708`
- `fm_adapt_mfT`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/newcrit_20260708`
- `fm_ranked_mfF`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_ranked/newcrit_20260708`
- `fm_ranked_mfT`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_ranked/newcrit_20260708`
- `fm_nonadapt_mfF`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/newcrit_20260708`
- `fm_nonadapt_mfT`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/newcrit_20260708`

**FM — old-criteria** (`docs/old_plots/`, original Dhruv runs):
- `fm_adapt_mfF`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/2026-06-24_23-51-39`
- `fm_adapt_mfT`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/2026-06-24_23-51-39`
- `fm_nonadapt_mfF`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/2026-06-24_23-51-39`
- `fm_nonadapt_mfT`: `adaptive_humanoid_standup_reach/outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_True_threshold_mode_dynamic_adapt_iter_30_alpha_0.1_sampling_mode_direct/2026-06-24_23-51-39`


# Final-state dispersion acquisition campaign (2026-07-26)

Acquisition scored by the **geometry of the predicted final-state cloud** instead of
label counts: `score(x) = mean pairwise distance among K=20 predicted endpoints`,
range-normalized per dimension and circular-aware. No `classify_attractor`, no
`attractor_radius`, no λ*/δ*/q̂ in the acquisition path — the success criterion is
confined to threshold optimization and evaluation.

**Provenance.** Trained on **Amarel** (`gpu-redhat`, 24 jobs, 2026-07-26/27), then
rsynced to the iLab tree. **The paths below are the iLab copies** and are relative to
`EXP` like the rest of this file. The Amarel originals remain at
`/scratch/st1122/adaptive_roa/experiments/...` under the same
`<system>/outputs/<config>/<timestamp>` layout, but iLab is authoritative — every plot
and analysis script reads `EXP`.

Grid: 4 systems × d2_ratio {0.5, 1.0} × selection_rule {greedy, greedy_diverse,
proportional} = 24 runs, all COMPLETED. Metrics are INLINE at
`epoch_XXX/results.json → full_roa.lambda_delta` (same layout as the newcrit FM runs),
so `plot_clf_vs_fm.py` reads them via `load_fm_inline` with `resolve_fm_dispersion`.

Arm names in `plot_clf_vs_fm.py` / `clf_vs_fm_metrics.csv`:
`fm_disp_{greedy,gdiv,prop}_{d05,d10}` where `gdiv` = `greedy_diverse`,
`prop` = `proportional`, `d05` = d2_ratio 0.5, `d10` = d2_ratio 1.0.

Output dirs are self-identifying as of commit `53a3490` — they carry `d2_ratio`,
`sampling_mode`, `selection_rule` and `exp_<SLURM_JOB_ID>`, so a directory names the
config that produced it and ties itself to `slurm_logs/<jobname>_<jobid>.out`.

## pendulum — dispersion (`adaptive_pendulum_lqr/outputs/`)

Common prefix: `training_index_0_d2_ratio_{D2}_warm_start_False_adapt_iter_19_sampling_mode_dispersion_selection_{RULE}_exp_{JOB}`

- `fm_disp_greedy_d05`: `..._d2_ratio_0.5_..._selection_greedy_exp_59182852/2026-07-26_20-52-59`
- `fm_disp_gdiv_d05`:   `..._d2_ratio_0.5_..._selection_greedy_diverse_exp_59183665/2026-07-26_21-02-08`
- `fm_disp_prop_d05`:   `..._d2_ratio_0.5_..._selection_proportional_exp_59183666/2026-07-26_21-02-08`
- `fm_disp_greedy_d10`: `..._d2_ratio_1.0_..._selection_greedy_exp_59183667/2026-07-26_21-02-08`
- `fm_disp_gdiv_d10`:   `..._d2_ratio_1.0_..._selection_greedy_diverse_exp_59183668/2026-07-26_21-02-08`
- `fm_disp_prop_d10`:   `..._d2_ratio_1.0_..._selection_proportional_exp_59183669/2026-07-26_21-02-08`

## cartpole — dispersion (`adaptive_cartpole_pybullet/outputs/`)

Common prefix: `training_index_0_d2_ratio_{D2}_warm_start_False_manifold_False_adapt_iter_15_sampling_mode_dispersion_selection_{RULE}_exp_{JOB}`

- `fm_disp_greedy_d05`: `..._d2_ratio_0.5_..._selection_greedy_exp_59183670/2026-07-26_21-02-08`
- `fm_disp_gdiv_d05`:   `..._d2_ratio_0.5_..._selection_greedy_diverse_exp_59183671/2026-07-26_21-02-08`
- `fm_disp_prop_d05`:   `..._d2_ratio_0.5_..._selection_proportional_exp_59183672/2026-07-26_21-02-08`
- `fm_disp_greedy_d10`: `..._d2_ratio_1.0_..._selection_greedy_exp_59183673/2026-07-26_21-02-08`  ⚠ **model collapse**
- `fm_disp_gdiv_d10`:   `..._d2_ratio_1.0_..._selection_greedy_diverse_exp_59183674/2026-07-26_21-17-29`
- `fm_disp_prop_d10`:   `..._d2_ratio_1.0_..._selection_proportional_exp_59183675/2026-07-26_21-17-29`

⚠ `fm_disp_greedy_d10` predicts **zero successes from epoch 8 onward** (F1 exactly 0.0,
precision = recall = 0). Accuracy still reads .994 because successes are rare — read F1,
never accuracy, for this run. Pure greedy with no diversity term and no random component
(d2=1.0) concentrates the batch into one pocket; `greedy_diverse` on the identical cell
produced the campaign's best cartpole F1 (.9974).

## quad2d — dispersion (`adaptive_quadrotor2d/outputs/`)

Common prefix: `training_index_0_d2_ratio_{D2}_warm_start_False_manifold_False_adapt_iter_20_sampling_mode_dispersion_selection_{RULE}_exp_{JOB}`

- `fm_disp_greedy_d05`: `..._d2_ratio_0.5_..._selection_greedy_exp_59203527/2026-07-27_01-29-08`
- `fm_disp_gdiv_d05`:   `..._d2_ratio_0.5_..._selection_greedy_diverse_exp_59183677/2026-07-26_21-36-45`
- `fm_disp_prop_d05`:   `..._d2_ratio_0.5_..._selection_proportional_exp_59183678/2026-07-26_21-36-45`
- `fm_disp_greedy_d10`: `..._d2_ratio_1.0_..._selection_greedy_exp_59183679/2026-07-26_21-36-45`
- `fm_disp_gdiv_d10`:   `..._d2_ratio_1.0_..._selection_greedy_diverse_exp_59183680/2026-07-26_21-36-45`
- `fm_disp_prop_d10`:   `..._d2_ratio_1.0_..._selection_proportional_exp_59183681/2026-07-26_21-36-45`

⚠ **Do not use** `..._selection_greedy_exp_59183676/2026-07-26_21-36-46` — an abandoned partial run
(1 epoch on disk). Job 59183676 landed on Amarel `gpu017`, whose log header read
`Unable to determine the device handle for GPU0 ... Unknown Error` alongside RTX 3090s
while every sibling drew an L40S. It reached 1 epoch in ~4h against 13 for same-system
runs launched the same second; SLURM reported the node healthy throughout. Cancelled and
requeued as **59203527** with `--exclude=gpu017`; the requeue reproduced the original's
epoch-0 diagnostics exactly. `resolve_fm_dispersion` takes `sorted(...)[-1]`, and
`59203527` sorts after `59183676`, so the plot script already picks the good run — but the
stale directory is still on disk.

## quad3d — dispersion (`adaptive_quadrotor3d/outputs/`)

Common prefix: `training_index_0_d2_ratio_{D2}_warm_start_False_manifold_False_adapt_iter_30_sampling_mode_dispersion_selection_{RULE}_exp_{JOB}`

- `fm_disp_greedy_d05`: `..._d2_ratio_0.5_..._selection_greedy_exp_59183682/2026-07-26_21-36-45`
- `fm_disp_gdiv_d05`:   `..._d2_ratio_0.5_..._selection_greedy_diverse_exp_59183683/2026-07-26_21-36-45`
- `fm_disp_prop_d05`:   `..._d2_ratio_0.5_..._selection_proportional_exp_59183684/2026-07-26_21-36-45`
- `fm_disp_greedy_d10`: `..._d2_ratio_1.0_..._selection_greedy_exp_59183685/2026-07-26_21-36-45`
- `fm_disp_gdiv_d10`:   `..._d2_ratio_1.0_..._selection_greedy_diverse_exp_59183686/2026-07-26_21-36-45`
- `fm_disp_prop_d10`:   `..._d2_ratio_1.0_..._selection_proportional_exp_59183687/2026-07-26_21-36-45`

## Outcome

**Dispersion does not beat the `ranked`/`direct` baselines for ROA classification.** At
matched trajectory budgets (`docs/plots/clf_vs_fm_metrics.csv`):

| system | best baseline | best dispersion | verdict |
|---|---|---|---|
| pendulum @971 | ranked .9966 | prop d2=1.0 **.9988** | dispersion wins |
| cartpole @1250 | ranked **.9963** | prop d2=1.0 .9849 | all 6 lose |
| quad2d @7608 | direct **.9634** | greedy d2=0.5 .8594 | all 6 lose, and lose to non-adaptive .8959 |
| quad3d @13000 | direct **.9646** | prop d2=0.5 .9538 | ties ranked, loses to direct |

Driver is FNR: .22–.27 on quad2d vs .05–.15 for the baselines. Dispersion starves the
success class because it conflates within-mode spread with between-mode separation, and
divergence regions are genuinely diffuse while being unambiguously failures.

**It does win on final-state regression.** At matched budget on quad2d the four most
accurate endpoint predictors in the whole comparison are dispersion arms
(`fm_disp_gdiv_d05` 1.4560 mean geodesic error vs `fm_direct` 1.5207, `fm_ranked` 1.5655,
`fm_nonadapt` 1.5621) — see `docs/plots/fm_endpoint_error_by_group.csv`, generated by
`docs/plots/plot_endpoint_errors.py`. Better regressors, worse basin classifiers, from the
same runs.

## Follow-up: offline score bake-off (no new training)

`scripts/experiment_mode_separation_bakeoff.py` re-scores frozen checkpoints from the runs
above with alternative label-free functionals and grades them against
`epoch_XXX/full_roa_per_point.npz` (start_states, true_labels, p_success). Results in
`docs/plots/mode_separation_bakeoff.csv`. Reproduces the campaign's failure in minutes:
dispersion's top-100 on cartpole contains 0% true successes and 0% boundary points — worse
than random — while threshold-free label entropy reaches 2.94× boundary lift on quad2d.
