# Main Results

## Quadrotor 2D

### Runs

| Label | d2_ratio | filter | train radius | Path |
|-------|----------|--------|--------------|------|
| d2=1.0, no filter | 1.0 | no | 0.2 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-14_12-42-40` |
| d2=0, no filter | 0 | no | 0.2 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-13_23-33-57` |
| d2=0, filtered | 0.0 | yes | 0.3 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-14_19-23-42` |
| d2=1.0, filtered | 1.0 | yes | 0.3 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-15_09-58-06` |

### Evaluation

All evaluated at `radius_0.3_alpha_0.1_mc_10_batch_100000` using `qhat_prediction_sets`.

### Config Differences

| Parameter | no filter runs | filtered runs |
|-----------|---------------|---------------|
| `conformal.attractor_radius` | 0.2 | 0.3 |
| `conformal.use_p_invalid_veto` | absent | true |
| `adaptive_v2.filter_confident_pairs` | absent | true |
| `adaptive_v2.filter_min_pairs` | absent | 100 |

### Flow Matching Results

#### 100% adaptive, no filter

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 17.04% | 0.7632 | 0.9038 | 0.6604 | 0.9955 |
| 4000 | 16.73% | 0.8046 | 0.9107 | 0.7206 | 0.9955 |
| 5000 | 13.03% | 0.8270 | 0.8977 | 0.7666 | 0.9937 |
| 6000 | 12.44% | 0.8435 | 0.9075 | 0.7880 | 0.9939 |
| 7000 | 12.57% | 0.8449 | 0.9088 | 0.7895 | 0.9944 |
| 8000 | 11.22% | 0.8617 | 0.9056 | 0.8219 | 0.9934 |
| 9000 | 9.75% | 0.8611 | 0.9169 | 0.8117 | 0.9943 |
| 10000 | 10.05% | 0.8700 | 0.9192 | 0.8258 | 0.9944 |
| 11000 | 8.94% | 0.8768 | 0.9170 | 0.8400 | 0.9940 |
| 12000 | 9.09% | 0.8782 | 0.9227 | 0.8377 | 0.9944 |

#### 0% adaptive, no filter

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 17.37% | 0.7547 | 0.9034 | 0.6481 | 0.9956 |
| 4000 | 14.95% | 0.8105 | 0.9101 | 0.7305 | 0.9950 |
| 5000 | 13.80% | 0.8243 | 0.9001 | 0.7602 | 0.9940 |
| 6000 | 12.74% | 0.8389 | 0.9093 | 0.7786 | 0.9944 |
| 7000 | 11.03% | 0.8489 | 0.9039 | 0.8002 | 0.9935 |
| 8000 | 10.60% | 0.8586 | 0.9109 | 0.8120 | 0.9938 |
| 9000 | 9.81% | 0.8665 | 0.9179 | 0.8207 | 0.9942 |
| 10000 | 9.18% | 0.8633 | 0.9107 | 0.8205 | 0.9936 |
| 11000 | 8.72% | 0.8758 | 0.9205 | 0.8352 | 0.9944 |
| 12000 | 9.05% | 0.8776 | 0.9148 | 0.8433 | 0.9937 |

#### 0% adaptive, filtered

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 16.69% | 0.7552 | 0.9066 | 0.6471 | 0.9957 |
| 4000 | 16.25% | 0.7874 | 0.9125 | 0.6924 | 0.9957 |
| 5000 | 13.59% | 0.8179 | 0.8956 | 0.7527 | 0.9935 |
| 6000 | 12.32% | 0.8444 | 0.9037 | 0.7924 | 0.9937 |
| 7000 | 11.57% | 0.8517 | 0.9128 | 0.7983 | 0.9943 |
| 8000 | 10.20% | 0.8536 | 0.9112 | 0.8028 | 0.9940 |
| 9000 | 10.32% | 0.8576 | 0.9069 | 0.8134 | 0.9937 |
| 10000 | 15.35% | 0.8922 | 0.9300 | 0.8573 | 0.9956 |
| 11000 | 13.91% | 0.8921 | 0.9292 | 0.8579 | 0.9950 |
| 12000 | 14.20% | 0.9029 | 0.9356 | 0.8724 | 0.9957 |

#### 100% adaptive, filtered

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 16.73% | 0.7560 | 0.9068 | 0.6482 | 0.9958 |
| 4000 | 16.17% | 0.7782 | 0.9074 | 0.6812 | 0.9956 |
| 5000 | 15.11% | 0.8167 | 0.9079 | 0.7421 | 0.9949 |
| 6000 | 12.20% | 0.8469 | 0.8965 | 0.8025 | 0.9932 |
| 7000 | 12.22% | 0.8612 | 0.9188 | 0.8103 | 0.9948 |
| 8000 | 16.89% | 0.8918 | 0.9307 | 0.8560 | 0.9958 |
| 9000 | 15.38% | 0.8901 | 0.9344 | 0.8499 | 0.9959 |
| 10000 | 15.67% | 0.8940 | 0.9336 | 0.8576 | 0.9957 |
| 11000 | 9.51% | 0.8699 | 0.9127 | 0.8309 | 0.9937 |
| 12000 | 13.98% | 0.8995 | 0.9319 | 0.8694 | 0.9953 |

### Lyapunov Neural Network

#### Conformal (per-trajectory count)

| Trajectories | Separatrix | Accuracy | Precision | Recall | F1 |
|-------------|-----------|----------|-----------|--------|------|
| 3000 | 0.32% | 0.9526 | 0.8593 | 0.4714 | 0.6088 |
| 4000 | 0.41% | 0.9580 | 0.8881 | 0.5238 | 0.6590 |
| 5000 | 0.26% | 0.9578 | 0.8736 | 0.5401 | 0.6675 |
| 6000 | 0.30% | 0.9623 | 0.9106 | 0.5734 | 0.7037 |
| 7000 | 0.26% | 0.9646 | 0.9054 | 0.6133 | 0.7313 |
| 8000 | 0.30% | 0.9674 | 0.8910 | 0.6651 | 0.7617 |
| 9000 | 0.34% | 0.9688 | 0.9059 | 0.6691 | 0.7697 |
| 10000 | 0.33% | 0.9675 | 0.9067 | 0.6515 | 0.7582 |
| 11000 | 0.36% | 0.9666 | 0.9078 | 0.6358 | 0.7478 |
| 12000 | 0.46% | 0.9695 | 0.8956 | 0.6856 | 0.7767 |

#### Neuromancer (direct Lyapunov evaluation)

| Setting | Separatrix | F1 | Precision | Recall | Accuracy |
|---------|-----------|------|-----------|--------|----------|
| With separatrix | 36.00% | 0.8040 | 0.7000 | 0.9450 | 0.9720 |
| Without separatrix | 0% | 0.5540 | 0.7000 | 0.4580 | 0.9410 |

### Adaptive Sampling Comparison (Joint Optimization)

#### Runs

| Label | d2_ratio | sampling_mode | optimize_mode | filter | train radius | Path |
|-------|----------|---------------|---------------|--------|--------------|------|
| Non-adaptive baseline | 0.0 | ranked | joint | yes | 0.3 | `.../sampling_mode_ranked/2026-02-17_14-07-33` |
| Adaptive direct | 1.0 | direct | joint | yes | 0.3 | `.../sampling_mode_direct/2026-02-17_14-12-47` |
| Adaptive ranked | 1.0 | ranked | joint | yes | 0.3 | `.../sampling_mode_ranked/2026-02-18_14-09-00` |

#### Evaluation

All evaluated at `radius_0.3_alpha_0.1_mc_10_batch_100000` using `lambda_delta`.

Note: `lambda_delta` and `qhat_prediction_sets` results are identical for Quadrotor 2D (q_hat=0 for all epochs).

#### Non-adaptive baseline (d2=0)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 10.55% | 0.5158 | 0.9428 | 0.3550 | 0.9989 |
| 4000 | 9.27% | 0.6798 | 0.9271 | 0.5366 | 0.9975 |
| 5000 | 11.69% | 0.7625 | 0.9096 | 0.6564 | 0.9960 |
| 6000 | 10.27% | 0.8038 | 0.9319 | 0.7067 | 0.9967 |
| 7000 | 10.12% | 0.7788 | 0.9482 | 0.6608 | 0.9980 |
| 8000 | 8.76% | 0.8413 | 0.9323 | 0.7666 | 0.9961 |
| 9000 | 9.54% | 0.7698 | 0.9502 | 0.6469 | 0.9983 |
| 10000 | 8.59% | 0.8324 | 0.9503 | 0.7405 | 0.9976 |
| 11000 | 8.24% | 0.8295 | 0.9473 | 0.7377 | 0.9975 |
| 12000 | 8.02% | 0.8433 | 0.9496 | 0.7584 | 0.9974 |

#### Adaptive direct

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 10.50% | 0.5180 | 0.9404 | 0.3575 | 0.9988 |
| 4000 | 8.14% | 0.7879 | 0.9314 | 0.6827 | 0.9969 |
| 4969 | 8.35% | 0.8641 | 0.9256 | 0.8103 | 0.9955 |
| 5709 | 7.51% | 0.8802 | 0.9336 | 0.8326 | 0.9958 |
| 6166 | 7.91% | 0.8958 | 0.9504 | 0.8471 | 0.9972 |
| 6514 | 7.09% | 0.8977 | 0.9482 | 0.8523 | 0.9968 |
| 6777 | 7.13% | 0.9048 | 0.9564 | 0.8585 | 0.9974 |
| 6981 | 6.55% | 0.9046 | 0.9490 | 0.8641 | 0.9968 |
| 7147 | 7.20% | 0.9000 | 0.9559 | 0.8504 | 0.9976 |
| 7294 | 6.54% | 0.9096 | 0.9505 | 0.8720 | 0.9969 |

#### Adaptive ranked

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 22.45% | 0.6678 | 0.9542 | 0.5136 | 0.9992 |
| 4000 | 17.29% | 0.8154 | 0.9627 | 0.7072 | 0.9989 |
| 5000 | 11.23% | 0.8370 | 0.9613 | 0.7412 | 0.9987 |
| 6000 | 10.04% | 0.8362 | 0.9652 | 0.7376 | 0.9989 |
| 7000 | 10.00% | 0.8913 | 0.9661 | 0.8272 | 0.9986 |
| 8000 | 9.26% | 0.8721 | 0.9698 | 0.7923 | 0.9989 |
| 9000 | 11.18% | 0.9063 | 0.9676 | 0.8524 | 0.9987 |
| 10000 | 10.86% | 0.9066 | 0.9675 | 0.8529 | 0.9986 |
| 11000 | 14.01% | 0.9253 | 0.9689 | 0.8855 | 0.9986 |
| 12000 | 13.97% | 0.9270 | 0.9675 | 0.8897 | 0.9986 |

### Adaptive Sampling Comparison (Delta Optimization)

#### Runs

| Label | d2_ratio | sampling_mode | optimize_mode | filter | train radius | Path |
|-------|----------|---------------|---------------|--------|--------------|------|
| Non-adaptive baseline | 0.0 | ranked | delta | yes | 0.3 | `.../sampling_mode_ranked/2026-02-19_13-50-18` |
| Adaptive direct | 1.0 | direct | delta | yes | 0.3 | `.../sampling_mode_direct/2026-02-19_13-51-06` |
| Adaptive ranked | 1.0 | ranked | delta | yes | 0.3 | `.../sampling_mode_ranked/2026-02-19_13-50-09` |

#### Evaluation

All evaluated at `radius_0.3_alpha_0.1_mc_10_batch_100000` using `qhat_prediction_sets`.

#### Non-adaptive baseline (d2=0)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 30.69% | 0.8337 | 0.9259 | 0.7582 | 0.9970 |
| 4000 | 29.68% | 0.8399 | 0.9335 | 0.7634 | 0.9973 |
| 5000 | 34.56% | 0.8742 | 0.9388 | 0.8179 | 0.9978 |
| 6000 | 31.27% | 0.8868 | 0.9290 | 0.8481 | 0.9970 |
| 7000 | 27.94% | 0.9019 | 0.9310 | 0.8745 | 0.9965 |
| 8000 | 27.18% | 0.9035 | 0.9335 | 0.8754 | 0.9967 |
| 9000 | 24.47% | 0.8974 | 0.9295 | 0.8675 | 0.9962 |
| 10000 | 25.67% | 0.9152 | 0.9365 | 0.8949 | 0.9963 |
| 11000 | 26.44% | 0.9141 | 0.9339 | 0.8951 | 0.9964 |
| 12000 | 23.35% | 0.9212 | 0.9441 | 0.8995 | 0.9967 |

#### Adaptive direct

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 30.74% | 0.8317 | 0.9295 | 0.7526 | 0.9971 |
| 4000 | 24.38% | 0.8516 | 0.9308 | 0.7848 | 0.9969 |
| 5000 | 33.25% | 0.9397 | 0.9402 | 0.9392 | 0.9966 |
| 6000 | 24.65% | 0.9218 | 0.9349 | 0.9091 | 0.9964 |
| 7000 | 25.31% | 0.9451 | 0.9435 | 0.9467 | 0.9966 |
| 8000 | 30.80% | 0.9598 | 0.9538 | 0.9659 | 0.9970 |
| 9000 | 20.75% | 0.9484 | 0.9478 | 0.9490 | 0.9966 |
| 10000 | 20.31% | 0.9546 | 0.9518 | 0.9574 | 0.9969 |
| 10731 | 18.15% | 0.9500 | 0.9445 | 0.9556 | 0.9960 |
| 11152 | 16.70% | 0.9494 | 0.9487 | 0.9501 | 0.9963 |

#### Adaptive ranked

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 3000 | 30.72% | 0.8365 | 0.9305 | 0.7598 | 0.9971 |
| 4000 | 22.51% | 0.8660 | 0.9430 | 0.8006 | 0.9975 |
| 5000 | 21.12% | 0.9149 | 0.9383 | 0.8927 | 0.9965 |
| 6000 | 23.19% | 0.9411 | 0.9440 | 0.9382 | 0.9965 |
| 7000 | 14.89% | 0.9324 | 0.9331 | 0.9317 | 0.9953 |
| 8000 | 12.83% | 0.9352 | 0.9351 | 0.9353 | 0.9953 |
| 9000 | 17.87% | 0.9493 | 0.9451 | 0.9536 | 0.9960 |
| 10000 | 19.40% | 0.9530 | 0.9503 | 0.9558 | 0.9965 |
| 11000 | 24.68% | 0.9598 | 0.9532 | 0.9665 | 0.9967 |
| 12000 | 25.99% | 0.9623 | 0.9580 | 0.9667 | 0.9971 |

### Summary: Joint vs Delta Optimization (Final Epoch)

| Method | Opt Mode | Trajectories | F1 | Sep% | Precision | Recall |
|--------|----------|-------------|------|------|-----------|--------|
| Non-adaptive baseline | joint | 12,000 | 0.8433 | 8.02% | 0.9496 | 0.7584 |
| Adaptive direct | joint | 7,294 | 0.9096 | 6.54% | 0.9505 | 0.8720 |
| Adaptive ranked | joint | 12,000 | 0.9270 | 13.97% | 0.9675 | 0.8897 |
| Non-adaptive baseline | delta | 12,000 | 0.9212 | 23.35% | 0.9441 | 0.8995 |
| Adaptive direct | delta | 11,152 | 0.9494 | 16.70% | 0.9487 | 0.9501 |
| **Adaptive ranked** | **delta** | **12,000** | **0.9623** | **25.99%** | **0.9580** | **0.9667** |

**Key findings:**
- Delta optimization achieves higher F1 but at the cost of much higher Sep% (16-26% vs 6-14%) — it's more conservative, abstaining on more points rather than committing
- Joint optimization is more aggressive — lower Sep% (tighter separatrix) but lower F1 because it commits on harder points
- Ranking: ranked > direct > non-adaptive holds for both optimization modes
- Best F1: Adaptive ranked (delta) at 0.9623, but with 26% Sep%
- Best F1/Sep% tradeoff: Adaptive direct (joint) — F1=0.9096 with only 6.54% Sep% using 7,294 trajectories

### Summary: Joint Optimization vs Classifier (Lambda-Delta)

| Method | Trajectories | F1 | Sep% | Precision | Recall |
|--------|-------------|------|------|-----------|--------|
| Lyapunov NN | 12,000 | 0.7767 | 0.46% | 0.8956 | 0.6856 |
| DeepReach (auto-threshold) | 12,000 | 0.2548 | 0% | 0.1526 | 0.7718 |
| Non-adaptive | 12,000 | 0.8433 | 8.02% | 0.9496 | 0.7584 |
| Adaptive (direct) | 7,294 | 0.9096 | 6.54% | 0.9505 | 0.8720 |
| **Adaptive (ranked)** | **12,000** | **0.9270** | **13.97%** | **0.9675** | **0.8897** |

## CartPole (PyBullet)

### Adaptive Sampling Comparison

#### Runs

| Label | d2_ratio | sampling_mode | optimize_mode | filter | train radius | Path |
|-------|----------|---------------|---------------|--------|--------------|------|
| Non-adaptive baseline | 0.0 | ranked | joint | yes | 0.2 | `.../adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.0_..._sampling_mode_ranked_w_0.9/2026-02-18_14-26-34` |
| Adaptive direct | 1.0 | direct | joint | yes | 0.2 | `.../adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_..._sampling_mode_direct_w_0.9/2026-02-18_14-26-32` |
| Adaptive ranked | 1.0 | ranked | joint | yes | 0.2 | `.../adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_..._sampling_mode_ranked_w_0.9/2026-02-18_19-22-14` |

#### Evaluation

All evaluated at `radius_0.2_alpha_0.1_mc_10_batch_100000` using `lambda_delta`.

Note: `lambda_delta` and `qhat_prediction_sets` results are identical for CartPole (q_hat=0 for all epochs).

#### Non-adaptive baseline (d2=0)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 300 | 33.96% | 0.9510 | 0.9327 | 0.9700 | 0.9874 |
| 350 | 20.65% | 0.9788 | 0.9766 | 0.9811 | 0.9970 |
| 400 | 16.95% | 0.9769 | 0.9708 | 0.9831 | 0.9958 |
| 450 | 15.23% | 0.9706 | 0.9781 | 0.9632 | 0.9973 |
| 500 | 15.14% | 0.9727 | 0.9807 | 0.9647 | 0.9978 |
| 550 | 17.32% | 0.9822 | 0.9832 | 0.9813 | 0.9979 |
| 600 | 16.30% | 0.9835 | 0.9833 | 0.9837 | 0.9976 |
| 650 | 14.00% | 0.9790 | 0.9734 | 0.9847 | 0.9962 |
| 700 | 13.72% | 0.9813 | 0.9880 | 0.9747 | 0.9985 |
| 750 | 8.61% | 0.9811 | 0.9716 | 0.9909 | 0.9948 |
| 800 | 13.28% | 0.9793 | 0.9724 | 0.9864 | 0.9958 |
| 850 | 12.40% | 0.9836 | 0.9914 | 0.9760 | 0.9989 |
| 900 | 10.30% | 0.9900 | 0.9927 | 0.9874 | 0.9989 |
| 950 | 8.87% | 0.9870 | 0.9908 | 0.9833 | 0.9985 |
| 1000 | 8.10% | 0.9823 | 0.9930 | 0.9718 | 0.9989 |
| 1050 | 9.96% | 0.9884 | 0.9918 | 0.9850 | 0.9988 |
| 1100 | 9.13% | 0.9911 | 0.9940 | 0.9882 | 0.9990 |
| 1150 | 8.74% | 0.9898 | 0.9928 | 0.9868 | 0.9989 |
| 1200 | 8.26% | 0.9903 | 0.9916 | 0.9890 | 0.9985 |
| 1250 | 9.22% | 0.9912 | 0.9922 | 0.9903 | 0.9987 |

#### Adaptive direct

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 300 | 33.88% | 0.9519 | 0.9329 | 0.9717 | 0.9874 |
| 350 | 20.79% | 0.9821 | 0.9843 | 0.9799 | 0.9980 |
| 400 | 13.25% | 0.9858 | 0.9810 | 0.9907 | 0.9970 |
| 450 | 11.07% | 0.9657 | 0.9909 | 0.9418 | 0.9989 |
| 500 | 9.33% | 0.9599 | 0.9954 | 0.9270 | 0.9994 |
| 550 | 10.93% | 0.9884 | 0.9926 | 0.9842 | 0.9989 |
| 600 | 7.39% | 0.9871 | 0.9927 | 0.9816 | 0.9987 |
| 650 | 8.89% | 0.9890 | 0.9967 | 0.9813 | 0.9995 |
| 700 | 5.15% | 0.9914 | 0.9968 | 0.9861 | 0.9994 |
| 750 | 3.56% | 0.9883 | 0.9965 | 0.9803 | 0.9993 |
| 800 | 2.77% | 0.9885 | 0.9955 | 0.9816 | 0.9991 |
| 850 | 1.62% | 0.9876 | 0.9920 | 0.9833 | 0.9983 |
| 900 | 1.44% | 0.9858 | 0.9958 | 0.9760 | 0.9991 |
| 950 | 1.34% | 0.9873 | 0.9961 | 0.9787 | 0.9992 |
| 1000 | 0.99% | 0.9878 | 0.9954 | 0.9803 | 0.9990 |
| 1050 | 0.86% | 0.9888 | 0.9951 | 0.9825 | 0.9990 |
| 1100 | 2.07% | 0.9946 | 0.9965 | 0.9927 | 0.9993 |
| 1150 | 2.27% | 0.9931 | 0.9913 | 0.9950 | 0.9981 |
| 1200 | 2.26% | 0.9951 | 0.9946 | 0.9956 | 0.9988 |
| 1250 | 2.42% | 0.9926 | 0.9884 | 0.9970 | 0.9974 |

#### Adaptive ranked

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 300 | 35.02% | 0.9584 | 0.9484 | 0.9687 | 0.9903 |
| 350 | 33.48% | 0.9744 | 0.9784 | 0.9704 | 0.9967 |
| 400 | 21.41% | 0.9741 | 0.9771 | 0.9711 | 0.9962 |
| 450 | 12.19% | 0.9677 | 0.9900 | 0.9464 | 0.9989 |
| 500 | 10.38% | 0.9814 | 0.9928 | 0.9702 | 0.9990 |
| 550 | 7.34% | 0.9829 | 0.9917 | 0.9742 | 0.9986 |
| 600 | 4.22% | 0.9807 | 0.9884 | 0.9732 | 0.9978 |
| 650 | 5.93% | 0.9870 | 0.9858 | 0.9882 | 0.9971 |
| 700 | 3.22% | 0.9808 | 0.9917 | 0.9700 | 0.9984 |
| 750 | 2.55% | 0.9845 | 0.9924 | 0.9767 | 0.9985 |
| 800 | 2.46% | 0.9860 | 0.9886 | 0.9834 | 0.9976 |
| 850 | 5.22% | 0.9933 | 0.9929 | 0.9938 | 0.9985 |
| 900 | 3.51% | 0.9914 | 0.9891 | 0.9937 | 0.9976 |
| 950 | 4.09% | 0.9948 | 0.9931 | 0.9965 | 0.9985 |
| 1000 | 1.72% | 0.9918 | 0.9937 | 0.9900 | 0.9986 |
| 1050 | 2.31% | 0.9960 | 0.9963 | 0.9957 | 0.9992 |
| 1100 | 1.64% | 0.9928 | 0.9912 | 0.9944 | 0.9981 |
| 1150 | 1.30% | 0.9932 | 0.9919 | 0.9946 | 0.9982 |
| 1200 | 1.43% | 0.9950 | 0.9969 | 0.9930 | 0.9993 |
| 1250 | 1.82% | 0.9954 | 0.9967 | 0.9940 | 0.9993 |

### Lyapunov Neural Network

| Trajectories | Separatrix | F1 |
|-------------|-----------|------|
| 1000 | 0.05% | 0.9450 |

### Summary: Joint Optimization vs Classifier (Lambda-Delta, @1000 trajectories)

| Method | Trajectories | F1 | Sep% | Precision | Recall |
|--------|-------------|------|------|-----------|--------|
| Lyapunov NN @1,000 | 1,000 | 0.7340 | 0.00% | 0.8590 | 0.6410 |
| DeepReach (auto-threshold) @1,000 | 1,000 | 0.7988 | 0.00% | 0.7338 | 0.8764 |
| Classification @1,000 | 1,000 | 0.9450 | 0.05% | 0.9550 | 0.9350 |
| Non-generative @1,000 | 1,000 | 0.9990 | 31.60% | 0.9970 | 1.0000 |
| MORALS @1,000 | 1,000 | 0.4438 | 0.00% | 0.5979 | 0.3528 |
| Non-adaptive | 1,000 | 0.9823 | 8.10% | 0.9930 | 0.9718 |
| Adaptive (direct) | 1,000 | 0.9878 | 0.99% | 0.9954 | 0.9803 |
| **Adaptive (ranked)** | **1,000** | **0.9918** | **1.72%** | **0.9937** | **0.9900** |

## Quadrotor 3D

### Runs

| Label | d2_ratio | filter | train radius | Path |
|-------|----------|--------|--------------|------|
| d2=0, filtered | 0.0 | yes | 0.3 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-14_19-23-57` |
| d2=0, no filter | 0 | no | 0.3 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-13_23-34-02` |
| d2=1.0, no filter | 1.0 | no | 0.3 | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-14_12-42-38` |

### Evaluation

All evaluated at `radius_0.3_alpha_0.1_mc_10_batch_100000` using `qhat_prediction_sets`.

### Config Differences

| Parameter | no filter runs | filtered run |
|-----------|---------------|--------------|
| `conformal.use_p_invalid_veto` | absent | true |
| `adaptive_v2.filter_confident_pairs` | absent | true |
| `adaptive_v2.filter_min_pairs` | absent | 100 |

Note: All three quad3d runs share `conformal.attractor_radius=0.3`.

### Flow Matching Results

#### 0% adaptive, filtered

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 43.93% | 0.9125 | 0.8766 | 0.9514 | 0.9500 |
| 6000 | 43.88% | 0.9304 | 0.9049 | 0.9573 | 0.9653 |
| 7000 | 38.81% | 0.9154 | 0.8805 | 0.9532 | 0.9548 |
| 8000 | 37.53% | 0.9118 | 0.8726 | 0.9548 | 0.9509 |
| 9000 | 35.56% | 0.9089 | 0.8687 | 0.9531 | 0.9502 |
| 10000 | 33.73% | 0.9065 | 0.8625 | 0.9551 | 0.9478 |
| 11000 | 32.25% | 0.9047 | 0.8624 | 0.9514 | 0.9490 |
| 12000 | 31.38% | 0.9183 | 0.8835 | 0.9560 | 0.9585 |
| 13000 | 31.27% | 0.9190 | 0.8838 | 0.9570 | 0.9584 |
| 14000 | 29.81% | 0.9170 | 0.8804 | 0.9567 | 0.9574 |

#### 0% adaptive, no filter

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 44.09% | 0.9117 | 0.8748 | 0.9518 | 0.9489 |
| 6000 | 43.05% | 0.9273 | 0.8994 | 0.9571 | 0.9623 |
| 7000 | 39.09% | 0.9090 | 0.8694 | 0.9524 | 0.9493 |
| 8000 | 36.83% | 0.9059 | 0.8625 | 0.9539 | 0.9467 |
| 9000 | 35.40% | 0.9112 | 0.8721 | 0.9539 | 0.9518 |
| 10000 | 33.92% | 0.9066 | 0.8637 | 0.9540 | 0.9483 |
| 11000 | 32.14% | 0.9121 | 0.8718 | 0.9563 | 0.9528 |
| 12000 | 32.08% | 0.9225 | 0.8891 | 0.9586 | 0.9606 |
| 13000 | 30.30% | 0.9076 | 0.8662 | 0.9532 | 0.9511 |
| 14000 | 29.44% | 0.9150 | 0.8792 | 0.9538 | 0.9573 |

#### 100% adaptive, no filter

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 43.95% | 0.9123 | 0.8761 | 0.9516 | 0.9498 |
| 6000 | 42.85% | 0.9269 | 0.8980 | 0.9577 | 0.9617 |
| 7000 | 39.37% | 0.9101 | 0.8699 | 0.9541 | 0.9488 |
| 8000 | 36.50% | 0.9058 | 0.8626 | 0.9535 | 0.9467 |
| 9000 | 35.36% | 0.9104 | 0.8728 | 0.9514 | 0.9522 |
| 10000 | 33.94% | 0.9078 | 0.8646 | 0.9556 | 0.9488 |
| 11000 | 32.16% | 0.9107 | 0.8698 | 0.9557 | 0.9521 |
| 12000 | 31.18% | 0.9128 | 0.8731 | 0.9564 | 0.9537 |
| 13000 | 30.22% | 0.9076 | 0.8638 | 0.9560 | 0.9501 |
| 14000 | 29.63% | 0.9182 | 0.8830 | 0.9564 | 0.9586 |

### Adaptive Sampling Comparison

#### Runs

| Label | d2_ratio | sampling_mode | optimize_mode | filter | train radius | Path |
|-------|----------|---------------|---------------|--------|--------------|------|
| Non-adaptive baseline | 0.0 | ranked | joint | yes | 0.3 | `.../adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_..._sampling_mode_ranked/2026-02-18_14-25-10` |
| Adaptive direct | 1.0 | direct | joint | yes | 0.3 | `.../adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_..._sampling_mode_direct/2026-02-18_14-25-13` |
| Adaptive ranked | 1.0 | ranked | joint | yes | 0.3 | `.../adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_..._sampling_mode_ranked/2026-02-18_14-25-29` |

Note: Adaptive ranked has 8/10 eval epochs. Q-hat results are unstable (some epochs have F1=0 due to degenerate threshold optimization); lambda_delta results are stable.

#### Evaluation

All evaluated at `radius_0.3_alpha_0.1_mc_10_batch_100000`.

#### Lambda-Delta Results

##### Non-adaptive baseline (d2=0)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 35.65% | 0.8441 | 0.9830 | 0.7396 | 0.9978 |
| 6000 | 39.84% | 0.9182 | 0.9721 | 0.8699 | 0.9945 |
| 7000 | 31.53% | 0.8988 | 0.9732 | 0.8350 | 0.9949 |
| 8000 | 30.91% | 0.9104 | 0.9707 | 0.8571 | 0.9942 |
| 9000 | 32.63% | 0.8868 | 0.9877 | 0.8047 | 0.9983 |
| 10000 | 27.05% | 0.9147 | 0.9703 | 0.8651 | 0.9938 |
| 11000 | 29.63% | 0.8776 | 0.9873 | 0.7898 | 0.9983 |
| 12000 | 27.38% | 0.9112 | 0.9880 | 0.8455 | 0.9979 |
| 13000 | 29.38% | 0.9363 | 0.9611 | 0.9127 | 0.9908 |
| 14000 | 26.05% | 0.9194 | 0.9769 | 0.8683 | 0.9956 |

##### Adaptive direct

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 35.64% | 0.8445 | 0.9826 | 0.7405 | 0.9978 |
| 6000 | 38.01% | 0.9206 | 0.9701 | 0.8759 | 0.9934 |
| 7000 | 47.63% | 0.9355 | 0.9794 | 0.8955 | 0.9965 |
| 8000 | 34.91% | 0.9172 | 0.9731 | 0.8674 | 0.9953 |
| 9000 | 24.68% | 0.9101 | 0.9722 | 0.8554 | 0.9943 |
| 10000 | 35.15% | 0.9212 | 0.9763 | 0.8719 | 0.9963 |
| 11000 | 31.39% | 0.9318 | 0.9847 | 0.8842 | 0.9972 |
| 12000 | 37.40% | 0.9493 | 0.9801 | 0.9204 | 0.9962 |
| 13000 | 27.98% | 0.9373 | 0.9817 | 0.8967 | 0.9962 |
| 13934 | 25.01% | 0.9290 | 0.9804 | 0.8828 | 0.9961 |

##### Adaptive ranked (8/10 eval epochs)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 34.84% | 0.8823 | 0.9785 | 0.8034 | 0.9963 |
| 6000 | 37.59% | 0.9190 | 0.9683 | 0.8744 | 0.9930 |
| 7000 | 35.34% | 0.9203 | 0.9753 | 0.8712 | 0.9951 |
| 8000 | 32.19% | 0.9248 | 0.9733 | 0.8809 | 0.9945 |
| 9000 | 36.06% | 0.8949 | 0.9853 | 0.8196 | 0.9982 |
| 10000 | 41.39% | 0.9397 | 0.9780 | 0.9043 | 0.9964 |
| 11000 | 30.51% | 0.9274 | 0.9813 | 0.8790 | 0.9966 |
| 12000 | 30.37% | 0.9396 | 0.9730 | 0.9084 | 0.9941 |

#### Q-hat Prediction Sets Results

Note: Q-hat is unstable for Quadrotor 3D — several epochs produce F1=0 (degenerate threshold classifies nothing as success).

##### Non-adaptive baseline (d2=0)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 45.43% | 0.8804 | 0.9916 | 0.7916 | 0.9990 |
| 6000 | 39.84% | 0.9182 | 0.9721 | 0.8699 | 0.9945 |
| 7000 | 55.14% | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 8000 | 44.70% | 0.9521 | 0.9852 | 0.9211 | 0.9968 |
| 9000 | 60.51% | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 10000 | 51.52% | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 11000 | 44.81% | 0.9360 | 0.9944 | 0.8842 | 0.9992 |
| 12000 | 44.05% | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 13000 | 35.02% | 0.9510 | 0.9611 | 0.9410 | 0.9898 |
| 14000 | 33.31% | 0.9378 | 0.9896 | 0.8912 | 0.9982 |

##### Adaptive direct

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 44.19% | 0.8955 | 0.9826 | 0.8225 | 0.9974 |
| 6000 | 62.75% | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7000 | 47.63% | 0.9355 | 0.9794 | 0.8955 | 0.9965 |
| 8000 | 41.40% | 0.9412 | 0.9731 | 0.9112 | 0.9947 |
| 9000 | 49.35% | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 10000 | 42.76% | 0.9421 | 0.9871 | 0.9010 | 0.9981 |
| 11000 | 37.06% | 0.9510 | 0.9847 | 0.9195 | 0.9969 |
| 12000 | 37.40% | 0.9493 | 0.9801 | 0.9204 | 0.9962 |
| 13000 | 33.49% | 0.9543 | 0.9817 | 0.9285 | 0.9958 |
| 13934 | 32.44% | 0.9471 | 0.9912 | 0.9068 | 0.9984 |

##### Adaptive ranked (8/10 eval epochs)

| Trajectories | Sep% | F1 | Precision | Recall | Specificity |
|-------------|------|------|-----------|--------|-------------|
| 5000 | 50.64% | 0.9400 | 0.9890 | 0.8956 | 0.9979 |
| 6000 | 52.87% | 0.9612 | 0.9839 | 0.9395 | 0.9960 |
| 7000 | 43.26% | 0.9429 | 0.9861 | 0.9034 | 0.9972 |
| 8000 | 39.97% | 0.9478 | 0.9851 | 0.9133 | 0.9969 |
| 9000 | 36.06% | 0.8949 | 0.9853 | 0.8196 | 0.9982 |
| 10000 | 41.39% | 0.9397 | 0.9780 | 0.9043 | 0.9964 |
| 11000 | 36.97% | 0.9508 | 0.9813 | 0.9222 | 0.9962 |
| 12000 | 37.64% | 0.9584 | 0.9849 | 0.9333 | 0.9967 |

### Lyapunov Neural Network

| Trajectories | Separatrix | Accuracy | Precision | Recall | F1 |
|-------------|-----------|----------|-----------|--------|------|
| 3000 | 1.24% | 0.8811 | 0.8196 | 0.5803 | 0.6795 |
| 4000 | 0.67% | 0.8840 | 0.7683 | 0.6713 | 0.7165 |
| 5000 | 1.58% | 0.8864 | 0.7587 | 0.6994 | 0.7279 |
| 6000 | 0.66% | 0.8922 | 0.7958 | 0.6811 | 0.7340 |
| 7000 | 0.78% | 0.8944 | 0.7778 | 0.7229 | 0.7493 |
| 8000 | 0.70% | 0.8955 | 0.8875 | 0.5939 | 0.7116 |
| 9000 | 0.61% | 0.8941 | 0.7620 | 0.7509 | 0.7564 |
| 10000 | 0.57% | 0.9024 | 0.8156 | 0.7150 | 0.7620 |
| 11000 | 0.57% | 0.9040 | 0.8171 | 0.7220 | 0.7666 |
| 12000 | 0.57% | 0.9016 | 0.8053 | 0.7255 | 0.7633 |

### Summary: Joint Optimization vs Classifier (Lambda-Delta, @12000 trajectories)

| Method | Trajectories | F1 | Sep% | Precision | Recall |
|--------|-------------|------|------|-----------|--------|
| Lyapunov NN | 12,000 | 0.7633 | 0.57% | 0.8053 | 0.7255 |
| DeepReach (auto-threshold) | 12,000 | 0.6183 | 0% | 0.6221 | 0.6145 |
| Non-adaptive | 12,000 | 0.9112 | 27.38% | 0.9880 | 0.8455 |
| Adaptive (direct) | 12,000 | 0.9493 | 37.40% | 0.9801 | 0.9204 |
| **Adaptive (ranked)** | **12,000** | **0.9396** | **30.37%** | **0.9730** | **0.9084** |

## Pendulum

### Runs

| Label | d2_ratio | sampling_mode | Path |
|-------|----------|---------------|------|
| Non-adaptive | 0.0 | ranked | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-56-20` |
| Adaptive (direct) | 1.0 | direct | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07` |
| Adaptive (ranked) | 1.0 | ranked | `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-55-55` |

### Evaluation

All evaluated at `radius_0.075_alpha_0.1_mc_10_batch_100000`. Lambda-delta = qhat (q_hat=0 for all epochs).

### Joint Optimization Results (Lambda-Delta)

#### Non-adaptive

| Epoch | Trajectories | F1 | Sep% | Precision | Recall |
|-------|-------------|------|------|-----------|--------|
| 0 | 100 | 0.9445 | 28.45% | 0.9917 | 0.9016 |
| 1 | 150 | 0.9869 | 25.96% | 0.9888 | 0.9850 |
| 2 | 200 | 0.9783 | 15.75% | 0.9911 | 0.9658 |
| 3 | 250 | 0.9860 | 6.84% | 0.9843 | 0.9877 |
| 4 | 300 | 0.9842 | 10.18% | 0.9876 | 0.9809 |
| 5 | 350 | 0.9691 | 4.34% | 0.9663 | 0.9720 |
| 6 | 400 | 0.9849 | 6.40% | 0.9898 | 0.9801 |
| 7 | 450 | 0.9841 | 7.81% | 0.9768 | 0.9916 |
| 8 | 500 | 0.9856 | 4.28% | 0.9785 | 0.9929 |
| 9 | 550 | 0.9857 | 4.50% | 0.9793 | 0.9921 |
| 10 | 600 | 0.9861 | 4.56% | 0.9792 | 0.9932 |

#### Adaptive (direct)

| Epoch | Trajectories | F1 | Sep% | Precision | Recall |
|-------|-------------|------|------|-----------|--------|
| 0 | 100 | 0.9448 | 28.83% | 0.9904 | 0.9032 |
| 1 | 150 | 0.9656 | 19.25% | 0.9894 | 0.9428 |
| 2 | 200 | 0.9714 | 11.91% | 0.9963 | 0.9477 |
| 3 | 250 | 0.9915 | 9.24% | 0.9976 | 0.9856 |
| 4 | 300 | 0.9776 | 1.91% | 0.9990 | 0.9571 |
| 5 | 350 | 0.9836 | 3.86% | 0.9960 | 0.9716 |
| 6 | 400 | 0.9709 | 1.90% | 0.9999 | 0.9434 |
| 7 | 450 | 0.9837 | 2.82% | 0.9989 | 0.9689 |
| 8 | 500 | 0.9739 | 5.95% | 0.9998 | 0.9493 |
| 9 | 550 | 0.9737 | 8.82% | 0.9999 | 0.9489 |
| 10 | 600 | 0.9766 | 5.06% | 1.0000 | 0.9544 |

#### Adaptive (ranked)

| Epoch | Trajectories | F1 | Sep% | Precision | Recall |
|-------|-------------|------|------|-----------|--------|
| 0 | 100 | 0.9436 | 28.78% | 0.9898 | 0.9015 |
| 1 | 150 | 0.9837 | 25.76% | 0.9962 | 0.9715 |
| 2 | 200 | 0.9747 | 9.94% | 0.9925 | 0.9575 |
| 3 | 250 | 0.9711 | 3.07% | 0.9964 | 0.9470 |
| 4 | 300 | 0.9771 | 4.57% | 0.9996 | 0.9556 |
| 5 | 350 | 0.9877 | 3.06% | 0.9986 | 0.9770 |
| 6 | 400 | 0.9944 | 4.21% | 0.9947 | 0.9941 |
| 7 | 450 | 0.9815 | 2.62% | 1.0000 | 0.9637 |
| 8 | 500 | 0.9644 | 11.70% | 0.9998 | 0.9313 |
| 9 | 550 | 0.9738 | 12.44% | 0.9987 | 0.9501 |
| 10 | 600 | 0.9655 | 9.26% | 1.0000 | 0.9334 |

### Summary: Joint Optimization (Lambda-Delta, @600 trajectories)

| Method | Trajectories | F1 | Sep% | Precision | Recall |
|--------|-------------|------|------|-----------|--------|
| Lyapunov NN @1,000 | 1,000 | 0.7550 | 0.00% | 0.8630 | 0.6700 |
| DeepReach (auto-threshold) @1,000 | 1,000 | 0.9752 | — | 0.9805 | 0.9700 |
| Classification @1,000 | 1,000 | 0.9850 | 0.19% | 0.9830 | 0.9880 |
| Non-generative @1,000 | 1,000 | 0.6440 | — | 0.9990 | 0.4760 |
| MORALS @1,000 | 1,000 | 0.9390 | 7.30% | 0.9430 | 0.9350 |
| Non-adaptive | 600 | 0.9861 | 4.56% | 0.9792 | 0.9932 |
| Adaptive (direct) | 600 | 0.9766 | 5.06% | 1.0000 | 0.9544 |
| Adaptive (ranked) | 600 | 0.9655 | 9.26% | 1.0000 | 0.9334 |

## DeepReach Baseline

DeepReach trains a SIREN neural network to approximate the value function V(t,x) via temporal consistency loss on trajectory data. States with V <= threshold are classified as safe. All models: 2x96 SIREN, `exact` model, 50k epochs.

Experiment directory: `/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/deepreach`

### Summary (threshold=0)

| System | Samples | F1 | Precision | Recall | Specificity | Balanced Acc |
|--------|---------|------|-----------|--------|-------------|--------------|
| Pendulum (2D) | 48,770 | **0.9607** | 0.9965 | 0.9273 | 0.9980 | 0.9626 |
| CartPole (4D) | 115,242 | 0.7133 | 0.5545 | 0.9997 | 0.8237 | 0.9117 |
| Quadrotor 2D (6D) | 488,789 | 0.0015 | 0.0012 | 0.0020 | 0.8508 | 0.4264 |
| Quadrotor 3D (13D) | 999,000 | 0.0025 | 0.4982 | 0.0012 | 0.9996 | 0.5004 |

### Summary (auto-threshold, optimized for F1 on cal_set)

| System | Samples | Threshold | F1 | Precision | Recall | Specificity | Balanced Acc |
|--------|---------|-----------|------|-----------|--------|-------------|--------------|
| Pendulum (2D) | 48,770 | 0.0772 | **0.9752** | 0.9805 | 0.9700 | 0.9878 | 0.9789 |
| CartPole (4D) | 115,242 | -0.1778 | 0.7988 | 0.7338 | 0.8764 | 0.9302 | 0.9033 |
| Quadrotor 2D (6D) | 488,789 | 0.2814 | 0.2548 | 0.1526 | 0.7718 | 0.6260 | 0.6989 |
| Quadrotor 3D (13D) | 999,000 | 0.5684 | 0.6183 | 0.6221 | 0.6145 | 0.8949 | 0.7547 |

### Pendulum (2D)

- **Checkpoint**: `model_epoch_50000.pth`, t_eval=5.0
- **Data**: `/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k`

| Metric | threshold=0 | auto-threshold (0.0772) |
|--------|-------------|-------------------------|
| TP / TN / FP / FN | 17,470 / 29,869 / 61 / 1,370 | 18,274 / 29,566 / 364 / 566 |
| Precision | 0.9965 | 0.9805 |
| Recall | 0.9273 | 0.9700 |
| Specificity | 0.9980 | 0.9878 |
| F1 | 0.9607 | 0.9752 |
| Accuracy | 0.9707 | 0.9809 |
| Balanced Accuracy | 0.9626 | 0.9789 |

### CartPole (4D)

- **Checkpoint**: `model_epoch_50000.pth`, t_eval=6.0
- **Data**: `/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet`

| Metric | threshold=0 | auto-threshold (-0.1778) |
|--------|-------------|--------------------------|
| TP / TN / FP / FN | 20,737 / 77,836 / 16,662 / 7 | 18,179 / 87,904 / 6,594 / 2,565 |
| Precision | 0.5545 | 0.7338 |
| Recall | 0.9997 | 0.8764 |
| Specificity | 0.8237 | 0.9302 |
| F1 | 0.7133 | 0.7988 |
| Accuracy | 0.8554 | 0.9205 |
| Balanced Accuracy | 0.9117 | 0.9033 |

Note: Threshold tuning shifted from 0 to -0.178, reducing FP from 16.6k to 6.6k. F1 improved 0.71 → 0.80.

### Quadrotor 2D (6D)

- **Checkpoint**: `model_epoch_50000.pth`, t_eval=6.0
- **Data**: `/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_rl`

| Metric | threshold=0 | auto-threshold (0.2814) |
|--------|-------------|--------------------------|
| TP / TN / FP / FN | 79 / 382,503 / 67,061 / 39,146 | 30,272 / 281,448 / 168,116 / 8,953 |
| Precision | 0.0012 | 0.1526 |
| Recall | 0.0020 | 0.7718 |
| Specificity | 0.8508 | 0.6260 |
| F1 | 0.0015 | 0.2548 |
| Accuracy | 0.7827 | 0.6377 |
| Balanced Accuracy | 0.4264 | 0.6989 |

Note: Auto-threshold recovers recall (0.002 → 0.77) but at massive cost to specificity (0.85 → 0.63). Model has not learned a meaningful decision boundary.

### Quadrotor 3D (13D)

- **Checkpoint**: `model_epoch_50000.pth`, t_eval=6.0
- **Data**: `/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr`

| Metric | threshold=0 | auto-threshold (0.5684) |
|--------|-------------|--------------------------|
| TP / TN / FP / FN | 273 / 779,284 / 275 / 219,168 | 134,857 / 697,648 / 81,911 / 84,584 |
| Precision | 0.4982 | 0.6221 |
| Recall | 0.0012 | 0.6145 |
| Specificity | 0.9996 | 0.8949 |
| F1 | 0.0025 | 0.6183 |
| Accuracy | 0.7803 | 0.8333 |
| Balanced Accuracy | 0.5004 | 0.7547 |

Note: Large threshold shift (0 → 0.57) recovers some discriminative ability (F1: 0.002 → 0.62), but still weak. Model has partial structure but far from reliable.
