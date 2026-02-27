# Main Results

Detailed per-system results (including all baselines) are in the [`results/`](results/) directory:
- [Quadrotor 2D](results/quadrotor2d.md)
- [CartPole (PyBullet)](results/cartpole.md)
- [Quadrotor 3D](results/quadrotor3d.md)
- [Pendulum](results/pendulum.md)

## Quadrotor 2D — Summary

### Joint vs Delta Optimization (Final Epoch)

| Method | Opt Mode | Trajectories | F1 | Sep% | Cons. F1 | Coverage | Precision | Recall |
|--------|----------|-------------|------|------|----------|----------|-----------|--------|
| Non-adaptive baseline | joint | 12,000 | 0.8576 | 7.97% | 0.6948 | 0.9057 | 0.9502 | 0.7814 |
| Adaptive direct (d2=1.0) | joint | 7,294 | 0.9163 | 6.39% | 0.7929 | 0.9261 | 0.9489 | 0.8859 |
| Adaptive direct (d2=0.75) | joint | 10,332 | 0.9210 | 10.59% | 0.6371 | 0.8876 | 0.9723 | 0.8748 |
| Adaptive direct (d2=0.5) | joint | 11,839 | 0.9232 | 15.05% | 0.6197 | 0.8434 | 0.9692 | 0.8815 |
| Adaptive ranked | joint | 12,000 | 0.9432 | 14.08% | 0.6198 | 0.8548 | 0.9703 | 0.9176 |
| Non-adaptive baseline | delta | 12,000 | 0.9212 | 23.35% | — | — | 0.9441 | 0.8995 |
| Adaptive direct | delta | 11,152 | 0.9494 | 16.70% | — | — | 0.9487 | 0.9501 |
| **Adaptive ranked** | **delta** | **12,000** | **0.9623** | **25.99%** | **—** | **—** | **0.9580** | **0.9667** |

All joint optimization runs evaluated with mc=20, cal_k=2000. Delta runs use mc=10 (not yet re-evaluated). Cons. F1 = conservative F1 (separatrix classified as failure).

**Key findings:**
- Delta optimization achieves higher F1 but at the cost of much higher Sep% (16-26% vs 6-14%) — it's more conservative, abstaining on more points rather than committing
- Joint optimization is more aggressive — lower Sep% (tighter separatrix) but lower F1 because it commits on harder points
- Ranking: ranked > direct > non-adaptive holds for both optimization modes
- Best F1: Adaptive ranked (delta) at 0.9623, but with 26% Sep%
- Best F1/Sep% tradeoff: Adaptive direct (d2=1.0, joint) — F1=0.9163 with only 6.39% Sep% using 7,294 trajectories, and highest conservative F1 (0.7929)
- Conservative F1 heavily penalizes high Sep%: ranked joint has F1=0.9432 but cons. F1=0.6198 (14% Sep%), while direct d2=1.0 has F1=0.9163 but cons. F1=0.7929 (6.4% Sep%)

### All Methods Comparison (Lambda-Delta)

| Method | Trajectories | F1 | Sep% | Cons. F1 | Coverage | Precision | Recall |
|--------|-------------|------|------|----------|----------|-----------|--------|
| Classification | 12,000 | 0.7767 | 0.46% | — | — | 0.8956 | 0.6856 |
| Non-generative | 12,000 | 0.0665 | 1% | — | — | 0.9733 | 0.0344 |
| Lyapunov NN | 12,000 | 0.8040 | 35.95% | — | — | 0.7000 | 0.9450 |
| DeepReach | 12,000 | 0.263 | 0% | — | — | — | — |
| Non-adaptive | 12,000 | 0.8576 | 7.97% | 0.6948 | 0.9057 | 0.9502 | 0.7814 |
| Adaptive (direct, d2=1.0) | 7,294 | 0.9163 | 6.39% | 0.7929 | 0.9261 | 0.9489 | 0.8859 |
| Adaptive (direct, d2=0.75) | 10,332 | 0.9210 | 10.59% | 0.6371 | 0.8876 | 0.9723 | 0.8748 |
| Adaptive (direct, d2=0.5) | 11,839 | 0.9232 | 15.05% | 0.6197 | 0.8434 | 0.9692 | 0.8815 |
| **Adaptive (ranked)** | **12,000** | **0.9432** | **14.08%** | **0.6198** | **0.8548** | **0.9703** | **0.9176** |

All flow matching runs evaluated with mc=20, cal_k=2000. Cons. F1 = conservative F1 (separatrix classified as failure).

## CartPole — Summary (@1,000 trajectories)

| Method | Trajectories | F1 | Sep% | Cons. F1 | Coverage | Precision | Recall |
|--------|-------------|------|------|----------|----------|-----------|--------|
| Lyapunov NN | 1,000 | 0.9520 | 26.80% | — | — | 0.9610 | 0.9440 |
| DeepReach | 1,000 | 0.543 | 0% | — | — | — | — |
| Classification | 1,000 | 0.9460 | 0.15% | — | — | 0.9668 | 0.9260 |
| Non-generative | 1,000 | 0.3457 | 1% | — | — | 0.9991 | 0.2090 |
| MORALS | 1,000 | 0.699 | 17.9% | — | — | — | — |
| Non-adaptive | 1,000 | 0.9839 | 7.59% | 0.8212 | 0.9200 | 0.9915 | 0.9764 |
| Adaptive (direct, d2=1.0) | 1,000 | 0.9896 | 0.99% | 0.9742 | 0.9865 | 0.9958 | 0.9836 |
| Adaptive (direct, d2=0.75) | 1,000 | 0.9867 | 1.68% | 0.9524 | 0.9788 | 0.9967 | 0.9769 |
| Adaptive (direct, d2=0.5) | 1,000 | 0.9947 | 3.93% | 0.9304 | 0.9590 | 0.9967 | 0.9926 |
| **Adaptive (ranked, d2=1.0)** | **1,000** | **0.9924** | **1.64%** | **0.9769** | **0.9809** | **0.9929** | **0.9918** |

All flow matching runs evaluated with mc=20. Cons. F1 = conservative F1 (separatrix classified as failure).

## Quadrotor 3D — Summary (@12,000 trajectories)

| Method | Trajectories | F1 | Sep% | Precision | Recall |
|--------|-------------|------|------|-----------|--------|
| Classification | 12,000 | 0.7633 | 0.57% | 0.8053 | 0.7255 |
| Non-generative | 12,000 | 0.0290 | 0.95% | 1.0000 | 0.0150 |
| Lyapunov NN | 12,000 | 0.9350 | 36.05% | 0.9310 | 0.9390 |
| DeepReach | 15,000 | 0.721 | 61% | — | — |
| Non-adaptive | 12,000 | 0.9112 | 27.38% | 0.9880 | 0.8455 |
| Adaptive (direct, 10 iter) | 12,000 | 0.9493 | 37.40% | 0.9801 | 0.9204 |
| Adaptive (direct, 20 iter) | 25,000 | 0.9525 | 25.22% | 0.9737 | 0.9322 |
| **Adaptive (ranked)** | **12,000** | **0.9396** | **30.37%** | **0.9730** | **0.9084** |

## Pendulum — Summary (@500 trajectories)

| Method | Trajectories | F1 | Sep% | Cons. F1 | Precision | Recall |
|--------|-------------|------|------|----------|-----------|--------|
| Lyapunov NN | 500 | 0.527 | 39.5% | — | — | — |
| Lyapunov NN | 1,000 | 0.837 | 34.6% | — | — | — |
| DeepReach | 1,000 | 0.968 | 27% | — | — | — |
| Classification | 500 | 0.9744 | 0.15% | — | 0.9514 | 0.9985 |
| Non-generative | 500 | 0.7681 | 0.61% | — | 0.9967 | 0.6248 |
| MORALS @1,000 | 1,000 | 0.9390 | 7.30% | — | 0.9430 | 0.9350 |
| Non-adaptive | 500 | 0.9861 | 4.22% | 0.9681 | 0.9781 | 0.9943 |
| Adaptive (direct, d2=1.0) | 500 | 0.9741 | 5.95% | 0.9622 | 1.0000 | 0.9495 |
| Adaptive (ranked, d2=1.0) | 500 | 0.9727 | 11.84% | 0.9417 | 1.0000 | 0.9468 |
| Adaptive (direct, d2=0.75) | 500 | 0.9849 | 1.33% | 0.9775 | 0.9998 | 0.9704 |
| **Adaptive (direct, d2=0.5)** | **500** | **0.9902** | **1.57%** | **0.9859** | **0.9949** | **0.9856** |

All flow matching runs evaluated with mc=20. Cons. F1 = conservative F1 (separatrix classified as failure).

## DeepReach Baseline — Summary

| System | Trajectories | F1 | Sep% | Notes |
|--------|-------------|------|------|-------|
| Pendulum (2D) | 900 | 0.968 | 27% | val+train, w=0.95 |
| CartPole (4D) | 900 | 0.543 | 0% | val, w=0.3 |
| Quadrotor 2D (6D) | 12,000 | 0.263 | 0% | val+train, w=0.3 |
| Quadrotor 3D (13D) | 15,000 | 0.721 | 61% | val, w=0.99 |

DeepReach degrades sharply with dimensionality. In 6D+ it fails to learn a meaningful decision boundary (F1 < 0.3 for Quad2D). The Quad3D best result (F1=0.721) requires extreme conservatism (w=0.99, 61% Sep%).

---

## Multi-Point Results (Low / Mid / High Trajectories)

### Pendulum — @100, @250, @500

| Method | @100 F1 | @100 Sep% | @250 F1 | @250 Sep% | @500 F1 | @500 Sep% |
|--------|---------|-----------|---------|-----------|---------|-----------|
| Classification | 0.9476 | 0.19% | 0.9694 | 0.02% | 0.9744 | 0.15% |
| Non-generative | 0.6922 | 0.61% | 0.7411 | 0.61% | 0.7681 | 0.61% |
| Lyapunov NN | — | — | — | — | 0.527 | 39.5% |
| DeepReach | — | — | — | — | 0.980 | 18% |
| Non-adaptive | 0.9503 | 30.72% | 0.9874 | 6.97% | 0.9861 | 4.22% |
| Adaptive (direct, d2=1.0) | 0.9492 | 30.68% | 0.9945 | 9.54% | 0.9741 | 5.95% |
| Adaptive (ranked, d2=1.0) | 0.9497 | 30.83% | 0.9742 | 2.99% | 0.9727 | 11.84% |
| Adaptive (direct, d2=0.75) | 0.9961 | 50.71% | 0.9913 | 5.36% | 0.9849 | 1.33% |
| **Adaptive (direct, d2=0.5)** | **0.9912** | **44.80%** | **0.9800** | **8.27%** | **0.9902** | **1.57%** |

Lyapunov NN: @50 F1=0.387 (39.5% uncertain), @1000 F1=0.837 (34.6%). DeepReach: @50 F1=0.962, @500 F1=0.980, @900 F1=0.968. MORALS: @1000 F1=0.939 (7.3%).

### CartPole — @300, @650, @1,000

| Method | @300 F1 | @300 Sep% | @650 F1 | @650 Sep% | @1000 F1 | @1000 Sep% |
|--------|---------|-----------|---------|-----------|----------|------------|
| Classification | 0.8882 | 0.07% | 0.9291 | 0.06% | 0.9460 | 0.15% |
| Non-generative | 0.4057 | 1% | 0.5021 | 1% | 0.3457 | 1% |
| Lyapunov NN | — | — | — | — | 0.9520 | 26.80% |
| DeepReach | 0.485 | 0% | 0.599 | 0% | 0.543 | 0% |
| MORALS | 0.396 | 10.4% | 0.221 | 62.9% | 0.699 | 17.9% |
| Non-adaptive | 0.9568 | 35.71% | 0.9801 | 13.65% | 0.9839 | 7.59% |
| Adaptive (direct, d2=1.0) | 0.9562 | 35.66% | 0.9904 | 8.12% | 0.9896 | 0.99% |
| Adaptive (ranked, d2=1.0) | 0.9631 | 36.82% | 0.9876 | 5.58% | 0.9924 | 1.64% |
| Adaptive (direct, d2=0.75) | 0.9624 | 36.85% | 0.9796 | 10.04% | 0.9867 | 1.68% |
| **Adaptive (direct, d2=0.5)** | **0.9647** | **36.83%** | **0.9803** | **9.76%** | **0.9947** | **3.93%** |

DeepReach: @900 F1=0.543 used for @1000 column (closest available).

### Quadrotor 2D — @3,000, @7,000, @12,000 (Joint Optimization)

| Method | @3000 F1 | @3000 Sep% | @7000 F1 | @7000 Sep% | @12000 F1 | @12000 Sep% |
|--------|----------|------------|----------|------------|-----------|-------------|
| Classification | 0.6088 | 0.32% | 0.7313 | 0.26% | 0.7767 | 0.46% |
| Non-generative | 0.0587 | 0.95% | 0.0808 | 1% | 0.0665 | 1% |
| Lyapunov NN | — | — | — | — | 0.8040 | 35.95% |
| DeepReach | 0.143 | 0% | 0.158 | 0% | 0.263 | 0% |
| Non-adaptive | 0.5468 | 10.67% | 0.8021 | 10.13% | 0.8576 | 7.97% |
| Adaptive (direct, d2=1.0) | 0.5478 | 10.63% | 0.9127 | 6.98% | 0.9163 | 6.39% |
| Adaptive (direct, d2=0.75) | 0.6477 | 17.88% | 0.8904 | 14.57% | 0.9210 | 10.59% |
| Adaptive (direct, d2=0.5) | 0.6471 | 17.84% | 0.8509 | 14.65% | 0.9232 | 15.05% |
| **Adaptive (ranked)** | **0.7094** | **23.27%** | **0.9077** | **9.75%** | **0.9432** | **14.08%** |

DeepReach: @2000 F1=0.143, @7500 F1=0.158, @12000 F1=0.263 (closest available points). Adaptive direct d2=1.0 final epoch at 7,294 trajectories (shown at @7000). Lyapunov NN only available @12,000.

### Quadrotor 3D — @5,000, @10,000, @12,000 (Lambda-Delta)

| Method | @5000 F1 | @5000 Sep% | @10000 F1 | @10000 Sep% | @12000 F1 | @12000 Sep% |
|--------|----------|------------|-----------|-------------|-----------|-------------|
| Classification | 0.7279 | 1.58% | 0.7620 | 0.57% | 0.7633 | 0.57% |
| Non-generative | 0.0200 | 1% | 0.0500 | 0.95% | 0.0290 | 0.95% |
| Lyapunov NN | — | — | — | — | 0.9350 | 36.05% |
| DeepReach | 0.502 | 9% | 0.375 | 0% | 0.721 | 61% |
| Non-adaptive | 0.8441 | 35.65% | 0.9147 | 27.05% | 0.9112 | 27.38% |
| Adaptive (direct, 10 iter) | 0.8445 | 35.64% | 0.9212 | 35.15% | 0.9493 | 37.40% |
| Adaptive (direct, 20 iter) | — | — | 0.9388 | 42.90% | 0.9359 | 31.79% |
| **Adaptive (ranked)** | **0.8823** | **34.84%** | **0.9397** | **41.39%** | **0.9396** | **30.37%** |

DeepReach: @5000 F1=0.502, @10000 F1=0.375, @15000 F1=0.721 (closest available points). All flow matching runs at mc=10. Lyapunov NN only available @12,000. The 20-iter direct run starts at 10k trajectories (initial_train_size=10000).
