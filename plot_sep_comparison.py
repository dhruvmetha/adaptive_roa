import matplotlib.pyplot as plt
import numpy as np

# Data for iter_15 (continues decreasing) - using q_hat prediction sets
train_traj_15 = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
sep_pct_15 = [
    32.201,   # epoch 0 qhat
    15.782,   # epoch 1 qhat
    5.822,    # epoch 2 qhat
    4.788,    # epoch 3 qhat
    4.593,    # epoch 4 qhat
    4.213,    # epoch 5 qhat
    2.575,    # epoch 6 qhat
    2.690,    # epoch 7 qhat
    2.201,    # epoch 8 qhat
    1.891,    # epoch 9 qhat
    2.657,    # epoch 10 qhat
    1.807,    # epoch 11 qhat
]
# F1 scores (from qhat_prediction_sets)
f1_15 = [
    0.0,     # epoch 0 qhat
    97.52,   # epoch 1 qhat
    98.42,   # epoch 2 qhat
    98.25,   # epoch 3 qhat
    98.61,   # epoch 4 qhat
    98.62,   # epoch 5 qhat
    98.06,   # epoch 6 qhat
    98.50,   # epoch 7 qhat
    98.63,   # epoch 8 qhat
    98.68,   # epoch 9 qhat
    98.83,   # epoch 10 qhat
    98.64,   # epoch 11 qhat
]

# Data for iter_1 (stagnates) - using q_hat prediction sets
train_traj_1 = [1000]
sep_pct_1 = [4.083]  # from final_results.json qhat_prediction_sets
f1_1 = [98.27]  # from final_results.json qhat

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Separatrix %
ax1.plot(train_traj_15, sep_pct_15, 'b-o', linewidth=2, markersize=8, label='adapt_iter=15 (q_hat)')
ax1.axhline(y=sep_pct_1[0], color='r', linestyle='--', linewidth=2, label=f'cartpole@1000 (q_hat, stagnates)')
ax1.set_xlabel('Training Trajectories', fontsize=12)
ax1.set_ylabel('Separatrix %', fontsize=12)
ax1.set_ylim(0, 35)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Right plot: F1 Score
ax2.plot(train_traj_15, f1_15, 'g-s', linewidth=2, markersize=6, label='adapt_iter=15 F1')
ax2.axhline(y=f1_1[0], color='orange', linestyle=':', linewidth=2, label=f'cartpole@1000 F1')
ax2.set_xlabel('Training Trajectories', fontsize=12)
ax2.set_ylabel('F1 Score (%)', fontsize=12)
ax2.set_ylim(90, 100)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('separatrix_comparison.png', dpi=150)
plt.savefig('separatrix_comparison.pdf', dpi=150)
print("Saved to separatrix_comparison.png and separatrix_comparison.pdf")
