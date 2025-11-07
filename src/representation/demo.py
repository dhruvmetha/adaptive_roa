"""
Demo script showing how to use the Trajectory MAE.

This script demonstrates:
1. Training a Trajectory MAE model
2. Loading a trained model
3. Extracting trajectory representations
4. Using representations for downstream tasks
"""

import numpy as np
import torch
from pathlib import Path

from inference import TrajectoryMAEInference


def demo_extract_representations():
    """Demo: Extract representations from trained model."""

    print("=" * 80)
    print("Demo: Extracting Trajectory Representations")
    print("=" * 80)

    # Path to trained checkpoint (update this to your actual checkpoint)
    checkpoint_path = "outputs/trajectory_mae_cartpole/version_0/checkpoints/last.ckpt"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train a model first using:")
        print("  python src/representation/train.py")
        return

    # Initialize inference module
    inferencer = TrajectoryMAEInference(checkpoint_path, device="cuda")

    # Print model info
    print("\nModel Info:")
    info = inferencer.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Load a sample trajectory
    data_dir = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_dmcontrol")
    traj_dir = data_dir / "trajectories"
    sample_file = list(traj_dir.glob("sequence_*.txt"))[0]

    print(f"\nLoading sample trajectory: {sample_file.name}")
    traj = np.loadtxt(sample_file, delimiter=',', dtype=np.float32)
    print(f"  Shape: {traj.shape}")

    # Define normalization bounds
    state_bounds = {
        'min': np.array([-2.4, -130.0, -10.0, -10.0]),
        'max': np.array([2.4, 130.0, 10.0, 10.0])
    }

    # Extract representation
    print("\nExtracting representation...")
    representation = inferencer.extract_representation(
        states=traj,
        aggregate='mean',
        normalize=False,
        state_bounds=state_bounds
    )

    print(f"Representation shape: {representation.shape}")
    print(f"Representation norm: {np.linalg.norm(representation):.4f}")
    print(f"Representation stats: mean={representation.mean():.4f}, std={representation.std():.4f}")

    # Extract representations for multiple trajectories
    print("\n" + "-" * 80)
    print("Extracting representations for 10 trajectories...")

    trajectory_files = sorted(traj_dir.glob("sequence_*.txt"))[:10]
    batch_representations = inferencer.extract_representations_from_files(
        trajectory_files,
        aggregate='mean',
        state_bounds=state_bounds,
        batch_size=4,
        show_progress=True
    )

    print(f"\nBatch representations shape: {batch_representations.shape}")
    print(f"Mean representation norm: {np.linalg.norm(batch_representations, axis=1).mean():.4f}")

    # Visualize reconstruction
    print("\n" + "-" * 80)
    print("Visualizing reconstruction...")

    original, masked_idx, reconstructed = inferencer.reconstruct_trajectory(
        states=traj,
        mask_ratio=0.75,
        normalize=False,
        state_bounds=state_bounds
    )

    print(f"Original shape: {original.shape}")
    print(f"Number of masked positions: {len(masked_idx)}")
    print(f"Reconstructed manifold features shape: {reconstructed.shape}")
    print(f"  Note: Reconstructed has 5 dims: (x, sin(theta), cos(theta), x_dot, theta_dot)")

    # Show sample of masked positions
    print(f"\nSample masked positions: {masked_idx[:5]}")
    print(f"Reconstructed values (first 3 masked positions):")
    for i in range(min(3, len(masked_idx))):
        idx = masked_idx[i]
        print(f"  Position {idx}: {reconstructed[i]}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demo_downstream_task():
    """Demo: Use representations for a downstream classification task."""

    print("\n" + "=" * 80)
    print("Demo: Using Representations for Classification")
    print("=" * 80)

    checkpoint_path = "outputs/trajectory_mae_cartpole/version_0/checkpoints/last.ckpt"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train a model first.")
        return

    # Initialize inference
    inferencer = TrajectoryMAEInference(checkpoint_path, device="cuda")

    # Load trajectories with labels
    data_dir = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_dmcontrol")
    traj_dir = data_dir / "trajectories"
    labels_file = data_dir / "roa_labels.txt"

    # Load labels
    labels_data = np.loadtxt(labels_file, delimiter=',')
    success_labels = labels_data[:, -1].astype(int)  # Last column is success flag

    print(f"\nTotal trajectories: {len(success_labels)}")
    print(f"Success: {(success_labels == 1).sum()}, Failure: {(success_labels == 0).sum()}")

    # Extract representations for first 100 trajectories
    num_samples = 100
    trajectory_files = sorted(traj_dir.glob("sequence_*.txt"))[:num_samples]
    labels = success_labels[:num_samples]

    state_bounds = {
        'min': np.array([-2.4, -130.0, -10.0, -10.0]),
        'max': np.array([2.4, 130.0, 10.0, 10.0])
    }

    print(f"\nExtracting representations for {num_samples} trajectories...")
    representations = inferencer.extract_representations_from_files(
        trajectory_files,
        aggregate='mean',
        state_bounds=state_bounds,
        batch_size=16,
        show_progress=True
    )

    print(f"Representations shape: {representations.shape}")

    # Train a simple classifier
    print("\nTraining a simple linear classifier on representations...")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nClassification Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print("\n" + "=" * 80)
    print("Downstream task demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run demos
    demo_extract_representations()

    # Uncomment to run downstream task demo
    # demo_downstream_task()
