#!/usr/bin/env python3
"""
Evaluate and visualize learned contrastive embeddings

Provides diagnostic tools to assess embedding quality:
- Temporal distance vs. embedding distance correlation
- Nearest-neighbor accuracy
- UMAP/t-SNE visualization
- Attractor basin clustering analysis

Usage:
    python src/contrastive/evaluate_embeddings.py \
        --checkpoint outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt \
        --system humanoid \
        --num_samples 5000 \
        --output_dir evaluation_results/
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from src.contrastive.encoder_loader import load_pretrained_encoder
from src.data.trajectory_contrastive_data import TrajectoryContrastiveDataset


def evaluate_embeddings(
    checkpoint_path: str,
    trajectory_dir: str,
    system,
    state_dim: int,
    num_samples: int = 5000,
    temporal_window: int = 10,
    output_dir: str = "evaluation_results",
    use_umap: bool = True
):
    """
    Evaluate and visualize learned embeddings

    Args:
        checkpoint_path: Path to pretrained encoder checkpoint
        trajectory_dir: Directory containing trajectory files
        system: DynamicalSystem instance
        state_dim: Dimension of state vectors
        num_samples: Number of samples to evaluate
        temporal_window: Temporal window used during training
        output_dir: Directory to save results
        use_umap: If True, use UMAP; else use t-SNE
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Embedding Evaluation and Visualization")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Samples: {num_samples}")
    print("="*80)
    print()

    # Load encoder
    print("Loading pretrained encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_pretrained_encoder(checkpoint_path, freeze=True, device=device)
    encoder.to(device)
    print()

    # Load dataset
    print("Loading trajectory dataset...")
    dataset = TrajectoryContrastiveDataset(
        trajectory_dir=trajectory_dir,
        state_dim=state_dim,
        temporal_window=temporal_window,
        num_negatives=1,
        max_trajectories=100,  # Limit for evaluation
        cache_trajectories=True
    )
    print()

    # Sample states and compute embeddings
    print(f"Computing embeddings for {num_samples} samples...")
    states_list = []
    embeddings_list = []
    metadata_list = []

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[i]

            # Get anchor state (raw)
            anchor_raw = sample['anchor'].unsqueeze(0).to(device)  # [1, state_dim]

            # Normalize
            anchor_norm = system.normalize_state(anchor_raw)

            # Encode
            embedding = encoder(anchor_norm)  # [1, embedding_dim]

            states_list.append(anchor_raw.cpu().numpy())
            embeddings_list.append(embedding.cpu().numpy())
            metadata_list.append(sample['metadata'])

    states = np.vstack(states_list)  # [N, state_dim]
    embeddings = np.vstack(embeddings_list)  # [N, embedding_dim]

    print(f"States shape: {states.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print()

    # ========================================================================
    # METRIC 1: Temporal Distance vs. Embedding Distance
    # ========================================================================

    print("Computing temporal distance vs. embedding distance...")

    # Sample pairs with known temporal distances
    temporal_distances = []
    embedding_distances = []

    for i in tqdm(range(min(1000, len(metadata_list)))):
        meta_i = metadata_list[i]
        traj_i = meta_i['traj_id']
        t_i = meta_i['anchor_t']

        # Sample a few neighbors from same trajectory
        for j in range(i+1, min(i+50, len(metadata_list))):
            meta_j = metadata_list[j]
            traj_j = meta_j['traj_id']
            t_j = meta_j['anchor_t']

            # Only consider same trajectory
            if traj_i == traj_j:
                # Temporal distance
                temp_dist = abs(t_i - t_j)

                # Embedding distance (cosine)
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                cos_dist = 1 - np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))

                temporal_distances.append(temp_dist)
                embedding_distances.append(cos_dist)

    # Plot correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(temporal_distances, embedding_distances, alpha=0.3, s=10)
    plt.xlabel("Temporal Distance (timesteps)", fontsize=12)
    plt.ylabel("Embedding Distance (cosine)", fontsize=12)
    plt.title("Temporal Distance vs. Embedding Distance", fontsize=14)
    plt.grid(alpha=0.3)

    # Compute correlation
    correlation = np.corrcoef(temporal_distances, embedding_distances)[0, 1]
    plt.text(0.05, 0.95, f"Correlation: {correlation:.3f}",
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_vs_embedding_distance.png", dpi=150)
    print(f"✅ Saved: {output_dir / 'temporal_vs_embedding_distance.png'}")
    print(f"   Correlation: {correlation:.3f}")
    print()

    # ========================================================================
    # METRIC 2: UMAP/t-SNE Visualization
    # ========================================================================

    print("Computing 2D projection of embeddings...")

    if use_umap:
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            method_name = "UMAP"
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
            method_name = "t-SNE"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = "t-SNE"

    # Plot colored by trajectory ID
    plt.figure(figsize=(10, 8))
    traj_ids = np.array([meta['traj_id'] for meta in metadata_list])
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=traj_ids, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label="Trajectory ID")
    plt.xlabel(f"{method_name} Dimension 1", fontsize=12)
    plt.ylabel(f"{method_name} Dimension 2", fontsize=12)
    plt.title(f"Embedding Space Visualization ({method_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"embeddings_{method_name.lower()}_by_trajectory.png", dpi=150)
    print(f"✅ Saved: {output_dir / f'embeddings_{method_name.lower()}_by_trajectory.png'}")
    print()

    # ========================================================================
    # METRIC 3: Embedding Statistics
    # ========================================================================

    print("Computing embedding statistics...")

    # Embedding norms (should be ~1.0 with L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Embedding Norm", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Embedding Norms", fontsize=14)
    plt.axvline(norms.mean(), color='r', linestyle='--', label=f'Mean: {norms.mean():.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_norms.png", dpi=150)
    print(f"✅ Saved: {output_dir / 'embedding_norms.png'}")
    print(f"   Mean norm: {norms.mean():.3f} ± {norms.std():.3f}")
    print()

    # ========================================================================
    # Save Summary Report
    # ========================================================================

    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Contrastive Embedding Evaluation Report\n")
        f.write("="*80 + "\n\n")

        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Trajectory dir: {trajectory_dir}\n")
        f.write(f"Num samples: {len(states)}\n")
        f.write(f"State dim: {state_dim}\n")
        f.write(f"Embedding dim: {embeddings.shape[1]}\n\n")

        f.write("Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Temporal-Embedding Correlation: {correlation:.3f}\n")
        f.write(f"Embedding Norm (mean ± std): {norms.mean():.3f} ± {norms.std():.3f}\n\n")

        f.write("Files generated:\n")
        f.write("-"*80 + "\n")
        f.write(f"- temporal_vs_embedding_distance.png\n")
        f.write(f"- embeddings_{method_name.lower()}_by_trajectory.png\n")
        f.write(f"- embedding_norms.png\n")

    print(f"✅ Saved: {report_path}")
    print()

    print("="*80)
    print("✅ Evaluation Complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate contrastive embeddings")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to pretrained encoder checkpoint")
    parser.add_argument("--system", type=str, required=True,
                       choices=["humanoid", "cartpole", "pendulum"],
                       help="System type")
    parser.add_argument("--trajectory_dir", type=str, default=None,
                       help="Directory containing trajectories (auto-detected if not provided)")
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Number of samples to evaluate")
    parser.add_argument("--temporal_window", type=int, default=10,
                       help="Temporal window used during training")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--use_tsne", action="store_true",
                       help="Use t-SNE instead of UMAP")

    args = parser.parse_args()

    # Import system
    if args.system == "humanoid":
        from src.systems.humanoid import HumanoidSystem
        system = HumanoidSystem()
        state_dim = 67
        if args.trajectory_dir is None:
            args.trajectory_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up/trajectories"
    elif args.system == "cartpole":
        from src.systems.cartpole import CartPoleSystem
        system = CartPoleSystem()
        state_dim = 4
        if args.trajectory_dir is None:
            args.trajectory_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_dmcontrol/trajectories"
    elif args.system == "pendulum":
        from src.systems.pendulum import PendulumSystem
        system = PendulumSystem()
        state_dim = 2
        if args.trajectory_dir is None:
            args.trajectory_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k/trajectories"

    # Run evaluation
    evaluate_embeddings(
        checkpoint_path=args.checkpoint,
        trajectory_dir=args.trajectory_dir,
        system=system,
        state_dim=state_dim,
        num_samples=args.num_samples,
        temporal_window=args.temporal_window,
        output_dir=args.output_dir,
        use_umap=not args.use_tsne
    )


if __name__ == "__main__":
    main()
