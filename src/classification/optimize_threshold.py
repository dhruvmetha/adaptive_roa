#!/usr/bin/env python3
"""
Optimize delta threshold for classifier with uncertainty region.

Threshold scheme:
- If prob < (0.5 - delta) -> predict failure (0)
- If prob > (0.5 + delta) -> predict success (1)
- If (0.5 - delta) <= prob <= (0.5 + delta) -> unknown

Optimization objective (minimize):
    loss = 0.9 * misclassification_rate + 0.1 * unknown_rate
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.model.simple_mlp import SimpleMLP


# ============================================================================
# Embedders for different systems
# ============================================================================

class Quadrotor2DEmbedder:
    """Embed Quadrotor 2D states with normalization and S^1 manifold embedding."""

    X_LIMIT = 0.96
    Z_MIN = 0.1
    Z_MAX = 1.5
    X_DOT_LIMIT = 1.2
    Z_DOT_LIMIT = 1.5
    THETA_DOT_LIMIT = 10.0

    @classmethod
    def embed_state(cls, state: np.ndarray) -> np.ndarray:
        x, z, theta, x_dot, z_dot, theta_dot = state
        x_norm = x / cls.X_LIMIT
        z_center = (cls.Z_MAX + cls.Z_MIN) / 2
        z_range = (cls.Z_MAX - cls.Z_MIN) / 2
        z_norm = (z - z_center) / z_range
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_dot_norm = x_dot / cls.X_DOT_LIMIT
        z_dot_norm = z_dot / cls.Z_DOT_LIMIT
        theta_dot_norm = theta_dot / cls.THETA_DOT_LIMIT
        return np.array([x_norm, z_norm, sin_theta, cos_theta,
                        x_dot_norm, z_dot_norm, theta_dot_norm], dtype=np.float32)

    @classmethod
    def embed_batch(cls, states: np.ndarray) -> np.ndarray:
        return np.array([cls.embed_state(s) for s in states], dtype=np.float32)


class Quadrotor3DEmbedder:
    """Embed Quadrotor 3D states (13D quaternion-based, no embedding needed)."""

    @classmethod
    def embed_batch(cls, states: np.ndarray) -> np.ndarray:
        # 13D state is used as-is (already normalized in data module)
        return states.astype(np.float32)


class CartPoleEmbedder:
    """Embed CartPole states with S^1 manifold embedding."""

    X_LIMIT = 2.4
    X_DOT_LIMIT = 10.0
    THETA_DOT_LIMIT = 10.0

    @classmethod
    def embed_state(cls, state: np.ndarray) -> np.ndarray:
        x, theta, x_dot, theta_dot = state
        x_norm = x / cls.X_LIMIT
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_dot_norm = x_dot / cls.X_DOT_LIMIT
        theta_dot_norm = theta_dot / cls.THETA_DOT_LIMIT
        return np.array([x_norm, sin_theta, cos_theta,
                        x_dot_norm, theta_dot_norm], dtype=np.float32)

    @classmethod
    def embed_batch(cls, states: np.ndarray) -> np.ndarray:
        return np.array([cls.embed_state(s) for s in states], dtype=np.float32)


EMBEDDERS = {
    'quadrotor2d': Quadrotor2DEmbedder,
    'quadrotor3d': Quadrotor3DEmbedder,
    'cartpole': CartPoleEmbedder,
}

INPUT_DIMS = {
    'quadrotor2d': 7,
    'quadrotor3d': 13,
    'cartpole': 5,
}

STATE_COLS = {
    'quadrotor2d': (6, 12),   # Columns 6-11 (6 values)
    'quadrotor3d': (13, 26),  # Columns 13-25 (13 values)
    'cartpole': (4, 8),       # Columns 4-7 (4 values)
}


# ============================================================================
# Main functions
# ============================================================================

def load_eval_states(eval_file: str, system: str):
    """Load eval_states.txt file."""
    data = np.loadtxt(eval_file, delimiter=',')
    start_col, end_col = STATE_COLS[system]
    states = data[:, start_col:end_col]
    labels = data[:, -1].astype(int)
    return states, labels


def load_model(checkpoint_path: str, system: str, device: str):
    """Load model from checkpoint."""
    input_dim = INPUT_DIMS[system]
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['hyper_parameters']

    model = SimpleMLP(
        input_dim=input_dim,
        output_dim=hparams['output_dim'],
        hidden_channels=hparams['hidden_channels'],
        lr=hparams['lr'],
        weight_decay=hparams['weight_decay'],
        max_epochs=hparams['max_epochs'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    return model


def get_predictions(model, states: np.ndarray, system: str, device: str, batch_size: int = 4096):
    """Get probability predictions for all states."""
    embedder = EMBEDDERS[system]
    embedded_states = embedder.embed_batch(states)

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(embedded_states), batch_size):
            batch = torch.tensor(
                embedded_states[i:i+batch_size],
                dtype=torch.float32,
                device=device
            )
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def evaluate_threshold(probs: np.ndarray, labels: np.ndarray, delta: float):
    """
    Evaluate classifier with given delta threshold.

    Returns:
        misclassification_rate: errors among classified samples
        unknown_rate: fraction of samples marked as unknown
        loss: 0.9 * misclassification_rate + 0.1 * unknown_rate
    """
    lower_thresh = 0.5 - delta
    upper_thresh = 0.5 + delta

    # Classify
    pred_failure = probs < lower_thresh
    pred_success = probs > upper_thresh
    pred_unknown = ~pred_failure & ~pred_success

    # Counts
    n_total = len(labels)
    n_unknown = np.sum(pred_unknown)
    n_classified = n_total - n_unknown

    # Among classified samples, count errors
    classified_mask = ~pred_unknown
    if n_classified > 0:
        classified_preds = np.where(pred_success[classified_mask], 1, 0)
        classified_labels = labels[classified_mask]
        n_errors = np.sum(classified_preds != classified_labels)
        misclassification_rate = n_errors / n_classified
    else:
        misclassification_rate = 0.0

    unknown_rate = n_unknown / n_total
    loss = 0.9 * misclassification_rate + 0.1 * unknown_rate

    return {
        'delta': delta,
        'lower_thresh': lower_thresh,
        'upper_thresh': upper_thresh,
        'misclassification_rate': misclassification_rate,
        'unknown_rate': unknown_rate,
        'loss': loss,
        'n_classified': n_classified,
        'n_unknown': n_unknown,
        'n_total': n_total,
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize delta threshold')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--eval-file', type=str, required=True,
                        help='Path to eval_states.txt')
    parser.add_argument('--system', type=str, required=True,
                        choices=['quadrotor2d', 'quadrotor3d', 'cartpole'],
                        help='System type')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of samples to use for optimization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.system, args.device)

    # Load eval states
    print(f"Loading eval states from: {args.eval_file}")
    states, labels = load_eval_states(args.eval_file, args.system)
    print(f"  Total samples: {len(states)}")

    # Sample subset for optimization
    if args.n_samples < len(states):
        indices = np.random.choice(len(states), size=args.n_samples, replace=False)
        states = states[indices]
        labels = labels[indices]
        print(f"  Using {args.n_samples} samples for optimization")

    print(f"  Labels: {np.sum(labels == 1)} success, {np.sum(labels == 0)} failure")

    # Get predictions
    print("\nGetting predictions...")
    probs = get_predictions(model, states, args.system, args.device)

    # Grid search over delta
    print("\n" + "=" * 70)
    print("GRID SEARCH OVER DELTA")
    print("=" * 70)
    print(f"{'Delta':>8} {'Lower':>8} {'Upper':>8} {'MisErr':>10} {'UnkRate':>10} {'Loss':>10}")
    print("-" * 70)

    deltas = np.arange(0.05, 0.50, 0.05)
    results = []

    for delta in deltas:
        result = evaluate_threshold(probs, labels, delta)
        results.append(result)
        print(f"{result['delta']:>8.2f} {result['lower_thresh']:>8.2f} "
              f"{result['upper_thresh']:>8.2f} {result['misclassification_rate']:>10.4f} "
              f"{result['unknown_rate']:>10.4f} {result['loss']:>10.4f}")

    # Find best delta
    best_result = min(results, key=lambda x: x['loss'])

    print("=" * 70)
    print("\nBEST THRESHOLD:")
    print(f"  Delta: {best_result['delta']:.2f}")
    print(f"  Thresholds: [{best_result['lower_thresh']:.2f}, {best_result['upper_thresh']:.2f}]")
    print(f"  Misclassification rate: {best_result['misclassification_rate']:.4f}")
    print(f"  Unknown rate: {best_result['unknown_rate']:.4f}")
    print(f"  Loss: {best_result['loss']:.4f}")
    print(f"  Classified: {best_result['n_classified']}/{best_result['n_total']}")
    print("=" * 70)

    return best_result


if __name__ == '__main__':
    main()
