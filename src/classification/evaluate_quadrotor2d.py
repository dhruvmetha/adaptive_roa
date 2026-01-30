"""
Evaluation script for Quadrotor 2D binary classification.

Loads a trained model and evaluates on eval_states.txt, outputting:
- CSV with initial state, predicted p_success, and ground truth label
- Metrics: Accuracy, Precision, Recall, Specificity, F1, Separatrix %

Usage:
    python src/classification/evaluate_quadrotor2d.py \
        --checkpoint outputs/quadrotor2d_classification/.../checkpoints/last.ckpt \
        --output results_q2d.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.model.simple_mlp import SimpleMLP
from src.data.quadrotor2d_classification_data import Quadrotor2DEvalDataset


def compute_separatrix_percentage(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute the percentage of predictions that fall on the separatrix.
    
    Separatrix = predictions where p_success is between 0.4 and 0.6 (uncertain region).
    """
    uncertain_mask = (y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)
    return 100.0 * np.sum(uncertain_mask) / len(y_pred_proba)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quadrotor 2D classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_lqr/eval_states.txt",
        help="Path to eval_states.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_quadrotor2d.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for p_success",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")

    # Model configuration (must match training)
    input_dim = 7  # 6D state with sin/cos embedding for theta
    output_dim = 1
    hidden_channels = [128, 256, 128]

    model = SimpleMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_channels=hidden_channels,
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=500,
    )

    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Load evaluation dataset
    print(f"Loading evaluation data from: {args.eval_file}")
    eval_dataset = Quadrotor2DEvalDataset(args.eval_file)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Run inference
    print("Running inference...")
    all_probs = []
    all_labels = []
    all_states = []

    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["inputs"].to(device)
            labels = batch["label"]
            original_states = batch["original_state"]

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze(-1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_states.extend(original_states.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_states = np.array(all_states)

    # Compute predictions
    all_preds = (all_probs >= args.threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Compute specificity (true negative rate)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Compute separatrix percentage
    separatrix_pct = compute_separatrix_percentage(all_labels, all_probs)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(all_labels)}")
    print(f"  Success (1): {int(np.sum(all_labels))}")
    print(f"  Failure (0): {int(len(all_labels) - np.sum(all_labels))}")
    print(f"\nThreshold: {args.threshold}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    print(f"\nMetrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  Separatrix:  {separatrix_pct:.2f}%")
    print("=" * 60)

    # Create output DataFrame with original (denormalized) states
    # State order: x, z, theta, x_dot, z_dot, theta_dot
    df = pd.DataFrame({
        "x": all_states[:, 0],
        "z": all_states[:, 1],
        "theta": all_states[:, 2],
        "x_dot": all_states[:, 3],
        "z_dot": all_states[:, 4],
        "theta_dot": all_states[:, 5],
        "p_success": all_probs,
        "predicted_label": all_preds,
        "ground_truth": all_labels.astype(int),
    })

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()


