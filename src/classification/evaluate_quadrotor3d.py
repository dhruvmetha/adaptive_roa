#!/usr/bin/env python3
"""
Quadrotor 3D Classification Evaluation Script

Generates a CSV file with:
- Initial state (x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r)
- p_success (predicted probability of success)
- ground_truth label

Usage:
    python src/classification/evaluate_quadrotor3d.py --checkpoint path/to/checkpoint.ckpt --output results.csv
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.simple_mlp import SimpleMLP
from src.data.quadrotor3d_classification_data import Quadrotor3DEvalDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quadrotor 3D classifier and generate CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="quadrotor3d_classification_results.csv", help="Output CSV path")
    parser.add_argument("--eval_file", type=str,
                        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr/eval_states.txt",
                        help="Path to eval_states.txt")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--hidden_channels", type=str, default="256,512,256",
                        help="Hidden layer sizes (comma-separated)")
    args = parser.parse_args()

    # Parse hidden channels
    hidden_channels = [int(x) for x in args.hidden_channels.split(',')]

    # Load eval dataset
    print("Loading eval data...")
    dataset = Quadrotor3DEvalDataset(args.eval_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(dataset)} samples")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Recreate model with new constructor
    model = SimpleMLP(
        input_dim=13,  # Full 13D state
        output_dim=1,
        hidden_channels=hidden_channels,
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=500,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Collect results
    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["inputs"].to(device)
            original_states = batch["original_state"].numpy()
            labels = batch["label"]

            # Get predictions
            logits = model(inputs).squeeze(-1)
            p_success = torch.sigmoid(logits).cpu().numpy()

            # Store results with original states
            for i in range(len(p_success)):
                results.append({
                    "x": original_states[i, 0],
                    "y": original_states[i, 1],
                    "z": original_states[i, 2],
                    "qw": original_states[i, 3],
                    "qx": original_states[i, 4],
                    "qy": original_states[i, 5],
                    "qz": original_states[i, 6],
                    "x_dot": original_states[i, 7],
                    "y_dot": original_states[i, 8],
                    "z_dot": original_states[i, 9],
                    "p": original_states[i, 10],
                    "q": original_states[i, 11],
                    "r": original_states[i, 12],
                    "p_success": p_success[i],
                    "ground_truth": int(labels[i].item()),
                })

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)

    for thresh in thresholds:
        pred = (df["p_success"] >= thresh).astype(int)
        gt = df["ground_truth"]

        tp = ((pred == 1) & (gt == 1)).sum()
        tn = ((pred == 0) & (gt == 0)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()

        accuracy = (tp + tn) / len(df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{thresh:<12.1f} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

    # Detailed metrics at threshold 0.5
    thresh = 0.5
    pred = (df["p_success"] >= thresh).astype(int)
    gt = df["ground_truth"]

    tp = ((pred == 1) & (gt == 1)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(df)

    p = df["p_success"]
    sep_30_70 = ((p >= 0.3) & (p <= 0.7)).sum()

    print("\n" + "=" * 60)
    print("Detailed Metrics (threshold=0.5)")
    print("=" * 60)
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Separatrix (0.3-0.7): {sep_30_70} ({100 * sep_30_70 / len(df):.2f}%)")

    print("\n" + "=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Success (ground truth): {df['ground_truth'].sum()} ({100 * df['ground_truth'].mean():.1f}%)")
    print(f"p_success range: [{df['p_success'].min():.4f}, {df['p_success'].max():.4f}]")
    print(f"p_success mean: {df['p_success'].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
