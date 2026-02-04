#!/usr/bin/env python3
"""
Evaluate trained classifier on eval_states.txt

Supports threshold-based separatrix classification:
- If prob < lower_thresh -> predict failure (0)
- If prob > upper_thresh -> predict success (1)
- If lower_thresh <= prob <= upper_thresh -> separatrix (excluded from F1)

Usage:
    python src/classification/evaluate_on_eval_states.py \
        --checkpoint path/to/model.ckpt \
        --eval-file path/to/eval_states.txt \
        --system quadrotor2d \
        --delta 0.15
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.model.simple_mlp import SimpleMLP
from src.data.quadrotor_classification_data import EMBEDDERS


# Input dimensions after embedding
INPUT_DIMS = {
    'quadrotor2d': 7,
    'quadrotor3d': 13,
}

# Column indices for extracting states from eval_states.txt
# Format: init_state, final_state, label
# We use the FINAL state (second set of columns) for evaluation
STATE_COLS = {
    'quadrotor2d': (6, 12),   # Columns 6-11 (6 values) - final state
    'quadrotor3d': (13, 26),  # Columns 13-25 (13 values) - final state
}


def load_eval_states(eval_file: str, system: str):
    """
    Load eval_states.txt file.

    Format: init_state, final_state, label (comma-separated)
    We extract the final state for evaluation.
    """
    data = np.loadtxt(eval_file, delimiter=',')
    start_col, end_col = STATE_COLS[system]
    states = data[:, start_col:end_col]
    labels = data[:, -1].astype(int)
    return states, labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate classifier on eval_states')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--eval-file', type=str, required=True,
                        help='Path to eval_states.txt')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--system', type=str, default='quadrotor2d',
                        choices=['quadrotor2d', 'quadrotor3d'],
                        help='System type for preprocessing')
    # Threshold arguments for separatrix detection
    parser.add_argument('--delta', type=float, default=None,
                        help='Delta for symmetric threshold (0.5 ± delta)')
    parser.add_argument('--lower-thresh', type=float, default=None,
                        help='Lower threshold for failure prediction')
    parser.add_argument('--upper-thresh', type=float, default=None,
                        help='Upper threshold for success prediction')
    args = parser.parse_args()

    # Determine thresholds
    if args.delta is not None:
        lower_thresh = 0.5 - args.delta
        upper_thresh = 0.5 + args.delta
    elif args.lower_thresh is not None and args.upper_thresh is not None:
        lower_thresh = args.lower_thresh
        upper_thresh = args.upper_thresh
    else:
        # No thresholds - standard 0.5 cutoff
        lower_thresh = 0.5
        upper_thresh = 0.5

    use_thresholds = (lower_thresh != upper_thresh)

    # Load model
    print(f"Loading model from: {args.checkpoint}")

    input_dim = INPUT_DIMS[args.system]

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
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
    model.to(args.device)

    # Load eval states
    print(f"Loading eval states from: {args.eval_file}")
    states, labels = load_eval_states(args.eval_file, args.system)
    print(f"  Total samples: {len(states):,}")
    print(f"  Labels: {np.sum(labels == 1):,} success, {np.sum(labels == 0):,} failure")

    # Embed states using same preprocessing as training
    print("\nEmbedding states...")
    embedder = EMBEDDERS[args.system]
    embedded_states = embedder.embed_batch(states)
    print(f"  Embedded shape: {embedded_states.shape}")

    # Run inference in batches
    print("\nRunning inference...")
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(embedded_states), args.batch_size):
            batch_states = torch.tensor(
                embedded_states[i:i+args.batch_size],
                dtype=torch.float32,
                device=args.device
            )

            logits = model(batch_states)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs.cpu().numpy())

            if (i + args.batch_size) % 100000 < args.batch_size:
                print(f"  Processed {min(i + args.batch_size, len(states)):,}/{len(states):,}")

    probabilities = np.concatenate(all_probs)

    # Classify with thresholds
    pred_failure = probabilities < lower_thresh
    pred_success = probabilities > upper_thresh
    pred_sep = ~pred_failure & ~pred_success

    predictions = np.where(pred_success, 1, np.where(pred_failure, 0, -1))

    # Compute metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if use_thresholds:
        print(f"\nThreshold Configuration:")
        print(f"  Lower threshold: {lower_thresh:.4f}")
        print(f"  Upper threshold: {upper_thresh:.4f}")
        if args.delta is not None:
            print(f"  Delta: {args.delta:.4f}")

    # Separatrix statistics
    n_total = len(labels)
    n_sep = np.sum(pred_sep)
    sep_pct = 100.0 * n_sep / n_total

    print(f"\nSeparatrix Statistics:")
    print(f"  Sep points: {n_sep:,}/{n_total:,} ({sep_pct:.2f}%)")

    classified_mask = ~pred_sep
    n_classified = np.sum(classified_mask)
    print(f"  Classified points: {n_classified:,}/{n_total:,} ({100.0 * n_classified / n_total:.2f}%)")

    if n_classified > 0:
        classified_preds = predictions[classified_mask]
        classified_labels = labels[classified_mask]

        accuracy = np.mean(classified_preds == classified_labels)
        print(f"\nAccuracy (excl. sep): {accuracy:.4f} ({np.sum(classified_preds == classified_labels):,}/{n_classified:,})")

        # Per-class accuracy
        success_mask = classified_labels == 1
        failure_mask = classified_labels == 0

        print(f"\nPer-class accuracy (excl. sep):")
        if np.sum(success_mask) > 0:
            success_acc = np.mean(classified_preds[success_mask] == 1)
            print(f"  Success (label=1): {success_acc:.4f} ({np.sum(classified_preds[success_mask] == 1):,}/{np.sum(success_mask):,})")
        else:
            print(f"  Success (label=1): N/A (no samples)")

        if np.sum(failure_mask) > 0:
            failure_acc = np.mean(classified_preds[failure_mask] == 0)
            print(f"  Failure (label=0): {failure_acc:.4f} ({np.sum(classified_preds[failure_mask] == 0):,}/{np.sum(failure_mask):,})")
        else:
            print(f"  Failure (label=0): N/A (no samples)")

        # Confusion matrix
        tp = np.sum((classified_preds == 1) & (classified_labels == 1))
        tn = np.sum((classified_preds == 0) & (classified_labels == 0))
        fp = np.sum((classified_preds == 1) & (classified_labels == 0))
        fn = np.sum((classified_preds == 0) & (classified_labels == 1))

        print(f"\nConfusion Matrix (excl. sep):")
        print(f"  TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nPrecision (excl. sep): {precision:.4f}")
        print(f"Recall (excl. sep): {recall:.4f}")
        print(f"F1 Score (excl. sep): {f1:.4f}")

        misclass_rate = 1.0 - accuracy
        print(f"\nMisclassification rate (excl. sep): {misclass_rate:.4f}")
    else:
        print("\nNo classified points (all in separatrix region)!")

    # Separatrix breakdown by true label
    if n_sep > 0:
        sep_labels = labels[pred_sep]
        n_sep_success = np.sum(sep_labels == 1)
        n_sep_failure = np.sum(sep_labels == 0)
        print(f"\nSeparatrix breakdown by true label:")
        print(f"  True success in sep: {n_sep_success:,} ({100.0 * n_sep_success / n_sep:.2f}%)")
        print(f"  True failure in sep: {n_sep_failure:,} ({100.0 * n_sep_failure / n_sep:.2f}%)")

    print("=" * 70)


if __name__ == '__main__':
    main()
