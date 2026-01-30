#!/usr/bin/env python3
"""
CartPole Classification Evaluation Script

Generates a CSV file with:
- Initial state (x, theta, x_dot, theta_dot)
- p_success (predicted probability of success)
- ground_truth label

Usage:
    python src/classification/evaluate_cartpole.py --checkpoint path/to/checkpoint.ckpt --output results.csv
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
from torch.utils.data import Dataset, DataLoader

from src.model.simple_mlp import SimpleMLP


class CartPoleEvalDatasetWithOriginal(Dataset):
    """Dataset that returns both embedded states and original states for CSV output."""
    
    def __init__(self, eval_file: str):
        self.data = []  # Original states
        self.labels = []
        
        # CartPole state bounds
        self.x_limit = 6.0
        self.x_dot_limit = 5.0
        self.theta_dot_limit = 5.0
        
        # Load data
        with open(eval_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    state = np.array([float(parts[i]) for i in range(4)], dtype=np.float32)
                    label = int(float(parts[8]))
                    self.data.append(state)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.data)} eval samples")
        print(f"  Success (1): {sum(self.labels)}")
        print(f"  Failure (0): {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """Embed state with sin/cos for theta."""
        x, theta, x_dot, theta_dot = state
        x_norm = x / self.x_limit
        x_dot_norm = x_dot / self.x_dot_limit
        theta_dot_norm = theta_dot / self.theta_dot_limit
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return np.array([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dtype=np.float32)
    
    def __getitem__(self, idx):
        state = self.data[idx]
        label = self.labels[idx]
        embedded = self._embed_state(state)
        
        return {
            "inputs": torch.from_numpy(embedded).float(),
            "original_state": torch.from_numpy(state).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CartPole classifier and generate CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="classification_results.csv", help="Output CSV path")
    parser.add_argument("--eval_file", type=str, 
                        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet/eval_states.txt",
                        help="Path to eval_states.txt")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()
    
    # Load eval dataset
    print("Loading eval data...")
    dataset = CartPoleEvalDatasetWithOriginal(args.eval_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # Recreate model
    from functools import partial
    
    class ModelConfig:
        def __init__(self):
            self.input_dim = 5  # Manifold embedding: (x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm)
            self.output_dim = 1
            self.hidden_channels = [128, 256, 128]
    
    model_config = ModelConfig()
    optimizer_partial = partial(torch.optim.AdamW, lr=1e-3)
    scheduler_partial = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=500)
    
    model = SimpleMLP(
        model=model_config,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
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
                    "theta": original_states[i, 1],
                    "x_dot": original_states[i, 2],
                    "theta_dot": original_states[i, 3],
                    "p_success": p_success[i],
                    "ground_truth": int(labels[i].item()),
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    
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
    
    print("\n" + "="*60)
    print("Detailed Metrics (threshold=0.5)")
    print("="*60)
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Separatrix (0.3-0.7): {sep_30_70} ({100*sep_30_70/len(df):.2f}%)")
    
    print("\n" + "="*60)
    print(f"Total samples: {len(df)}")
    print(f"Success (ground truth): {df['ground_truth'].sum()} ({100*df['ground_truth'].mean():.1f}%)")
    print(f"p_success range: [{df['p_success'].min():.4f}, {df['p_success'].max():.4f}]")
    print(f"p_success mean: {df['p_success'].mean():.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
