#!/usr/bin/env python3
"""
Evaluation script for Humanoid Baseline Classifier.
Generates detailed metrics and plots (ROC, Confusion Matrix).
Includes Separatrix Analysis (Success > 0.6, Failure < 0.4, Separatrix in between).
"""
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score

# Add src to path
# src/classification/humanoid/mlp -> src/
sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.classification.humanoid.mlp.classifier import HumanoidBaselineClassifier
from src.data.humanoid_roa_data import HumanoidROADataModule
from src.systems.humanoid import HumanoidSystem

@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_humanoid_mlp")
def main(cfg: DictConfig):
    print("🧪 Evaluation Mode")
    
    # Check if user provided a checkpoint path in config or CLI override
    checkpoint_path = cfg.get("checkpoint_path", None)
    
    if not checkpoint_path:
        print("⚠️ No checkpoint_path provided. Please provide via checkpoint_path=/path/to/ckpt")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize System
    # We need to ensure bounds are consistent if used in training
    system_cfg = cfg.get("system", {})
    if "_target_" in system_cfg:
        system = HumanoidSystem(
            bounds_file=system_cfg.get("bounds_file"),
            use_dynamic_bounds=system_cfg.get("use_dynamic_bounds", False)
        )
    else:
        system = HumanoidSystem()

    # Load Model
    # We need to map to CPU first to avoid device mismatch during load if cuda is not available/wanted
    # but actually, standard behavior is to map_location to device.
    # We will let lightning handle it, but explicitly move inputs to model device later.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = HumanoidBaselineClassifier.load_from_checkpoint(
        checkpoint_path,
        system=system,
        map_location=device
    )
    model.to(device)
    model.eval()
    
    # Data Module
    data_module = HumanoidROADataModule(
        data_file=cfg.data.file,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    
    # Collect predictions
    all_probs = []
    all_labels = []
    
    print("Predicting on test set...")
    with torch.no_grad():
        for x, y in test_loader:
            # Move batch to device
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(y)
            
    all_probs = torch.cat(all_probs).cpu().numpy().flatten()
    all_labels = torch.cat(all_labels).cpu().numpy().flatten()
    
    total_samples = len(all_labels)
    print(f"\n📊 Evaluation on {total_samples} samples")

    # --- Separatrix Analysis ---
    # Rules:
    # > 0.6 : Success (Predicted 1)
    # < 0.4 : Failure (Predicted 0)
    # 0.4 - 0.6 : Separatrix (Uncertain)
    
    upper_thresh = 0.6
    lower_thresh = 0.4
    
    # Identify indices
    success_indices = all_probs > upper_thresh
    failure_indices = all_probs < lower_thresh
    separatrix_indices = (all_probs >= lower_thresh) & (all_probs <= upper_thresh)
    
    # Counts
    num_separatrix = np.sum(separatrix_indices)
    separatrix_pct = (num_separatrix / total_samples) * 100
    
    print(f"\n🔮 Separatrix Analysis:")
    print(f"   Separatrix Thresholds: < {lower_thresh} (Fail) | {lower_thresh}-{upper_thresh} (Uncertain) | > {upper_thresh} (Success)")
    print(f"   Total Points: {total_samples}")
    print(f"   Separatrix Points: {num_separatrix} ({separatrix_pct:.2f}%)")
    
    # Filter confident points for metrics
    confident_indices = ~separatrix_indices
    confident_probs = all_probs[confident_indices]
    confident_labels = all_labels[confident_indices]
    
    if len(confident_probs) == 0:
        print("⚠️ No confident predictions found!")
        return

    # Convert confident probs to binary predictions (using 0.5 cutoff effectively, since we removed middle)
    # Actually, since > 0.6 is 1 and < 0.4 is 0, essentially anything > 0.5 in this subset is 1.
    confident_preds = (confident_probs > 0.5).astype(int)
    
    # Calculate Metrics on confident subset
    acc = accuracy_score(confident_labels, confident_preds)
    prec = precision_score(confident_labels, confident_preds)
    rec = recall_score(confident_labels, confident_preds)
    
    # Specificity: TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(confident_labels, confident_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"\n📈 Metrics on Confident Subset (excluding Separatrix):")
    print(f"   Confident Samples: {len(confident_labels)} ({100 - separatrix_pct:.2f}%)")
    print(f"   Accuracy:    {acc:.4f}")
    print(f"   Precision:   {prec:.4f}")
    print(f"   Recall:      {rec:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # --- Plots ---
    
    # 1. Standard ROC (using all data)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Baseline MLP')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_curve_baseline.png")
    print("\nSaved roc_curve_baseline.png")
    
    # 2. Confusion Matrix (Confident Data Only)
    cm = confusion_matrix(confident_labels, confident_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failure", "Success"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Confident Only)\nExcluded {separatrix_pct:.1f}% Separatrix")
    plt.savefig("confusion_matrix_confident.png")
    print("Saved confusion_matrix_confident.png")
    
    # 3. Histogram of probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(all_probs, bins=50, range=(0,1), alpha=0.7, color='purple')
    plt.axvline(lower_thresh, color='red', linestyle='--', label='Fail Threshold (0.4)')
    plt.axvline(upper_thresh, color='green', linestyle='--', label='Success Threshold (0.6)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.savefig("probability_distribution.png")
    print("Saved probability_distribution.png")

if __name__ == "__main__":
    main()
