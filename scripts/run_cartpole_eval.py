"""
CartPole evaluation script for two models:
1. LR=0.001, 500 epochs (patience=500)
2. LR=0.001, 2000 epochs (patience=2000)
"""
import numpy as np
import torch
from tqdm import tqdm
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher import CartPoleLatentConditionalFlowMatcher

# Data paths
dataset_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet"
eval_states_file = f"{dataset_dir}/eval_states.txt"

# Model paths - direct checkpoint files
models = {
    "LR0.001_500ep": "/common/users/rm1838/adaptive_cartpole/outputs/adaptive_cartpole_pybullet/cp_lr001_p500_2026-01-10_19-10-12/epoch_000/checkpoints/best-271-0.0013.ckpt",
    "LR0.001_2000ep": "/common/users/rm1838/adaptive_cartpole/outputs/adaptive_cartpole_pybullet/cp_lr001_e2000_2026-01-11_00-31-52/epoch_000/checkpoints/best-1715-0.0011.ckpt",
}

# Evaluation parameters
samples = 100
batch_size = 2048
device = "cuda:0"

def evaluate_model(model_name, ckpt_path):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Load system
    system = CartPoleSystem(dataset_dir=dataset_dir)

    # Load model
    flow_matcher = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(ckpt_path, device=device)

    # Load data (eval_states.txt format: x_s, θ_s, ẋ_s, θ̇_s, x_e, θ_e, ẋ_e, θ̇_e, label)
    eval_data = np.loadtxt(eval_states_file, delimiter=",")
    # Extract start states (first 4 columns) and labels (last column)
    inp = torch.from_numpy(eval_data[:, :4]).float().to(device)
    labels = torch.from_numpy(eval_data[:, -1]).long().to(device)

    print(f"Total samples: {len(eval_data)}")
    print(f"Success rate in data: {np.mean(eval_data[:, -1] == 1)*100:.2f}%")

    # Run inference
    is_success = np.zeros((len(eval_data), samples))

    for batch_start in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(eval_data))
        batch_inp = inp[batch_start:batch_end, :]

        for sample_idx in range(samples):
            model_input = batch_inp.clone()
            pred = flow_matcher.predict_endpoint(model_input)
            is_success[batch_start:batch_end, sample_idx] = system.classify_attractor(pred, 0.2).cpu().numpy()

    # Compute predictions
    batch_labels = labels.cpu().numpy()
    pred_labels = np.ones_like(batch_labels) * -1  # Default to separatrix

    failure = (is_success == -1).sum(axis=1) / samples > 0.6
    success = (is_success == 1).sum(axis=1) / samples > 0.6

    pred_labels[failure] = 0
    pred_labels[success] = 1

    # Compute metrics
    tp = np.sum((batch_labels == 1) & (pred_labels == 1))
    tn = np.sum((batch_labels == 0) & (pred_labels == 0))
    fp = np.sum((batch_labels == 0) & (pred_labels == 1))
    fn = np.sum((batch_labels == 1) & (pred_labels == 0))
    sep_count = np.sum(pred_labels == -1)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    sep_perc = sep_count / len(eval_data)

    print(f"\n--- Results for {model_name} ---")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1: {f1*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    print(f"Separatrix: {sep_perc*100:.2f}%")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "separatrix": sep_perc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

if __name__ == "__main__":
    results = []
    for name, path in models.items():
        result = evaluate_model(name, path)
        results.append(result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Sep%':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['model']:<20} {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['f1']*100:>9.2f}% {r['separatrix']*100:>9.2f}%")
