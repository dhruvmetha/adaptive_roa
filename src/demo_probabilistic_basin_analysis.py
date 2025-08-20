"""
Probabilistic attractor basin analysis demo using LCFM.

Generates matplotlib visualizations:
- Basin map (with separatrix overlay)
- Basin statistics plots
- Uncertainty heatmaps (entropy, pmax)
- Probability heatmap (pmax-based with 0.5 default for no attractor)
- Standard deviation maps (endpoint prediction uncertainty across samples)
  * Combined visualization showing magnitude and θ/ω components
  * Individual plots for magnitude, θ std, and ω std
"""

import argparse
import os
from pathlib import Path
import torch

from src.systems.pendulum_config import PendulumConfig
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.demo_lcfm_inference import load_trained_lcfm_model


def main():
    parser = argparse.ArgumentParser(description="Probabilistic basin analysis with LCFM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained LCFM checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="prob_basin_analysis", help="Output directory for plots")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid resolution")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for grid evaluation")
    parser.add_argument("--num_samples", type=int, default=64, help="Samples per grid point for probability estimation")
    parser.add_argument("--steps", type=int, default=64, help="Integration steps for LCFM RK4")
    parser.add_argument("--entropy_thr", type=float, default=0.9, help="Entropy threshold for separatrix")
    parser.add_argument("--pmax_thr", type=float, default=0.55, help="Max-probability threshold for separatrix")
    parser.add_argument("--margin_thr", type=float, default=0.15, help="Margin threshold for separatrix")
    parser.add_argument("--device_id", type=int, default=None, help="GPU device ID (e.g., 0, 1, 2). If not specified, uses default device selection")
    args = parser.parse_args()

    # Set GPU device if specified
    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={args.device_id}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load LCFM inference
    inference = load_trained_lcfm_model(args.checkpoint)
    # If the inference supports steps adjustment, apply it
    if hasattr(inference, 'steps'):
        inference.steps = args.steps
    # ensure on device if model uses device internally
    # (LatentCircularInference uses model.eval() and works with cpu tensors; keep cpu to reduce VRAM unless needed)

    # Analyzer
    config = PendulumConfig()
    analyzer = AttractorBasinAnalyzer(config)

    thresholds = {"entropy": args.entropy_thr, "pmax": args.pmax_thr, "margin": args.margin_thr}

    # Run probabilistic analysis
    results = analyzer.analyze_attractor_basins(
        inference,
        resolution=args.resolution,
        batch_size=args.batch_size,
        use_probabilistic=True,
        num_samples=args.num_samples,
        thresholds=thresholds,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all visualizations and data
    analyzer.save_analysis_results(out_dir, results)

    print(f"\nDone. Visualizations saved under: {out_dir}")
    print(f"\nGenerated visualizations include:")
    print(f"- Basin maps and statistics")
    print(f"- Uncertainty heatmaps (entropy, max probability)")
    print(f"- Probability heatmaps")
    print(f"- Standard deviation maps showing endpoint prediction uncertainty across {args.num_samples} samples")
    print(f"  * Higher std dev indicates regions where the model's endpoint predictions vary more")
    print(f"  * Useful for identifying regions of model uncertainty beyond attractor classification")


if __name__ == "__main__":
    main()


