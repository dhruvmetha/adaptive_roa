#!/usr/bin/env python3
"""
Demo: Region of Attraction (RoA) Analysis for Latent Conditional Flow Matching

This script demonstrates how to perform RoA analysis on a trained LCFM model:
1. Load trained checkpoint
2. Create inference wrapper
3. Run deterministic basin analysis
4. Run probabilistic basin analysis (with uncertainty)
5. Visualize and save results

Usage:
    # Basic deterministic analysis
    python demo_roa_analysis.py

    # With custom checkpoint
    python demo_roa_analysis.py --checkpoint outputs/pendulum_latent_conditional_fm/checkpoints/best.ckpt

    # Probabilistic analysis with uncertainty
    python demo_roa_analysis.py --probabilistic --num-samples 100
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import argparse
from pathlib import Path

from src.flow_matching.latent_conditional import LatentConditionalFlowMatcher, LatentConditionalFlowMatchingInference
from src.systems.pendulum_lcfm import PendulumSystemLCFM
from src.systems.pendulum_config import PendulumConfig
from src.model.latent_conditional_unet1d import LatentConditionalUNet1D
from src.visualization.attractor_analysis import AttractorBasinAnalyzer


def load_trained_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load trained latent conditional flow matcher from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        flow_matcher: Loaded and ready for inference
    """
    print("="*80)
    print("Loading Trained Model")
    print("="*80)

    # Create system
    print("  Creating system...")
    system = PendulumSystemLCFM()

    # Create model (architecture must match training config!)
    print("  Creating model architecture...")
    model = LatentConditionalUNet1D(
        embedded_dim=3,              # (sin θ, cos θ, θ̇_norm)
        latent_dim=2,                # Gaussian latent variable
        condition_dim=3,             # Embedded start state
        time_emb_dim=64,             # Time embedding dimension
        hidden_dims=[256, 512, 256], # UNet hidden layers
        output_dim=2,                # Velocity in tangent space
        use_input_embeddings=False
    )

    # Create flow matcher
    print("  Creating flow matcher...")
    flow_matcher = LatentConditionalFlowMatcher(
        system=system,
        model=model,
        optimizer=None,
        scheduler=None,
        config={},
        latent_dim=2
    )

    # Load checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    flow_matcher.load_state_dict(checkpoint['state_dict'])
    flow_matcher.eval()
    flow_matcher = flow_matcher.to(device)

    print("✓ Model loaded successfully!")
    print()

    return flow_matcher


def run_deterministic_roa_analysis(flow_matcher, resolution=0.1, batch_size=1000):
    """
    Run deterministic RoA analysis (samples random latent for each point)

    Args:
        flow_matcher: Trained flow matcher
        resolution: Grid resolution
        batch_size: Batch size for processing

    Returns:
        results: Analysis results dictionary
    """
    print("="*80)
    print("Deterministic RoA Analysis")
    print("="*80)

    # Create inference wrapper
    inferencer = LatentConditionalFlowMatchingInference(
        flow_matcher=flow_matcher,
        num_integration_steps=100,
        integration_method="rk4"
    )

    # Create analyzer
    config = PendulumConfig()
    analyzer = AttractorBasinAnalyzer(config)

    # Run analysis
    print(f"Running basin analysis with resolution={resolution}...")
    results = analyzer.analyze_attractor_basins(
        inferencer=inferencer,
        resolution=resolution,
        batch_size=batch_size,
        use_probabilistic=False
    )

    # Save results
    output_dir = Path("roa_analysis_deterministic")
    print(f"\nSaving results to {output_dir}...")
    analyzer.save_analysis_results(output_dir, results)

    print("\n✓ Deterministic analysis complete!")
    print(f"  Resolution: {resolution}")
    print(f"  Grid points: {results['statistics']['total_points']}")
    print(f"  Output: {output_dir}")
    print()

    return results


def run_probabilistic_roa_analysis(flow_matcher, resolution=0.1,
                                   num_samples=64, batch_size=500):
    """
    Run probabilistic RoA analysis with uncertainty estimation

    Args:
        flow_matcher: Trained flow matcher
        resolution: Grid resolution
        num_samples: Number of samples for Monte Carlo estimation
        batch_size: Batch size for processing

    Returns:
        results: Analysis results dictionary with uncertainty
    """
    print("="*80)
    print("Probabilistic RoA Analysis")
    print("="*80)

    # Create inference wrapper
    inferencer = LatentConditionalFlowMatchingInference(
        flow_matcher=flow_matcher,
        num_integration_steps=100,
        integration_method="rk4"
    )

    # Create analyzer
    config = PendulumConfig()
    analyzer = AttractorBasinAnalyzer(config)

    # Run probabilistic analysis
    print(f"Running probabilistic basin analysis...")
    print(f"  Resolution: {resolution}")
    print(f"  Samples per point: {num_samples}")
    print(f"  This may take a while...")

    results = analyzer.analyze_attractor_basins(
        inferencer=inferencer,
        resolution=resolution,
        batch_size=batch_size,
        use_probabilistic=True,
        num_samples=num_samples,
        thresholds={
            'entropy': 0.9,   # Entropy threshold for separatrix
            'pmax': 0.55,     # Max probability threshold
            'margin': 0.15    # Margin between top-2 probabilities
        }
    )

    # Save results
    output_dir = Path("roa_analysis_probabilistic")
    print(f"\nSaving results to {output_dir}...")
    analyzer.save_analysis_results(output_dir, results)

    # Print uncertainty statistics
    if 'entropy' in results:
        print("\n📊 Uncertainty Statistics:")
        print(f"  Mean entropy: {results['entropy'].mean():.4f}")
        print(f"  Max entropy: {results['entropy'].max():.4f}")
        print(f"  Mean pmax: {results['pmax'].mean():.4f}")
        print(f"  Mean margin: {results['margin'].mean():.4f}")

    if 'endpoint_std' in results:
        std_data = results['endpoint_std']
        valid_std = std_data[std_data > 0]
        if len(valid_std) > 0:
            print(f"\n📏 Endpoint Variability:")
            print(f"  Mean std magnitude: {valid_std.mean():.4f}")
            print(f"  Max std magnitude: {valid_std.max():.4f}")
            print(f"  Median std magnitude: {float(torch.tensor(valid_std).median()):.4f}")

    print("\n✓ Probabilistic analysis complete!")
    print(f"  Resolution: {resolution}")
    print(f"  Grid points: {results['statistics']['total_points']}")
    print(f"  Samples per point: {num_samples}")
    print(f"  Output: {output_dir}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="RoA Analysis for LCFM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (default: find latest)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Grid resolution for discretization (default: 0.1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        help="Run probabilistic analysis with uncertainty"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of samples for probabilistic analysis (default: 64)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both deterministic and probabilistic analyses"
    )

    args = parser.parse_args()

    # Find checkpoint if not provided
    if args.checkpoint is None:
        print("🔍 Searching for latest checkpoint...")
        checkpoint_dir = Path("outputs/pendulum_latent_conditional_fm/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                # Sort by modification time, get latest
                args.checkpoint = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
                print(f"✓ Found checkpoint: {args.checkpoint}\n")
            else:
                print("❌ No checkpoints found in outputs/pendulum_latent_conditional_fm/checkpoints/")
                print("   Please train a model first or specify --checkpoint")
                return
        else:
            print("❌ Checkpoint directory not found!")
            print("   Please train a model first or specify --checkpoint")
            return

    # Load model
    flow_matcher = load_trained_model(args.checkpoint)

    # Run analysis
    if args.both:
        # Run both analyses
        run_deterministic_roa_analysis(flow_matcher, args.resolution, args.batch_size)
        run_probabilistic_roa_analysis(flow_matcher, args.resolution,
                                      args.num_samples, args.batch_size)
    elif args.probabilistic:
        # Run only probabilistic
        run_probabilistic_roa_analysis(flow_matcher, args.resolution,
                                      args.num_samples, args.batch_size)
    else:
        # Run only deterministic
        run_deterministic_roa_analysis(flow_matcher, args.resolution, args.batch_size)

    print("="*80)
    print("🎉 RoA Analysis Complete!")
    print("="*80)
    print("\n📁 Results saved to:")
    if args.both:
        print("  - roa_analysis_deterministic/")
        print("  - roa_analysis_probabilistic/")
    elif args.probabilistic:
        print("  - roa_analysis_probabilistic/")
    else:
        print("  - roa_analysis_deterministic/")

    print("\n📊 Generated files:")
    print("  - attractor_basins.png          (basin map)")
    print("  - basin_statistics.png          (statistics plots)")
    print("  - basin_analysis_data.npz       (raw data)")
    print("  - basin_analysis_report.txt     (text summary)")
    if args.probabilistic or args.both:
        print("  - uncertainty_entropy.png       (entropy map)")
        print("  - uncertainty_pmax.png          (max probability map)")
        print("  - probability_heatmap_pmax.png  (probability heatmap)")
        print("  - endpoint_std_combined.png     (std deviation maps)")
    print()


if __name__ == "__main__":
    main()
