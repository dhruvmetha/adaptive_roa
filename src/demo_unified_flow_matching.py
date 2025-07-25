"""
Unified demo script for both standard and circular flow matching
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from pathlib import Path

# Import unified flow matching framework
from flow_matching.standard.inference import StandardFlowMatchingInference
from flow_matching.circular.inference import CircularFlowMatchingInference
from evaluation.evaluator import FlowMatchingEvaluator
from visualization.attractor_analysis import AttractorBasinAnalyzer
from systems.pendulum_config import PendulumConfig


def demo_flow_matching_variant(variant: str, checkpoint_path: str, output_dir: str):
    """
    Demo a specific flow matching variant
    
    Args:
        variant: 'standard' or 'circular'
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save results
    """
    print(f"\\n{'='*60}")
    print(f"DEMO: {variant.upper()} FLOW MATCHING")
    print(f"{'='*60}")
    
    # Initialize configuration
    config = PendulumConfig()
    
    # Load appropriate inferencer
    print(f"Loading {variant} flow matching model from: {checkpoint_path}")
    try:
        if variant == 'standard':
            inferencer = StandardFlowMatchingInference(checkpoint_path)
        elif variant == 'circular':
            inferencer = CircularFlowMatchingInference(checkpoint_path)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        print(f"✓ {variant.capitalize()} flow matching model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading {variant} model: {e}")
        return
    
    # Create output directory for this variant
    variant_output_dir = Path(output_dir) / f"{variant}_flow_matching"
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Demo 1: Single prediction
    print(f"\\n{'-'*40}")
    print(f"Demo 1: Single Prediction ({variant})")
    print(f"{'-'*40}")
    
    # Test prediction
    q_start = 1.5
    q_dot_start = -0.8
    print(f"Start state: angle={q_start:.3f} rad, velocity={q_dot_start:.3f} rad/s")
    
    endpoint = inferencer.predict_single(q_start, q_dot_start)
    print(f"Predicted endpoint: angle={endpoint[0]:.3f} rad, velocity={endpoint[1]:.3f} rad/s")
    
    # Check closest attractor
    attractors = config.ATTRACTORS
    closest_idx, distances = config.get_closest_attractor(endpoint.reshape(1, -1))
    closest_name = config.ATTRACTOR_NAMES[closest_idx[0]]
    print(f"Closest attractor: {closest_name} (distance: {distances[0]:.3f})")
    
    # Demo 2: Batch prediction
    print(f"\\n{'-'*40}")
    print(f"Demo 2: Batch Prediction ({variant})")
    print(f"{'-'*40}")
    
    test_states = [
        [0.5, 1.0],
        [-1.0, -0.5],
        [2.0, 0.1],
        [-2.5, 1.5],
    ]
    
    print("Test states:")
    for i, state in enumerate(test_states):
        print(f"  {i+1}: angle={state[0]:.3f}, velocity={state[1]:.3f}")
    
    batch_endpoints = inferencer.batch_predict(test_states)
    print("\\nPredicted endpoints:")
    for i, endpoint in enumerate(batch_endpoints):
        print(f"  {i+1}: angle={endpoint[0]:.3f}, velocity={endpoint[1]:.3f}")
    
    # Demo 3: Flow path visualization
    print(f"\\n{'-'*40}")
    print(f"Demo 3: Flow Path Visualization ({variant})")
    print(f"{'-'*40}")
    
    demo_start = [1.8, -1.2]
    print(f"Generating flow path from: angle={demo_start[0]:.3f}, velocity={demo_start[1]:.3f}")
    
    try:
        # Get flow path
        endpoint, path = inferencer.predict_single(demo_start[0], demo_start[1], return_path=True)
        
        # Create basic visualization
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Phase space path
        ax1.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Flow path')
        ax1.scatter(demo_start[0], demo_start[1], color='green', s=100, 
                   marker='o', label='Start', zorder=5)
        ax1.scatter(endpoint[0], endpoint[1], color='red', s=100, 
                   marker='s', label='End', zorder=5)
        
        # Add attractors
        for i, (attractor, name, color) in enumerate(zip(
            config.ATTRACTORS, config.ATTRACTOR_NAMES, config.ATTRACTOR_COLORS
        )):
            circle = plt.Circle(attractor, config.ATTRACTOR_RADIUS, 
                              color=color, alpha=0.4, label=name if i < 3 else '')
            ax1.add_patch(circle)
        
        ax1.set_xlabel('Angle (q)')
        ax1.set_ylabel('Angular Velocity (q̇)')
        ax1.set_title(f'{variant.capitalize()} Flow Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-np.pi, np.pi)
        ax1.set_ylim(-2*np.pi, 2*np.pi)
        
        # Plot 2: Time evolution
        time_steps = np.linspace(0, 1, len(path))
        ax2.plot(time_steps, path[:, 0], 'b-', label='Angle', linewidth=2)
        ax2.plot(time_steps, path[:, 1], 'r-', label='Angular Velocity', linewidth=2)
        ax2.set_xlabel('Flow Time')
        ax2.set_ylabel('State Value')
        ax2.set_title('State Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = variant_output_dir / f"{variant}_flow_path_demo.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Flow path visualization saved as '{save_path}'")
        
    except Exception as e:
        print(f"✗ Error creating flow path visualization: {e}")
    
    # Demo 4: Attractor basin analysis
    print(f"\\n{'-'*40}")
    print(f"Demo 4: Attractor Basin Analysis ({variant})")
    print(f"{'-'*40}")
    
    try:
        analyzer = AttractorBasinAnalyzer(config)
        
        # Run basin analysis with smaller grid for demo
        basin_results = analyzer.analyze_attractor_basins(
            inferencer,
            resolution=0.2,  # Coarser for demo speed
            batch_size=500
        )
        
        # Save basin analysis
        basin_dir = variant_output_dir / "basin_analysis"
        analyzer.save_analysis_results(basin_dir, basin_results)
        
        print(f"✓ Basin analysis completed and saved to {basin_dir}")
        
    except Exception as e:
        print(f"✗ Error in basin analysis: {e}")
    
    print(f"\\n{variant.capitalize()} flow matching demo completed!")
    print(f"Results saved to: {variant_output_dir}")


def main():
    """Main demo function"""
    print("=" * 60)
    print("UNIFIED FLOW MATCHING DEMO")
    print("=" * 60)
    print("This demo showcases both standard and circular flow matching variants")
    print("using the unified framework architecture.")
    
    # Configuration
    output_dir = "unified_flow_matching_results"
    
    # Model checkpoints (update these paths as needed)
    checkpoints = {
        'standard': "outputs/flow_matching/checkpoints/epoch=epoch=199-step=step=3600-val_loss=val_loss=0.000156-v1.ckpt",
        'circular': "outputs/circular_flow_matching/checkpoints/epoch=189-step=6650-val_loss=0.000308.ckpt"
    }
    
    # Demo both variants
    for variant, checkpoint_path in checkpoints.items():
        if Path(checkpoint_path).exists():
            demo_flow_matching_variant(variant, checkpoint_path, output_dir)
        else:
            print(f"\\n⚠️  Checkpoint not found for {variant} flow matching: {checkpoint_path}")
            print(f"   Skipping {variant} demo...")
    
    print(f"\\n{'='*60}")
    print("UNIFIED DEMO COMPLETE!")
    print(f"{'='*60}")
    print("The unified flow matching framework provides:")
    print("• Automatic variant detection")
    print("• Shared base functionality with specialized implementations") 
    print("• Consistent APIs across standard and circular variants")
    print("• Unified evaluation and visualization pipelines")
    print("• Reduced code duplication and improved maintainability")
    print(f"\\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()