"""
Refactored flow matching evaluation using the new unified modules
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.inference_flow_matching import FlowMatchingInference
from src.evaluation.evaluator import FlowMatchingEvaluator
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.systems.pendulum_config import PendulumConfig


@hydra.main(config_path="../configs", config_name="train_flow_matching.yaml")
def main(cfg: DictConfig):
    """
    Refactored evaluation script using the new unified evaluation modules
    """
    # Configuration
    checkpoint_path = cfg.get("checkpoint_path", "path/to/checkpoint.ckpt")
    output_dir = Path("evaluation_results_refactored")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize components using new unified modules
    config = PendulumConfig()
    inferencer = FlowMatchingInference(checkpoint_path)
    evaluator = FlowMatchingEvaluator(
        model_name="Flow Matching",
        config=config,
        use_circular_metrics=False
    )
    
    # Load test data
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    print(f"Evaluating on {len(data_module.test_dataset)} test samples...")
    
    # === STANDARD EVALUATION ===
    print("\\n" + "="*50)
    print("STANDARD EVALUATION")
    print("="*50)
    
    # Run evaluation using unified evaluator
    results = evaluator.evaluate_on_dataloader(
        inferencer, 
        test_loader, 
        data_module,
        max_samples=5000  # Limit for faster demo
    )
    
    # Print results to console
    evaluator.print_results()
    
    # Save standard evaluation results
    standard_dir = output_dir / "standard_evaluation"
    evaluator.save_results(standard_dir)
    
    # Create sample flow paths
    print("\\nCreating sample flow paths...")
    evaluator.create_sample_flow_paths(
        inferencer, 
        n_samples=5,
        output_dir=standard_dir
    )
    
    # === ATTRACTOR BASIN ANALYSIS ===
    print("\\n" + "="*50)
    print("ATTRACTOR BASIN ANALYSIS")
    print("="*50)
    
    # Initialize basin analyzer
    basin_analyzer = AttractorBasinAnalyzer(config)
    
    # Run basin analysis
    basin_results = basin_analyzer.analyze_attractor_basins(
        inferencer,
        resolution=0.1,
        batch_size=1000
    )
    
    # Save basin analysis results
    basin_dir = output_dir / "basin_analysis"
    basin_analyzer.save_analysis_results(basin_dir, basin_results)
    
    # === COMPREHENSIVE REPORT ===
    print("\\n" + "="*50)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*50)
    
    # Create comprehensive evaluation report
    create_comprehensive_report(
        output_dir, 
        results, 
        basin_results,
        checkpoint_path
    )
    
    print("\\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print("\\nGenerated analysis:")
    print("1. Standard Evaluation:")
    print("   - Prediction accuracy metrics")
    print("   - Error distribution plots") 
    print("   - Phase space comparisons")
    print("   - Sample flow path visualizations")
    print("\\n2. Attractor Basin Analysis:")
    print("   - State space discretization")
    print("   - Basin boundary mapping")
    print("   - Separatrix point detection")
    print("   - Statistical analysis")
    print("\\n3. Comprehensive Report:")
    print("   - Combined analysis summary")
    print("   - Model performance insights")


def create_comprehensive_report(output_dir: Path, 
                              eval_results: dict, 
                              basin_results: dict,
                              checkpoint_path: str):
    """Create a comprehensive evaluation report combining all analyses"""
    
    report_path = output_dir / "comprehensive_report.txt"
    
    eval_metrics = eval_results['metrics']
    basin_stats = basin_results['statistics']
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE FLOW MATCHING EVALUATION REPORT\\n")
        f.write("=" * 70 + "\\n\\n")
        
        f.write(f"Model: {checkpoint_path}\\n")
        f.write(f"Evaluation Date: {Path().cwd()}\\n\\n")
        
        # Standard metrics section
        f.write("1. STANDARD EVALUATION METRICS\\n")
        f.write("-" * 35 + "\\n")
        f.write(f"Test Samples: {eval_metrics['n_samples']}\\n")
        f.write(f"Overall MSE: {eval_metrics['mse']:.6f}\\n")
        f.write(f"Overall MAE: {eval_metrics['mae']:.6f}\\n")
        f.write(f"Angle MSE: {eval_metrics['mse_angle']:.6f}\\n")
        f.write(f"Angle MAE: {eval_metrics['mae_angle']:.6f}\\n")
        f.write(f"Velocity MSE: {eval_metrics['mse_velocity']:.6f}\\n")
        f.write(f"Velocity MAE: {eval_metrics['mae_velocity']:.6f}\\n")
        f.write(f"Attractor Prediction Accuracy: {eval_metrics['attractor_prediction_accuracy']:.1f}%\\n\\n")
        
        # Basin analysis section
        f.write("2. ATTRACTOR BASIN ANALYSIS\\n")
        f.write("-" * 28 + "\\n")
        f.write(f"Grid Resolution: {basin_results['resolution']}\\n")
        f.write(f"Total Grid Points: {basin_stats['total_points']}\\n")
        f.write(f"Separatrix Points: {basin_stats['separatrix_count']} ({basin_stats['separatrix_percentage']:.1f}%)\\n\\n")
        
        f.write("Basin Distribution:\\n")
        basin_counts = basin_stats['basin_counts']
        f.write(f"  Center Attractor Basin: {basin_counts['attractor_0']} ({basin_counts['attractor_0_percent']:.1f}%)\\n")
        f.write(f"  Right Attractor Basin: {basin_counts['attractor_1']} ({basin_counts['attractor_1_percent']:.1f}%)\\n")
        f.write(f"  Left Attractor Basin: {basin_counts['attractor_2']} ({basin_counts['attractor_2_percent']:.1f}%)\\n")
        f.write(f"  Separatrix Region: {basin_counts['separatrix']} ({basin_counts['separatrix_percent']:.1f}%)\\n\\n")
        
        # Performance insights
        f.write("3. PERFORMANCE INSIGHTS\\n")
        f.write("-" * 23 + "\\n")
        
        # Accuracy assessment
        if eval_metrics['attractor_prediction_accuracy'] > 90:
            accuracy_assessment = "Excellent"
        elif eval_metrics['attractor_prediction_accuracy'] > 80:
            accuracy_assessment = "Good"
        elif eval_metrics['attractor_prediction_accuracy'] > 70:
            accuracy_assessment = "Moderate"
        else:
            accuracy_assessment = "Poor"
        
        f.write(f"Model Accuracy Assessment: {accuracy_assessment}\\n")
        f.write(f"Average Distance to Attractors: {eval_metrics['avg_distance_to_closest_attractor_pred']:.4f}\\n")
        
        # Basin structure assessment
        separatrix_ratio = basin_stats['separatrix_percentage']
        if separatrix_ratio < 5:
            basin_quality = "Well-defined basins with clear boundaries"
        elif separatrix_ratio < 15:
            basin_quality = "Mostly well-defined basins with some unclear regions"
        else:
            basin_quality = "Significant separatrix regions - basin boundaries may be unclear"
        
        f.write(f"Basin Structure Quality: {basin_quality}\\n")
        
        # Recommendations
        f.write("\\n4. RECOMMENDATIONS\\n")
        f.write("-" * 17 + "\\n")
        
        if eval_metrics['attractor_prediction_accuracy'] < 85:
            f.write("• Consider retraining with more data or different hyperparameters\\n")
        
        if eval_metrics['mae_angle'] > 0.1:
            f.write("• Angular prediction accuracy could be improved\\n")
            
        if eval_metrics['mae_velocity'] > 0.2:
            f.write("• Velocity prediction accuracy could be improved\\n")
        
        if separatrix_ratio > 10:
            f.write("• High separatrix ratio suggests model uncertainty in some regions\\n")
            f.write("• Consider analyzing separatrix regions for training data gaps\\n")
        
        f.write("\\n5. FILES GENERATED\\n")
        f.write("-" * 17 + "\\n")
        f.write("Standard Evaluation:\\n")
        f.write("  • prediction_scatter.png - Prediction vs ground truth\\n")
        f.write("  • error_distribution.png - Error histograms\\n")
        f.write("  • phase_space_comparison.png - Phase space overlay\\n")
        f.write("  • sample_flow_path_*.png - Individual trajectory examples\\n\\n")
        f.write("Basin Analysis:\\n")
        f.write("  • attractor_basins.png - Basin visualization\\n")
        f.write("  • basin_statistics.png - Statistical analysis\\n")
        f.write("  • basin_analysis_data.npz - Raw analysis data\\n")
    
    print(f"✓ Comprehensive report saved to: {report_path}")


if __name__ == "__main__":
    main()