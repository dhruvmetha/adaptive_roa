"""
Command line interface for EndpointCFM
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from .orchestration import EndpointCFM


def train_cli():
    """CLI for training EndpointCFM models"""
    parser = argparse.ArgumentParser(description="Train EndpointCFM model")
    parser.add_argument("--trajectory-files", nargs="+", required=True,
                        help="List of trajectory files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for model and πta")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", default="auto",
                        help="Device for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    
    args = parser.parse_args()
    
    # Initialize and train
    cfm = EndpointCFM()
    
    try:
        checkpoint = cfm.train(
            trajectory_files=args.trajectory_files,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            num_workers=args.num_workers
        )
        print(f"✅ Training completed successfully!")
        print(f"Best checkpoint: {checkpoint}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def infer_cli():
    """CLI for inference with EndpointCFM models"""
    parser = argparse.ArgumentParser(description="Run inference with EndpointCFM")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--start-states", required=True,
                        help="Path to numpy file with start states [N, 2]")
    parser.add_argument("--output", required=True,
                        help="Output path for final states")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples per start state")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="ODE integration steps")
    parser.add_argument("--method", default="rk4",
                        choices=["euler", "rk4"],
                        help="Integration method")
    
    args = parser.parse_args()
    
    # Load start states
    try:
        start_states = np.load(args.start_states)
        print(f"📊 Loaded {len(start_states)} start states from {args.start_states}")
    except Exception as e:
        print(f"❌ Error loading start states: {e}")
        sys.exit(1)
    
    # Initialize and load model
    cfm = EndpointCFM()
    
    try:
        cfm.load_model(args.checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    try:
        final_states = cfm.get_final_states(
            start_states=start_states,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            method=args.method
        )
        
        # Save results
        np.save(args.output, final_states)
        print(f"✅ Inference completed!")
        print(f"Final states shape: {final_states.shape}")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        sys.exit(1)