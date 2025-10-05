#!/usr/bin/env python3
"""
Generate and cache CartPole LCFM endpoints for later analysis

This script generates endpoints for all CartPole start states using the LCFM model
and saves them to a cache file for fast repeated analysis.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time

from src.flow_matching.cartpole_latent_conditional.inference import CartPoleLatentConditionalFlowMatchingInference


def load_ground_truth_data(file_path):
    """Load ground truth CartPole ROA labels from file"""
    print(f"Loading ground truth data from: {file_path}")
    
    # Check file format
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    
    # Try to detect file format
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    if ',' in first_line and '.txt' in first_line:
        # Format: trajectory_filename,label
        print(f"Detected format: trajectory_filename,label")
        
        trajectories_dir = file_path.parent / "trajectories"
        if not trajectories_dir.exists():
            raise FileNotFoundError(f"Trajectories directory not found: {trajectories_dir}")
        
        # Load trajectory filenames and labels
        trajectory_files = []
        labels = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        filename = parts[0]
                        label = int(parts[1])
                        trajectory_files.append(filename)
                        labels.append(label)
        
        # Extract start states from trajectory files
        start_states = []
        valid_labels = []
        
        print(f"Extracting start states from {len(trajectory_files)} trajectory files...")
        
        for trajectory_file, label in tqdm(zip(trajectory_files, labels), total=len(trajectory_files)):
            trajectory_path = trajectories_dir / trajectory_file
            
            try:
                # Read first line of trajectory (start state)
                with open(trajectory_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                
                # Parse start state: x,x_dot,theta,theta_dot
                start_state = [float(x) for x in first_line.split(',')]
                if len(start_state) != 4:
                    continue
                    
                start_states.append(start_state)
                valid_labels.append(label)
                
            except Exception as e:
                print(f"Warning: Could not read {trajectory_file}: {e}")
                continue
        
        start_states = np.array(start_states)
        labels = np.array(valid_labels)
        
    else:
        # Format: x, x_dot, theta, theta_dot, label (space-separated)
        print(f"Detected format: x x_dot theta theta_dot label")
        data = np.loadtxt(file_path)
        
        start_states = data[:, :4]  # x, x_dot, theta, theta_dot columns
        labels = data[:, 4].astype(int)  # label column
    
    # Print data statistics
    n_total = len(labels)
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)
    
    print(f"Ground truth statistics:")
    print(f"  Total samples: {n_total:,}")
    print(f"  Positive (1): {n_positive:,} ({n_positive/n_total*100:.1f}%)")
    print(f"  Negative (0): {n_negative:,} ({n_negative/n_total*100:.1f}%)")
    print(f"  State range - x: [{np.min(start_states[:, 0]):.3f}, {np.max(start_states[:, 0]):.3f}]")
    print(f"  State range - ẋ: [{np.min(start_states[:, 1]):.3f}, {np.max(start_states[:, 1]):.3f}]")
    print(f"  State range - θ: [{np.min(start_states[:, 2]):.3f}, {np.max(start_states[:, 2]):.3f}]")
    print(f"  State range - θ̇: [{np.min(start_states[:, 3]):.3f}, {np.max(start_states[:, 3]):.3f}]")
    
    return {
        'start_states': start_states,
        'labels': labels
    }


def generate_endpoints_batch(inferencer, start_states, num_samples, batch_size, num_steps=50):
    """
    Generate endpoints for all CartPole start states in batches
    
    Args:
        inferencer: CartPole LCFM inference object
        start_states: Array of start states [N, 4] (x, ẋ, θ, θ̇)
        num_samples: Number of samples per start state
        batch_size: Batch size for processing
        num_steps: Integration steps for trajectory generation
        
    Returns:
        endpoints: Array of endpoints [N, num_samples, 4]
    """
    n_points = len(start_states)
    all_endpoints = np.zeros((n_points, num_samples, 4))
    
    print(f"\n🔄 Generating CartPole endpoints:")
    print(f"  Start states: {n_points:,}")
    print(f"  Samples per state: {num_samples}")
    print(f"  Total predictions: {n_points * num_samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Integration steps: {num_steps}")
    print(f"  dt: {1.0/num_steps:.4f}")
    
    # Process in batches
    for i in tqdm(range(0, n_points, batch_size), desc="Generating endpoints"):
        end_idx = min(i + batch_size, n_points)
        batch_states = start_states[i:end_idx]
        batch_size_actual = len(batch_states)
        
        # Generate multiple samples for each start state
        batch_endpoints = []
        
        # Convert batch to tensor
        batch_tensor = torch.tensor(batch_states, dtype=torch.float32, device=inferencer.device)
        
        # Generate multiple endpoint samples for all states in batch
        batch_endpoints_tensor = inferencer.predict_endpoint(
            batch_tensor,  # [batch_size_actual, 4]
            num_samples=num_samples,
            num_steps=num_steps
        )
        
        # Convert to numpy if needed
        if hasattr(batch_endpoints_tensor, 'cpu'):
            batch_endpoints_tensor = batch_endpoints_tensor.cpu().numpy()
        
        # Reshape from [batch_size_actual*num_samples, 4] to [batch_size_actual, num_samples, 4]
        batch_endpoints = batch_endpoints_tensor.reshape(batch_size_actual, num_samples, 4)
        
        # Store endpoints
        all_endpoints[i:end_idx] = batch_endpoints
    
    return all_endpoints


def save_endpoints(endpoints_data, output_file):
    """Save endpoints data to NPZ file"""
    print(f"\n💾 Saving endpoints to: {output_file}")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    np.savez_compressed(
        output_file,
        **endpoints_data
    )
    
    # Print file info
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Endpoints saved successfully!")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Start states: {endpoints_data['start_states'].shape[0]:,}")
    print(f"   Samples per state: {endpoints_data['num_samples']}")
    print(f"   Total endpoints: {endpoints_data['endpoints'].size // 4:,}")


def main():
    parser = argparse.ArgumentParser(description="Generate CartPole LCFM endpoints and cache them for analysis")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to CartPole LCFM checkpoint file (.ckpt)")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth CartPole ROA labels file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path for cached endpoints (.npz)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per start state (default: 100)")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Batch size for processing (default: 50)")
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Number of integration steps for endpoint generation (default: 100)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU ID to use")
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎯 Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.device_count()} devices")
        print(f"📍 Current device: {torch.cuda.current_device()}")
    else:
        print("💻 Running on CPU")
    
    # Load model
    print(f"\n🤖 Loading CartPole LCFM model from: {args.model_path}")
    try:
        inferencer = CartPoleLatentConditionalFlowMatchingInference(args.model_path)
        print("✅ Model loaded successfully!")
        
        # Set random seed for reproducible stochastic sampling
        random_seed = 42  # Fixed seed for reproducibility
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        print(f"🎲 Set random seed: {random_seed} for reproducible sampling")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load ground truth data
    ground_truth_data = load_ground_truth_data(args.ground_truth)
    start_states = ground_truth_data['start_states']
    labels = ground_truth_data['labels']
    
    # Limit samples if specified (for testing)
    if args.max_samples is not None:
        print(f"🔄 Limiting to {args.max_samples} samples for testing")
        indices = np.random.choice(len(start_states), 
                                 min(args.max_samples, len(start_states)), 
                                 replace=False)
        start_states = start_states[indices]
        labels = labels[indices]
    
    # Generate endpoints
    start_time = time.time()
    
    endpoints = generate_endpoints_batch(
        inferencer, 
        start_states,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps
    )
    
    generation_time = time.time() - start_time
    print(f"⏱️  Endpoint generation completed in {generation_time:.1f} seconds")
    
    # Get system info for metadata
    system = inferencer.system
    attractors = system.attractors()
    
    # Prepare data for saving
    endpoints_data = {
        'start_states': start_states,
        'labels': labels,  # Include ground truth labels
        'endpoints': endpoints,
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'model_path': args.model_path,
        'ground_truth_path': args.ground_truth,
        'attractors': np.array(attractors),
        'generation_time': generation_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': random_seed
    }
    
    # Save endpoints
    save_endpoints(endpoints_data, args.output)
    
    print(f"\n🎉 CartPole endpoint generation complete!")
    print(f"📁 Cache file saved to: {args.output}")
    print(f"⚡ Next: Use 'analyze_cartpole_endpoints.py' for fast analysis")


if __name__ == "__main__":
    main()