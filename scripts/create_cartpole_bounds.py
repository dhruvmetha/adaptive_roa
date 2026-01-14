#!/usr/bin/env python3
"""
Create CartPole bounds pickle file from dataset_description.json.

This script reads the achieved bounds from the dataset description and creates
a pickle file compatible with CartPoleSystem.

The CartPoleSystem expects a nested dictionary format:
    bounds_data['bounds']['x']['min'], bounds_data['bounds']['x']['max'], etc.
"""
import pickle
import json
from pathlib import Path

# Read dataset description
dataset_desc_path = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet/dataset_description.json"
with open(dataset_desc_path, 'r') as f:
    desc = json.load(f)

# Extract achieved bounds (actual data range)
achieved = desc['achieved_bounds']

# Create bounds dictionary matching CartPoleSystem expected format
# Format: {'bounds': {'x': {'min': ..., 'max': ...}, ...}}
bounds_data = {
    'bounds': {
        'x': {
            'min': achieved['x']['min'],
            'max': achieved['x']['max'],
        },
        'theta': {
            'min': achieved['theta']['min'],
            'max': achieved['theta']['max'],
        },
        'x_dot': {
            'min': achieved['x_dot']['min'],
            'max': achieved['x_dot']['max'],
        },
        'theta_dot': {
            'min': achieved['theta_dot']['min'],
            'max': achieved['theta_dot']['max'],
        },
    },
    # Also include termination thresholds for reference
    'termination': desc['generation_parameters']['termination_thresholds'],
    # Include dataset statistics
    'statistics': desc['dataset_statistics'],
}

print("CartPole Bounds (from dataset_description.json):")
print(f"  [0] x:         [{bounds_data['bounds']['x']['min']:.4f}, {bounds_data['bounds']['x']['max']:.4f}] m")
print(f"  [1] theta:     [{bounds_data['bounds']['theta']['min']:.4f}, {bounds_data['bounds']['theta']['max']:.4f}] rad")
print(f"  [2] x_dot:     [{bounds_data['bounds']['x_dot']['min']:.4f}, {bounds_data['bounds']['x_dot']['max']:.4f}] m/s")
print(f"  [3] theta_dot: [{bounds_data['bounds']['theta_dot']['min']:.4f}, {bounds_data['bounds']['theta_dot']['max']:.4f}] rad/s")

# Save to pickle
output_path = Path("/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(bounds_data, f)

print(f"\nSaved bounds to: {output_path}")
