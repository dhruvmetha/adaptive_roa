"""
Demo script for circular flow matching model
"""
import torch
import numpy as np
from src.inference_circular_flow_matching import CircularFlowMatchingInference

def main():
    # Path to your trained circular flow matching checkpoint
    checkpoint_path = "outputs/circular_flow_matching/checkpoints/epoch=189-step=6650-val_loss=0.000308.ckpt"
    
    print("Loading trained circular flow matching model...")
    try:
        inferencer = CircularFlowMatchingInference(checkpoint_path)
        print("✓ Circular flow matching model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please update the checkpoint_path in this script or train the model first.")
        return
    
    # Example 1: Single prediction
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Prediction (Circular-Aware)")
    print("="*50)
    
    # Define a start state: (angle, angular_velocity)
    q_start = 1.5      # Initial angle (radians)
    q_dot_start = -0.8 # Initial angular velocity (rad/s)
    
    print(f"Start state: angle={q_start:.3f} rad, velocity={q_dot_start:.3f} rad/s")
    
    # Predict endpoint using circular flow matching
    endpoint = inferencer.predict_single(q_start, q_dot_start)
    print(f"Predicted endpoint: angle={endpoint[0]:.3f} rad, velocity={endpoint[1]:.3f} rad/s")
    
    # Check which attractor it's closest to (circular distance)
    attractors = np.array([[0, 0], [2.1, 0], [-2.1, 0]])  # Note: ±π are same on circle
    
    # Use circular distance for angle comparison
    def circular_distance(theta1, theta2):
        diff = theta1 - theta2
        return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    
    distances = []
    for attr in attractors:
        angle_dist = circular_distance(endpoint[0], attr[0])
        vel_dist = abs(endpoint[1] - attr[1])
        total_dist = np.sqrt(angle_dist**2 + vel_dist**2)
        distances.append(total_dist)
    
    closest_attractor = np.argmin(distances)
    attractor_names = ["Center (0,0)", "Right (π,0)", "Left (-π,0)"]
    
    print(f"Closest attractor: {attractor_names[closest_attractor]} (distance: {distances[closest_attractor]:.3f})")
    
    # Example 2: Batch prediction
    print("\n" + "="*50) 
    print("EXAMPLE 2: Batch Prediction (Circular-Aware)")
    print("="*50)
    
    # Multiple start states
    start_states = [
        [0.5, 1.0],        # Small angle, positive velocity
        [-1.0, -0.5],      # Negative angle, negative velocity
        [2.8, 0.1],        # Near π boundary
        [-2.9, 1.5],       # Near -π boundary
        [3.1, -0.2],       # Past π (should wrap)
        [-3.0, 0.8],       # Past -π (should wrap)
    ]
    
    print("Start states:")
    for i, state in enumerate(start_states):
        print(f"  {i+1}: angle={state[0]:.3f}, velocity={state[1]:.3f}")
    
    # Predict all endpoints
    endpoints = inferencer.batch_predict(start_states)
    
    print("\nPredicted endpoints:")
    for i, endpoint in enumerate(endpoints):
        print(f"  {i+1}: angle={endpoint[0]:.3f}, velocity={endpoint[1]:.3f}")
    
    # Example 3: Visualize circular flow path
    print("\n" + "="*50)
    print("EXAMPLE 3: Circular Flow Path Visualization")
    print("="*50)
    
    # Pick an interesting start state near the boundary
    demo_start = [2.8, -1.2]  # Close to π boundary
    print(f"Visualizing circular flow path from: angle={demo_start[0]:.3f}, velocity={demo_start[1]:.3f}")
    
    try:
        # This will create and show a plot
        fig = inferencer.visualize_flow_path(
            demo_start, 
            save_path="circular_flow_path.png",
            figsize=(14, 6)
        )
        print("✓ Circular flow path visualization saved as 'circular_flow_path.png'")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
    
    # Example 3.5: Multiple flow paths visualization
    print("\n" + "="*50)
    print("EXAMPLE 3.5: Multiple Flow Paths Visualization (50 Samples)")
    print("="*50)
    
    # Generate 50 random start states
    np.random.seed(42)  # For reproducibility
    n_samples = 50
    
    # Sample angles uniformly from [-π, π] and velocities from [-2, 2]
    angles = np.random.uniform(-np.pi, np.pi, n_samples)
    velocities = np.random.uniform(-2.0*np.pi, 2.0*np.pi, n_samples)
    
    random_start_states = [[angle, vel] for angle, vel in zip(angles, velocities)]
    
    print(f"Generating {n_samples} random start states...")
    print(f"  Angle range: [{angles.min():.2f}, {angles.max():.2f}] rad")
    print(f"  Velocity range: [{velocities.min():.2f}, {velocities.max():.2f}] rad/s")
    
    try:
        # Visualize all 50 flow paths
        inferencer.visualize_multiple_flow_paths(
            random_start_states,
            save_path="circular_flow_paths_50_samples.png",
            figsize=(16, 8)
        )
        print("✓ Multiple flow paths visualization saved as 'circular_flow_paths_50_samples.png'")
    except Exception as e:
        print(f"✗ Error creating multiple paths visualization: {e}")
    
    # Example 4: Boundary behavior test
    print("\n" + "="*50)
    print("EXAMPLE 4: Circular Boundary Behavior Test") 
    print("="*50)
    
    # Test states near ±π boundary
    boundary_states = [
        [3.0, 0.5],   # Just past π
        [-3.1, -0.3], # Just past -π
        [3.14, 0.8],  # Very close to π
        [-3.14, -0.6] # Very close to -π
    ]
    
    print("Testing circular boundary behavior:")
    for i, state in enumerate(boundary_states):
        endpoint = inferencer.predict_single(state[0], state[1])
        wrapped_start = np.arctan2(np.sin(state[0]), np.cos(state[0]))  # Wrap input
        
        print(f"  Start: ({state[0]:.3f}, {state[1]:.3f}) → Wrapped: ({wrapped_start:.3f}, {state[1]:.3f})")
        print(f"         → End: ({endpoint[0]:.3f}, {endpoint[1]:.3f})")
    
    # Example 5: Comparison with non-circular model
    print("\n" + "="*50)
    print("EXAMPLE 5: Circular vs Non-Circular Comparison")
    print("="*50)
    
    # Test case: states that should map to same result due to circular symmetry
    test_pairs = [
        ([3.1, 0.5], [-3.1, 0.5]),      # π vs -π region
        ([2.9, -0.3], [-2.9, -0.3]),   # Near boundaries
    ]
    
    print("Testing circular symmetry (similar states should give similar results):")
    for i, (state1, state2) in enumerate(test_pairs):
        end1 = inferencer.predict_single(state1[0], state1[1])
        end2 = inferencer.predict_single(state2[0], state2[1])
        
        # Compare using circular distance
        angle_diff = circular_distance(end1[0], end2[0])
        vel_diff = abs(end1[1] - end2[1])
        
        print(f"  Pair {i+1}:")
        print(f"    State A: ({state1[0]:.3f}, {state1[1]:.3f}) → ({end1[0]:.3f}, {end1[1]:.3f})")
        print(f"    State B: ({state2[0]:.3f}, {state2[1]:.3f}) → ({end2[0]:.3f}, {end2[1]:.3f})")
        print(f"    Circular angle diff: {angle_diff:.3f}, Velocity diff: {vel_diff:.3f}")

    print("\n" + "="*50)
    print("Demo completed! The circular flow matching model:")
    print("• Properly handles circular angle boundaries")
    print("• Uses geodesic interpolation on S¹ × ℝ")
    print("• Should give more accurate predictions near ±π boundaries")
    print("• Respects the circular topology of pendulum angles")
    print("="*50)

if __name__ == "__main__":
    main()