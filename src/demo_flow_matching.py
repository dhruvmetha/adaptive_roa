"""
Demo script showing how to use the trained flow matching model
"""
import torch
import numpy as np
from src.inference_flow_matching import FlowMatchingInference

def main():
    # Path to your trained checkpoint (update this!)
    # Old checkpoint (from previous run with wrong naming):
    # checkpoint_path = "logs/vae_training/version_2/checkpoints/epoch=309-step=5580.ckpt"
    
    # New checkpoint location (after configuration fix):
    checkpoint_path = "outputs/flow_matching/checkpoints/epoch=epoch=199-step=step=3600-val_loss=val_loss=0.000156-v1.ckpt"
    
    print("Loading trained flow matching model...")
    try:
        inferencer = FlowMatchingInference(checkpoint_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please update the checkpoint_path in this script.")
        return
    
    # Example 1: Single prediction
    print("\\n" + "="*50)
    print("EXAMPLE 1: Single Prediction")
    print("="*50)
    
    # Define a start state: (angle, angular_velocity)
    q_start = 1.5      # Initial angle (radians)
    q_dot_start = -0.8 # Initial angular velocity (rad/s)
    
    print(f"Start state: angle={q_start:.3f} rad, velocity={q_dot_start:.3f} rad/s")
    
    # Predict endpoint
    endpoint = inferencer.predict_single(q_start, q_dot_start)
    print(f"Predicted endpoint: angle={endpoint[0]:.3f} rad, velocity={endpoint[1]:.3f} rad/s")
    
    # Check which attractor it's closest to
    attractors = np.array([[0, 0], [2.1, 0], [-2.1, 0]])
    distances = [np.linalg.norm(endpoint - attr) for attr in attractors]
    closest_attractor = np.argmin(distances)
    attractor_names = ["Center (0,0)", "Right (2.1,0)", "Left (-2.1,0)"]
    
    print(f"Closest attractor: {attractor_names[closest_attractor]} (distance: {distances[closest_attractor]:.3f})")
    
    # Example 2: Batch prediction
    print("\\n" + "="*50) 
    print("EXAMPLE 2: Batch Prediction")
    print("="*50)
    
    # Multiple start states
    start_states = [
        [0.5, 1.0],    # Small angle, positive velocity
        [-1.0, -0.5],  # Negative angle, negative velocity
        [2.0, 0.1],    # Large angle, small velocity
        [-2.5, 1.5],   # Large negative angle, positive velocity
    ]
    
    print("Start states:")
    for i, state in enumerate(start_states):
        print(f"  {i+1}: angle={state[0]:.3f}, velocity={state[1]:.3f}")
    
    # Predict all endpoints
    endpoints = inferencer.batch_predict(start_states)
    
    print("\\nPredicted endpoints:")
    for i, endpoint in enumerate(endpoints):
        print(f"  {i+1}: angle={endpoint[0]:.3f}, velocity={endpoint[1]:.3f}")
    
    # Example 3: Visualize flow path
    print("\\n" + "="*50)
    print("EXAMPLE 3: Flow Path Visualization")
    print("="*50)
    
    # Pick an interesting start state
    demo_start = [1.8, -1.2]
    print(f"Visualizing flow path from: angle={demo_start[0]:.3f}, velocity={demo_start[1]:.3f}")
    
    try:
        # This will create and show a plot
        fig = inferencer.visualize_flow_path(
            demo_start, 
            save_path="demo_flow_path.png",
            figsize=(12, 6)
        )
        print("✓ Flow path visualization saved as 'demo_flow_path.png'")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
    
    # Example 4: Flow path analysis
    print("\\n" + "="*50)
    print("EXAMPLE 4: Flow Path Analysis") 
    print("="*50)
    
    start_state = [1.0, 0.5]
    endpoint, path = inferencer.predict_single(start_state[0], start_state[1], return_path=True)
    
    print(f"Start: ({start_state[0]:.3f}, {start_state[1]:.3f})")
    print(f"End: ({endpoint[0]:.3f}, {endpoint[1]:.3f})")
    print(f"Path length: {len(path)} steps")
    
    # Analyze path properties
    total_distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    max_speed = np.max(np.linalg.norm(np.diff(path, axis=0), axis=1))
    
    print(f"Total path distance: {total_distance:.3f}")
    print(f"Maximum step size: {max_speed:.3f}")
    
    # Show key waypoints
    n_waypoints = 5
    waypoint_indices = np.linspace(0, len(path)-1, n_waypoints, dtype=int)
    print("\\nKey waypoints along the path:")
    for i, idx in enumerate(waypoint_indices):
        t = idx / (len(path) - 1)
        point = path[idx]
        print(f"  t={t:.2f}: ({point[0]:.3f}, {point[1]:.3f})")
    
    # Example 5: Visualize 250 flow paths in one plot
    print("\\n" + "="*50)
    print("EXAMPLE 5: 250-Point Flow Visualization")
    print("="*50)
    
    # Generate 250 random starting points within the phase space bounds
    np.random.seed(42)  # For reproducible results
    start_points = []
    for _ in range(250):
        angle = np.random.uniform(-np.pi, np.pi)
        velocity = np.random.uniform(-2*np.pi, 2*np.pi)
        start_points.append([angle, velocity])
    
    try:
        import matplotlib.pyplot as plt
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Generate colormap for 250 trajectories
        from matplotlib.cm import tab10, Set3, hsv
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i / len(start_points)) for i in range(len(start_points))]
        
        # Add attractor regions first
        attractors = [[0, 0], [2.1, 0], [-2.1, 0]]
        attractor_labels = ['Center (0,0)', 'Right (2.1,0)', 'Left (-2.1,0)']
        attractor_colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        for j, (attr, label, color) in enumerate(zip(attractors, attractor_labels, attractor_colors)):
            circle = plt.Circle(attr, 0.1, color=color, alpha=0.4, label=label)
            ax.add_patch(circle)
        
        print(f"Processing {len(start_points)} flow paths...")
        
        # Batch prediction for efficiency
        all_endpoints = inferencer.batch_predict(start_points)
        
        # Collect paths for visualization  
        paths_data = []
        print("Collecting flow paths for visualization...")
        
        for i, start_point in enumerate(start_points):
            if i % 50 == 0:
                print(f"  Processing trajectory {i+1}/{len(start_points)}")
                
            # Get flow path
            endpoint, path = inferencer.predict_single(start_point[0], start_point[1], return_path=True)
            
            # Convert to numpy if needed
            if isinstance(path, torch.Tensor):
                path_np = path.cpu().numpy()
            else:
                path_np = path
                
            paths_data.append(path_np)
        
        print("Plotting trajectories...")
        
        for i, (start_point, path_np, color) in enumerate(zip(start_points, paths_data, colors)):
            start_np = np.array(start_point)
            end_np = path_np[-1]  # Last point in path
            
            # Plot flow path (thinner lines for 250 paths)
            ax.plot(path_np[:, 0], path_np[:, 1], color=color, linewidth=0.8, 
                    alpha=0.7)
            
            # Plot start points (smaller markers)
            ax.scatter(start_np[0], start_np[1], color='darkgreen', s=15, marker='o', 
                      zorder=5, alpha=0.8)
            
            # Plot end points (smaller markers)
            ax.scatter(end_np[0], end_np[1], color='darkred', s=15, marker='s', 
                      zorder=5, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Angle (q) [radians]', fontsize=12)
        ax.set_ylabel('Angular Velocity (q̇) [rad/s]', fontsize=12)
        ax.set_title('Flow Matching: 250 Trajectories from Random Start States to Attractors', 
                    fontsize=14, fontweight='bold')
        
        # Set specified dimensions: -π to π × -2π to 2π
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-2*np.pi, 2*np.pi)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add axis lines for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Add custom legend elements for start/end points
        from matplotlib.lines import Line2D
        custom_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                   markersize=10, markeredgecolor='black', markeredgewidth=2, label='Start Points'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', 
                   markersize=10, markeredgecolor='black', markeredgewidth=2, label='End Points')
        ]
        
        # Create simplified legend for 250 trajectories
        ax.legend(custom_elements + [plt.Circle((0, 0), 0.1, color='lightcoral', alpha=0.4),
                                   plt.Circle((0, 0), 0.1, color='lightblue', alpha=0.4),
                                   plt.Circle((0, 0), 0.1, color='lightgreen', alpha=0.4)], 
                 ['Start Points', 'End Points', 'Center (0,0)', 'Right (2.1,0)', 'Left (-2.1,0)'],
                 loc='upper right', fontsize=10)
        
        # Add π-based tick labels
        pi_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        pi_labels = ['-π', '-π/2', '0', 'π/2', 'π']
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        
        pi2_ticks = [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi]
        pi2_labels = ['-2π', '-π', '0', 'π', '2π']
        ax.set_yticks(pi2_ticks)
        ax.set_yticklabels(pi2_labels)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig("250_flow_paths_demo.png", dpi=150, bbox_inches='tight')
        print("✓ 250-point flow visualization saved as '250_flow_paths_demo.png'")
        
        plt.show()
        
    except Exception as e:
        print(f"✗ Error creating 250-point visualization: {e}")

    print("\\n" + "="*50)
    print("Demo completed! The trained flow matching model can:")
    print("• Predict single endpoints from start states")
    print("• Handle batch predictions efficiently") 
    print("• Visualize the learned flow paths")
    print("• Provide detailed path analysis")
    print("• Show multiple trajectories in phase space")
    print("="*50)

if __name__ == "__main__":
    main()