import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.module.circular_flow_matching import CircularFlowMatching
from src.model.circular_unet1d import CircularUNet1D

class CircularFlowMatchingInference:
    def __init__(self, checkpoint_path: str):
        """
        Initialize circular flow matching inference with trained checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model architecture first
        model_net = CircularUNet1D(
            input_dim=6,   # current_state(3) + condition(3)
            output_dim=3,  # velocity on S¹ × ℝ
            hidden_dims=[64, 128, 256],
            time_emb_dim=128
        )
        
        # Load model from checkpoint
        self.model = CircularFlowMatching.load_from_checkpoint(
            checkpoint_path, 
            model=model_net,
            strict=False  # In case of minor mismatches
        )
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def embed_state(self, state):
        """Convert (θ, θ̇) → (sin(θ), cos(θ), θ̇)"""
        if isinstance(state, (list, tuple)):
            theta, theta_dot = state
        else:
            theta, theta_dot = state[..., 0], state[..., 1]
        return torch.stack([torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)
    
    def extract_state(self, embedded):
        """Convert (sin(θ), cos(θ), θ̇) → (θ, θ̇)"""
        if torch.is_tensor(embedded):
            sin_theta, cos_theta, theta_dot = embedded[..., 0], embedded[..., 1], embedded[..., 2]
            theta = torch.atan2(sin_theta, cos_theta)
            return torch.stack([theta, theta_dot], dim=-1)
        else:
            sin_theta, cos_theta, theta_dot = embedded
            theta = np.arctan2(sin_theta, cos_theta)
            return np.array([theta, theta_dot])
    
    def project_to_manifold(self, x):
        """Project back to S¹ × ℝ manifold"""
        sin_theta, cos_theta, theta_dot = x[..., 0], x[..., 1], x[..., 2]
        
        # Normalize (sin, cos) to unit circle
        norm = torch.sqrt(sin_theta**2 + cos_theta**2)
        sin_theta_proj = sin_theta / (norm + 1e-8)
        cos_theta_proj = cos_theta / (norm + 1e-8)
        
        return torch.stack([sin_theta_proj, cos_theta_proj, theta_dot], dim=-1)
    
    @torch.no_grad()
    def predict_endpoint(self, start_states, num_steps=100, return_path=False):
        """
        Predict endpoint(s) for given start state(s) using circular flow matching
        
        Args:
            start_states: tensor [batch_size, 2] or [2] for single prediction (θ, θ̇)
            num_steps: number of integration steps
            return_path: if True, return the full integration path
            
        Returns:
            predicted_endpoints: tensor [batch_size, 2] or [2] (θ, θ̇)
            paths (optional): tensor [batch_size, num_steps+1, 2] - full integration path
        """
        # Handle single state input
        single_input = False
        if start_states.dim() == 1:
            start_states = start_states.unsqueeze(0)
            single_input = True
            
        start_states = start_states.to(self.device)
        batch_size = start_states.shape[0]
        
        # Convert to embedded representation
        start_embedded = self.embed_state(start_states)
        
        # Initialize flow at start state
        x = start_embedded.clone()
        dt = 1.0 / num_steps
        
        # Store path if requested
        if return_path:
            path = [self.extract_state(x.clone())]
        
        # Integrate using Euler method
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * i * dt
            
            # Predict velocity on manifold
            velocity = self.model.model(x, t, condition=start_embedded)
            
            # Euler step
            x = x + velocity * dt
            
            # Project back to manifold
            x = self.project_to_manifold(x)
            
            if return_path:
                path.append(self.extract_state(x.clone()))
        
        # Convert final state back to (θ, θ̇)
        predicted_endpoints = self.extract_state(x)
        
        # Handle single output
        if single_input:
            predicted_endpoints = predicted_endpoints.squeeze(0)
            if return_path:
                path_tensor = torch.stack(path, dim=0)  # [num_steps+1, 2]
                return predicted_endpoints, path_tensor
        
        if return_path:
            path_tensor = torch.stack(path, dim=1)  # [batch_size, num_steps+1, 2]
            return predicted_endpoints, path_tensor
        
        return predicted_endpoints
    
    def predict_single(self, q, q_dot, return_path=False):
        """
        Convenient method for single prediction
        
        Args:
            q: angle (float)
            q_dot: angular velocity (float) 
            return_path: if True, return integration path
            
        Returns:
            (q_end, q_dot_end): predicted endpoint
            path (optional): integration path
        """
        start_state = torch.tensor([q, q_dot], dtype=torch.float32)
        
        if return_path:
            endpoint, path = self.predict_endpoint(start_state, return_path=True)
            return endpoint.cpu().numpy(), path.cpu().numpy()
        else:
            endpoint = self.predict_endpoint(start_state)
            return endpoint.cpu().numpy()
    
    def visualize_flow_path(self, start_state, save_path=None, figsize=(12, 6)):
        """
        Visualize the circular flow path from start to endpoint
        
        Args:
            start_state: [q, q_dot] initial state
            save_path: path to save the plot
            figsize: figure size
        """
        start_tensor = torch.tensor(start_state, dtype=torch.float32)
        endpoint, path = self.predict_endpoint(start_tensor, return_path=True)
        
        # Convert to numpy
        path_np = path.cpu().numpy()
        start_np = start_state
        end_np = endpoint.cpu().numpy()
        
        # Fix shape: squeeze any singleton dimensions except the last one
        if path_np.ndim == 3 and path_np.shape[1] == 1:
            path_np = path_np.squeeze(1)  # Remove the singleton batch dimension
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.plot(path_np[:, 0], path_np[:, 1], 'b-', alpha=0.7, linewidth=2, label='Circular flow path')
        ax1.scatter(start_np[0], start_np[1], color='green', s=100, marker='o', 
                   label='Start', zorder=5, edgecolor='black')
        ax1.scatter(end_np[0], end_np[1], color='red', s=100, marker='s', 
                   label='Predicted end', zorder=5, edgecolor='black')
        
        # Add attractor regions (theoretical)
        attractors = [[0, 0], [2.1, 0], [-2.1, 0]]  # Note: ±π are the same
        attractor_labels = ['Center (0,0)', 'Right (2.1,0)', 'Left (-2.1,0)']
        for i, attr in enumerate(attractors):
            circle = plt.Circle(attr, 0.1, color='gray', alpha=0.3, label=attractor_labels[i])
            ax1.add_patch(circle)
            
        
        ax1.set_xlabel('Angle θ [radians]')
        ax1.set_ylabel('Angular velocity θ̇ [rad/s]')
        ax1.set_title('Phase Space: Circular Flow Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis to show circular nature
        ax1.set_xlim(-np.pi - 0.5, np.pi + 0.5)
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # Plot 2: Time evolution
        time_steps = np.linspace(0, 1, len(path_np))
        ax2.plot(time_steps, path_np[:, 0], 'b-', label='Angle θ', linewidth=2)
        ax2.plot(time_steps, path_np[:, 1], 'r-', label='Angular velocity θ̇', linewidth=2)
        ax2.scatter(0, start_np[0], color='green', s=60, marker='o', zorder=5)
        ax2.scatter(0, start_np[1], color='green', s=60, marker='o', zorder=5)
        ax2.scatter(1, end_np[0], color='red', s=60, marker='s', zorder=5)
        ax2.scatter(1, end_np[1], color='red', s=60, marker='s', zorder=5)
        
        ax2.set_xlabel('Flow time t')
        ax2.set_ylabel('State value')
        ax2.set_title('State Evolution Over Flow Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def visualize_multiple_flow_paths(self, start_states_list, save_path=None, figsize=(12, 6)):
        """
        Visualize multiple circular flow paths in the same plot
        
        Args:
            start_states_list: list of [q, q_dot] initial states
            save_path: path to save the plot
            figsize: figure size
        """
        start_tensor = torch.tensor(start_states_list, dtype=torch.float32)
        endpoints, paths = self.predict_endpoint(start_tensor, return_path=True)
        
        # Convert to numpy
        paths_np = paths.cpu().numpy()  # [batch_size, num_steps+1, 2]
        endpoints_np = endpoints.cpu().numpy()  # [batch_size, 2]
        start_np = np.array(start_states_list)  # [batch_size, 2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Phase space trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(start_states_list)))
        
        for i, (path, start, end, color) in enumerate(zip(paths_np, start_np, endpoints_np, colors)):
            # Plot flow path
            ax1.plot(path[:, 0], path[:, 1], '-', alpha=0.6, linewidth=1, color=color)
            # Plot start point
            ax1.scatter(start[0], start[1], color=color, s=30, marker='o', 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
            # Plot end point
            ax1.scatter(end[0], end[1], color=color, s=30, marker='s', 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add attractor regions
        attractors = [[0, 0], [2.1, 0], [-2.1, 0]]
        attractor_labels = ['Center (0,0)', 'Right (2.1,0)', 'Left (-2.1,0)']
        for i, attr in enumerate(attractors):
            circle = plt.Circle(attr, 0.1, color='gray', alpha=0.3, label=attractor_labels[i])
            ax1.add_patch(circle)
        
        ax1.set_xlabel('Angle θ [radians]')
        ax1.set_ylabel('Angular velocity θ̇ [rad/s]')
        ax1.set_title(f'Phase Space: {len(start_states_list)} Circular Flow Paths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis to show circular nature
        ax1.set_xlim(-np.pi - 0.5, np.pi + 0.5)
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # Plot 2: Endpoint distribution
        ax2.scatter(start_np[:, 0], start_np[:, 1], c=colors, s=40, marker='o', 
                   alpha=0.7, edgecolor='black', linewidth=0.5, label='Start states')
        ax2.scatter(endpoints_np[:, 0], endpoints_np[:, 1], c=colors, s=40, marker='s', 
                   alpha=0.7, edgecolor='black', linewidth=0.5, label='Predicted endpoints')
        
        # Add attractor regions
        for i, attr in enumerate(attractors):
            circle = plt.Circle(attr, 0.1, color='gray', alpha=0.3)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('Angle θ [radians]')
        ax2.set_ylabel('Angular velocity θ̇ [rad/s]')
        ax2.set_title('Start vs Predicted Endpoints')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis to show circular nature
        ax2.set_xlim(-np.pi - 0.5, np.pi + 0.5)
        ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def batch_predict(self, start_states_list):
        """
        Predict endpoints for a batch of start states
        
        Args:
            start_states_list: list of [q, q_dot] pairs
            
        Returns:
            predictions: numpy array [n_samples, 2]
        """
        start_tensor = torch.tensor(start_states_list, dtype=torch.float32)
        predictions = self.predict_endpoint(start_tensor)
        return predictions.cpu().numpy()

# Example usage script
if __name__ == "__main__":
    # Initialize inference (you'll need to provide actual checkpoint path)
    checkpoint_path = "path/to/your/circular_checkpoint.ckpt"
    inferencer = CircularFlowMatchingInference(checkpoint_path)
    
    # Single prediction
    q, q_dot = 1.5, -0.5  # Example start state
    endpoint = inferencer.predict_single(q, q_dot)
    print(f"Start: ({q}, {q_dot}) → Predicted end: ({endpoint[0]:.3f}, {endpoint[1]:.3f})")
    
    # Visualize flow path
    inferencer.visualize_flow_path([q, q_dot], save_path="circular_flow_path.png")