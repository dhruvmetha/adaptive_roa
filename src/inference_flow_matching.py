import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.module.flow_matching import FlowMatching
from src.model.unet1d import UNet1D
from src.data.endpoint_data import EndpointDataset

class FlowMatchingInference:
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """
        Initialize inference with trained checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model architecture first
        model_net = UNet1D(
            input_dim=4,  # Current state (2D) + condition (2D)
            output_dim=2,  # Velocity prediction (2D)
            hidden_dims=[64, 128, 256],
            time_emb_dim=128
        )
        
        # Load model from checkpoint with the architecture
        self.model = FlowMatching.load_from_checkpoint(
            checkpoint_path, 
            model=model_net,
            strict=False  # In case of minor mismatches
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # For denormalization
        self.min_bounds = np.array([-np.pi, -2*np.pi])
        self.max_bounds = np.array([np.pi, 2*np.pi])
    
    def normalize_state(self, state):
        """Normalize state to [0, 1] range"""
        state = np.array(state)
        return (state - self.min_bounds) / (self.max_bounds - self.min_bounds)
    
    def denormalize_state(self, state):
        """Denormalize state from [0, 1] back to original range"""
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        return state * (self.max_bounds - self.min_bounds) + self.min_bounds
    
    @torch.no_grad()
    def predict_endpoint(self, start_states, num_steps=100, return_path=False):
        """
        Predict endpoint(s) for given start state(s)
        
        Args:
            start_states: tensor [batch_size, 2] or [2] for single prediction
            num_steps: number of integration steps
            return_path: if True, return the full integration path
            
        Returns:
            predicted_endpoints: tensor [batch_size, 2] or [2]
            paths (optional): tensor [batch_size, num_steps+1, 2] - full integration path
        """
        # Handle single state input
        single_input = False
        if start_states.dim() == 1:
            start_states = start_states.unsqueeze(0)
            single_input = True
            
        start_states = start_states.to(self.device)
        batch_size = start_states.shape[0]
        
        # Normalize input
        start_states_norm = torch.tensor(
            [self.normalize_state(s.cpu().numpy()) for s in start_states], 
            device=self.device, dtype=torch.float32
        )
        
        # Initialize at start state (t=0)
        x = start_states_norm.clone()
        dt = 1.0 / num_steps
        
        # Store path if requested
        if return_path:
            path = [x.clone()]
        
        # Integrate using Euler method
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * i * dt
            
            # Predict velocity
            velocity = self.model.model(x, t, condition=start_states_norm)
            
            # Euler step
            x = x + velocity * dt
            
            if return_path:
                path.append(x.clone())
        
        # Denormalize results
        predicted_endpoints = torch.tensor(
            [self.denormalize_state(s) for s in x], 
            device=x.device, dtype=torch.float32
        )
        
        # Handle single output
        if single_input:
            predicted_endpoints = predicted_endpoints.squeeze(0)
        
        if return_path:
            # Denormalize path
            path_tensor = torch.stack(path, dim=1)  # [batch_size, num_steps+1, 2]
            path_denorm = torch.tensor(
                [[self.denormalize_state(step) for step in trajectory] 
                 for trajectory in path_tensor], 
                device=path_tensor.device, dtype=torch.float32
            )
            
            if single_input:
                path_denorm = path_denorm.squeeze(0)
                
            return predicted_endpoints, path_denorm
        
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
    
    def visualize_flow_path(self, start_state, save_path=None, figsize=(10, 6)):
        """
        Visualize the flow path from start to endpoint
        
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Phase space trajectory
        ax1.plot(path_np[:, 0], path_np[:, 1], 'b-', alpha=0.7, linewidth=2, label='Flow path')
        ax1.scatter(start_np[0], start_np[1], color='green', s=100, marker='o', 
                   label='Start', zorder=5, edgecolor='black')
        ax1.scatter(end_np[0], end_np[1], color='red', s=100, marker='s', 
                   label='Predicted end', zorder=5, edgecolor='black')
        
        # Add attractor regions
        attractors = [[0, 0], [2.1, 0], [-2.1, 0]]
        for attr in attractors:
            circle = plt.Circle(attr, 0.1, color='gray', alpha=0.3)
            ax1.add_patch(circle)
        
        ax1.set_xlabel('Angle (q)')
        ax1.set_ylabel('Angular velocity (q̇)')
        ax1.set_title('Phase Space Flow Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time evolution
        time_steps = np.linspace(0, 1, len(path_np))
        ax2.plot(time_steps, path_np[:, 0], 'b-', label='Angle (q)', linewidth=2)
        ax2.plot(time_steps, path_np[:, 1], 'r-', label='Angular velocity (q̇)', linewidth=2)
        ax2.scatter(0, start_np[0], color='green', s=60, marker='o', zorder=5)
        ax2.scatter(0, start_np[1], color='green', s=60, marker='o', zorder=5)
        ax2.scatter(1, end_np[0], color='red', s=60, marker='s', zorder=5)
        ax2.scatter(1, end_np[1], color='red', s=60, marker='s', zorder=5)
        
        ax2.set_xlabel('Flow time (t)')
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
    checkpoint_path = "path/to/your/checkpoint.ckpt"
    inferencer = FlowMatchingInference(checkpoint_path)
    
    # Single prediction
    q, q_dot = 1.5, -0.5  # Example start state
    endpoint = inferencer.predict_single(q, q_dot)
    print(f"Start: ({q}, {q_dot}) → Predicted end: ({endpoint[0]:.3f}, {endpoint[1]:.3f})")
    
    # Visualize flow path
    inferencer.visualize_flow_path([q, q_dot], save_path="flow_path.png")