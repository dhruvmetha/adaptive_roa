"""
Normalized-space integrator for CartPole LCFM

Integrates in normalized coordinates [-1,1] × [-1,1] × [-π,π] × [-1,1]
for numerical stability and consistency with training.
"""
import torch
from typing import List, Tuple, Callable
from ..systems.base import DynamicalSystem


class NormalizedCartPoleIntegrator:
    """
    Integrator that works in normalized CartPole coordinates:
    - x_norm ∈ [-1, 1] (normalized cart position)
    - ẋ_norm ∈ [-1, 1] (normalized cart velocity)  
    - θ ∈ [-π, π] (pole angle - natural S¹ coordinate)
    - θ̇_norm ∈ [-1, 1] (normalized angular velocity)
    
    Benefits:
    - Numerical stability (bounded integration)
    - Consistency with training
    - Single denormalization at the end
    """
    
    def __init__(self, system: DynamicalSystem):
        """
        Initialize normalized integrator
        
        Args:
            system: CartPole system with bounds for final denormalization
        """
        self.system = system
    
    def integrate_step(self, 
                      state_norm: torch.Tensor,     # [B, 4] normalized state
                      velocity_norm: torch.Tensor,  # [B, 4] normalized velocity  
                      dt: float) -> torch.Tensor:
        """
        Integrate one step in normalized coordinates
        
        Args:
            state_norm: [B, 4] as [x_norm, ẋ_norm, θ, θ̇_norm]
            velocity_norm: [B, 4] as [dx_norm/dt, dẋ_norm/dt, dθ/dt, dθ̇_norm/dt]
            dt: Integration time step
            
        Returns:
            new_state_norm: [B, 4] next state in normalized coordinates
        """
        # Extract components
        x_norm = state_norm[:, 0:1]        # [B, 1] cart position (normalized)
        x_dot_norm = state_norm[:, 1:2]    # [B, 1] cart velocity (normalized)
        theta = state_norm[:, 2]           # [B] pole angle (natural S¹)
        theta_dot_norm = state_norm[:, 3:4] # [B, 1] angular velocity (normalized)
        
        v_x_norm = velocity_norm[:, 0:1]   # [B, 1] cart position velocity
        v_x_dot_norm = velocity_norm[:, 1:2] # [B, 1] cart velocity
        v_theta = velocity_norm[:, 2]      # [B] angular velocity  
        v_theta_dot_norm = velocity_norm[:, 3:4] # [B, 1] angular acceleration
        
        # ℝ components: Standard Euler integration (stays bounded)
        x_norm_new = x_norm + v_x_norm * dt              # [-1,1] → [-1,1]
        x_dot_norm_new = x_dot_norm + v_x_dot_norm * dt  # [-1,1] → [-1,1]  
        theta_dot_norm_new = theta_dot_norm + v_theta_dot_norm * dt # [-1,1] → [-1,1]
        
        # S¹ component: Proper SO(2) integration using Theseus exponential map
        theta_new = self._integrate_so2_theseus(theta, v_theta, dt)
        
        # Ensure theta_new has correct shape [B, 1]
        if theta_new.dim() == 1:
            theta_new = theta_new.unsqueeze(-1)
        
        # Clamp ℝ components to stay in bounds (safety)
        x_norm_new = torch.clamp(x_norm_new, -1.0, 1.0)
        x_dot_norm_new = torch.clamp(x_dot_norm_new, -1.0, 1.0)
        theta_dot_norm_new = torch.clamp(theta_dot_norm_new, -1.0, 1.0)
        
        return torch.cat([x_norm_new, x_dot_norm_new, theta_new, theta_dot_norm_new], dim=1)
    
    def integrate_batch(self, 
                       start_states_norm: torch.Tensor,  # [B, 4] normalized
                       velocity_func: Callable,          # (t, x_norm) -> v_norm
                       num_steps: int = 100) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Integrate batch of trajectories in normalized space
        
        Args:
            start_states_norm: [B, 4] initial states in normalized coordinates
            velocity_func: Function (t, x_norm) -> velocity_norm  
            num_steps: Number of integration steps
            
        Returns:
            Tuple of (final_states_norm, trajectory_list_norm)
        """
        dt = 1.0 / num_steps
        x_current = start_states_norm.clone()
        trajectory = [x_current.clone()]
        
        for step in range(num_steps):
            t = step * dt
            
            # Get normalized velocity from model
            velocity_norm = velocity_func(t, x_current)
            
            # Integrate one step in normalized space
            x_next = self.integrate_step(x_current, velocity_norm, dt)
            
            x_current = x_next  
            trajectory.append(x_current.clone())
        
        return x_current, trajectory
    
    def denormalize_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Convert final result from normalized back to raw coordinates
        
        Args:
            state_norm: [B, 4] normalized state [x_norm, ẋ_norm, θ, θ̇_norm]
            
        Returns:
            state_raw: [B, 4] raw state [x, ẋ, θ, θ̇]
        """
        # Extract normalized components
        x_norm = state_norm[:, 0:1]        # [-1, 1]
        x_dot_norm = state_norm[:, 1:2]    # [-1, 1] 
        theta = state_norm[:, 2:3]         # [-π, π] (already in natural coordinates)
        theta_dot_norm = state_norm[:, 3:4] # [-1, 1]
        
        # Denormalize ℝ components back to original scale  
        x_raw = x_norm * self.system.cart_limit                    # [-1,1] → [-cart_limit, cart_limit]
        x_dot_raw = x_dot_norm * self.system.velocity_limit        # [-1,1] → [-velocity_limit, velocity_limit]
        theta_raw = theta                                          # [-π,π] unchanged (natural coordinate)
        theta_dot_raw = theta_dot_norm * self.system.angular_velocity_limit # [-1,1] → [-angular_velocity_limit, angular_velocity_limit]
        
        return torch.cat([x_raw, x_dot_raw, theta_raw, theta_dot_raw], dim=1)
    
    def normalize_state(self, state_raw: torch.Tensor) -> torch.Tensor:
        """
        Convert raw coordinates to normalized space
        
        Args:
            state_raw: [B, 4] raw state [x, ẋ, θ, θ̇]
            
        Returns:
            state_norm: [B, 4] normalized state [x_norm, ẋ_norm, θ, θ̇_norm]
        """
        # Extract raw components
        x_raw = state_raw[:, 0:1]
        x_dot_raw = state_raw[:, 1:2] 
        theta_raw = state_raw[:, 2:3]
        theta_dot_raw = state_raw[:, 3:4]
        
        # Normalize ℝ components to [-1, 1]
        x_norm = x_raw / self.system.cart_limit                    
        x_dot_norm = x_dot_raw / self.system.velocity_limit        
        theta_norm = torch.atan2(torch.sin(theta_raw), torch.cos(theta_raw))  # Wrap to [-π,π]
        theta_dot_norm = theta_dot_raw / self.system.angular_velocity_limit
        
        # Clamp to bounds (safety)
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        x_dot_norm = torch.clamp(x_dot_norm, -1.0, 1.0) 
        theta_dot_norm = torch.clamp(theta_dot_norm, -1.0, 1.0)
        
        return torch.cat([x_norm, x_dot_norm, theta_norm, theta_dot_norm], dim=1)
    
    def _integrate_so2_theseus(self, theta: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Proper SO(2) integration using Theseus exponential map
        
        Args:
            theta: Current angles [B]
            omega: Angular velocities [B] 
            dt: Time step
            
        Returns:
            new_theta: Next angles [B] properly integrated on S¹
        """
        try:
            import theseus as th
            
            batch_size = theta.shape[0]
            device = theta.device
            dtype = theta.dtype
            
            # Create SE(2) with zero translation for pure SO(2) behavior
            zero_translation = torch.zeros(batch_size, 2, device=device, dtype=dtype)
            current_poses = torch.cat([zero_translation, theta.unsqueeze(-1)], dim=-1)  # [B, 3]
            current_se2 = th.SE2(x_y_theta=current_poses)
            
            # Create tangent vector in SE(2) space (zero translation, pure rotation)
            zero_vel = torch.zeros(batch_size, 2, device=device, dtype=dtype)
            tangent_vec = torch.cat([zero_vel, omega.unsqueeze(-1) * dt], dim=-1)  # [B, 3]
            
            # Apply exponential map
            exp_tangent = th.SE2.exp_map(tangent_vec)
            next_se2 = current_se2.compose(exp_tangent)
            
            # Extract angle and ensure proper shape
            new_theta = next_se2.theta()  # [B]
            return new_theta
            
        except Exception as e:
            print(f"Warning: Theseus SO(2) integration failed ({e}), using fallback")
            # Fallback: Simple Euler + wrapping
            theta_new = theta + omega * dt
            return torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
    
    def __repr__(self) -> str:
        return f"NormalizedCartPoleIntegrator(space=[-1,1]²×[-π,π]×[-1,1], system={self.system}, SO2=Theseus)"