"""
Universal manifold integrator using Theseus Lie groups
"""
import torch
from typing import List, Dict, Any, Optional, Tuple
from src.systems.base import DynamicalSystem, ManifoldComponent

try:
    import theseus as th
    THESEUS_AVAILABLE = True
except ImportError:
    THESEUS_AVAILABLE = False
    print("Warning: Theseus not available. Using fallback integration methods.")


class TheseusIntegrator:
    """
    Universal integrator that handles multiple manifold types using Theseus
    
    Supports proper integration on:
    - SO(2): 2D rotations (circles) via SE(2) with zero translation
    - SO(3): 3D rotations via native Theseus SO(3)
    - SE(2): 2D poses via native Theseus SE(2)  
    - SE(3): 3D poses via native Theseus SE(3)
    - ℝ: Real line via standard Euler integration
    """
    
    def __init__(self, system: DynamicalSystem):
        """
        Initialize integrator for a specific dynamical system
        
        Args:
            system: DynamicalSystem defining manifold structure
        """
        self.system = system
        self.manifold_components = system.manifold_components
        
        # Verify Theseus availability for required manifolds
        self._check_theseus_requirements()
    
    def _check_theseus_requirements(self):
        """Check if Theseus is available for required manifold types"""
        theseus_manifolds = ["SO2", "SO3", "SE2", "SE3"]
        requires_theseus = any(comp.manifold_type in theseus_manifolds 
                             for comp in self.manifold_components)
        
        if requires_theseus and not THESEUS_AVAILABLE:
            print(f"Warning: System {self.system} requires Theseus for manifold integration, "
                  f"but Theseus is not available. Falling back to manual methods.")
    
    def integrate_step(self, 
                      state: torch.Tensor, 
                      velocity: torch.Tensor, 
                      dt: float) -> torch.Tensor:
        """
        Integrate one step on the manifold using proper Lie group operations
        
        Args:
            state: Current state [..., state_dim] in raw coordinates
            velocity: Tangent velocity [..., tangent_dim] 
            dt: Integration time step
            
        Returns:
            next_state: Next state [..., state_dim] after integration
        """
        # Decompose state and velocity by manifold components
        state_components = self._decompose_state(state)
        velocity_components = self._decompose_velocity(velocity)
        
        # Integrate each component on its manifold
        new_components = []
        
        for comp, state_comp, vel_comp in zip(
            self.manifold_components, state_components, velocity_components
        ):
            if comp.manifold_type == "SO2":
                new_comp = self._integrate_so2(state_comp, vel_comp, dt)
            elif comp.manifold_type == "SO3":
                new_comp = self._integrate_so3(state_comp, vel_comp, dt)
            elif comp.manifold_type == "SE2":
                new_comp = self._integrate_se2(state_comp, vel_comp, dt)
            elif comp.manifold_type == "SE3":
                new_comp = self._integrate_se3(state_comp, vel_comp, dt)
            elif comp.manifold_type == "Real":
                new_comp = self._integrate_real(state_comp, vel_comp, dt)
            else:
                raise ValueError(f"Unknown manifold type: {comp.manifold_type}")
            
            new_components.append(new_comp)
        
        # Recombine components
        return torch.cat(new_components, dim=-1)
    
    def integrate_batch(self, 
                       start_states: torch.Tensor,
                       velocity_func,
                       num_steps: int = 100) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Integrate a batch of trajectories using ODE solving
        
        Args:
            start_states: Initial states [B, state_dim] 
            velocity_func: Function (t, x) -> velocity [B, tangent_dim]
            num_steps: Number of integration steps
            
        Returns:
            Tuple of (final_states, trajectory_list)
        """
        dt = 1.0 / num_steps
        x_current = start_states.clone()
        trajectory = [x_current.clone()]
        
        for step in range(num_steps):
            t = step * dt
            
            # Get velocity at current time and state
            velocity = velocity_func(t, x_current)
            
            # Integrate one step on manifold
            x_next = self.integrate_step(x_current, velocity, dt)
            
            x_current = x_next
            trajectory.append(x_current.clone())
        
        return x_current, trajectory
    
    def _decompose_state(self, state: torch.Tensor) -> List[torch.Tensor]:
        """Decompose state tensor by manifold components"""
        components = []
        start_idx = 0
        
        for comp in self.manifold_components:
            end_idx = start_idx + comp.dim
            components.append(state[..., start_idx:end_idx])
            start_idx = end_idx
            
        return components
    
    def _decompose_velocity(self, velocity: torch.Tensor) -> List[torch.Tensor]:
        """Decompose velocity tensor by tangent space components"""
        components = []
        start_idx = 0
        
        for comp in self.manifold_components:
            end_idx = start_idx + comp.tangent_dim
            components.append(velocity[..., start_idx:end_idx])
            start_idx = end_idx
            
        return components
    
    def _integrate_so2(self, angle: torch.Tensor, angular_vel: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Integrate on SO(2) using Theseus SE(2) with zero translation
        
        Args:
            angle: Current angle [..., 1]
            angular_vel: Angular velocity [..., 1]  
            dt: Time step
            
        Returns:
            new_angle: Next angle [..., 1] wrapped to [-π, π]
        """
        if THESEUS_AVAILABLE:
            try:
                batch_shape = angle.shape[:-1]
                batch_size = torch.prod(torch.tensor(batch_shape)).item()
                
                # Flatten for Theseus operations
                angle_flat = angle.view(batch_size, 1)
                angular_vel_flat = angular_vel.view(batch_size, 1)
                
                # Create SE(2) with zero translation for SO(2) behavior
                zero_translation = torch.zeros(batch_size, 2, device=angle.device, dtype=angle.dtype)
                current_poses = torch.cat([zero_translation, angle_flat], dim=-1)  # [batch, 3]
                current_se2 = th.SE2(x_y_theta=current_poses)
                
                # Create tangent vector (only rotation component)
                zero_vel = torch.zeros(batch_size, 2, device=angular_vel.device, dtype=angular_vel.dtype)
                tangent_vec = torch.cat([zero_vel, angular_vel_flat * dt], dim=-1)  # [batch, 3]
                
                # Apply exponential map (create SE2 from tangent vector)
                exp_tangent = th.SE2.exp_map(tangent_vec)
                next_se2 = current_se2.compose(exp_tangent)
                
                # Extract angle and reshape
                new_angle = next_se2.theta().unsqueeze(-1).view(angle.shape)
                return new_angle
                
            except Exception as e:
                print(f"Warning: Theseus SO(2) integration failed ({e}), using fallback")
        
        # Fallback: Manual angle wrapping
        new_angle = angle + angular_vel * dt
        return torch.atan2(torch.sin(new_angle), torch.cos(new_angle))
    
    def _integrate_so3(self, quaternion: torch.Tensor, angular_vel: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Integrate on SO(3) using Theseus
        
        Args:
            quaternion: Current quaternion [..., 4] as (w, x, y, z)
            angular_vel: Angular velocity [..., 3]
            dt: Time step
            
        Returns:
            new_quaternion: Next quaternion [..., 4] normalized
        """
        if THESEUS_AVAILABLE:
            try:
                batch_shape = quaternion.shape[:-1]
                batch_size = torch.prod(torch.tensor(batch_shape)).item()
                
                # Flatten for Theseus operations  
                quat_flat = quaternion.view(batch_size, 4)
                angular_vel_flat = angular_vel.view(batch_size, 3)
                
                # Create SO(3) objects
                current_so3 = th.SO3(quaternion=quat_flat)
                
                # Create tangent vector
                tangent_vec = angular_vel_flat * dt
                
                # Apply exponential map
                exp_tangent = th.SO3.exp_map(tangent_vec)
                next_so3 = current_so3.compose(exp_tangent)
                
                # Extract quaternion and reshape
                new_quaternion = next_so3.to_quaternion().view(quaternion.shape)
                return new_quaternion
                
            except Exception as e:
                print(f"Warning: Theseus SO(3) integration failed ({e}), using fallback")
        
        # Fallback: Simple quaternion integration (not geometrically correct)
        # This is a placeholder - proper SO(3) integration without Theseus is complex
        print("Warning: SO(3) integration without Theseus is not implemented. Using identity.")
        return quaternion
    
    def _integrate_se2(self, pose: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Integrate on SE(2) using Theseus
        
        Args:
            pose: Current pose [..., 3] as (x, y, θ)
            velocity: Velocity [..., 3] as (vx, vy, ω)
            dt: Time step
            
        Returns:
            new_pose: Next pose [..., 3]
        """
        if THESEUS_AVAILABLE:
            try:
                batch_shape = pose.shape[:-1]
                batch_size = torch.prod(torch.tensor(batch_shape)).item()
                
                # Flatten for Theseus operations
                pose_flat = pose.view(batch_size, 3)
                vel_flat = velocity.view(batch_size, 3)
                
                # Create SE(2) objects
                current_se2 = th.SE2(x_y_theta=pose_flat)
                
                # Create tangent vector
                tangent_vec = vel_flat * dt
                
                # Apply exponential map (create SE2 from tangent vector)
                exp_tangent = th.SE2.exp_map(tangent_vec)
                next_se2 = current_se2.compose(exp_tangent)
                
                # Extract pose and reshape (reconstruct x_y_theta format)
                xy = next_se2.xy()
                theta = next_se2.theta()
                new_pose = torch.cat([xy, theta], dim=-1).view(pose.shape)
                return new_pose
                
            except Exception as e:
                print(f"Warning: Theseus SE(2) integration failed ({e}), using fallback")
        
        # Fallback: Separate x, y, θ integration
        x, y, theta = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3]
        vx, vy, omega = velocity[..., 0:1], velocity[..., 1:2], velocity[..., 2:3]
        
        new_x = x + vx * dt
        new_y = y + vy * dt  
        new_theta = torch.atan2(torch.sin(theta + omega * dt), torch.cos(theta + omega * dt))
        
        return torch.cat([new_x, new_y, new_theta], dim=-1)
    
    def _integrate_se3(self, pose: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Integrate on SE(3) using Theseus
        
        Args:
            pose: Current pose [..., 7] as (x, y, z, qw, qx, qy, qz)
            velocity: Velocity [..., 6] as (vx, vy, vz, ωx, ωy, ωz)
            dt: Time step
            
        Returns:
            new_pose: Next pose [..., 7]
        """
        if THESEUS_AVAILABLE:
            try:
                batch_shape = pose.shape[:-1]
                batch_size = torch.prod(torch.tensor(batch_shape)).item()
                
                # Flatten for Theseus operations
                pose_flat = pose.view(batch_size, 7)
                vel_flat = velocity.view(batch_size, 6)
                
                # Create SE(3) objects
                current_se3 = th.SE3(tensor=pose_flat)
                
                # Create tangent vector
                tangent_vec = vel_flat * dt
                
                # Apply exponential map
                exp_tangent = th.SE3.exp_map(tangent_vec)
                next_se3 = current_se3.compose(exp_tangent)
                
                # Extract pose and reshape
                new_pose = next_se3.tensor.view(pose.shape)
                return new_pose
                
            except Exception as e:
                print(f"Warning: Theseus SE(3) integration failed ({e}), using fallback")
        
        # Fallback: Not implemented - SE(3) integration is complex
        print("Warning: SE(3) integration without Theseus is not implemented. Using identity.")
        return pose
    
    def _integrate_real(self, position: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Integrate on real line using standard Euler method
        
        Args:
            position: Current position [..., dim]
            velocity: Velocity [..., dim]
            dt: Time step
            
        Returns:
            new_position: Next position [..., dim]
        """
        return position + velocity * dt
    
    def __repr__(self) -> str:
        manifold_types = [comp.manifold_type for comp in self.manifold_components]
        return f"TheseusIntegrator(manifolds={manifold_types}, theseus_available={THESEUS_AVAILABLE})"