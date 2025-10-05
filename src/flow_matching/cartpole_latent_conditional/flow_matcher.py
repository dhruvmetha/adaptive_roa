"""
CartPole Latent Conditional Flow Matching implementation
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl

from ..base.flow_matcher import BaseFlowMatcher
from ...systems.base import DynamicalSystem


class CartPoleLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    CartPole Latent Conditional Flow Matching:
    - Flows from noisy input to data endpoint
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in ℝ² × S¹ × ℝ tangent space (x, θ, ẋ, θ̇)
    """
    
    def __init__(self, 
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[dict] = None,
                 latent_dim: int = 2):
        """
        Initialize CartPole latent conditional flow matcher
        
        Args:
            system: DynamicalSystem (should be CartPole with ℝ² × S¹ × ℝ structure)
            model: CartPoleLatentConditionalUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            config: Configuration dict
            latent_dim: Dimension of latent space
        """
        self.system = system
        self.latent_dim = latent_dim
        super().__init__(model, optimizer, scheduler, config)
    
    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input in ℝ² × S¹ × ℝ space
        
        Args:
            batch_size: Number of samples
            device: Device to create tensors on
            
        Returns:
            Noisy states [batch_size, 4] as (x, θ, ẋ, θ̇)
        """
        # x ~ Uniform[-cart_limit, +cart_limit] (symmetric bounds)
        x = torch.rand(batch_size, 1, device=device) * (self.system.cart_limit * 2) - self.system.cart_limit
        
        # ẋ ~ Uniform[-velocity_limit, +velocity_limit] (symmetric bounds)
        x_dot = torch.rand(batch_size, 1, device=device) * (self.system.velocity_limit * 2) - self.system.velocity_limit
        
        # θ ~ Uniform[-π, π] (wrapped angle)
        theta = torch.rand(batch_size, 1, device=device) * 2 * torch.pi - torch.pi
        
        # θ̇ ~ Uniform[-angular_velocity_limit, +angular_velocity_limit] (symmetric bounds)
        theta_dot = torch.rand(batch_size, 1, device=device) * (self.system.angular_velocity_limit * 2) - self.system.angular_velocity_limit
        
        return torch.cat([x, theta, x_dot, theta_dot], dim=1)
    
    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian latent vector
        
        Args:
            batch_size: Number of samples
            device: Device to create tensors on
            
        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, theta, x_dot, theta_dot) → (x_norm, theta, x_dot_norm, theta_dot_norm)
        
        Args:
            state: [B, 4] raw cartpole state (theta already wrapped to [-π, π])
            
        Returns:
            [B, 4] normalized state (theta unchanged)
        """
        # Extract components
        x = state[:, 0]
        theta = state[:, 1]  # Keep as-is (already wrapped)
        x_dot = state[:, 2]
        theta_dot = state[:, 3]
        
        # Normalize linear quantities to [-1, 1] range using symmetric bounds
        x_norm = x / self.system.cart_limit
        x_dot_norm = x_dot / self.system.velocity_limit
        theta_dot_norm = theta_dot / self.system.angular_velocity_limit
        
        return torch.stack([x_norm, theta, x_dot_norm, theta_dot_norm], dim=1)
    
    def embed_normalized_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state → (x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm)
        
        Args:
            normalized_state: [B, 4] normalized state
            
        Returns:
            [B, 5] embedded state
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]
        
        # Embed circular angle as sin/cos
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        return torch.stack([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dim=1)
    
    def interpolate_r2_s1_r(self, 
                           x_noise: torch.Tensor,    # [B, 4] (x, θ, ẋ, θ̇)
                           x_data: torch.Tensor,     # [B, 4] (x, θ, ẋ, θ̇) 
                           t: torch.Tensor           # [B]
                           ) -> torch.Tensor:
        """
        Interpolate in ℝ² × S¹ × ℝ space
        - ℝ components (x, ẋ, θ̇): linear interpolation
        - S¹ component (θ): geodesic interpolation using Theseus SO(2) exp map
        
        Returns:
            Interpolated states [B, 4] in ℝ² × S¹ × ℝ as (x, θ, ẋ, θ̇)
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.expand(x_noise.shape[0])
        t = t.unsqueeze(-1)  # [B, 1]
        
        # Extract components
        x_noise_pos = x_noise[:, 0:1]      # [B, 1] cart position
        theta_noise = x_noise[:, 1]        # [B] pole angle
        x_noise_vel = x_noise[:, 2:3]      # [B, 1] cart velocity  
        theta_dot_noise = x_noise[:, 3:4]  # [B, 1] angular velocity
        
        x_data_pos = x_data[:, 0:1]        # [B, 1] cart position
        theta_data = x_data[:, 1]          # [B] pole angle  
        x_data_vel = x_data[:, 2:3]        # [B, 1] cart velocity
        theta_dot_data = x_data[:, 3:4]    # [B, 1] angular velocity
        
        # Linear interpolation on ℝ components (x, ẋ, θ̇)
        x_t = (1 - t) * x_noise_pos + t * x_data_pos # t * x_data_pos - t * x_noise_pos + x_noise_pos
        x_dot_t = (1 - t) * x_noise_vel + t * x_data_vel
        theta_dot_t = (1 - t) * theta_dot_noise + t * theta_dot_data
        
        # Geodesic interpolation on S¹ for θ using Theseus SO(2) directly
        import theseus as th
        
        # Create SO(2) rotations from angles
        noise_so2 = th.SO2(theta=theta_noise.unsqueeze(-1))  # [B] -> [B, 1]
        data_so2 = th.SO2(theta=theta_data.unsqueeze(-1))    # [B] -> [B, 1]
        
        # Compute relative rotation: noise^{-1} * data
        relative_so2 = noise_so2.inverse().compose(data_so2) #  x_data_pos - x_noise_pos 
        
        # Get tangent vector (log map) - gives shortest angular distance
        log_relative = relative_so2.log_map()  # [B, 1] tangent vector # x_data_pos - x_noise_pos 
        
        # Scale by interpolation parameter t
        t_expanded = t.squeeze(-1).unsqueeze(-1)  # [B, 1]
        scaled_tangent = t_expanded * log_relative  # [B, 1]  t * (x_data_pos - x_noise_pos)  
        
        # Apply exponential map from noise position: noise ∘ exp(t * log(noise^{-1} * data))
        displacement_so2 = th.SO2.exp_map(scaled_tangent)
        interpolated_so2 = noise_so2.compose(displacement_so2) 
        
        # Extract interpolated angle (already wrapped to [-π, π])
        theta_t = interpolated_so2.theta().squeeze(-1)  # [B, 1] -> [B]
        
        return torch.cat([x_t, theta_t.unsqueeze(-1), x_dot_t, theta_dot_t], dim=1)
    
    def compute_target_velocity_r2_s1_r(self,
                                       x_t: torch.Tensor,        # [B, 4] current normalized state
                                       x_endpoint: torch.Tensor, # [B, 4] target normalized state
                                       t: torch.Tensor           # [B] (not used, kept for compatibility)
                                       ) -> torch.Tensor:
        """
        Compute target velocity from current position to endpoint in normalized coordinate space
        
        Args:
            x_t: Current interpolated normalized state [B, 4] (x_norm, theta, x_dot_norm, theta_dot_norm)
            x_endpoint: Target endpoint normalized state [B, 4] (x_norm, theta, x_dot_norm, theta_dot_norm)
            t: Time parameter (not used in direct velocity computation)
            
        Returns:
            Target velocity [B, 4] as (dx_norm/dt, dθ/dt, dx_dot_norm/dt, dθ_dot_norm/dt)
        """
        import theseus as th
        
        batch_size = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        
        # Extract angle components (θ - pole angle)
        theta_current = x_t[:, 1]        # [B]
        theta_target = x_endpoint[:, 1]  # [B]
        
        # denom = (1- t).clamp(min=1e-3).unsqueeze(-1)
        
        # For S¹ component: use Theseus SO(2) directly for angular velocity
        current_so2 = th.SO2(theta=theta_current.unsqueeze(-1))  # [B] -> [B, 1]
        target_so2 = th.SO2(theta=theta_target.unsqueeze(-1))    # [B] -> [B, 1]
        
        # Compute relative rotation: current^{-1} * target
        relative_so2 = current_so2.inverse().compose(target_so2)
        
        # Get angular velocity using log map - gives shortest angular path
        theta_velocity = relative_so2.log_map()  # [B, 1] - direct angular velocity
        
        # For ℝ components: simple differences (inputs already normalized)
        x_velocity = (x_endpoint[:, 0:1] - x_t[:, 0:1])           # [B, 1] position velocity
        x_dot_velocity = (x_endpoint[:, 2:3] - x_t[:, 2:3])     # [B, 1] velocity difference  
        theta_dot_velocity = (x_endpoint[:, 3:4] - x_t[:, 3:4])   # [B, 1] angular velocity difference
    

        return torch.cat([x_velocity, theta_velocity, x_dot_velocity, theta_dot_velocity], dim=1)
    
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute CartPole latent conditional flow matching loss
        
        Args:
            batch: Dictionary containing 'raw_start_state' and 'raw_end_state' in format [x, θ, ẋ, θ̇]
            
        Returns:
            Flow matching loss
        """
        # Extract data endpoints and start states (raw ℝ² × S¹ × ℝ format)
        start_states = batch["raw_start_state"]  # [B, 4] (x, θ, ẋ, θ̇)
        data_endpoints = batch["raw_end_state"]  # [B, 4] (x, θ, ẋ, θ̇)
        
        batch_size = start_states.shape[0]
        device = self.device
        
        # Sample noisy inputs in ℝ² × S¹ × ℝ
        x_noise = self.sample_noisy_input(batch_size, device)
        
        # Sample random times
        t = torch.rand(batch_size, device=device)
        
        # Sample latent vectors
        z = self.sample_latent(batch_size, device)
        
        # Interpolate in ℝ² × S¹ × ℝ between noise and data
        x_t_raw = self.interpolate_r2_s1_r(x_noise, data_endpoints, t)         
        
        x_noise_normalized = self.normalize_state(x_noise)
        
        # Normalize states that will be used by the model  
        x_t_normalized = self.normalize_state(x_t_raw)
        start_normalized = self.normalize_state(start_states)
        data_normalized = self.normalize_state(data_endpoints)
        
        # Compute target velocity from interpolated point to endpoint using normalized states
        target_velocity = self.compute_target_velocity_r2_s1_r(x_noise_normalized, data_normalized, t)  # check: x_end - x_start
        
        
        # Embed normalized states for neural network input  
        x_t_embedded = self.embed_normalized_state(x_t_normalized) # theta -> sin theta, cos theta
        start_embedded = self.embed_normalized_state(start_normalized)
        
        # Predict velocity using the model
        predicted_velocity = self.forward(x_t_embedded, t, z, condition=start_embedded)
        
        # Compute MSE loss between predicted and target velocities
        
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def forward(self, 
                x_t: torch.Tensor, 
                t: torch.Tensor, 
                z: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x_t: Embedded interpolated state [B, 5]
            t: Time parameter [B]
            z: Latent vector [B, 2]
            condition: Embedded start state [B, 5]
            
        Returns:
            Predicted velocity [B, 4] in tangent space
        """
        return self.model(x_t, t, z, condition)
    
    # ========================================================================================
    # UNIFIED INFERENCE METHODS - Used by training, validation, and standalone inference
    # ========================================================================================
    
    def predict_endpoint(self, 
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        **UNIFIED INFERENCE METHOD**
        Predict endpoints from start states using proper ODE integration.
        
        This method is used by:
        - Validation callback during training
        - Standalone inference scripts (generate_lcfm_endpoints.py)
        - Any other inference needs
        
        Args:
            start_states: Start states [B, 4] in raw coordinates (x, θ, ẋ, θ̇)
            num_steps: Number of integration steps for ODE solving
            latent: Optional latent vectors [B, latent_dim]. If None, will sample.
            
        Returns:
            Predicted endpoints [B, 4] in raw coordinates
        """
        batch_size = start_states.shape[0]
        device = start_states.device
        
        # Ensure model is in eval mode for inference
        was_training = self.training
        self.eval()
        
        try:
            with torch.no_grad():
                # Sample noisy inputs using existing method
                x_noise_raw = self.sample_noisy_input(batch_size, device)
                
                # Sample or use provided latent vectors
                if latent is None:
                    z = torch.randn(batch_size, self.latent_dim, device=device)
                else:
                    z = latent
                
                # Normalize states (same as training pipeline)
                x_noise_normalized = self.normalize_state(x_noise_raw)
                start_normalized = self.normalize_state(start_states)
                
                # Embed normalized start states for conditioning
                # start_embedded = self.embed_normalized_state(start_normalized)
                
                # Integration in normalized coordinates
                final_states_normalized = self._integrate_normalized_trajectory(
                    x_noise_normalized, z, start_normalized, num_steps
                )
                
                # Denormalize back to raw coordinates
                final_states_raw = self._denormalize_state(final_states_normalized)
                
                return final_states_normalized, final_states_raw
                
        finally:
            # Restore original training mode'
            if was_training:
                self.train()
    
    def predict_endpoints_batch(self,
                               start_states: torch.Tensor,
                               num_steps: int = 100,
                               num_samples: int = 1) -> torch.Tensor:
        """
        **UNIFIED INFERENCE METHOD**
        Predict multiple endpoint samples per start state (for stochastic models).
        
        Args:
            start_states: Start states [B, 4] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state
            
        Returns:
            Predicted endpoints [B*num_samples, 4] in raw coordinates
        """
        if num_samples == 1:
            return self.predict_endpoint(start_states, num_steps)
        
        batch_size = start_states.shape[0]
        all_endpoints = []
        
        for _ in range(num_samples):
            # Sample different latent vectors for each sample
            endpoints = self.predict_endpoint(start_states, num_steps, latent=None)
            all_endpoints.append(endpoints)
        
        # Concatenate all samples: [B*num_samples, 4]
        return torch.cat(all_endpoints, dim=0)
    
    
    def _integrate_normalized_trajectory(self,
                                       x_start_norm: torch.Tensor,
                                       z: torch.Tensor,
                                       condition_normalized: torch.Tensor,
                                       num_steps: int) -> torch.Tensor:
        """
        Core integration logic in normalized coordinates using Euler method.
        
        Args:
            x_start_norm: Starting normalized state [B, 4]
            z: Latent vectors [B, latent_dim]
            condition_embedded: Embedded conditioning [B, 5]
            num_steps: Number of integration steps
            
        Returns:
            Final normalized states [B, 4]
        """
        batch_size = x_start_norm.shape[0]
        device = x_start_norm.device
        
        # Initialize current state
        x_t = x_start_norm.clone()
        dt = 1.0 / num_steps
        
        condition_embedded = self.embed_normalized_state(condition_normalized)
        
        # Integration loop
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            
            # Embed current normalized state
            x_t_embedded = self.embed_normalized_state(x_t) # theta -> sin theta, cos theta
            
            # Get velocity from model (using existing forward method)
            velocity_normalized = self.model(x_t_embedded, t, z, condition_embedded)
            
            # Update state using proper manifold integration
            # Euclidean components (x, ẋ, θ̇): Standard Euler integration
            x_t[:, [0, 2, 3]] = x_t[:, [0, 2, 3]] + velocity_normalized[:, [0, 2, 3]] * dt
            
            # Circular component (θ): Use Theseus SO(2) exponential map for proper manifold integration
            import theseus as th
            
            # Current angles and angular velocities
            current_angles = x_t[:, 1]  # [B]
            angular_velocities = velocity_normalized[:, 1]  # [B] dθ/dt
            
            # Create SO(2) rotations from current angles
            current_rotations = th.SO2(theta=current_angles)
            
            # Apply exponential map: exp(ω * dt) where ω is angular velocity
            # This gives us the rotation update in the tangent space
            angular_displacement = angular_velocities * dt  # [B]
            
            # Use exp_map to convert tangent vector to rotation
            # For SO(2), the tangent vector is just the scalar angular displacement
            displacement_rotations = th.SO2.exp_map(angular_displacement.unsqueeze(-1))  # [B, 1] -> SO(2)
            
            # Compose rotations: new_rotation = current_rotation ∘ exp(ω * dt)
            new_rotations = current_rotations.compose(displacement_rotations)
            
            # Extract updated angles (already wrapped to [-π, π] by Theseus)
            x_t[:, 1] = new_rotations.theta().squeeze(-1)  # [B, 1] -> [B] - note the method call
        
        return x_t
    
    
    def _denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.
        
        Args:
            normalized_state: [B, 4] normalized state (x_norm, theta, x_dot_norm, theta_dot_norm)
            
        Returns:
            [B, 4] raw state (x, theta, x_dot, theta_dot)
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]  # Already in natural coordinates [-π, π]
        x_dot_norm = normalized_state[:, 2] 
        theta_dot_norm = normalized_state[:, 3]
        
        # Denormalize using system bounds
        x = x_norm * self.system.cart_limit
        x_dot = x_dot_norm * self.system.velocity_limit
        theta_dot = theta_dot_norm * self.system.angular_velocity_limit
        
        return torch.stack([x, theta, x_dot, theta_dot], dim=1)
    
    # ========================================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ========================================================================================
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        **UNIFIED CHECKPOINT LOADING**
        Load a trained CartPole LCFM model from checkpoint for inference.
        
        Loads configuration from both Hydra config and Lightning hparams for robustness:
        1. First tries .hydra/config.yaml (most complete)
        2. Falls back to checkpoint hyperparameters
        3. Combines both sources for model instantiation
        
        Usage:
            model = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint("path/to/model.ckpt")
            endpoints = model.predict_endpoint(start_states)
        
        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt)
            device: Device to load model on ("cuda", "cpu", or None for auto)
            
        Returns:
            Loaded model ready for inference
        """
        import torch
        import yaml
        from pathlib import Path
        from omegaconf import OmegaConf
        from ...systems.cartpole_lcfm import CartPoleSystemLCFM
        from ...model.cartpole_latent_conditional_unet1d import CartPoleLatentConditionalUNet1D
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint_path = Path(checkpoint_path)
        print(f"🤖 Loading CartPole LCFM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")
        
        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Find the training directory (parent of checkpoints/)
        # Structure: .../timestamp/checkpoints/model.ckpt
        if checkpoint_path.parent.name == "checkpoints":
            training_dir = checkpoint_path.parent.parent
        else:
            # Assume checkpoint is in training directory
            training_dir = checkpoint_path.parent
        
        print(f"🗂️  Training directory: {training_dir}")
        
        # ========================================================================
        # LOAD HYDRA CONFIG (Primary source - most complete)
        # ========================================================================
        hydra_config = None
        hydra_config_path = training_dir / ".hydra" / "config.yaml"
        
        if hydra_config_path.exists():
            try:
                print(f"📋 Loading Hydra config: {hydra_config_path}")
                with open(hydra_config_path, 'r') as f:
                    hydra_config = yaml.safe_load(f)
                print("✅ Hydra config loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load Hydra config: {e}")
                hydra_config = None
        else:
            print(f"⚠️  Hydra config not found at: {hydra_config_path}")
        
        # ========================================================================
        # LOAD LIGHTNING CHECKPOINT (Fallback source)
        # ========================================================================
        print(f"📦 Loading Lightning checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        print("✅ Lightning checkpoint loaded")
        
        # ========================================================================
        # COMBINE CONFIGURATION SOURCES
        # ========================================================================
        
        # Extract latent_dim (prefer Lightning hparams)
        latent_dim = hparams.get("latent_dim")
        if latent_dim is None and hydra_config:
            latent_dim = hydra_config.get("flow_matching", {}).get("latent_dim", 2)
        if latent_dim is None:
            latent_dim = 2
            print(f"⚠️  Using default latent_dim: {latent_dim}")
        
        # Extract model config (prefer Lightning hparams, supplement with Hydra)
        model_config = hparams.get("config", {})
        if not model_config and hydra_config:
            model_config = hydra_config.get("model", {})
        
        # Ensure latent_dim is set in model config
        model_config["latent_dim"] = latent_dim
        
        print(f"📋 Final config - latent_dim: {latent_dim}")
        print(f"📋 Model config keys: {list(model_config.keys())}")
        
        # ========================================================================
        # INITIALIZE SYSTEM AND MODEL
        # ========================================================================
        
        # Initialize system (try to restore from hparams first)
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new CartPole system (not found in hparams)")
            system = CartPoleSystemLCFM()
        else:
            print("✅ Restored CartPole system from checkpoint")
        
        # Create model architecture with robust defaults
        model = CartPoleLatentConditionalUNet1D(
            embedded_dim=model_config.get('embedded_dim', 5),
            latent_dim=model_config.get('latent_dim', latent_dim),
            condition_dim=model_config.get('condition_dim', 5),
            time_emb_dim=model_config.get('time_emb_dim', 64),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 256]),
            output_dim=model_config.get('output_dim', 4),
            use_input_embeddings=model_config.get('use_input_embeddings', False),
            input_emb_dim=model_config.get('input_emb_dim', 64)
        )
        
        # Create flow matcher instance (no optimizer/scheduler needed for inference)
        flow_matcher = cls(
            system=system,
            model=model,
            optimizer=None,  # Not needed for inference
            scheduler=None,  # Not needed for inference
            config=model_config,
            latent_dim=latent_dim
        )
        
        # ========================================================================
        # LOAD MODEL WEIGHTS
        # ========================================================================
        
        print("🔄 Loading model state dict...")
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        
        if not model_state_dict:
            raise ValueError("No model weights found in checkpoint! Keys: " + str(list(state_dict.keys())[:10]))
        
        flow_matcher.model.load_state_dict(model_state_dict)
        
        # Move to device and set eval mode
        flow_matcher = flow_matcher.to(device)
        flow_matcher.eval()
        
        # ========================================================================
        # SUCCESS SUMMARY
        # ========================================================================
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Checkpoint: {checkpoint_path.name}")
        print(f"   Config sources: {'Hydra + Lightning' if hydra_config else 'Lightning only'}")
        print(f"   System: {type(system).__name__}")
        print(f"   System bounds: cart±{system.cart_limit:.1f}, vel±{system.velocity_limit:.1f}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        
        return flow_matcher