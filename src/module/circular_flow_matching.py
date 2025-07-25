import torch
import torch.nn as nn
import lightning.pytorch as pl
import numpy as np
from torchmetrics import MeanMetric

class CircularFlowMatching(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        
        # The circular-aware neural network model
        self.model = model
        
        # Optimizer and scheduler
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x, t, condition):
        """
        Forward pass
        x: current state on S¹ × ℝ [batch_size, 3]
        t: time [batch_size]
        condition: start state [batch_size, 3]
        """
        return self.model(x, t, condition)
    
    def embed_state(self, state):
        """Convert (θ, θ̇) → (sin(θ), cos(θ), θ̇)"""
        theta, theta_dot = state[..., 0], state[..., 1]
        return torch.stack([torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)
    
    def extract_state(self, embedded):
        """Convert (sin(θ), cos(θ), θ̇) → (θ, θ̇)"""
        sin_theta, cos_theta, theta_dot = embedded[..., 0], embedded[..., 1], embedded[..., 2]
        theta = torch.atan2(sin_theta, cos_theta)
        return torch.stack([theta, theta_dot], dim=-1)
    
    def circular_distance(self, theta1, theta2):
        """Compute circular distance between angles"""
        diff = theta1 - theta2
        return torch.atan2(torch.sin(diff), torch.cos(diff))
    
    def geodesic_interpolation(self, x0_embedded, x1_embedded, t):
        """
        Interpolate on S¹ × ℝ using geodesics
        
        Args:
            x0_embedded: start state (sin(θ₀), cos(θ₀), θ̇₀)
            x1_embedded: end state (sin(θ₁), cos(θ₁), θ̇₁)
            t: interpolation parameter [0,1]
            
        Returns:
            x_t: interpolated state
            target_velocity: target velocity field
        """
        batch_size = x0_embedded.shape[0]
        
        # Extract angles from embedded representation
        theta0 = torch.atan2(x0_embedded[..., 0], x0_embedded[..., 1])  # [batch_size]
        theta1 = torch.atan2(x1_embedded[..., 0], x1_embedded[..., 1])  # [batch_size]
        
        # Compute shortest angular path
        angular_diff = self.circular_distance(theta1, theta0)  # [batch_size]
        
        # Ensure t has correct shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1).squeeze(-1)  # Ensure [batch_size] shape
        
        # Interpolate angle along geodesic (shortest path on circle)
        theta_t = theta0 + t * angular_diff  # [batch_size]
        
        # Linear interpolation for angular velocity (on ℝ)
        theta_dot_0 = x0_embedded[..., 2]  # [batch_size]
        theta_dot_1 = x1_embedded[..., 2]  # [batch_size]
        theta_dot_t = (1 - t) * theta_dot_0 + t * theta_dot_1  # [batch_size]
        
        # Convert interpolated state back to embedded form
        x_t = torch.stack([torch.sin(theta_t), torch.cos(theta_t), theta_dot_t], dim=-1)
        
        # Compute target velocity (time derivative of geodesic)
        # d/dt [sin(θ_t), cos(θ_t), θ̇_t]
        dtheta_dt = angular_diff  # dθ/dt
        dtheta_dot_dt = theta_dot_1 - theta_dot_0  # dθ̇/dt
        
        # Chain rule: d/dt sin(θ_t) = cos(θ_t) * dθ/dt
        target_velocity = torch.stack([
            torch.cos(theta_t) * dtheta_dt,    # d/dt sin(θ_t)
            -torch.sin(theta_t) * dtheta_dt,   # d/dt cos(θ_t)  
            dtheta_dot_dt                      # d/dt θ̇_t
        ], dim=-1)
        
        return x_t, target_velocity
    
    def training_step(self, batch, batch_idx):
        # Get embedded states (already converted by dataset)
        start_states = batch["start_state"]  # (sin,cos,θ̇) [batch_size, 3]
        end_states = batch["end_state"]      # (sin,cos,θ̇) [batch_size, 3]
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Get geodesic interpolation and target velocity
        x_t, target_velocity = self.geodesic_interpolation(start_states, end_states, t)
        
        # Predict velocity using our circular-aware model
        pred_velocity = self.model(x_t, t, condition=start_states)
        
        # Compute MSE loss between predicted and target velocity
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        
        # Log metrics
        self.train_loss.update(loss)
        self.log("train_loss", self.train_loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get embedded states
        start_states = batch["start_state"]
        end_states = batch["end_state"]
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Get geodesic interpolation and target velocity
        x_t, target_velocity = self.geodesic_interpolation(start_states, end_states, t)
        
        # Predict velocity
        pred_velocity = self.model(x_t, t, condition=start_states)
        
        # Compute loss
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        
        # Log metrics
        self.val_loss.update(loss)
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test by sampling from the learned flow"""
        start_states = batch["start_state"]
        end_states = batch["end_state"]
        
        # Generate samples using the learned flow
        generated_ends = self.sample(start_states, num_steps=100)
        
        # Compute MSE between generated and true endpoints
        mse_loss = nn.functional.mse_loss(generated_ends, end_states)
        
        self.log("test_mse", mse_loss, on_epoch=True, prog_bar=True)
        
        return {"test_mse": mse_loss, "generated": generated_ends, "true": end_states}
    
    @torch.no_grad()
    def sample(self, start_states, num_steps=100):
        """
        Sample from the learned circular flow
        start_states: [batch_size, 3] - conditioning states (embedded)
        """
        batch_size = start_states.shape[0]
        device = start_states.device
        
        # Start from the actual start states (not random noise)
        x = start_states.clone()
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(batch_size, device=device) * i * dt
            
            # Predict velocity
            with torch.no_grad():
                velocity = self.model(x, t, condition=start_states)
            
            # Euler step
            x = x + velocity * dt
            
            # Project back to S¹ × ℝ manifold
            x = self.project_to_manifold(x)
        
        return x
    
    def project_to_manifold(self, x):
        """Project back to S¹ × ℝ manifold"""
        sin_theta, cos_theta, theta_dot = x[..., 0], x[..., 1], x[..., 2]
        
        # Normalize (sin, cos) to unit circle
        norm = torch.sqrt(sin_theta**2 + cos_theta**2)
        sin_theta_proj = sin_theta / (norm + 1e-8)
        cos_theta_proj = cos_theta / (norm + 1e-8)
        
        # θ̇ component stays unchanged (no constraint on ℝ)
        return torch.stack([sin_theta_proj, cos_theta_proj, theta_dot], dim=-1)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.parameters())
        scheduler = self.scheduler_partial(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
            },
        }