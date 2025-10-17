# 🚀 Adding a New System to Facebook Flow Matching

**Complete Guide for Implementing Flow Matching on a New Dynamical System**

This guide walks you through adding support for a new system (e.g., Acrobot, Double Pendulum, Quadcopter) to the unified Facebook Flow Matching training framework.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [File Checklist](#file-checklist)
5. [Testing](#testing)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### **What You'll Create**

For a new system (e.g., "Acrobot"), you'll create:

1. **Manifold** - Defines the geometry (e.g., S¹×S¹×ℝ²)
2. **System** - Defines system parameters and state embedding
3. **Flow Matcher** - Implements Facebook FM with your manifold
4. **Model** - Neural network architecture
5. **Data Module** - Handles dataset loading
6. **Config** - YAML configuration file
7. **Tests** - Validation tests

Then you can train by running:
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_acrobot_lcfm
```

---

## Prerequisites

### **Understand Your System**

Before starting, answer these questions:

1. **State space structure**: What is the manifold?
   - Example: Pendulum is S¹×ℝ (angle + angular velocity)
   - Example: CartPole is ℝ²×S¹×ℝ (position, angle, velocities)
   - Example: Acrobot is S¹×S¹×ℝ² (two angles + two angular velocities)

2. **Which components are circular?**
   - Identify which state variables wrap around (e.g., angles)
   - These require special geodesic treatment

3. **Normalization bounds**: What are the limits?
   - Position limits: [-x_max, x_max]
   - Velocity limits: [-v_max, v_max]
   - Angles are always in [-π, π]

4. **State embedding**: How to make it neural network friendly?
   - Angles: θ → (sin θ, cos θ)
   - Linear quantities: normalize to [-1, 1]

---

## Step-by-Step Implementation

### **Step 1: Define the Manifold**

**File:** `src/utils/fb_manifolds.py`

Add a new manifold class for your system's geometry.

#### **Template:**

```python
class AcrobotManifold(Manifold):
    """
    S¹ × S¹ × ℝ² manifold for Acrobot system

    State: (θ₁, θ₂, θ̇₁, θ̇₂) where:
    - θ₁, θ₂ ∈ S¹: angles (wrapped to [-π, π])
    - θ̇₁, θ̇₂ ∈ ℝ: angular velocities (normalized)
    """

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Exponential map: move from x along tangent vector u

        Args:
            x: Point on manifold [B, 4] = (θ₁, θ₂, θ̇₁, θ̇₂)
            u: Tangent vector [B, 4]

        Returns:
            Transported point [B, 4]
        """
        # Extract components
        theta1 = x[..., 0:1]     # First angle (S¹)
        theta2 = x[..., 1:2]     # Second angle (S¹)
        omega1 = x[..., 2:3]     # First angular velocity (ℝ)
        omega2 = x[..., 3:4]     # Second angular velocity (ℝ)

        u_theta1 = u[..., 0:1]
        u_theta2 = u[..., 1:2]
        u_omega1 = u[..., 2:3]
        u_omega2 = u[..., 3:4]

        # S¹ components: wrap to [-π, π]
        new_theta1 = torch.atan2(
            torch.sin(theta1 + u_theta1),
            torch.cos(theta1 + u_theta1)
        )
        new_theta2 = torch.atan2(
            torch.sin(theta2 + u_theta2),
            torch.cos(theta2 + u_theta2)
        )

        # ℝ components: standard addition
        new_omega1 = omega1 + u_omega1
        new_omega2 = omega2 + u_omega2

        return torch.cat([new_theta1, new_theta2, new_omega1, new_omega2], dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector from x to y

        Args:
            x: Source point [B, 4]
            y: Target point [B, 4]

        Returns:
            Tangent vector [B, 4] from x to y
        """
        # Extract components
        theta1_x, theta2_x, omega1_x, omega2_x = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
        theta1_y, theta2_y, omega1_y, omega2_y = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

        # S¹: shortest angular distance (wrapped)
        delta_theta1 = torch.atan2(
            torch.sin(theta1_y - theta1_x),
            torch.cos(theta1_y - theta1_x)
        )
        delta_theta2 = torch.atan2(
            torch.sin(theta2_y - theta2_x),
            torch.cos(theta2_y - theta2_x)
        )

        # ℝ: standard difference
        delta_omega1 = omega1_y - omega1_x
        delta_omega2 = omega2_y - omega2_x

        return torch.cat([delta_theta1, delta_theta2, delta_omega1, delta_omega2], dim=-1)

    def projx(self, x: Tensor) -> Tensor:
        """Project point onto manifold (wrap angles to [-π, π])"""
        theta1 = torch.atan2(torch.sin(x[..., 0:1]), torch.cos(x[..., 0:1]))
        theta2 = torch.atan2(torch.sin(x[..., 1:2]), torch.cos(x[..., 1:2]))
        omega1 = x[..., 2:3]
        omega2 = x[..., 3:4]
        return torch.cat([theta1, theta2, omega1, omega2], dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project vector onto tangent space (identity for product manifolds)"""
        return u
```

#### **Key Principles:**

1. **S¹ components (angles)**:
   - Use `atan2(sin(...), cos(...))` for wrapping
   - Compute shortest angular distance in `logmap`

2. **ℝ components (linear quantities)**:
   - Simple addition/subtraction
   - No special treatment

3. **Product manifolds**:
   - `proju` is usually identity
   - Each component handled independently

---

### **Step 2: Create the System Class**

**File:** `src/systems/acrobot_lcfm.py`

Define system parameters and state transformations.

#### **Template:**

```python
class AcrobotSystemLCFM:
    """
    Acrobot system for Latent Conditional Flow Matching

    State: (θ₁, θ₂, θ̇₁, θ̇₂)
    - Angles in [-π, π]
    - Angular velocities normalized
    """

    def __init__(self):
        # Load or define system bounds
        self.angle_velocity_limit_1 = 10.0  # Max angular velocity for joint 1
        self.angle_velocity_limit_2 = 10.0  # Max angular velocity for joint 2

        # Data bounds (load from saved file if available)
        self.load_bounds()

    def load_bounds(self):
        """Load data bounds from file or use defaults"""
        import pickle
        from pathlib import Path

        bounds_file = Path("/path/to/acrobot_data_bounds.pkl")

        if bounds_file.exists():
            with open(bounds_file, 'rb') as f:
                bounds = pickle.load(f)
                self.angle_velocity_limit_1 = bounds['theta_dot_1_max']
                self.angle_velocity_limit_2 = bounds['theta_dot_2_max']
        else:
            print("Using default bounds")

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state for neural network

        Args:
            state: [B, 4] raw state (θ₁, θ₂, θ̇₁, θ̇₂)

        Returns:
            [B, 4] normalized state (angles unchanged, velocities normalized)
        """
        theta1 = state[:, 0]
        theta2 = state[:, 1]
        omega1 = state[:, 2]
        omega2 = state[:, 3]

        # Normalize angular velocities to [-1, 1]
        omega1_norm = omega1 / self.angle_velocity_limit_1
        omega2_norm = omega2 / self.angle_velocity_limit_2

        return torch.stack([theta1, theta2, omega1_norm, omega2_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Denormalize back to raw coordinates"""
        theta1 = normalized_state[:, 0]
        theta2 = normalized_state[:, 1]
        omega1_norm = normalized_state[:, 2]
        omega2_norm = normalized_state[:, 3]

        # Denormalize angular velocities
        omega1 = omega1_norm * self.angle_velocity_limit_1
        omega2 = omega2_norm * self.angle_velocity_limit_2

        return torch.stack([theta1, theta2, omega1, omega2], dim=1)

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed state for neural network input

        Args:
            state: [B, 4] (θ₁, θ₂, θ̇₁_norm, θ̇₂_norm)

        Returns:
            [B, 6] (sin θ₁, cos θ₁, sin θ₂, cos θ₂, θ̇₁_norm, θ̇₂_norm)
        """
        theta1 = state[:, 0]
        theta2 = state[:, 1]
        omega1_norm = state[:, 2]
        omega2_norm = state[:, 3]

        # Embed angles as sin/cos
        sin_theta1 = torch.sin(theta1)
        cos_theta1 = torch.cos(theta1)
        sin_theta2 = torch.sin(theta2)
        cos_theta2 = torch.cos(theta2)

        return torch.stack([
            sin_theta1, cos_theta1,
            sin_theta2, cos_theta2,
            omega1_norm, omega2_norm
        ], dim=1)

    def __str__(self):
        return (f"AcrobotSystemLCFM("
                f"ω₁_max={self.angle_velocity_limit_1:.1f}, "
                f"ω₂_max={self.angle_velocity_limit_2:.1f})")
```

---

### **Step 3: Create the Flow Matcher**

**File:** `src/flow_matching/acrobot_latent_conditional/flow_matcher_fb.py`

Implement the Facebook FM flow matcher for your system.

#### **Template:**

```python
"""
Acrobot Latent Conditional Flow Matching using Facebook Flow Matching
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
sys.path.append('/path/to/flow_matching')

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper

from ..base.flow_matcher import BaseFlowMatcher
from ...systems.base import DynamicalSystem
from ...utils.fb_manifolds import AcrobotManifold


class AcrobotLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Acrobot Latent Conditional Flow Matching using Facebook FM

    State space: S¹×S¹×ℝ² (two angles + two angular velocities)
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[dict] = None,
                 latent_dim: int = 2):
        self.system = system
        self.latent_dim = latent_dim
        super().__init__(model, optimizer, scheduler, config)

        # Facebook FM components
        self.manifold = AcrobotManifold()
        self.path = GeodesicProbPath(
            scheduler=CondOTScheduler(),
            manifold=self.manifold
        )

        print("✅ Initialized with Facebook Flow Matching:")
        print(f"   - Manifold: AcrobotManifold (S¹×S¹×ℝ²)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noisy input in S¹×S¹×ℝ² space"""
        # Angles: uniform in [-π, π]
        theta1 = torch.rand(batch_size, 1, device=device) * 2 * torch.pi - torch.pi
        theta2 = torch.rand(batch_size, 1, device=device) * 2 * torch.pi - torch.pi

        # Angular velocities: uniform in [-limit, limit]
        omega1 = torch.rand(batch_size, 1, device=device) * (2 * self.system.angle_velocity_limit_1) - self.system.angle_velocity_limit_1
        omega2 = torch.rand(batch_size, 1, device=device) * (2 * self.system.angle_velocity_limit_2) - self.system.angle_velocity_limit_2

        return torch.cat([theta1, theta2, omega1, omega2], dim=1)

    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample Gaussian latent vector"""
        return torch.randn(batch_size, self.latent_dim, device=device)

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute flow matching loss using Facebook FM"""
        # Extract data
        start_states = batch["raw_start_state"]  # [B, 4]
        data_endpoints = batch["raw_end_state"]  # [B, 4]

        batch_size = start_states.shape[0]
        device = self.device

        # Sample noise, time, latent
        x_noise = self.sample_noisy_input(batch_size, device)
        t = torch.rand(batch_size, device=device)
        z = self.sample_latent(batch_size, device)

        # Normalize for model
        x_noise_normalized = self.system.normalize_state(x_noise)
        data_normalized = self.system.normalize_state(data_endpoints)
        start_normalized = self.system.normalize_state(start_states)

        # AUTOMATIC geodesic interpolation + velocity via Facebook FM
        path_sample = self.path.sample(
            x_0=x_noise_normalized,
            x_1=data_normalized,
            t=t
        )

        # Embed for neural network
        x_t_embedded = self.system.embed_state(path_sample.x_t)
        start_embedded = self.system.embed_state(start_normalized)

        # Predict velocity
        predicted_velocity = self.forward(x_t_embedded, t, z, condition=start_embedded)

        # AUTOMATIC target velocity (via autodiff!)
        target_velocity = path_sample.dx_t

        # Loss
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x_t, t, z, condition)

    def predict_endpoint(self, start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None,
                        method: str = "euler") -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict endpoints using RiemannianODESolver"""
        batch_size = start_states.shape[0]
        device = start_states.device

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Sample noise and latent
                x_noise = self.sample_noisy_input(batch_size, device)
                x_noise_normalized = self.system.normalize_state(x_noise)

                if latent is None:
                    z = torch.randn(batch_size, self.latent_dim, device=device)
                else:
                    z = latent

                # Prepare conditioning
                start_normalized = self.system.normalize_state(start_states)
                start_embedded = self.system.embed_state(start_normalized)

                # Create velocity model wrapper
                velocity_model = _AcrobotVelocityModelWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_embedded,
                    embed_fn=lambda x: self.system.embed_state(x)
                )

                # Solve ODE with manifold projection
                solver = RiemannianODESolver(
                    manifold=self.manifold,
                    velocity_model=velocity_model
                )

                final_states_normalized = solver.sample(
                    x_init=x_noise_normalized,
                    step_size=1.0/num_steps,
                    method=method,
                    projx=True,
                    proju=True,
                    time_grid=torch.tensor([0.0, 1.0], device=device)
                )

                # Denormalize
                final_states_raw = self.system.denormalize_state(final_states_normalized)

                return final_states_normalized, final_states_raw

        finally:
            if was_training:
                self.train()


class _AcrobotVelocityModelWrapper(ModelWrapper):
    """Model wrapper for RiemannianODESolver"""

    def __init__(self, model: nn.Module, latent: torch.Tensor,
                 condition: torch.Tensor, embed_fn):
        super().__init__(model)
        self.latent = latent
        self.condition = condition
        self.embed_fn = embed_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        x_embedded = self.embed_fn(x)

        batch_size = x.shape[0]
        z = self.latent.expand(batch_size, -1) if self.latent.shape[0] == 1 else self.latent
        cond = self.condition.expand(batch_size, -1) if self.condition.shape[0] == 1 else self.condition

        return self.model(x_embedded, t, z, cond)
```

---

### **Step 4: Create the Model**

**File:** `src/model/acrobot_latent_conditional_unet1d.py`

Create a UNet model for your system.

#### **Template:**

```python
"""
1D UNet for Acrobot Latent Conditional Flow Matching
"""
import torch
import torch.nn as nn


class AcrobotLatentConditionalUNet1D(nn.Module):
    """
    UNet for Acrobot flow matching

    Input: embedded state (6D) + time (64D) + latent (2D) + condition (6D)
    Output: velocity in tangent space (4D)
    """

    def __init__(self,
                 embedded_dim: int = 6,      # (sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁, ω₂)
                 latent_dim: int = 2,
                 condition_dim: int = 6,
                 time_emb_dim: int = 64,
                 hidden_dims: list = [256, 512, 256],
                 output_dim: int = 4):       # (dθ₁, dθ₂, dω₁, dω₂)

        super().__init__()

        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU()
        )

        # Total input dimension
        total_input_dim = embedded_dim + time_emb_dim + latent_dim + condition_dim

        # UNet architecture
        layers = []
        prev_dim = total_input_dim

        # Encoder
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedded state [B, 6]
            t: Time [B] or [B, 1]
            z: Latent [B, latent_dim]
            condition: Embedded start state [B, 6]

        Returns:
            Velocity [B, 4]
        """
        # Ensure t is [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Time embedding
        t_emb = self.time_mlp(t)

        # Concatenate all inputs
        combined = torch.cat([x, t_emb, z, condition], dim=-1)

        # Forward through network
        return self.network(combined)

    def get_model_info(self):
        """Return model information"""
        return {
            'model_type': self.__class__.__name__,
            'embedded_dim': self.embedded_dim,
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'time_emb_dim': self.time_emb_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
```

---

### **Step 5: Create the Data Module**

**File:** `src/data/acrobot_endpoint_data.py`

#### **Template:**

```python
"""
Data module for Acrobot endpoint datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class AcrobotEndpointDataset(Dataset):
    """Dataset for Acrobot endpoints"""

    def __init__(self, data_file: str):
        self.data = self.load_data(data_file)

    def load_data(self, data_file: str):
        """Load endpoint data from file"""
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                # Assuming format: θ₁_start, θ₂_start, ω₁_start, ω₂_start,
                #                  θ₁_end, θ₂_end, ω₁_end, ω₂_end
                start_state = values[:4]
                end_state = values[4:8]
                data.append({
                    'raw_start_state': torch.tensor(start_state, dtype=torch.float32),
                    'raw_end_state': torch.tensor(end_state, dtype=torch.float32)
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AcrobotEndpointDataModule(pl.LightningDataModule):
    """Lightning data module for Acrobot endpoints"""

    def __init__(self, data_file: str, batch_size: int = 256,
                 num_workers: int = 4, train_split: float = 0.8,
                 val_split: float = 0.1, shuffle: bool = True):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.shuffle = shuffle

    def setup(self, stage=None):
        """Setup train/val/test splits"""
        full_dataset = AcrobotEndpointDataset(self.data_file)

        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
```

---

### **Step 6: Create the Config File**

**File:** `configs/train_acrobot_lcfm.yaml`

#### **Template:**

```yaml
# Acrobot Latent Conditional Flow Matching (Facebook FM)
# Uses: S¹×S¹×ℝ² manifold, GeodesicProbPath, RiemannianODESolver

name: acrobot_latent_conditional_fm
seed: 42
batch_size: 256
base_lr: 1e-4
num_workers: 4

# System: Acrobot dynamics with S¹×S¹×ℝ² state space
system:
  _target_: src.systems.acrobot_lcfm.AcrobotSystemLCFM

# Flow Matcher: Facebook FM version with automatic geodesics
flow_matcher:
  _target_: src.flow_matching.acrobot_latent_conditional.flow_matcher_fb.AcrobotLatentConditionalFlowMatcher

# Model: Latent Conditional UNet for Acrobot
model:
  _target_: src.model.acrobot_latent_conditional_unet1d.AcrobotLatentConditionalUNet1D
  embedded_dim: 6              # (sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁, ω₂)
  latent_dim: 2
  condition_dim: 6
  time_emb_dim: 64
  hidden_dims: [256, 512, 256]
  output_dim: 4                # (dθ₁, dθ₂, dω₁, dω₂)

# Data: Acrobot endpoint dataset
data:
  _target_: src.data.acrobot_endpoint_data.AcrobotEndpointDataModule
  data_file: /path/to/acrobot_endpoints.txt
  batch_size: ${batch_size}
  num_workers: ${num_workers}
  train_split: 0.8
  val_split: 0.1
  shuffle: true

# Optimizer: AdamW
optimizer:
  _target_: torch.optim.AdamW
  lr: ${base_lr}
  weight_decay: 1e-5
  betas: [0.9, 0.999]

# Scheduler: Reduce on plateau
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10
  verbose: true
  min_lr: 1e-6

# Trainer: PyTorch Lightning
trainer:
  _target_: lightning.pytorch.Trainer
  max_epochs: 100
  accelerator: gpu
  devices: [1]
  precision: 32
  gradient_clip_val: 1.0
  log_every_n_steps: 10
  check_val_every_n_epoch: 1
  enable_progress_bar: true
  logger:
    _target_: lightning.pytorch.loggers.TensorBoardLogger
    save_dir: outputs/${name}
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: val_loss
      mode: min
      save_top_k: 3
      filename: "epoch{epoch:02d}-val_loss{val_loss:.4f}"
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: val_loss
      mode: min
      patience: 20

# Flow matching settings
flow_matching:
  latent_dim: 2
  noise_distribution: uniform
  num_integration_steps: 100

# Hydra output
hydra:
  run:
    dir: outputs/${name}/${now:%Y-%m-%d_%H-%M-%S}
```

---

### **Step 7: Create Test File**

**File:** `test_acrobot_fb_fm.py`

#### **Template:**

```python
"""
Test script for Acrobot Facebook Flow Matching
"""
import torch
import sys
import math
sys.path.append('/path/to/flow_matching')

from src.utils.fb_manifolds import AcrobotManifold
from src.flow_matching.acrobot_latent_conditional.flow_matcher_fb import AcrobotLatentConditionalFlowMatcher
from src.systems.acrobot_lcfm import AcrobotSystemLCFM
from src.model.acrobot_latent_conditional_unet1d import AcrobotLatentConditionalUNet1D
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler


def test_manifold():
    """Test manifold operations"""
    print("="*80)
    print("TEST 1: AcrobotManifold Operations")
    print("="*80)

    manifold = AcrobotManifold()

    # Test geodesic
    x = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([[1.0, 0.5, 2.0, 3.0]])

    tangent = manifold.logmap(x, y)
    print(f"✓ Logmap: {tangent}")

    halfway = manifold.expmap(x, tangent * 0.5)
    print(f"✓ Expmap halfway: {halfway}")

    # Test wrapping
    out_of_range = torch.tensor([[3.5, -4.0, 0.0, 0.0]])
    wrapped = manifold.projx(out_of_range)
    print(f"✓ Wrapped angles: {wrapped}")

    print("✅ Manifold operations PASSED\n")


def test_flow_matcher():
    """Test flow matcher"""
    print("="*80)
    print("TEST 2: AcrobotLatentConditionalFlowMatcher")
    print("="*80)

    system = AcrobotSystemLCFM()
    model = AcrobotLatentConditionalUNet1D()

    flow_matcher = AcrobotLatentConditionalFlowMatcher(
        system=system,
        model=model,
        optimizer=None,
        scheduler=None,
        latent_dim=2
    )

    # Test loss
    batch = {
        "raw_start_state": torch.randn(4, 4),
        "raw_end_state": torch.randn(4, 4),
    }

    loss = flow_matcher.compute_flow_loss(batch)
    print(f"✓ Loss: {loss.item():.6f}")

    print("✅ Flow matcher PASSED\n")


def test_inference():
    """Test inference"""
    print("="*80)
    print("TEST 3: Inference")
    print("="*80)

    system = AcrobotSystemLCFM()
    model = AcrobotLatentConditionalUNet1D()

    flow_matcher = AcrobotLatentConditionalFlowMatcher(
        system=system,
        model=model,
        optimizer=None,
        scheduler=None,
        latent_dim=2
    )

    flow_matcher.eval()

    start_states = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        _, pred = flow_matcher.predict_endpoint(start_states, num_steps=50, method="euler")

    print(f"✓ Prediction: {pred}")

    print("✅ Inference PASSED\n")


if __name__ == "__main__":
    test_manifold()
    test_flow_matcher()
    test_inference()
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
```

---

## File Checklist

- [ ] **Manifold**: `src/utils/fb_manifolds.py` (add AcrobotManifold)
- [ ] **System**: `src/systems/acrobot_lcfm.py`
- [ ] **Flow Matcher**: `src/flow_matching/acrobot_latent_conditional/flow_matcher_fb.py`
- [ ] **Model**: `src/model/acrobot_latent_conditional_unet1d.py`
- [ ] **Data Module**: `src/data/acrobot_endpoint_data.py`
- [ ] **Config**: `configs/train_acrobot_lcfm.yaml`
- [ ] **Test**: `test_acrobot_fb_fm.py`

---

## Testing

### **Run Tests**
```bash
python test_acrobot_fb_fm.py
```

### **Train**
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_acrobot_lcfm
```

---

## Common Patterns

### **Manifold Structure by System**

| System | Manifold | State Format | Embedded Dim |
|--------|----------|--------------|--------------|
| Pendulum | S¹×ℝ | (θ, θ̇) | 3 |
| CartPole | ℝ²×S¹×ℝ | (x, θ, ẋ, θ̇) | 5 |
| Acrobot | S¹×S¹×ℝ² | (θ₁, θ₂, θ̇₁, θ̇₂) | 6 |
| Double Pendulum | S¹×S¹×ℝ² | (θ₁, θ₂, θ̇₁, θ̇₂) | 6 |

### **State Dimension Calculation**

```
Embedded Dim = (# angles × 2) + (# linear quantities)

Examples:
- Pendulum: 1 angle × 2 + 1 velocity = 3
- CartPole: 1 angle × 2 + 3 linear = 5
- Acrobot: 2 angles × 2 + 2 velocities = 6
```

---

## Troubleshooting

### **Issue: Angles not wrapping correctly**
**Solution**: Ensure `projx` uses `atan2(sin(θ), cos(θ))` for all angles

### **Issue: Loss is NaN**
**Checklist**:
- [ ] Check state normalization ranges
- [ ] Verify embedded_dim matches system.embed_state output
- [ ] Check output_dim matches state dimension
- [ ] Reduce learning rate

### **Issue: Model doesn't learn**
**Checklist**:
- [ ] Verify geodesic computation (test manifold separately)
- [ ] Check data format (start_state, end_state correct?)
- [ ] Verify latent_dim is consistent across model and flow_matcher
- [ ] Check condition_dim matches embedded_dim

---

## Summary

To add a new system:

1. ✅ Define the **manifold** (geometry)
2. ✅ Create the **system** (parameters, embedding)
3. ✅ Implement the **flow matcher** (Facebook FM)
4. ✅ Design the **model** (UNet architecture)
5. ✅ Setup the **data module** (dataset loading)
6. ✅ Write the **config** (Hydra YAML)
7. ✅ Create **tests** (validation)
8. ✅ **Train**!

Then just run:
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_yoursystem_lcfm
```

**No changes to the training script needed!** 🎉
