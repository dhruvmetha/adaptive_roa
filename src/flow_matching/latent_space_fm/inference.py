import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

from ..base.inference import BaseFlowMatchingInference

def embed(theta_theta_dot: torch.Tensor) -> torch.Tensor:
    """
    (θ, θ̇) -> (sinθ, cosθ, θ̇)  ; accepts [B,2] or [2]
    """
    if theta_theta_dot.dim() == 1:
        theta_theta_dot = theta_theta_dot[None, :]
    theta = theta_theta_dot[:, 0]
    theta_dot = theta_theta_dot[:, 1]
    return torch.stack([torch.sin(theta), torch.cos(theta), theta_dot], dim=1)

def extract(x_emb: torch.Tensor) -> torch.Tensor:
    """
    (sinθ, cosθ, θ̇) -> (θ, θ̇) with safe atan2; returns [B,2]
    """
    if x_emb.dim() == 1:
        x_emb = x_emb[None, :]
    s, c, thdot = x_emb[:, 0], x_emb[:, 1], x_emb[:, 2]
    
    # Safe atan2 with normalization to handle numerical issues
    norm = torch.sqrt(s**2 + c**2).clamp_min(1e-8)
    s_norm = s / norm
    c_norm = c / norm
    theta = torch.atan2(s_norm, c_norm)
    
    return torch.stack([theta, thdot], dim=1)

class LatentCircularFlowMatchingInference(BaseFlowMatchingInference):
    """
    Inference wrapper for latent circular FM.
    """
    def _load_model(self, model: nn.Module, checkpoint_path: Optional[str] = None) -> nn.Module:
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state["state_dict"], strict=False)
        return model.eval()

    @torch.no_grad()
    def predict_endpoint(self, start_state: torch.Tensor, already_embedded: bool=False) -> Dict[str, torch.Tensor]:
        """
        start_state: [B,2] (θ, θ̇) if already_embedded=False; else [B,3] embedded
        returns dict with 'endpoint_emb', 'endpoint_raw'
        """
        device = next(self.model.parameters()).device

        x0_emb = start_state if already_embedded else embed(start_state)
        x0_emb = x0_emb.to(device)

        # encode to latent
        z0 = self.model.encode(x0_emb)

        # integrate dz/dt = f(z,t|z0) from t=0->1 with fixed steps
        steps = self.config.num_integration_steps
        dt = 1.0 / steps
        z = z0.clone()
        for i in range(steps):
            t = torch.full((z.shape[0],), (i + 0.5) * dt, device=device)  # mid-point time
            v = self.model(z, t, condition=z0)  # predicted v_z
            z = z + dt * v

        # decode and extract
        x1_emb = self.model.decode(z)
        x1_raw = extract(x1_emb)

        return {"endpoint_emb": x1_emb, "endpoint_raw": x1_raw}