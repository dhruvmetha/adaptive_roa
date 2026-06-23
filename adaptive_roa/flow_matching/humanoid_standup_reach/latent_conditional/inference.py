"""Inference helpers for HumanoidStandUpReach LCFM."""
import torch
from adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher import (
    HumanoidStandUpReachLatentConditionalFlowMatcher,
)


def load_model(checkpoint_path: str, dataset_dir: str = None):
    """Load a trained flow matcher from a Lightning checkpoint."""
    return HumanoidStandUpReachLatentConditionalFlowMatcher.load_from_checkpoint(checkpoint_path)


@torch.no_grad()
def predict_endpoints(model, start_states: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
    model.eval()
    return model.predict_endpoint(start_states, num_steps=num_steps)
