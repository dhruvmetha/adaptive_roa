#!/usr/bin/env python3
"""Train HumanoidStandUpReach Latent Conditional Flow Matching (Facebook FM).

Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-D state, 67-D ambient tangent).
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl

from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

if not OmegaConf.has_resolver("net_id"):
    OmegaConf.register_new_resolver("net_id", lambda default="": get_net_id() or default)
if not OmegaConf.has_resolver("exp_dir"):
    OmegaConf.register_new_resolver("exp_dir", lambda default="": get_exp_dir() or default)
if not OmegaConf.has_resolver("shared_data_base"):
    OmegaConf.register_new_resolver("shared_data_base", lambda default="": get_shared_data_base() or default)


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_humanoid_standup_reach")
def main(cfg: DictConfig):
    print("=" * 80)
    print("HumanoidStandUpReach Latent Conditional Flow Matching (Facebook FM)")
    print(f"   Manifold: R^34 x S^2 x R^30 (67-D) | seed={cfg.seed}")
    print("=" * 80)
    pl.seed_everything(cfg.seed)

    system = hydra.utils.instantiate(cfg.system)
    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        use_loss_weights=cfg.flow_matching.get("use_loss_weights", False),
        use_manifold=cfg.flow_matching.get("use_manifold", True),
        clamp_noise=cfg.flow_matching.get("clamp_noise", True),
        zero_latent=cfg.flow_matching.get("zero_latent", False),
        noise_scale=cfg.flow_matching.get("noise_scale", 1.0),
        _recursive_=False,
    )

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(flow_matcher, data_module)
    print("Training complete")


if __name__ == "__main__":
    main()
