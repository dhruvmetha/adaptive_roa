import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from ..base.config import FlowMatchingConfig
from .flow_matcher import LatentCircularFlowMatcher

@hydra.main(config_path="../../../configs", config_name="train_latent_circular_flow_matching.yaml")
def main(cfg: DictConfig):
    """Main training function for latent circular flow matching"""
    
    # optional device pinning
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    config = FlowMatchingConfig(
        sigma=cfg.get("sigma", 0.0),
        hidden_dims=tuple(cfg.model.get("hidden_dims", [128, 128])),
        time_emb_dim=cfg.model.get("time_emb_dim", 64),
        num_integration_steps=cfg.get("num_integration_steps", 100),
    )

    # hydra instantiate everything
    data_module = hydra.utils.instantiate(cfg.data)
    model_net = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer) if cfg.get("optimizer") else None
    scheduler = hydra.utils.instantiate(cfg.get("scheduler")) if cfg.get("scheduler") else None

    flow_model = LatentCircularFlowMatcher(
        model=model_net,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    print("Latent Circular Flow Matching model:")
    print(flow_model)
    print(f"Total parameters: {sum(p.numel() for p in flow_model.parameters()):,}")

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model=flow_model, datamodule=data_module)

    print("\\nLatent circular flow matching training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()