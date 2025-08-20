"""
Training script for latent circular flow matching
"""
import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.flow_matching.latent_circular.flow_matcher import LatentCircularFlowMatcher


@hydra.main(config_path="../../../configs", config_name="train_latent_circular.yaml")
def main(cfg: DictConfig):
    """Main training function for latent circular flow matching"""
    
    # Set GPU device from config
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Initialize data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model architecture (MLP)
    model_net = hydra.utils.instantiate(cfg.model)
    
    # Initialize optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.optimizer) if cfg.get("optimizer") else None
    scheduler = hydra.utils.instantiate(cfg.get("scheduler")) if cfg.get("scheduler") else None
    
    # Initialize latent circular flow matching module
    flow_model = LatentCircularFlowMatcher(
        model=model_net,
        optimizer=optimizer,
        scheduler=scheduler,
        latent_dim=cfg.latent.latent_dim,
        kl_weight=cfg.latent.kl_weight,
        endpoint_weight=0.0,  # Start with 0, can be adjusted
        posterior_std_min=1e-5
    )
    
    print("Latent Circular Flow Matching Model architecture:")
    print(flow_model)
    print(f"Total parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Initialize trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train the model
    trainer.fit(model=flow_model, datamodule=data_module)
    
    print("\\nLatent circular flow matching training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()