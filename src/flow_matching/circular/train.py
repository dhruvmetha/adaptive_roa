"""
Training script for circular flow matching
"""
import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.flow_matching.circular.flow_matcher import CircularFlowMatcher
from src.flow_matching.base.config import FlowMatchingConfig


@hydra.main(config_path="../../../configs", config_name="train_circular_flow_matching.yaml")
def main(cfg: DictConfig):
    """Main training function for circular flow matching"""
    
    # Set GPU device from config
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Create configuration
    config = FlowMatchingConfig(
        sigma=cfg.get("sigma", 0.0),
        hidden_dims=tuple(cfg.model.get("hidden_dims", [64, 128, 256])),
        time_emb_dim=cfg.model.get("time_emb_dim", 128),
        num_integration_steps=cfg.get("num_integration_steps", 100)
    )
    
    # Initialize data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model architecture (CircularUNet1D)
    model_net = hydra.utils.instantiate(cfg.model)
    
    # Initialize optimizer and scheduler (like old implementation)
    optimizer = hydra.utils.instantiate(cfg.optimizer) if cfg.get("optimizer") else None
    scheduler = hydra.utils.instantiate(cfg.get("scheduler")) if cfg.get("scheduler") else None
    
    # Initialize circular flow matching module
    flow_model = CircularFlowMatcher(
        model=model_net,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    print("Circular Flow Matching Model architecture:")
    print(flow_model)
    print(f"Total parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Initialize trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train the model
    trainer.fit(model=flow_model, datamodule=data_module)
    
    print("\\nCircular flow matching training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()