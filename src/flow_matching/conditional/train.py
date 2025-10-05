"""
Training script for conditional flow matching
"""
import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.flow_matching.conditional.flow_matcher import ConditionalFlowMatcher
from src.flow_matching.base.config import FlowMatchingConfig


@hydra.main(config_path="../../../configs", config_name="train_conditional_flow_matching.yaml")
def main(cfg: DictConfig):
    """Main training function for conditional flow matching"""
    
    # Set GPU device from config
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Create configuration
    config = FlowMatchingConfig(
        sigma=cfg.flow_matching.get("sigma", 0.0),
        hidden_dims=tuple(cfg.model.get("hidden_dims", [64, 128, 256])),
        time_emb_dim=cfg.model.get("time_emb_dim", 128),
        num_integration_steps=cfg.flow_matching.get("num_integration_steps", 100),
        # Add conditional flow matching specific config
        noise_distribution=cfg.flow_matching.get("noise_distribution", "uniform"),
        noise_scale=cfg.flow_matching.get("noise_scale", 1.0),
        noise_bounds=cfg.flow_matching.get("noise_bounds", None)
    )
    
    # Initialize data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model architecture (ConditionalUNet1D)
    model_net = hydra.utils.instantiate(cfg.model)
    
    # Initialize optimizer and scheduler (like existing implementation)
    optimizer = hydra.utils.instantiate(cfg.optimizer) if cfg.get("optimizer") else None
    scheduler = hydra.utils.instantiate(cfg.get("scheduler")) if cfg.get("scheduler") else None
    
    # Initialize conditional flow matching module with optional latent support
    # Support both boolean use_latent flag and integer latent_dim
    use_latent = cfg.flow_matching.get("use_latent", None)
    latent_dim = cfg.flow_matching.get("latent_dim", None)
    
    # Handle boolean use_latent flag
    if use_latent is not None:
        if isinstance(use_latent, bool):
            if use_latent:
                # Default to 2D latent if enabled but no dimension specified
                latent_dim = latent_dim if latent_dim is not None else 2
            else:
                # Explicitly disable latent
                latent_dim = None
        else:
            raise ValueError(f"use_latent must be boolean, got {type(use_latent)}")
    # Otherwise use existing latent_dim logic: None = no latent, int = latent dimension
    
    # If using latent, need to adjust model condition_dim
    if latent_dim is not None:
        print(f"🧬 Using latent conditional flow matching with latent_dim={latent_dim}")
        # Recreate model with expanded condition dimension
        original_condition_dim = model_net.condition_dim
        expanded_condition_dim = original_condition_dim + latent_dim
        
        # Create new model with expanded condition dimension
        model_net = hydra.utils.instantiate(cfg.model, condition_dim=expanded_condition_dim)
        print(f"📐 Expanded condition dimension: {original_condition_dim} → {expanded_condition_dim}")
    
    flow_model = ConditionalFlowMatcher(
        model=model_net,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        latent_dim=latent_dim  # Pass latent dimension
    )
    
    model_type = "Latent Conditional" if latent_dim else "Standard Conditional"
    print(f"{model_type} Flow Matching Model architecture:")
    print(flow_model)
    print(f"Total parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    if latent_dim:
        print(f"🧬 Latent dimension: {latent_dim}")
        print(f"🎯 Using Gaussian latent variable sampling for controllable multi-modality")
    
    # Initialize trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train the model
    trainer.fit(model=flow_model, datamodule=data_module)
    
    print("\nConditional flow matching training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()