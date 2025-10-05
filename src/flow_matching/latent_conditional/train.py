#!/usr/bin/env python3
"""
Training script for Latent Conditional Flow Matching
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.flow_matching.latent_conditional.flow_matcher import LatentConditionalFlowMatcher
from src.model.latent_conditional_unet1d import LatentConditionalUNet1D
from src.data.circular_endpoint_data import CircularEndpointDataModule
from src.systems.pendulum_lcfm import PendulumSystemLCFM


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_latent_conditional_flow_matching")
def main(cfg: DictConfig):
    # Set random seeds
    pl.seed_everything(cfg.seed)
    
    # System definition
    system = PendulumSystemLCFM()
    
    # Data module using Hydra config
    data_module = CircularEndpointDataModule(
        data_file=cfg.data.data_file,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    
    # Model architecture - override latent_dim from flow_matching config
    latent_dim = cfg.flow_matching.latent_dim if cfg.flow_matching.latent_dim else 2
    
    # Update model config with correct latent_dim
    model_cfg = OmegaConf.structured(cfg.model)
    model_cfg.latent_dim = latent_dim
    
    # Create model from config
    model = hydra.utils.instantiate(model_cfg)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture: {model.hidden_dims}")
    print(f"Time embedding dim: {model.time_emb_dim}")
    print(f"Latent dim: {model.latent_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Input dimensions: embedded={model.embedded_dim}, latent={model.latent_dim}, condition={model.condition_dim}")
    print(f"Total input dim: {model.embedded_dim + model.time_emb_dim + model.latent_dim + model.condition_dim}")
    
    # Create optimizer from config
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Create scheduler from config  
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    
    # Flow matcher - pass model config so it gets saved in hyperparameters
    flow_matcher = LatentConditionalFlowMatcher(
        system=system,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=OmegaConf.to_container(model_cfg, resolve=True),
        latent_dim=latent_dim
    )
    
    # Create trainer with callbacks from config
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train
    print("Starting Latent Conditional Flow Matching training...")
    print(f"System: {system}")
    print(f"Model: {model}")
    print(f"Latent dim: {latent_dim}")
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    
    trainer.fit(flow_matcher, data_module)
    
    print("Training completed!")


if __name__ == "__main__":
    main()