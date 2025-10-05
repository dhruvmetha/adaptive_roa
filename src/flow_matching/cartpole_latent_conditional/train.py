#!/usr/bin/env python3
"""
Training script for CartPole Latent Conditional Flow Matching
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

from src.flow_matching.cartpole_latent_conditional.flow_matcher import CartPoleLatentConditionalFlowMatcher
from src.model.cartpole_latent_conditional_unet1d import CartPoleLatentConditionalUNet1D
from src.data.cartpole_endpoint_data import CartPoleEndpointDataModule
from src.systems.cartpole_lcfm import CartPoleSystemLCFM


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_cartpole_latent_conditional_flow_matching")
def main(cfg: DictConfig):
    # Set random seeds
    pl.seed_everything(cfg.seed)
    
    # System definition
    system = CartPoleSystemLCFM()
    
    # Data module using Hydra instantiation
    data_module = hydra.utils.instantiate(cfg.data)
    
    # Model architecture - override latent_dim from flow_matching config
    latent_dim = cfg.flow_matching.latent_dim if cfg.flow_matching.latent_dim else 2
    
    # Update model config with correct latent_dim
    model_cfg = OmegaConf.structured(cfg.model)
    model_cfg.latent_dim = latent_dim
    
    # Create model from config
    model = hydra.utils.instantiate(model_cfg)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Model architecture: {model_info['hidden_dims']}")
    print(f"Time embedding dim: {model_info['time_emb_dim']}")
    print(f"Latent dim: {model_info['latent_dim']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Input dimensions: embedded={model_info['embedded_dim']}, latent={model_info['latent_dim']}, condition={model_info['condition_dim']}")
    print(f"Output dimensions: {model_info['output_dim']}")
    print(f"Total input dim: {model_info['embedded_dim'] + model_info['time_emb_dim'] + model_info['latent_dim'] + model_info['condition_dim']}")
    
    # Create optimizer from config
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Create scheduler from config  
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    
    # Flow matcher - pass model config so it gets saved in hyperparameters
    flow_matcher = CartPoleLatentConditionalFlowMatcher(
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
    print("Starting CartPole Latent Conditional Flow Matching training...")
    print(f"System: {system}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dim: {latent_dim}")
    print(f"Dataset: {data_module.data_file}")
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    
    trainer.fit(flow_matcher, data_module)
    
    print("Training completed!")


if __name__ == "__main__":
    main()