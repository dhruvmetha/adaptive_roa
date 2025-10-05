"""
Universal training script for flow matching on any dynamical system
"""
import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import torch

# Universal framework imports
from src.systems import PendulumSystem, CartPoleSystem
from src.flow_matching.universal import (
    UniversalFlowMatcher,
    UniversalFlowMatchingConfig
)
from src.model.universal_unet import UniversalUNet

# System factory
SYSTEM_REGISTRY = {
    "pendulum": PendulumSystem,
    "cartpole": CartPoleSystem,
    # Add more systems here as implemented
}


@hydra.main(config_path="configs", config_name="train_universal_flow_matching.yaml", version_base=None)
def main(cfg: DictConfig):
    """Universal training function for any dynamical system"""
    
    # Set GPU device from config
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Create dynamical system
    system_name = cfg.system.name.lower()
    if system_name not in SYSTEM_REGISTRY:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(SYSTEM_REGISTRY.keys())}")
    
    system_class = SYSTEM_REGISTRY[system_name]
    system_kwargs = cfg.system.get("params", {})
    system = system_class(**system_kwargs)
    
    print(f"System: {system}")
    print(f"Manifold structure: {[comp.name for comp in system.manifold_components]}")
    print(f"Dimensions: {system.state_dim} → {system.embedding_dim} → {system.tangent_dim}")
    
    # Create universal configuration
    config = UniversalFlowMatchingConfig.for_system(
        system=system,
        num_integration_steps=cfg.get("num_integration_steps", 100),
        hidden_dims=tuple(cfg.model.get("hidden_dims", [64, 128, 256])),
        time_emb_dim=cfg.model.get("time_emb_dim", 128),
        sigma=cfg.get("sigma", 0.0)
    )
    
    print(f"Config: {config}")
    print("System info:")
    for key, value in config.get_system_info().items():
        print(f"  {key}: {value}")
    
    # Initialize data module (system-specific)
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Initialize universal model architecture
    model_net = UniversalUNet(
        input_dim=config.model_input_dim,
        output_dim=config.model_output_dim,
        hidden_dims=list(config.hidden_dims),
        time_emb_dim=config.time_emb_dim
    )
    
    print(f"Model: {model_net}")
    print("Architecture info:")
    arch_info = model_net.get_architecture_info()
    for key, value in arch_info.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # Initialize optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model_net.parameters())
    scheduler = hydra.utils.instantiate(cfg.get("scheduler"), optimizer=optimizer) if cfg.get("scheduler") else None
    
    # Initialize universal flow matching module
    flow_model = UniversalFlowMatcher(
        system=system,
        model=model_net,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    print(f"Flow matcher: {flow_model}")
    
    # Initialize trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train the model
    print(f"\\nStarting training for {system_name} system...")
    trainer.fit(model=flow_model, datamodule=data_module)
    
    print(f"\\nUniversal flow matching training completed for {system_name}!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    # Save system info for inference
    system_info_path = os.path.join(trainer.log_dir, "system_info.yaml")
    
    # Create serializable system info - only essential data for inference
    dimensions_info = config.get_system_info()
    serializable_system_info = {
        "system_name": system_name,
        "dimensions": {
            "state_dim": int(dimensions_info.get("state_dim", 0)),
            "embedding_dim": int(dimensions_info.get("embedding_dim", 0)),
            "tangent_dim": int(dimensions_info.get("tangent_dim", 0)),
            "model_input_dim": int(dimensions_info.get("model_input_dim", 0)),
            "model_output_dim": int(dimensions_info.get("model_output_dim", 0))
        }
    }
    
    # Write as simple text file instead of YAML to avoid serialization issues
    with open(system_info_path.replace('.yaml', '.txt'), "w") as f:
        f.write(f"System: {system_name}\n")
        f.write(f"State dimension: {serializable_system_info['dimensions']['state_dim']}\n")
        f.write(f"Embedding dimension: {serializable_system_info['dimensions']['embedding_dim']}\n")
        f.write(f"Tangent dimension: {serializable_system_info['dimensions']['tangent_dim']}\n")
        f.write(f"Model input dimension: {serializable_system_info['dimensions']['model_input_dim']}\n")
        f.write(f"Model output dimension: {serializable_system_info['dimensions']['model_output_dim']}\n")
    
    print(f"System info saved to: {system_info_path}")


if __name__ == "__main__":
    main()