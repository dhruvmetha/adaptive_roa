import os
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@hydra.main(config_path="../configs", config_name="train_circular_flow_matching.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    
    # Instantiate data module
    data_module = hydra.utils.instantiate(cfg.data)
    
    # Instantiate model components
    model_net = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    scheduler = hydra.utils.instantiate(cfg.scheduler)
    
    # Instantiate the Lightning module  
    lightning_module = hydra.utils.instantiate(
        cfg.module,
        model=model_net,
        optimizer=optimizer, 
        scheduler=scheduler
    )
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir="outputs",
        name="circular_flow_matching"
    )
    
    # Setup checkpointing  
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"outputs/circular_flow_matching/checkpoints",
        filename="epoch={epoch}-step={step}-val_loss={val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    # Instantiate trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    print("=" * 50)
    print("CIRCULAR FLOW MATCHING TRAINING")
    print("=" * 50)
    print(f"Model: {model_net}")
    print(f"Data: {data_module}")
    print(f"Trainer: {trainer}")
    print("=" * 50)
    
    # Train the model
    trainer.fit(lightning_module, data_module)
    
    print("Training completed!")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()