import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train_flow_matching.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
        
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    print(model)
    
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Training
    trainer.fit(model=model, datamodule=data_module)
    
if __name__ == "__main__":
    main()