import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import hydra
import lightning.pytorch as pl

@hydra.main(config_path="../configs", config_name="evaluate_reachability.yaml")
def main(cfg):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
        
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    print(model)
    
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Training
    # trainer.fit(model=model, datamodule=data_module)
    
    # ckpt_path = "outputs/2025-05-09/17-10-46/logs/vae_training/version_0/checkpoints/epoch=999-step=250000.ckpt"
    
    print(os.getcwd())
    
    print(os.path.exists("logs/vae_training/version_0/checkpoints"))
    # print(os.path.exists("outputs/2025-05-09/17-10-46/logs/vae_training/version_0/checkpoints"))
    ckpt_path = "logs/vae_training/version_0/checkpoints/epoch=999-step=250000.ckpt"
    # if not os.path.exists(ckpt_path):
    #     raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    trainer.test(model=model, datamodule=data_module, ckpt_path=ckpt_path)
    
    
if __name__ == "__main__":
    main()