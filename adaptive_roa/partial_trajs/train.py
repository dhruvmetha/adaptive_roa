"""Train the deterministic partial-trajectory T-step dynamics regressor.

Usage:
    python adaptive_roa/partial_trajs/train.py                     # pendulum default
    python adaptive_roa/partial_trajs/train.py system=cartpole \
        dataset_dir=.../partial_deterministic/cartpole_pybullet/cartpole_pybullet_T25 \
        hidden_dims='[256,512,1024,512,256]'

After training it reports metric #1 (per-horizon T-step error) on the val split,
overall and stratified by freeze/motion.
"""
from __future__ import annotations

from typing import Dict, Optional

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf

from adaptive_roa.utils.env_config import get_shared_data_base

# Register OmegaConf resolver BEFORE Hydra loads config.
if not OmegaConf.has_resolver("shared_data_base"):
    OmegaConf.register_new_resolver(
        "shared_data_base", lambda default="": get_shared_data_base() or default
    )

from adaptive_roa.partial_trajs.systems import make_verifier_system
from adaptive_roa.partial_trajs.data.datamodule import HorizonDataModule
from adaptive_roa.partial_trajs.model.regressor import DynamicsRegressor
from adaptive_roa.partial_trajs.model.generative import GenerativeDynamics
from adaptive_roa.partial_trajs.eval.metrics import horizon_error_report


def build_model(cfg: DictConfig, system):
    """Build the dynamics backend selected by ``cfg.backend``."""
    backend = str(cfg.get("backend", "deterministic"))
    if backend == "deterministic":
        return DynamicsRegressor(
            system,
            hidden_dims=list(cfg.hidden_dims),
            lr=float(cfg.lr),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if backend == "generative":
        return GenerativeDynamics(
            system,
            hidden_dims=list(cfg.hidden_dims),
            latent_dim=int(cfg.get("latent_dim", 1)),
            num_integration_steps=int(cfg.get("num_integration_steps", 50)),
            lr=float(cfg.lr),
        )
    raise ValueError(f"unknown backend {backend!r} (expected deterministic|generative)")


@torch.no_grad()
def horizon_error_over_loader(model, system, loader, max_batches: Optional[int] = None) -> Dict[str, float]:
    """Metric #1: per-horizon T-step error over a dataloader."""
    model.eval()
    preds, targets, freezes, motions = [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        preds.append(model.predict(batch["x_start"]))
        targets.append(batch["x_end"])
        freezes.append(batch["is_freeze"])
        motions.append(batch["motion"])
    return horizon_error_report(
        torch.cat(preds),
        torch.cat(targets),
        torch.cat(freezes),
        torch.cat(motions),
        circular_indices=system.get_circular_indices(),
    )


def run(cfg: DictConfig):
    """Build system/data/model, fit, and return ``(model, metric#1 report)``."""
    pl.seed_everything(int(cfg.get("seed", 0)), workers=True)

    system = make_verifier_system(str(cfg.system), cfg.get("system_dataset_dir"))
    dm = HorizonDataModule(
        cfg.dataset_dir,
        batch_size=int(cfg.batch_size),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.get("seed", 0)),
    )
    model = build_model(cfg, system)
    trainer = pl.Trainer(
        max_epochs=int(cfg.max_epochs),
        accelerator=str(cfg.get("accelerator", "cpu")),
        devices=cfg.get("devices", 1),
        logger=False,
        enable_checkpointing=bool(cfg.get("enable_checkpointing", True)),
        enable_progress_bar=bool(cfg.get("enable_progress_bar", True)),
        limit_train_batches=cfg.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.get("limit_val_batches", 1.0),
    )
    dm.setup()
    trainer.fit(model, dm)

    report = horizon_error_over_loader(
        model, system, dm.val_dataloader(), max_batches=cfg.get("eval_max_batches")
    )
    return model, report


@hydra.main(
    version_base=None,
    config_path="../../configs/partial_trajs",
    config_name="train_partial_trajs",
)
def main(cfg: DictConfig) -> None:
    _, report = run(cfg)
    print("Val per-horizon T-step error (#1):", report)


if __name__ == "__main__":
    main()
