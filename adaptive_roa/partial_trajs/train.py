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

import glob
from pathlib import Path
from typing import Dict, Optional

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
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


def build_callbacks(cfg: DictConfig, ckpt_dir):
    """Checkpoint-only callbacks for the non-adaptive partial-trajectory runs.

    Keeps the best-val checkpoint (+ ``last.ckpt``) but deliberately omits
    ``EarlyStopping``: these runs are not adaptive and train the full ``max_epochs``.
    """
    return [
        ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor="val_loss", mode="min",
            save_top_k=1, save_last=True, filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
    ]


def _resolve_output_dir(cfg: DictConfig) -> str:
    """The training run directory: Hydra's output dir if running under @hydra.main,
    else ``cfg.output_dir`` (used by tests that call ``run`` without Hydra)."""
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().runtime.output_dir
    except Exception:
        return str(cfg.get("output_dir", "."))


def run(cfg: DictConfig):
    """Build system/data/model, fit, and return ``(model, metric#1 report)``.

    When ``enable_checkpointing`` is set (real runs) the trainer keeps a best-val
    ``ModelCheckpoint`` (+ ``last.ckpt``), a ``CSVLogger``, and gradient clipping,
    and reloads the best-val weights into the returned model after ``fit``. Unlike
    the adaptive ``ClassifierTrainer`` there is NO ``EarlyStopping`` — these
    non-adaptive runs train the full ``max_epochs``.
    """
    pl.seed_everything(int(cfg.get("seed", 0)), workers=True)

    system = make_verifier_system(str(cfg.system), cfg.get("system_dataset_dir"))
    max_traj = cfg.get("max_trajectories")
    dm = HorizonDataModule(
        cfg.dataset_dir,
        batch_size=int(cfg.batch_size),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.get("seed", 0)),
        max_trajectories=None if max_traj is None else int(max_traj),
        iid_horizons=bool(cfg.get("iid_horizons", False)),
    )
    model = build_model(cfg, system)

    checkpointing = bool(cfg.get("enable_checkpointing", True))
    callbacks = []
    logger = False
    ckpt_dir: Optional[Path] = None
    if checkpointing:
        ckpt_dir = Path(_resolve_output_dir(cfg)) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = build_callbacks(cfg, ckpt_dir)
        logger = CSVLogger(save_dir=_resolve_output_dir(cfg), name="partial_trajs_logs")

    trainer = pl.Trainer(
        max_epochs=int(cfg.max_epochs),
        accelerator=str(cfg.get("accelerator", "cpu")),
        devices=cfg.get("devices", 1),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=checkpointing,
        enable_progress_bar=bool(cfg.get("enable_progress_bar", True)),
        gradient_clip_val=float(cfg.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 50)),
        limit_train_batches=cfg.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.get("limit_val_batches", 1.0),
    )
    dm.setup()
    trainer.fit(model, dm)

    # Reload best-val weights into the in-memory model (load_from_checkpoint cannot
    # reconstruct the system/mlp constructor args; mirrors ClassifierTrainer).
    if ckpt_dir is not None:
        best = glob.glob(str(ckpt_dir / "best*.ckpt"))
        if best:
            state = torch.load(best[0], map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=False)

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
