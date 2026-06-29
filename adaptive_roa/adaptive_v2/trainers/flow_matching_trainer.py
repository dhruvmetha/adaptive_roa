"""Flow-matching trainer adapter for adaptive v2."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, open_dict

from adaptive_roa.data.cartpole_endpoint_data import CartPoleEndpointDataModule
from adaptive_roa.data.humanoid_standup_reach_endpoint_data import HumanoidStandUpReachEndpointDataModule
from adaptive_roa.data.pendulum_endpoint_data import PendulumEndpointDataModule
from adaptive_roa.data.quadrotor2d_endpoint_data import Quadrotor2DEndpointDataModule
from adaptive_roa.data.quadrotor3d_endpoint_data import Quadrotor3DEndpointDataModule
from adaptive_roa.data.trajectory_data import TrajectoryDataModule


# Global prediction mode: endpoint data modules (start → end pairs)
_DATAMODULES = {
    "pendulum": PendulumEndpointDataModule,
    "cartpole_pybullet": CartPoleEndpointDataModule,
    "quadrotor2d": Quadrotor2DEndpointDataModule,
    "quadrotor3d": Quadrotor3DEndpointDataModule,
    "humanoid_standup_reach": HumanoidStandUpReachEndpointDataModule,
}


class FlowMatchingTrainer:
    """Adapter over legacy flow-matching training code paths."""

    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name
        self.prediction_mode = str(cfg.get("prediction_mode", "global"))

        if system_name not in _DATAMODULES:
            raise ValueError(f"Unsupported system for v2 trainer: {system_name}")

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    @property
    def _flow_matching_cfg(self):
        pred = self._predictor_cfg
        if "flow_matching" in pred:
            return pred.flow_matching
        return self.cfg.flow_matching

    @property
    def is_local(self) -> bool:
        return self.prediction_mode == "local"

    def _create_datamodule(self, dataset_files: dict):
        """
        Create data module from dataset files dict.

        For global mode: uses endpoint pair files (train/val).
        For local mode: uses trajectory index files (train_trajectories/val_trajectories).
        """
        if self.is_local:
            # Get angle indices from system for proper wrapping
            # (e.g., [0] for pendulum, [1] for cartpole, [2] for quadrotor2d, None for quadrotor3d)
            angle_indices = getattr(self.system, 'angle_indices', None)
            return TrajectoryDataModule(
                train_trajectory_file=dataset_files["train_trajectories"],
                val_trajectory_file=dataset_files["val_trajectories"],
                trajectories_dir=self.cfg.data_source.trajectories_dir,
                sequence_length=self._flow_matching_cfg.get("sequence_length", 32),
                batch_size=self._predictor_cfg.get("batch_size", 256),
                val_batch_size=self._predictor_cfg.get("val_batch_size", 2048),
                num_workers=self._predictor_cfg.get("num_workers", self.cfg.get("num_workers", 4)),
                angle_indices=angle_indices,
            )

        # Global: endpoint data module
        dm_cls = _DATAMODULES[self.system_name]
        kwargs = {
            "data_file": dataset_files["train"],
            "validation_file": dataset_files["val"],
            "test_file": dataset_files["val"],
            "batch_size": self._predictor_cfg.get("batch_size", 256),
            "val_batch_size": self._predictor_cfg.get("val_batch_size", 2048),
            "num_workers": self._predictor_cfg.get("num_workers", self.cfg.get("num_workers", 4)),
        }
        if self.system_name in {"quadrotor3d", "humanoid_standup_reach"}:
            kwargs["dataset_dir"] = self.cfg.system.get("dataset_dir")
        return dm_cls(**kwargs)

    def fit(
        self,
        dataset_files: dict,
        output_dir: str,
        resume_checkpoint: str | None = None,
    ):
        data_module = self._create_datamodule(dataset_files)

        flow_matching = self._flow_matching_cfg
        use_loss_weights = flow_matching.get("use_loss_weights", False)
        use_manifold = flow_matching.get("use_manifold", True)
        use_log_loss_weights = flow_matching.get("use_log_loss_weights", False)
        clamp_noise = flow_matching.get("clamp_noise", True)
        zero_latent = flow_matching.get("zero_latent", False)
        noise_scale = flow_matching.get("noise_scale", 1.0)

        if self.system_name == "quadrotor3d":
            model_output_dim = 12 if use_manifold else 13
            if self.cfg.model.get("output_dim", 12) != model_output_dim:
                print(
                    "   Auto-adjusting model.output_dim: "
                    f"{self.cfg.model.get('output_dim', 12)} -> {model_output_dim} "
                    f"(use_manifold={use_manifold})"
                )
                with open_dict(self.cfg):
                    self.cfg.model.output_dim = model_output_dim

        model = hydra.utils.instantiate(self.cfg.model)

        flow_matcher_kwargs = {
            "system": self.system,
            "model": model,
            "optimizer": self.cfg.optimizer,
            "scheduler": self.cfg.scheduler,
            "model_config": OmegaConf.to_container(self.cfg.model, resolve=True),
            "latent_dim": flow_matching.latent_dim,
            "mae_val_frequency": flow_matching.mae_val_frequency,
            "use_loss_weights": use_loss_weights,
            "use_manifold": use_manifold,
            "use_log_loss_weights": use_log_loss_weights,
            "clamp_noise": clamp_noise,
            "zero_latent": zero_latent,
            "noise_scale": noise_scale,
            "val_error_log_file": str(Path(output_dir) / "validation_errors.txt"),
            "_recursive_": False,
        }
        if self.system_name == "quadrotor3d":
            flow_matcher_kwargs["quat_loss_weight"] = flow_matching.get("quat_loss_weight", 1.0)
        if self.is_local:
            flow_matcher_kwargs["sequence_length"] = flow_matching.get("sequence_length", 32)
            flow_matcher_kwargs["history_length"] = flow_matching.get("history_length", 1)

        flow_matcher = hydra.utils.instantiate(self.cfg.flow_matcher, **flow_matcher_kwargs)

        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"Warm start: Loading weights from {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint["state_dict"]
            model_state_dict = {
                key.replace("model.", ""): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
            flow_matcher.model.load_state_dict(model_state_dict)
            print(f"   Loaded model weights ({len(model_state_dict)} tensors)")

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})
        callbacks = []
        for cb_cfg in trainer_cfg.get("callbacks", []):
            callback = hydra.utils.instantiate(cb_cfg)
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = str(checkpoint_dir)
            callbacks.append(callback)

        if not callbacks:
            callbacks = [
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    filename="best-{epoch:02d}-{val_loss:.4f}",
                )
            ]

        logger = TensorBoardLogger(save_dir=output_dir, name="", version=None)

        trainer = pl.Trainer(
            max_epochs=trainer_cfg.get("max_epochs", 1000),
            accelerator=trainer_cfg.get("accelerator", "gpu") if torch.cuda.is_available() else "cpu",
            devices=trainer_cfg.get("devices", 1),
            precision=trainer_cfg.get("precision", 32),
            limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
            limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
            gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
            log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
            check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
            enable_progress_bar=trainer_cfg.get("enable_progress_bar", True),
            enable_model_summary=trainer_cfg.get("enable_model_summary", True),
            callbacks=callbacks,
            logger=logger,
        )

        trainer.fit(flow_matcher, data_module)

        checkpoints = glob.glob(str(checkpoint_dir / "best*.ckpt"))
        if checkpoints:
            try:
                flow_matcher = type(flow_matcher).load_from_checkpoint(
                    checkpoints[0],
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except Exception as e:
                # Fallback: load state dict directly (e.g., when Hydra config not on disk)
                print(f"Warning: load_from_checkpoint failed ({e}), loading state dict directly")
                checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
                state_dict = checkpoint["state_dict"]
                model_state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
                flow_matcher.model.load_state_dict(model_state_dict)

        return flow_matcher
