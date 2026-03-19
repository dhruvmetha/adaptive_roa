"""
Standalone training script for Pendulum Trajectory Flow Matcher.

Usage:
    python -m adaptive_roa.flow_matching.pendulum.trajectory.train \
        --trajectories_dir /path/to/trajectories \
        --shuffled_indices_file /path/to/shuffled_indices.txt \
        --output_dir /path/to/output \
        --sequence_length 32 \
        --max_epochs 500

This trains the trajectory-level flow matcher independently of the adaptive loop,
useful for development and ablation studies.
"""

import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.model.temporal_transformer import TemporalTransformer
from adaptive_roa.data.trajectory_data import TrajectoryDataset, TrajectoryEndpointDataModule
from adaptive_roa.flow_matching.pendulum.trajectory.flow_matcher import PendulumTrajectoryFlowMatcher


def main():
    parser = argparse.ArgumentParser(description="Train Pendulum Trajectory Flow Matcher")

    # Data
    parser.add_argument("--trajectories_dir", type=str, required=True)
    parser.add_argument("--shuffled_indices_file", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Pendulum system dataset dir (defaults to parent of trajectories_dir)")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Train endpoint file (for index extraction)")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Val endpoint file (for index extraction)")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/trajectory_fm")

    # Trajectory params
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--history_length", type=int, default=1)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--mae_val_frequency", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    # Flow matching
    parser.add_argument("--clamp_noise", action="store_true", default=True)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--use_manifold", action="store_true", default=True)

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir or str(Path(args.trajectories_dir).parent)

    # System
    system = PendulumSystem(dataset_dir=dataset_dir)

    # Model: TemporalTransformer
    embed_dim = 3  # sin(theta), cos(theta), theta_dot_norm
    condition_dim = 3  # same embedding for start state
    output_dim = 2  # tangent space dimension

    model = TemporalTransformer(
        sequence_length=args.sequence_length,
        input_dim=embed_dim,
        output_dim=output_dim,
        latent_dim=args.latent_dim,
        condition_dim=condition_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer config (will be instantiated by Lightning)
    optimizer_config = {
        "_target_": "torch.optim.AdamW",
        "lr": args.lr,
        "weight_decay": 1e-4,
    }
    scheduler_config = {
        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 20,
    }

    # Flow matcher
    model_config = {
        "_target_": "adaptive_roa.model.temporal_transformer.TemporalTransformer",
        "sequence_length": args.sequence_length,
        "input_dim": embed_dim,
        "output_dim": output_dim,
        "latent_dim": args.latent_dim,
        "condition_dim": condition_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }

    flow_matcher = PendulumTrajectoryFlowMatcher(
        system=system,
        model=model,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        model_config=model_config,
        latent_dim=args.latent_dim,
        mae_val_frequency=args.mae_val_frequency,
        use_loss_weights=False,
        use_manifold=args.use_manifold,
        clamp_noise=args.clamp_noise,
        noise_scale=args.noise_scale,
        sequence_length=args.sequence_length,
        history_length=args.history_length,
        val_error_log_file=str(output_dir / "validation_errors.txt"),
    )

    # Data: if train/val files provided, use them; otherwise create simple split
    if args.train_file and args.val_file:
        data_module = TrajectoryEndpointDataModule(
            data_file=args.train_file,
            validation_file=args.val_file,
            test_file=args.val_file,
            trajectories_dir=args.trajectories_dir,
            shuffled_indices_file=args.shuffled_indices_file,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        # Simple split: first 80% train, last 20% val
        from adaptive_roa.data.trajectory_data import TrajectoryDataset
        from torch.utils.data import DataLoader

        trajectories_dir = Path(args.trajectories_dir)
        with open(args.shuffled_indices_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        all_files = [trajectories_dir / fname for fname in filenames]

        n_total = len(all_files)
        n_train = int(0.8 * n_total)
        train_files = all_files[:n_train]
        val_files = all_files[n_train:]

        train_dataset = TrajectoryDataset(
            trajectories_dir=str(trajectories_dir),
            trajectory_files=train_files,
            sequence_length=args.sequence_length,
        )
        val_dataset = TrajectoryDataset(
            trajectories_dir=str(trajectories_dir),
            trajectory_files=val_files,
            sequence_length=args.sequence_length,
        )

        class SimpleDataModule(pl.LightningDataModule):
            def __init__(self, train_ds, val_ds, batch_size, num_workers):
                super().__init__()
                self.train_ds = train_ds
                self.val_ds = val_ds
                self.batch_size = batch_size
                self.num_workers = num_workers

            def train_dataloader(self):
                return DataLoader(self.train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)

            def val_dataloader(self):
                return DataLoader(self.val_ds, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)

        data_module = SimpleDataModule(train_dataset, val_dataset,
                                       args.batch_size, args.num_workers)

    # Trainer
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            auto_insert_metric_name=False,
        )
    ]

    logger = TensorBoardLogger(save_dir=str(output_dir), name="", version=None)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=32,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(flow_matcher, data_module)

    print(f"\nTraining complete. Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
