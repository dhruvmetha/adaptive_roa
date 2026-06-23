"""Resume an interrupted adaptive training run.

Detects the last fully completed epoch (has artifacts_v2.json or results.json),
cleans up any crashed partial epochs, restores pool state, and continues
the epoch loop.

Examples:
    # Resume with original settings
    python scripts/resume_adaptive.py --run-dir outputs/quadrotor3d_adaptive/2025-...

    # Resume and extend to 25 total epochs
    python scripts/resume_adaptive.py --run-dir outputs/quadrotor3d_adaptive/2025-... --n-epochs 25

    # Resume with different eval frequency
    python scripts/resume_adaptive.py --run-dir outputs/quadrotor3d_adaptive/2025-... --eval-every 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from adaptive_roa.utils.env_config import (
    get_data_dir,
    get_env_config,
    get_exp_dir,
    get_net_id,
    get_shared_data_base,
)


def register_resolvers():
    """Register the same OmegaConf resolvers used by run_adaptive.py."""
    if not OmegaConf.has_resolver("net_id"):
        OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
    if not OmegaConf.has_resolver("exp_dir"):
        OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    if not OmegaConf.has_resolver("shared_data_base"):
        OmegaConf.register_new_resolver(
            "shared_data_base",
            lambda default="": get_shared_data_base() or default,
        )
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env",
            lambda key, default="": os.environ.get(key, get_env_config().get(key, default)),
        )


def detect_completed_epochs(run_dir: Path) -> list[int]:
    """Return sorted list of epoch numbers that have artifacts_v2.json or results.json."""
    completed = []
    for d in run_dir.iterdir():
        m = re.match(r"epoch_(\d+)", d.name)
        if m and d.is_dir():
            epoch_num = int(m.group(1))
            has_artifacts = (d / "artifacts_v2.json").exists()
            has_results = (d / "results.json").exists()
            if has_artifacts or has_results:
                completed.append(epoch_num)
    return sorted(completed)


def detect_crashed_epochs(run_dir: Path, last_completed: int) -> list[int]:
    """Return epoch numbers > last_completed that exist as directories."""
    crashed = []
    for d in run_dir.iterdir():
        m = re.match(r"epoch_(\d+)", d.name)
        if m and d.is_dir():
            epoch_num = int(m.group(1))
            if epoch_num > last_completed:
                crashed.append(epoch_num)
    return sorted(crashed)


def load_epoch_results(run_dir: Path, completed_epochs: list[int]) -> list[dict]:
    """Load epoch results from final_results.json or individual epoch artifacts."""
    final_results_path = run_dir / "final_results.json"
    if final_results_path.exists():
        with open(final_results_path) as f:
            data = json.load(f)
        results = data.get("epoch_results", [])
        # Filter to only completed epochs
        results = [r for r in results if r["epoch"] in set(completed_epochs)]
        if len(results) == len(completed_epochs):
            return results

    # Reconstruct from per-epoch files
    results = []
    for epoch_num in completed_epochs:
        epoch_dir = run_dir / f"epoch_{epoch_num:03d}"
        # Try artifacts_v2.json first, then results.json
        artifacts_path = epoch_dir / "artifacts_v2.json"
        results_path = epoch_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results.append(json.load(f))
        elif artifacts_path.exists():
            with open(artifacts_path) as f:
                artifacts = json.load(f)
            # Build a minimal epoch_result from artifacts
            results.append({
                "epoch": artifacts["epoch"],
                "train_trajectories": artifacts["train_trajectories"],
                "sampling_mode": artifacts["sampling_mode"],
                "full_roa": artifacts.get("eval_metrics", {}),
                "endpoint_error": artifacts.get("endpoint_error", {}),
            })
    return results


def find_best_checkpoint(run_dir: Path, epoch_num: int) -> str | None:
    """Find the best checkpoint from a completed epoch."""
    epoch_dir = run_dir / f"epoch_{epoch_num:03d}"
    ckpts = glob.glob(str(epoch_dir / "checkpoints" / "best*.ckpt"))
    return ckpts[0] if ckpts else None


def main():
    parser = argparse.ArgumentParser(description="Resume an interrupted adaptive training run")
    parser.add_argument("--run-dir", required=True, help="Path to existing run output directory")
    parser.add_argument("--n-epochs", type=int, default=None, help="New total epoch count (default: keep original)")
    parser.add_argument("--eval-every", type=int, default=None, help="Override eval_every (default: keep original)")
    parser.add_argument("--d2-ratio", type=float, default=None, help="Override d2_ratio (default: keep original)")
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="Override samples_per_epoch (default: keep original)")
    parser.add_argument("--device", default=None, help="GPU device (default: from config)")
    parser.add_argument(
        "--no-delete-crashed",
        action="store_true",
        help="Don't delete crashed epoch directories (default: delete them)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        sys.exit(1)

    # --- Register resolvers before loading config ---
    register_resolvers()

    # --- Load config ---
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        print(f"ERROR: No .hydra/config.yaml found in {run_dir}")
        sys.exit(1)

    cfg = OmegaConf.load(hydra_config_path)
    assert isinstance(cfg, DictConfig)

    # Override output_dir to point to existing run
    OmegaConf.update(cfg, "output_dir", str(run_dir), force_add=True)

    # Override n_epochs if specified
    original_n_epochs = int(cfg.get("n_epochs", 10))
    if args.n_epochs is not None:
        OmegaConf.update(cfg, "n_epochs", args.n_epochs)

    if args.eval_every is not None:
        OmegaConf.update(cfg, "eval_every", args.eval_every)

    if args.d2_ratio is not None:
        OmegaConf.update(cfg, "d2_ratio", args.d2_ratio)

    if args.samples_per_epoch is not None:
        OmegaConf.update(cfg, "samples_per_epoch", args.samples_per_epoch)

    if args.device is not None:
        OmegaConf.update(cfg, "device", args.device)

    n_epochs = int(cfg.get("n_epochs", 10))

    # --- Detect epochs ---
    completed_epochs = detect_completed_epochs(run_dir)
    if not completed_epochs:
        print("ERROR: No completed epochs found. Nothing to resume from.")
        sys.exit(1)

    last_completed = max(completed_epochs)
    start_epoch = last_completed + 1

    if start_epoch >= n_epochs:
        print(f"All {n_epochs} epochs already completed (last completed: {last_completed}).")
        print(f"Use --n-epochs to extend beyond {n_epochs}.")
        sys.exit(0)

    crashed = detect_crashed_epochs(run_dir, last_completed)

    print("=" * 70)
    print("RESUME ADAPTIVE TRAINING")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"System: {cfg.adaptive_v2.get('system_name', 'unknown')}")
    print(f"Completed epochs: {completed_epochs}")
    print(f"Last completed: epoch_{last_completed:03d}")
    print(f"Crashed epochs: {crashed if crashed else 'none'}")
    print(f"Original n_epochs: {original_n_epochs}")
    print(f"Target n_epochs: {n_epochs}")
    print(f"Resuming from epoch {start_epoch} to {n_epochs - 1}")
    print()

    # --- Clean crashed epochs ---
    if crashed and not args.no_delete_crashed:
        for epoch_num in crashed:
            epoch_dir = run_dir / f"epoch_{epoch_num:03d}"
            print(f"Deleting crashed epoch directory: {epoch_dir}")
            shutil.rmtree(epoch_dir)

    # --- Create engine and restore state ---
    # Import here to avoid circular imports and ensure resolvers are registered
    import lightning.pytorch as pl

    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine

    engine = AdaptiveEngine(cfg)

    # Seed for reproducibility (same as run())
    pl.seed_everything(engine.seed, workers=True)
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Restore pool state from saved file (do NOT call pool.initialize())
    state_file = run_dir / "dataset_builder_state.json"
    if not state_file.exists():
        print(f"ERROR: No dataset_builder_state.json found in {run_dir}")
        sys.exit(1)

    print(f"Loading pool state from {state_file}")
    engine.pool.dataset_builder.load_state(str(state_file))

    # Rebuild dataset files
    print("Rebuilding dataset files...")
    dataset_files = engine.pool.build_all_datasets()

    # Find previous best checkpoint
    previous_best_checkpoint = find_best_checkpoint(run_dir, last_completed)
    if previous_best_checkpoint:
        print(f"Previous best checkpoint: {previous_best_checkpoint}")
    else:
        print("WARNING: No best checkpoint found in last completed epoch")

    # Load existing epoch results
    epoch_results = load_epoch_results(run_dir, completed_epochs)
    print(f"Loaded {len(epoch_results)} epoch results")

    # Compute n_existing_rows from the rebuilt train file
    n_existing_rows = engine._count_file_rows(dataset_files["train"])
    print(f"Existing train rows: {n_existing_rows}")

    print()
    print(f"Starting epoch loop from {start_epoch} to {n_epochs - 1}...")
    print()

    # --- Run the loop ---
    result = engine._run_loop(
        start_epoch=start_epoch,
        n_epochs=n_epochs,
        epoch_results=epoch_results,
        previous_best_checkpoint=previous_best_checkpoint,
        dataset_files=dataset_files,
        n_existing_rows=n_existing_rows,
    )

    print()
    print("=" * 70)
    print("RESUME COMPLETE")
    print("=" * 70)
    print(f"Total epochs in results: {len(result['epoch_results'])}")

    return result


if __name__ == "__main__":
    main()
