#!/usr/bin/env python3
"""Resume an adaptive run one round at a time until a metric plateaus.

The ordinary resume script accepts a total epoch count, so it cannot inspect an
evaluation metric between adaptive rounds.  This driver invokes it for exactly
one new epoch at a time, reads the completed artifact, and stops at either a
trajectory cap or metric-based patience.  Each invocation remains independently
resumable through dataset_builder_state.json.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRIC = "full_roa.conservative_lambda_delta.f1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _completed_results(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "final_results.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    results = _load_json(path).get("epoch_results", [])
    return sorted(results, key=lambda item: int(item["epoch"]))


def _pool_size(run_dir: Path) -> int:
    path = run_dir / "dataset_builder_state.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return len(_load_json(path).get("train_indices", []))


def _nested_metric(result: dict[str, Any], dotted_path: str) -> float:
    value: Any = result
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(
                f"metric {dotted_path!r} missing from epoch {result.get('epoch')}"
            )
        value = value[key]
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(
            f"metric {dotted_path!r} is not finite at epoch {result.get('epoch')}: {metric}"
        )
    return metric


def improvement_state(
    results: list[dict[str, Any]], metric_path: str, min_delta: float
) -> tuple[float, int, list[dict[str, Any]]]:
    """Return best metric, consecutive stale rounds, and compact history."""
    best = -math.inf
    stale_rounds = 0
    history: list[dict[str, Any]] = []
    for result in results:
        metric = _nested_metric(result, metric_path)
        improved = metric > best + min_delta
        if improved:
            best = metric
            stale_rounds = 0
        else:
            stale_rounds += 1
        history.append(
            {
                "epoch": int(result["epoch"]),
                "train_trajectories": int(result["train_trajectories"]),
                "metric": metric,
                "improved": improved,
            }
        )
    return best, stale_rounds, history


def _write_status(
    run_dir: Path,
    *,
    status: str,
    metric_path: str,
    min_delta: float,
    patience: int,
    max_trajectories: int,
    best: float,
    stale_rounds: int,
    history: list[dict[str, Any]],
) -> None:
    payload = {
        "status": status,
        "metric_path": metric_path,
        "min_delta": min_delta,
        "patience": patience,
        "max_trajectories": max_trajectories,
        "best_metric": best,
        "stale_rounds": stale_rounds,
        "history": history,
    }
    path = run_dir / "adaptive_stop.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume adaptive rounds until a trajectory cap or metric plateau"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-trajectories", type=int, default=2000)
    parser.add_argument("--samples-per-round", type=int, default=50)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers passed to each resumed round; 0 is NFS-safe",
    )
    parser.add_argument(
        "--max-new-rounds",
        type=int,
        default=None,
        help="Testing/operations cap; stop after this many new rounds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be positive")
    if args.samples_per_round <= 0:
        raise ValueError("--samples-per-round must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be nonnegative")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if args.max_new_rounds is not None and args.max_new_rounds <= 0:
        raise ValueError("--max-new-rounds must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")

    lock_path = run_dir / "adaptive_continue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another continuation owns {lock_path}") from exc

        new_rounds = 0
        while True:
            results = _completed_results(run_dir)
            if not results:
                raise RuntimeError("no completed epochs to continue")
            pool_size = _pool_size(run_dir)
            best, stale_rounds, history = improvement_state(
                results, args.metric, args.min_delta
            )
            last = results[-1]
            last_size = int(last["train_trajectories"])

            print(
                f"State: epoch={last['epoch']} evaluated={last_size} pool={pool_size} "
                f"best={best:.6f} stale={stale_rounds}/{args.patience}",
                flush=True,
            )

            if last_size >= args.max_trajectories:
                status = "max_trajectories"
            elif stale_rounds >= args.patience:
                status = "metric_plateau"
            elif args.max_new_rounds is not None and new_rounds >= args.max_new_rounds:
                status = "round_limit"
            else:
                status = "running"

            _write_status(
                run_dir,
                status=status,
                metric_path=args.metric,
                min_delta=args.min_delta,
                patience=args.patience,
                max_trajectories=args.max_trajectories,
                best=best,
                stale_rounds=stale_rounds,
                history=history,
            )
            if status != "running":
                print(f"Stopping: {status}", flush=True)
                return 0

            if pool_size > args.max_trajectories:
                raise RuntimeError(
                    f"saved pool ({pool_size}) exceeds cap ({args.max_trajectories})"
                )
            if pool_size != last_size + args.samples_per_round:
                raise RuntimeError(
                    "expected the saved pool to be exactly one acquisition batch ahead "
                    f"of evaluation, got evaluated={last_size}, pool={pool_size}, "
                    f"batch={args.samples_per_round}"
                )

            next_epoch = int(last["epoch"]) + 1
            target_epoch_count = next_epoch + 1
            acquisition_size = (
                0 if pool_size == args.max_trajectories else args.samples_per_round
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("resume_adaptive.py")),
                "--run-dir",
                str(run_dir),
                "--n-epochs",
                str(target_epoch_count),
                "--samples-per-epoch",
                str(acquisition_size),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.device is not None:
                command.extend(["--device", args.device])

            print(
                f"Starting epoch {next_epoch}: train/evaluate {pool_size} trajectories; "
                f"then acquire {acquisition_size}",
                flush=True,
            )
            subprocess.run(command, check=True)

            updated = _completed_results(run_dir)
            if int(updated[-1]["epoch"]) != next_epoch:
                raise RuntimeError(
                    f"resume returned without completing expected epoch {next_epoch}"
                )
            if int(updated[-1]["train_trajectories"]) != pool_size:
                raise RuntimeError(
                    f"epoch {next_epoch} evaluated an unexpected pool size: "
                    f"{updated[-1]['train_trajectories']} != {pool_size}"
                )
            new_rounds += 1


if __name__ == "__main__":
    raise SystemExit(main())
