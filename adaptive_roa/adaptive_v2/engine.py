"""Unified adaptive training engine (v2)."""

from __future__ import annotations

import glob
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.adaptive.endpoint_evaluation import (
    compute_endpoint_prediction_error,
    sample_val_data_for_optimization,
)
from adaptive_roa.adaptive_v2.pool.trajectory_pool import TrajectoryPool
from adaptive_roa.adaptive_v2.filters.confidence_filter import ConfidencePairFilter
from adaptive_roa.adaptive_v2.types import AcquisitionResult, EpochArtifacts


def _instantiate(cfg_node, *args, **kwargs):
    """Resolve _target_ from cfg_node and construct with cfg_node as first arg."""
    from hydra.utils import get_class
    cls = get_class(cfg_node._target_)
    return cls(cfg_node, *args, **kwargs)


def _convert_numpy(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _convert_numpy(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


class AdaptiveEngine:
    """Decision-complete adaptive loop used by the unified run script."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 42))
        requested_device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"CUDA device requested ({requested_device}) but unavailable; falling back to cpu")
            requested_device = "cpu"
        self.device = requested_device

        self.system_name = cfg.adaptive_v2.get("system_name", "pendulum")
        self.smoke_mode = bool(cfg.adaptive_v2.get("smoke_mode", False))
        self.filter_confident_pairs = bool(cfg.adaptive_v2.get("filter_confident_pairs", False))
        self.filter_min_pairs = int(cfg.adaptive_v2.get("filter_min_pairs", 100))
        self.save_legacy_results_json = bool(cfg.adaptive_v2.get("save_legacy_results_json", False))
        self.system = hydra.utils.instantiate(cfg.system)

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pool = TrajectoryPool(
            data_source_cfg=cfg.data_source,
            output_dir=str(self.output_dir / "datasets"),
            val_ratio=cfg.get("val_ratio", 0.1),
            test_ratio=cfg.get("test_ratio", 0.1),
            candidate_mode=str(cfg.get("candidate_mode", "start")),
        )
        from hydra.utils import get_class
        self.predictor_type = str(cfg.predictor.type)
        trainer_cls = get_class(cfg.predictor.trainer_target)
        self.trainer             = trainer_cls(cfg.predictor, self.system, self.system_name)
        self.probability_backend = _instantiate(cfg.probability,  self.system, self.device)
        self.threshold_backend   = _instantiate(cfg.threshold,    self.system, self.device)
        self.calibration_backend = _instantiate(cfg.calibration,  self.system, self.device)
        self.acquisition         = _instantiate(cfg.acquisition)
        self.evaluator           = _instantiate(cfg.eval,         self.system, self.device)

    @staticmethod
    def _count_file_rows(filepath: str) -> int:
        """Count non-empty lines in a text file."""
        with open(filepath) as f:
            return sum(1 for line in f if line.strip())

    def run(self) -> dict[str, Any]:
        pl.seed_everything(self.seed, workers=True)
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        dataset_kind = "classification" if self.predictor_type == "classifier" else "endpoint"
        dataset_files = self.pool.initialize(
            int(self.cfg.get("initial_train_size", 100)), dataset_kind=dataset_kind
        )

        n_epochs = int(self.cfg.get("n_epochs", 10))
        samples_per_epoch = int(self.cfg.get("samples_per_epoch", 50))
        d2_ratio = float(self.cfg.get("d2_ratio", 0.5))
        warm_start = bool(self.cfg.get("warm_start", False))
        acquisition_mode = self.acquisition.mode
        eval_every = int(self.cfg.get("eval_every", 1))

        epoch_results: list[dict[str, Any]] = []
        previous_best_checkpoint: str | None = None
        # Track unfiltered row count for confidence filter (rows, not trajectories)
        n_existing_rows = self._count_file_rows(dataset_files["train"])

        for epoch in range(n_epochs):
            print("\n" + "=" * 70)
            print(f"EPOCH {epoch}")
            print("=" * 70)

            train_trajectories_this_epoch = self.pool.train_size
            is_last_epoch = (epoch == n_epochs - 1)
            run_eval = (eval_every > 0 and (epoch % eval_every == 0 or is_last_epoch)) and not self.smoke_mode
            epoch_output_dir = self.output_dir / f"epoch_{epoch:03d}"
            epoch_output_dir.mkdir(parents=True, exist_ok=True)

            resume_ckpt = previous_best_checkpoint if (warm_start and previous_best_checkpoint) else None

            model_handle = self.trainer.fit(
                dataset_files=dataset_files,
                output_dir=str(epoch_output_dir),
                resume_checkpoint=resume_ckpt,
            )
            model_handle.eval()
            model_handle.to(self.device)

            epoch_ckpts = glob.glob(str(epoch_output_dir / "checkpoints" / "best*.ckpt"))
            if epoch_ckpts:
                previous_best_checkpoint = epoch_ckpts[0]

            self.probability_backend.bind_model(model_handle)
            self.threshold_backend.bind_model(model_handle)
            self.calibration_backend.bind_model(model_handle, predictor_type=self.predictor_type)

            n_d1_target = int(samples_per_epoch * (1.0 - d2_ratio))
            n_d2_target = samples_per_epoch - n_d1_target
            skip_ranked_for_eval_off = (
                acquisition_mode == "ranked" and eval_every <= 0 and d2_ratio <= 0.0
            )
            need_d2_acquisition = n_d2_target > 0 and not skip_ranked_for_eval_off

            X_val, y_val = sample_val_data_for_optimization(
                self.pool.dataset_builder,
            )
            threshold_state = self.threshold_backend.optimize(X_val, y_val)

            if self.smoke_mode or not run_eval or self.predictor_type == "classifier":
                # classifier has no generated endpoints -> no endpoint-prediction error
                endpoint_error = {"skipped": True}
            else:
                endpoint_error = compute_endpoint_prediction_error(
                    flow_matcher=model_handle,
                    dataset_builder=self.pool.dataset_builder,
                    batch_size=self.cfg.get("val_batch_size", 512),
                    device=self.device,
                    verbose=self.calibration_backend.verbose,
                )

            d1_states, d1_indices = self.pool.sample_candidates_without_marking(n_d1_target)
            if n_d1_target > 0 and len(d1_indices) == 0:
                print("No more trajectories available, stopping.")
                break

            self.pool.mark_indices_as_used(d1_indices)
            self.pool.add_to_training_balanced(d1_indices)

            q_hat = None
            test_metrics = {"coverage": None, "f1": None, "unknown_rate": None}
            if acquisition_mode == "conformal" and need_d2_acquisition:
                d1_labels = self.pool.get_labels(d1_indices)
                q_hat = self.calibration_backend.calibrate(d1_states, d1_labels, threshold_state)
                threshold_state.q_hat = q_hat
                X_test, y_test = self.pool.get_val_labels()
                predictor = self.threshold_backend.predictor
                if predictor is None:
                    raise RuntimeError("Threshold backend predictor missing after bind_model")
                test_metrics = predictor.evaluate(X_test, y_test, verbose=self.calibration_backend.verbose)
            elif acquisition_mode == "conformal":
                print("Skipping q_hat calibration because d2_target=0")

            if need_d2_acquisition:
                acquisition = self.acquisition.select(
                    pool=self.pool,
                    probability_backend=self.probability_backend,
                    threshold_backend=self.threshold_backend,
                    threshold_state=threshold_state,
                    target_count=n_d2_target,
                    exclude=set(d1_indices),
                )
            else:
                skipped_reason = "d2_target_zero"
                if skip_ranked_for_eval_off:
                    skipped_reason = "ranked_disabled_when_eval_every_zero"
                    print("Skipping ranked acquisition because eval_every=0")
                acquisition = AcquisitionResult(
                    diagnostics={"skipped_reason": skipped_reason},
                )
            acquisition.d1_indices = list(d1_indices)

            if acquisition.d2_indices:
                self.pool.mark_indices_as_used(acquisition.d2_indices)
                self.pool.add_to_training_balanced(acquisition.d2_indices)

            q_hat_eval = None
            n_cal_eval = 0
            cal_file = self.cfg.data_source.get("cal_set_file", None)
            if run_eval and cal_file:
                X_cal_eval, _, y_cal_eval = load_eval_states(
                    cal_file, max_rows=self.evaluator.max_eval_rows
                )
                q_hat_eval = self.calibration_backend.calibrate_eval(
                    X_cal_eval, y_cal_eval, threshold_state
                )
                n_cal_eval = len(X_cal_eval)

            threshold_state.q_hat_eval = q_hat_eval

            if not run_eval:
                full_roa_metrics: dict[str, Any] = {"skipped": True}
            else:
                full_roa_metrics = self.evaluator.evaluate_epoch(
                    model_handle,
                    threshold_state,
                    {
                        "eval_states_file": self.cfg.data_source.test_set_file,
                        "batch_size": int(self.cfg.get("val_batch_size", 2048)),
                        "output_dir": str(epoch_output_dir),
                    },
                )
            full_roa_metrics["q_hat_training"] = float(q_hat) if q_hat is not None else None
            full_roa_metrics["q_hat_eval"] = float(q_hat_eval) if q_hat_eval is not None else None
            full_roa_metrics["n_cal_eval"] = int(n_cal_eval)

            dataset_files = self.pool.build_all_datasets(dataset_kind=dataset_kind)

            # n_existing_rows = rows from previous epochs (keep as-is)
            # After build_all_datasets, file has old rows + new rows in order.
            # Update n_existing_rows to total unfiltered count for next epoch.
            n_total_rows = self._count_file_rows(dataset_files["train"])

            filter_diagnostics = None
            prediction_mode = str(self.cfg.get("prediction_mode", "global"))
            if (self.filter_confident_pairs and not self.smoke_mode
                    and prediction_mode != "local" and self.predictor_type != "classifier"):
                _, _, train_labels = self.pool.get_training_data()
                pair_filter = ConfidencePairFilter(
                    probability_backend=self.probability_backend,
                    decision_rule=self.calibration_backend.decision_rule,
                    min_pairs_floor=self.filter_min_pairs,
                )
                filtered_path, filter_diagnostics = pair_filter.filter_train_file(
                    train_file=dataset_files["train"],
                    threshold_state=threshold_state,
                    labels=train_labels,
                    n_existing=n_existing_rows,
                    output_file=str(Path(dataset_files["train"]).parent / "train_filtered.txt"),
                )
                dataset_files["train"] = filtered_path

            n_existing_rows = n_total_rows

            epoch_result = {
                "epoch": int(epoch),
                "train_trajectories": int(train_trajectories_this_epoch),
                "sampling_mode": acquisition_mode,
                "n_d1_added": int(len(d1_indices)),
                "n_d2_added": int(len(acquisition.d2_indices)),
                "n_d2_uncertain": int(len(acquisition.d2_indices) - acquisition.n_invalid_added),
                "n_d2_invalid": int(acquisition.n_invalid_added),
                "n_discarded_certain": int(acquisition.n_certain_discarded),
                "n_ranked_candidates_evaluated": acquisition.diagnostics.get("n_ranked_candidates_evaluated"),
                "ranked_score_threshold": acquisition.diagnostics.get("ranked_score_threshold"),
                "lambda_star": float(threshold_state.lambda_star),
                "delta_star": float(threshold_state.delta_star),
                "q_hat": float(q_hat) if q_hat is not None else None,
                "q_hat_eval": float(q_hat_eval) if q_hat_eval is not None else None,
                "n_cal_eval": int(n_cal_eval),
                "optimize_mode": self.threshold_backend.optimize_mode,
                "test_coverage": test_metrics.get("coverage"),
                "test_f1": test_metrics.get("f1"),
                "test_unknown_rate": test_metrics.get("unknown_rate"),
                "endpoint_error": endpoint_error,
                "full_roa": full_roa_metrics,
                "filter_diagnostics": _convert_numpy(filter_diagnostics.__dict__) if filter_diagnostics else None,
            }
            epoch_results.append(epoch_result)

            conformal_state = None
            if self.threshold_backend.predictor is not None:
                conformal_state = _convert_numpy(self.threshold_backend.predictor.get_state())

            if self.save_legacy_results_json:
                with open(epoch_output_dir / "results.json", "w") as f:
                    json.dump(_convert_numpy(epoch_result), f, indent=2)

                if conformal_state is not None:
                    with open(epoch_output_dir / "conformal_state.json", "w") as f:
                        json.dump(conformal_state, f, indent=2)

            epoch_artifacts = EpochArtifacts(
                epoch=epoch,
                train_trajectories=train_trajectories_this_epoch,
                sampling_mode=acquisition_mode,
                threshold_state=threshold_state,
                acquisition=acquisition,
                endpoint_error=endpoint_error,
                eval_metrics=full_roa_metrics,
                d1_eval_metrics=test_metrics if acquisition_mode == "conformal" else None,
                conformal_state=conformal_state,
                extra={
                    "n_cal_eval": n_cal_eval,
                    "optimize_mode": self.threshold_backend.optimize_mode,
                },
            )
            with open(epoch_output_dir / "artifacts_v2.json", "w") as f:
                json.dump(_convert_numpy(epoch_artifacts.__dict__), f, indent=2)

            hydra_src = self.output_dir / ".hydra"
            if hydra_src.exists():
                hydra_dst = epoch_output_dir / ".hydra"
                if not hydra_dst.exists():
                    shutil.copytree(hydra_src, hydra_dst)

            self.pool.save_state(str(self.output_dir / "dataset_builder_state.json"))

        stats = self.pool.get_statistics()
        final_payload = {
            "epoch_results": epoch_results,
            "final_stats": stats,
            "adaptive_v2": {
                "artifact_schema_version": 2,
                "system_name": self.system_name,
            },
        }
        with open(self.output_dir / "final_results.json", "w") as f:
            json.dump(_convert_numpy(final_payload), f, indent=2)

        return final_payload
