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
    sample_endpoint_data_for_optimization,
)
from adaptive_roa.adaptive_v2.eval.full_roa import FullROAEvaluator
from adaptive_roa.adaptive_v2.pool.trajectory_pool import TrajectoryPool
from adaptive_roa.adaptive_v2.probability.endpoint_mc import EndpointMCProbabilityBackend
from adaptive_roa.adaptive_v2.strategy.conformal import ConformalAcquisitionStrategy
from adaptive_roa.adaptive_v2.strategy.direct import DirectAcquisitionStrategy
from adaptive_roa.adaptive_v2.strategy.ranked import RankedAcquisitionStrategy
from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend
from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import FlowMatchingTrainer
from adaptive_roa.adaptive_v2.types import AcquisitionResult, EpochArtifacts
from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.calibrator import Calibrator


def _convert_numpy(obj: Any) -> Any:
    if is_dataclass(obj):
        return _convert_numpy(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def _build_strategy(mode: str):
    if mode == "ranked":
        return RankedAcquisitionStrategy()
    if mode == "conformal":
        return ConformalAcquisitionStrategy()
    return DirectAcquisitionStrategy()


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
        self.save_legacy_results_json = bool(cfg.adaptive_v2.get("save_legacy_results_json", False))
        self.system = hydra.utils.instantiate(cfg.system)

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pool = TrajectoryPool(
            data_source_cfg=cfg.data_source,
            output_dir=str(self.output_dir / "datasets"),
            val_ratio=cfg.get("val_ratio", 0.1),
            test_ratio=cfg.get("test_ratio", 0.1),
        )
        self.trainer = FlowMatchingTrainer(cfg, self.system, self.system_name)
        self.probability_backend = EndpointMCProbabilityBackend(self.system, cfg, self.device)
        self.threshold_backend = ConformalThresholdBackend(self.system, cfg, self.device)
        self.evaluator = FullROAEvaluator(self.system, cfg, self.device)

    def run(self) -> dict[str, Any]:
        pl.seed_everything(self.seed, workers=True)

        dataset_files = self.pool.initialize(int(self.cfg.get("initial_train_size", 100)))

        n_epochs = int(self.cfg.get("n_epochs", 10))
        samples_per_epoch = int(self.cfg.get("samples_per_epoch", 50))
        d2_ratio = float(self.cfg.get("d2_ratio", 0.5))
        warm_start = bool(self.cfg.get("warm_start", False))
        sampling_mode = str(self.cfg.get("sampling_mode", "conformal"))

        epoch_results: list[dict[str, Any]] = []
        previous_best_checkpoint: str | None = None

        for epoch in range(n_epochs):
            print("\n" + "=" * 70)
            print(f"EPOCH {epoch}")
            print("=" * 70)

            train_trajectories_this_epoch = self.pool.train_size
            epoch_output_dir = self.output_dir / f"epoch_{epoch:03d}"
            epoch_output_dir.mkdir(parents=True, exist_ok=True)

            resume_ckpt = previous_best_checkpoint if (warm_start and previous_best_checkpoint) else None

            model_handle = self.trainer.fit(
                train_file=dataset_files["train"],
                val_file=dataset_files["val"],
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

            X_train, y_train = sample_endpoint_data_for_optimization(
                self.pool.dataset_builder,
            )
            threshold_state = self.threshold_backend.optimize(X_train, y_train)

            if self.smoke_mode:
                endpoint_error = {"smoke_mode": True}
            else:
                endpoint_error = compute_endpoint_prediction_error(
                    flow_matcher=model_handle,
                    dataset_builder=self.pool.dataset_builder,
                    batch_size=self.cfg.get("val_batch_size", 512),
                    device=self.device,
                    verbose=self.cfg.conformal.get("verbose", True),
                )

            n_d1_target = int(samples_per_epoch * (1.0 - d2_ratio))
            n_d2_target = samples_per_epoch - n_d1_target

            d1_states, d1_indices = self.pool.sample_candidates_without_marking(n_d1_target)
            if len(d1_indices) == 0:
                print("No more trajectories available, stopping.")
                break

            self.pool.mark_indices_as_used(d1_indices)
            self.pool.add_to_training_balanced(d1_indices)

            q_hat = None
            test_metrics = {"coverage": None, "f1": None, "unknown_rate": None}
            if sampling_mode == "conformal":
                d1_labels = self.pool.get_labels(d1_indices)
                q_hat = self.threshold_backend.calibrate_qhat(d1_states, d1_labels, threshold_state)
                threshold_state.q_hat = q_hat
                X_test, y_test = self.pool.get_test_labels()
                test_metrics = self.threshold_backend.predictor.evaluate(
                    X_test,
                    y_test,
                    verbose=self.cfg.conformal.get("verbose", True),
                )

            strategy = _build_strategy(sampling_mode)
            acquisition = strategy.select(
                pool=self.pool,
                probability_backend=self.probability_backend,
                threshold_backend=self.threshold_backend,
                threshold_state=threshold_state,
                cfg=self.cfg,
                target_count=n_d2_target,
                exclude=set(d1_indices),
            )
            acquisition.d1_indices = list(d1_indices)

            if acquisition.d2_indices:
                self.pool.mark_indices_as_used(acquisition.d2_indices)
                self.pool.add_to_training_balanced(acquisition.d2_indices)

            # Held-out calibration for evaluation-time q_hat
            q_hat_eval = None
            n_cal_eval = 0
            cal_file = self.cfg.data_source.get("cal_set_file", None)
            if cal_file:
                X_cal_eval, _, y_cal_eval = load_eval_states(cal_file)
                cal_probs = self.probability_backend.estimate(X_cal_eval)
                eval_conformal = ConformalConfig(
                    delta=threshold_state.delta_star,
                    alpha=self.cfg.conformal.get("alpha_eval", 0.1),
                    decision_rule=self.cfg.conformal.get("decision_rule", "two_sided"),
                )
                eval_calibrator = Calibrator(eval_conformal)
                p_failure = cal_probs.p_failure if eval_conformal.decision_rule == "two_sided" else None
                q_hat_eval = float(
                    eval_calibrator.calibrate(
                        cal_probs.p_success,
                        y_cal_eval,
                        threshold_state.lambda_star,
                        threshold_state.delta_star,
                        p_failure=p_failure,
                        verbose=self.cfg.conformal.get("verbose", True),
                    )
                )
                n_cal_eval = len(X_cal_eval)

            threshold_state.q_hat_eval = q_hat_eval

            if self.smoke_mode:
                full_roa_metrics = {"smoke_mode": True}
            else:
                full_roa_metrics = self.evaluator.evaluate_epoch(
                    model_handle,
                    threshold_state,
                    {
                        "eval_states_file": self.cfg.data_source.test_set_file,
                        "num_mc_samples": int(self.cfg.conformal.get("num_mc_samples_eval", 20)),
                        "batch_size": int(self.cfg.get("val_batch_size", 2048)),
                        "attractor_radius": float(self.cfg.conformal.get("attractor_radius", 0.2)),
                        "output_dir": str(epoch_output_dir),
                        "verbose": bool(self.cfg.conformal.get("verbose", True)),
                        "decision_rule": self.cfg.conformal.get("decision_rule", "two_sided"),
                    },
                )
            full_roa_metrics["q_hat_training"] = float(q_hat) if q_hat is not None else None
            full_roa_metrics["q_hat_eval"] = float(q_hat_eval) if q_hat_eval is not None else None
            full_roa_metrics["n_cal_eval"] = int(n_cal_eval)

            dataset_files = self.pool.build_all_datasets()

            epoch_result = {
                "epoch": int(epoch),
                "train_trajectories": int(train_trajectories_this_epoch),
                "sampling_mode": sampling_mode,
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
                "optimize_mode": self.cfg.conformal.get("optimize_mode", "lambda"),
                "test_coverage": test_metrics.get("coverage"),
                "test_f1": test_metrics.get("f1"),
                "test_unknown_rate": test_metrics.get("unknown_rate"),
                "endpoint_error": endpoint_error,
                "full_roa": full_roa_metrics,
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
                sampling_mode=sampling_mode,
                threshold_state=threshold_state,
                acquisition=acquisition,
                endpoint_error=endpoint_error,
                eval_metrics=full_roa_metrics,
                d1_eval_metrics=test_metrics if sampling_mode == "conformal" else None,
                conformal_state=conformal_state,
                extra={
                    "n_cal_eval": n_cal_eval,
                    "optimize_mode": self.cfg.conformal.get("optimize_mode", "lambda"),
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
