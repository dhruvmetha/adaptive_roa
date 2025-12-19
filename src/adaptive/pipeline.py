"""
Adaptive Sampling Pipeline.

Orchestrates the full adaptive sampling loop combining conformal prediction
with flow matching for efficient data collection.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

from src.conformal.config import ConformalConfig
from src.conformal.predictor import ConformalPredictor
from src.adaptive.data_manager import DataManager, DataManagerConfig
from src.adaptive.sampler import StateSampler
from src.adaptive.simulator import Simulator


@dataclass
class AdaptivePipelineConfig:
    """Configuration for adaptive sampling pipeline."""
    # Sampling
    n_samples_per_epoch: int = 50   # New states to sample each epoch
    d1_ratio: float = 0.5           # Fraction of samples for calibration (D1)

    # Stopping criteria
    max_epochs: int = 10
    convergence_threshold: float = 0.01  # Stop if improvement < this

    # Output
    output_dir: str = "outputs/adaptive"
    save_intermediate: bool = True


class AdaptiveSamplingPipeline:
    """
    Main pipeline for adaptive sampling with conformal prediction.

    Orchestrates the loop:
    1. Sample new initial states
    2. Split into D1 (calibration) and D2 (selection pool)
    3. Simulate D1
    4. Update conformal predictor
    5. Select uncertain points from D2
    6. Simulate uncertain D2
    7. Retrain flow matcher
    8. Evaluate on test set

    Attributes:
        system: Dynamical system
        simulator: Simulator interface
        conformal_config: ConformalConfig
        pipeline_config: AdaptivePipelineConfig
        device: Device for computation
    """

    def __init__(
        self,
        system,
        simulator: Simulator,
        conformal_config: ConformalConfig,
        pipeline_config: Optional[AdaptivePipelineConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize adaptive sampling pipeline.

        Args:
            system: Dynamical system instance
            simulator: Simulator for running/looking up trajectories
            conformal_config: ConformalConfig for conformal prediction
            pipeline_config: AdaptivePipelineConfig (uses defaults if None)
            device: Device for computation
        """
        self.system = system
        self.simulator = simulator
        self.conformal_config = conformal_config
        self.pipeline_config = pipeline_config or AdaptivePipelineConfig()
        self.device = device

        # Get state dimension from system
        components = system.define_manifold_structure()
        self.state_dim = sum(c.dim for c in components)

        # Initialize components
        self.data_manager = DataManager(
            self.state_dim,
            DataManagerConfig()
        )
        self.sampler = StateSampler(system, device)

        # Flow matcher and conformal predictor (set during run)
        self.flow_matcher = None
        self.conformal_predictor = None

        # Output directory
        self.output_dir = Path(self.pipeline_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize_with_data(
        self,
        trajectory_file: str,
        flow_matcher_factory: Callable[[], Any],
        verbose: bool = True
    ) -> Dict:
        """
        Initialize pipeline with existing trajectory data.

        Args:
            trajectory_file: Path to initial trajectory file
            flow_matcher_factory: Callable that creates a new flow matcher instance
            verbose: Print initialization info

        Returns:
            Dict with initialization statistics
        """
        if verbose:
            print("=" * 70)
            print("INITIALIZING ADAPTIVE SAMPLING PIPELINE")
            print("=" * 70)

        # Load initial data
        stats = self.data_manager.initialize_from_file(
            trajectory_file,
            self.system,
            self.conformal_config.attractor_radius
        )

        if verbose:
            print(f"\nLoaded initial data:")
            print(f"  Training: {stats['train_size']} trajectories")
            print(f"  Test: {stats['test_size']} trajectories")
            print(f"  Success rate (train): {stats['success_rate_train']:.2%}")
            print(f"  Success rate (test): {stats['success_rate_test']:.2%}")

        # Store factory for creating fresh flow matchers
        self.flow_matcher_factory = flow_matcher_factory

        return stats

    def train_flow_matcher(
        self,
        trainer_fn: Callable[[str], Any],
        verbose: bool = True
    ):
        """
        Train flow matcher on current accumulated data.

        Args:
            trainer_fn: Function that takes trajectory file path and returns trained model
            verbose: Print training info
        """
        if verbose:
            print(f"\n[FM] Training flow matcher on {len(self.data_manager.labels)} trajectories...")

        # Save current data to file
        temp_file = self.output_dir / "current_trajectories.txt"
        self.data_manager.save_trajectory_file(str(temp_file))

        # Train flow matcher
        self.flow_matcher = trainer_fn(str(temp_file))

        # Create new conformal predictor with fresh flow matcher
        self.conformal_predictor = ConformalPredictor(
            self.flow_matcher,
            self.system,
            self.conformal_config,
            self.device
        )

        if verbose:
            print(f"[FM] Training complete")

    def fit_conformal_predictor(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Fit conformal predictor with fresh calibration data.

        Args:
            X_cal: [N, state_dim] calibration states
            y_cal: [N] calibration labels
            verbose: Print fitting info

        Returns:
            Dict with fitting info
        """
        if self.conformal_predictor is None:
            raise RuntimeError("Must train flow matcher before fitting conformal predictor")

        # Get training data for lambda optimization
        X_train, y_train, _, _ = self.data_manager.get_cp_data_split(cal_ratio=0.0)

        # Fit conformal predictor
        return self.conformal_predictor.fit(
            X_train, y_train, X_cal, y_cal, verbose=verbose
        )

    def run_epoch(
        self,
        epoch: int,
        trainer_fn: Callable[[str], Any],
        verbose: bool = True
    ) -> Dict:
        """
        Run one epoch of adaptive sampling.

        Steps:
        1. Sample new initial states
        2. Split into D1 (calibration) and D2 (pool)
        3. Simulate D1
        4. Update conformal predictor (refit)
        5. Select uncertain points from D2
        6. Simulate uncertain D2 only
        7. Retrain flow matcher
        8. Evaluate on test set

        Args:
            epoch: Current epoch number
            trainer_fn: Function to train flow matcher
            verbose: Print progress

        Returns:
            Dict with epoch results
        """
        config = self.pipeline_config

        if verbose:
            print("\n" + "=" * 70)
            print(f"EPOCH {epoch}")
            print("=" * 70)

        # Step 1: Sample new initial states
        if verbose:
            print(f"\n[1] Sampling {config.n_samples_per_epoch} new initial states...")

        new_states = self.sampler.sample_uniform(config.n_samples_per_epoch, as_tensor=False)

        # Step 2: Split into D1 and D2
        D1, D2 = self.sampler.split_d1_d2(new_states, config.d1_ratio)
        if verbose:
            print(f"    D1 (calibration): {len(D1)} points")
            print(f"    D2 (selection pool): {len(D2)} points")

        # Step 3: Simulate D1 (always)
        if verbose:
            print(f"\n[2] Simulating D1 ({len(D1)} simulations)...")

        D1_ends, D1_labels = self.simulator.simulate(D1)

        # Add D1 to data (for FM training)
        self.data_manager.add_trajectories(D1, D1_ends, D1_labels)

        # Step 4: Retrain flow matcher on all data so far
        if verbose:
            print(f"\n[3] Retraining flow matcher...")
        self.train_flow_matcher(trainer_fn, verbose=verbose)

        # Step 5: Fit conformal predictor with D1 as calibration
        if verbose:
            print(f"\n[4] Fitting conformal predictor...")
        fit_info = self.fit_conformal_predictor(D1, D1_labels, verbose=verbose)

        # Step 6: Select uncertain points from D2
        if verbose:
            print(f"\n[5] Evaluating D2 for uncertain points...")

        uncertain_mask, uncertain_indices, p_success_D2 = self.conformal_predictor.select_uncertain(D2)
        n_uncertain = np.sum(uncertain_mask)
        n_confident = len(D2) - n_uncertain

        if verbose:
            print(f"    Uncertain: {n_uncertain} points")
            print(f"    Confident: {n_confident} points (skipped)")

        # Step 7: Simulate uncertain D2 only
        n_d2_simulated = 0
        if n_uncertain > 0:
            if verbose:
                print(f"\n[6] Simulating uncertain points ({n_uncertain} simulations)...")

            D2_uncertain = D2[uncertain_mask]
            D2_ends, D2_labels = self.simulator.simulate(D2_uncertain)

            # Add to data
            self.data_manager.add_trajectories(D2_uncertain, D2_ends, D2_labels)
            n_d2_simulated = n_uncertain
        else:
            if verbose:
                print(f"\n[6] No uncertain points to simulate")

        # Total simulations this epoch
        n_simulated = len(D1) + n_d2_simulated
        n_skipped = n_confident

        # Step 8: Evaluate on test set
        if verbose:
            print(f"\n[7] Evaluating on test set...")

        X_test, y_test, _ = self.data_manager.get_test_data()
        test_metrics = self.conformal_predictor.evaluate(X_test, y_test, verbose=verbose)

        # Record epoch
        self.data_manager.record_epoch(
            n_simulated, n_skipped,
            self.conformal_predictor.lambda_star,
            self.conformal_predictor.q_hat,
            test_metrics
        )

        # Epoch summary
        epoch_results = {
            'epoch': epoch,
            'n_simulated': n_simulated,
            'n_skipped': n_skipped,
            'total_data': len(self.data_manager.labels),
            'lambda_star': self.conformal_predictor.lambda_star,
            'q_hat': self.conformal_predictor.q_hat,
            'test_metrics': test_metrics,
            'fit_info': fit_info,
        }

        if verbose:
            print("\n" + "-" * 70)
            print(f"EPOCH {epoch} SUMMARY")
            print("-" * 70)
            print(f"  Simulations: {n_simulated} (saved {n_skipped})")
            print(f"  Total data: {epoch_results['total_data']}")
            print(f"  λ* = {epoch_results['lambda_star']:.4f}, q_hat = {epoch_results['q_hat']:.4f}")
            print(f"  Unknown rate: {test_metrics['unknown_rate']:.2%}")
            print(f"  F1 score: {test_metrics['f1']:.2%}")
            print(f"  Coverage: {test_metrics['coverage']:.2%}")

        # Save intermediate results
        if config.save_intermediate:
            self._save_epoch_results(epoch, epoch_results)

        return epoch_results

    def run(
        self,
        trainer_fn: Callable[[str], Any],
        n_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run full adaptive sampling pipeline.

        Args:
            trainer_fn: Function to train flow matcher (takes trajectory file, returns model)
            n_epochs: Number of epochs (uses config default if None)
            verbose: Print progress

        Returns:
            Dict with final results and history
        """
        n_epochs = n_epochs or self.pipeline_config.max_epochs

        if verbose:
            print("\n" + "=" * 70)
            print("STARTING ADAPTIVE SAMPLING PIPELINE")
            print("=" * 70)
            print(f"System: {self.system}")
            print(f"Epochs: {n_epochs}")
            print(f"Samples per epoch: {self.pipeline_config.n_samples_per_epoch}")
            print(f"D1 ratio: {self.pipeline_config.d1_ratio}")
            print(f"Output: {self.output_dir}")

        # Initial flow matcher training
        if verbose:
            print("\n[INITIAL] Training flow matcher on initial data...")
        self.train_flow_matcher(trainer_fn, verbose=verbose)

        # Initial conformal predictor fit
        X_train, y_train, X_cal, y_cal = self.data_manager.get_cp_data_split(cal_ratio=0.3)
        self.fit_conformal_predictor(X_cal, y_cal, verbose=verbose)

        # Initial evaluation
        X_test, y_test, _ = self.data_manager.get_test_data()
        initial_metrics = self.conformal_predictor.evaluate(X_test, y_test, verbose=verbose)

        # Run epochs
        epoch_results = []
        for epoch in range(n_epochs):
            results = self.run_epoch(epoch, trainer_fn, verbose=verbose)
            epoch_results.append(results)

            # Check convergence
            if epoch > 0:
                prev_f1 = epoch_results[epoch - 1]['test_metrics']['f1']
                curr_f1 = results['test_metrics']['f1']
                improvement = curr_f1 - prev_f1

                if abs(improvement) < self.pipeline_config.convergence_threshold:
                    if verbose:
                        print(f"\n[CONVERGED] F1 improvement ({improvement:.4f}) below threshold")
                    break

        # Final summary
        final_results = {
            'initial_metrics': initial_metrics,
            'final_metrics': epoch_results[-1]['test_metrics'],
            'n_epochs_run': len(epoch_results),
            'epoch_history': epoch_results,
            'data_statistics': self.data_manager.get_statistics(),
        }

        if verbose:
            print("\n" + "=" * 70)
            print("ADAPTIVE SAMPLING COMPLETE")
            print("=" * 70)
            stats = self.data_manager.get_statistics()
            print(f"Total simulations: {stats['n_simulations_total']}")
            print(f"Simulations saved: {stats['n_simulations_saved']}")
            print(f"Savings rate: {stats['savings_rate']:.2%}")
            print(f"Final data size: {stats['n_trajectories']}")
            print(f"\nInitial F1: {initial_metrics['f1']:.2%}")
            print(f"Final F1: {final_results['final_metrics']['f1']:.2%}")
            print(f"Initial unknown rate: {initial_metrics['unknown_rate']:.2%}")
            print(f"Final unknown rate: {final_results['final_metrics']['unknown_rate']:.2%}")

        # Save final results
        self._save_final_results(final_results)

        return final_results

    def _save_epoch_results(self, epoch: int, results: Dict):
        """Save results for a single epoch."""
        import json
        epoch_file = self.output_dir / f"epoch_{epoch:03d}.json"

        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        with open(epoch_file, 'w') as f:
            json.dump(convert(results), f, indent=2)

    def _save_final_results(self, results: Dict):
        """Save final results."""
        import json

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        # Save JSON summary
        summary_file = self.output_dir / "final_results.json"
        with open(summary_file, 'w') as f:
            json.dump(convert(results), f, indent=2)

        # Save final trajectory file
        final_traj_file = self.output_dir / "final_trajectories.txt"
        self.data_manager.save_trajectory_file(str(final_traj_file))

        print(f"\nResults saved to {self.output_dir}")
