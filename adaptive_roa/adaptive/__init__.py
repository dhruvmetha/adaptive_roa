"""
Adaptive Sampling module for efficient trajectory collection.

This module provides adaptive sampling that uses conformal prediction
to focus simulations on uncertain regions (near the separatrix).

Main components:
- TrajectoryDataSource: Load and access trajectory data pool
- AdaptiveDatasetBuilder: Build endpoint datasets from trajectory indices
- DataManager: Track trajectory and classification data across epochs
- StateSampler: Sample initial states from state space
- Simulator: Interface for running simulations
- AdaptiveSamplingPipeline: Main pipeline orchestrating the loop
"""

from adaptive_roa.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder
from adaptive_roa.adaptive.data_manager import DataManager
from adaptive_roa.adaptive.sampler import StateSampler
from adaptive_roa.adaptive.simulator import Simulator, FileBasedSimulator, PoolSimulator
from adaptive_roa.adaptive.pipeline import AdaptiveSamplingPipeline, AdaptivePipelineConfig

__all__ = [
    "TrajectoryDataSource",
    "TrajectoryDataSourceConfig",
    "AdaptiveDatasetBuilder",
    "DataManager",
    "StateSampler",
    "Simulator",
    "FileBasedSimulator",
    "PoolSimulator",
    "AdaptiveSamplingPipeline",
    "AdaptivePipelineConfig",
]
