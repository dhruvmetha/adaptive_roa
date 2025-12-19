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

from src.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from src.adaptive.dataset_builder import AdaptiveDatasetBuilder
from src.adaptive.data_manager import DataManager
from src.adaptive.sampler import StateSampler
from src.adaptive.simulator import Simulator, FileBasedSimulator, PoolSimulator
from src.adaptive.pipeline import AdaptiveSamplingPipeline, AdaptivePipelineConfig

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
