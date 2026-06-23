# tests/test_humanoid_standup_reach_eval_config.py
from pathlib import Path
import numpy as np
import torch
import pytest
from omegaconf import OmegaConf
import yaml

REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"
pytestmark = pytest.mark.skipif(not Path(DATASET_DIR).exists(), reason="shared humanoid dataset not available")


def test_eval_config_targets_humanoid_flow_matcher():
    cfg = yaml.safe_load((REPO / "configs/evaluate_humanoid_standup_reach_roa.yaml").read_text())
    assert cfg["system"]["module"] == "adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher"
    assert cfg["system"]["class"] == "HumanoidStandUpReachLatentConditionalFlowMatcher"
    assert cfg["checkpoint"]["base_name"] == "humanoid_standup_reach_latent_conditional_fm"
    # 67-D state → batch sized down; attractor_radius present (unused by humanoid classify but required by API)
    assert cfg["evaluation"]["num_steps"] >= 1


def test_eval_file_rows_classify_without_angle_wrapping():
    # The fps test set is (state[67], final[67], label) comma-delimited, 135 cols.
    fps = Path(DATASET_DIR) / "train_test_splits/test_set_fps.txt"
    data = np.loadtxt(fps, delimiter=",", max_rows=16)
    assert data.shape[1] == 135
    states = torch.tensor(data[:, :67], dtype=torch.float32)
    from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem
    system = HumanoidStandUpReachSystem(dataset_dir=DATASET_DIR)
    labels = system.classify_attractor(states)
    assert labels.shape == (16,)
    assert set(labels.tolist()) <= {1, -1}
