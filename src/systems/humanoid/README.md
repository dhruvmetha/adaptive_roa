# Humanoid System Scripts

This directory contains scripts for creating and analyzing humanoid trajectory datasets.

## Scripts

### `create_humanoid_datasets.py`

Creates reach and stable_split datasets from humanoid trajectory data.

**Usage:**
```bash
# Create both reach and stable_split datasets
python src/systems/humanoid/create_humanoid_datasets.py humanoid_get_up_slow
python src/systems/humanoid/create_humanoid_datasets.py humanoid_get_up_medium

# Only regenerate shuffled_indices.txt (fast)
python src/systems/humanoid/create_humanoid_datasets.py humanoid_get_up_medium --only-shuffled-indices

# Create only reach dataset
python src/systems/humanoid/create_humanoid_datasets.py humanoid_get_up_slow --skip-stable-split

# Create only stable_split dataset
python src/systems/humanoid/create_humanoid_datasets.py humanoid_get_up_slow --skip-reach
```

**What it creates:**
- `{dataset_name}_reach/`: Trajectories truncated at first success point
- `{dataset_name}_stable_split/`: Trajectories recursively split at every success entry
- `roa_labels.txt`: ROA labels for each trajectory
- `shuffled_indices.txt`: Shuffled indices for train/test splits
- `dataset_description.json`: Complete dataset metadata

**Success criteria:**
- Head height ≥ 1.4m (index 21)
- Torso vertical z-component ≥ 0.9 (index 36)
- Horizontal speed ≤ 0.2 m/s (indices 37-38)

### `analyze_stable_split_transitions.py`

Analyzes transition statistics for stable_split datasets.

**Usage:**
```bash
python src/systems/humanoid/analyze_stable_split_transitions.py
```

**What it computes:**
- Success → Success: Segments starting and ending in success region
- Success → Failure: Segments starting in success but ending in failure
- Failure → Success: Segments starting in failure but ending in success
- Failure → Failure: Segments starting and ending in failure region

**Output:**
Prints detailed transition statistics for both medium and slow stable_split datasets.

## Dataset Locations

Default locations:
- **Source**: `/common/users/shared/pracsys/genMoPlan/data_trajectories/{dataset_name}`
- **Output**: `/common/users/dm1487/arcmg_datasets/{dataset_name}_reach` and `{dataset_name}_stable_split`

## Success Criteria

All three conditions must be met:
1. Head height ≥ 1.4m
2. Torso nearly upright (z ≥ 0.9)
3. Low horizontal speed (≤ 0.2 m/s)
