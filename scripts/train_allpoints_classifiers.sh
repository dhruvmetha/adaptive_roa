#!/bin/bash
# =============================================================================
# All-Points Classification Training Pipeline
# =============================================================================
# Usage:
#   ./scripts/train_allpoints_classifiers.sh quadrotor3d 6000    # Train Quadrotor 3D with 6000 trajectories
#   ./scripts/train_allpoints_classifiers.sh quadrotor2d 4400    # Train Quadrotor 2D with 4400 trajectories
#   ./scripts/train_allpoints_classifiers.sh cartpole 5000       # Train CartPole with 5000 trajectories
#   ./scripts/train_allpoints_classifiers.sh quadrotor3d         # Default: 15000 trajectories
# =============================================================================

set -e  # Exit on error

# Activate environment
export PATH="/common/users/dm1487/envs/arcmg/bin:$PATH"

# Change to project directory
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier

SYSTEM=${1:-"quadrotor3d"}
NUM_TRAJ=${2:-15000}

# Calculate splits (90% train, 10% val)
TRAIN_END=$((NUM_TRAJ * 90 / 100))
VAL_END=$NUM_TRAJ

# =============================================================================
# Helper Functions
# =============================================================================

build_dataset() {
    local system=$1
    local config="build_${system}_allpoints"
    local dest_dir="/common/users/dm1487/arcmg_datasets/${system}_allpoints"

    echo "============================================================"
    echo "Building ${system} all-points dataset"
    echo "============================================================"
    echo "Total trajectories: $NUM_TRAJ"
    echo "  Train: 0 - $TRAIN_END ($TRAIN_END trajectories)"
    echo "  Val:   $TRAIN_END - $VAL_END ($((VAL_END - TRAIN_END)) trajectories)"
    echo "============================================================"

    mkdir -p "$dest_dir"

    # Train set (all points)
    echo "Building train set..."
    python src/build_shuffled_endpoint_dataset.py --config-name=$config \
        start=0 end=$TRAIN_END increment=train type=train

    # Validation set (all points)
    echo "Building validation set..."
    python src/build_shuffled_endpoint_dataset.py --config-name=$config \
        start=$TRAIN_END end=$VAL_END increment=val type=train

    echo "Dataset built at: $dest_dir"
    ls -lh "$dest_dir"/*.txt 2>/dev/null || echo "No files found yet"
    echo ""
}

train_model() {
    local system=$1
    local config="train_${system}_allpoints_classification"

    echo "============================================================"
    echo "Training ${system} all-points classifier"
    echo "============================================================"
    echo "Config: $config"
    echo ""

    python src/classification/train_allpoints.py --config-name=$config
}

run_pipeline() {
    local system=$1

    echo ""
    echo "############################################################"
    echo "#                    ${system^^}                          "
    echo "############################################################"
    echo ""

    # Build dataset
    build_dataset "$system"

    # Train
    train_model "$system"
}

# =============================================================================
# Main
# =============================================================================

echo "============================================================"
echo "All-Points Classification Training Pipeline"
echo "============================================================"
echo "System: $SYSTEM"
echo "Trajectories: $NUM_TRAJ"
echo "Splits: train=$TRAIN_END (90%), val=$((VAL_END - TRAIN_END)) (10%)"
echo "Environment: $(which python)"
echo "Working directory: $(pwd)"
echo "============================================================"
echo ""

case $SYSTEM in
    quadrotor3d|quadrotor2d|cartpole)
        run_pipeline "$SYSTEM"
        ;;
    *)
        echo "Unknown system: $SYSTEM"
        echo "Usage: $0 {quadrotor3d|quadrotor2d|cartpole} [num_trajectories]"
        echo ""
        echo "Examples:"
        echo "  $0 quadrotor3d 6000    # Train Quadrotor 3D with 6000 trajectories"
        echo "  $0 quadrotor2d 4400    # Train Quadrotor 2D with 4400 trajectories"
        echo "  $0 cartpole 5000       # Train CartPole with 5000 trajectories"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
