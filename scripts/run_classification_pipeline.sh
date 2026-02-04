#!/bin/bash
# =============================================================================
# Full Classification Pipeline Script
# =============================================================================
# Builds datasets, trains classifiers, and evaluates on eval_states.txt
#
# Usage:
#   ./scripts/run_classification_pipeline.sh quadrotor2d   # Run Q2D pipeline
#   ./scripts/run_classification_pipeline.sh quadrotor3d   # Run Q3D pipeline
#   ./scripts/run_classification_pipeline.sh all           # Run both
# =============================================================================

set -e  # Exit on error

SYSTEM=${1:-"quadrotor2d"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Classification Pipeline${NC}"
echo -e "${GREEN}======================================================================${NC}"

run_pipeline() {
    local SYSTEM=$1

    echo -e "\n${YELLOW}>>> System: ${SYSTEM}${NC}\n"

    # Step 1: Build datasets
    echo -e "${GREEN}Step 1: Building expanded datasets${NC}"

    # Train: splits 0-7 (80%)
    echo "  Building train set..."
    python src/classification/build_expanded_dataset.py \
        --system ${SYSTEM} \
        --split train \
        --indices 0 1 2 3 4 5 6 7

    # Val: split 8 (10%)
    echo "  Building val set..."
    python src/classification/build_expanded_dataset.py \
        --system ${SYSTEM} \
        --split val \
        --indices 8

    # Test: split 9 (10%)
    echo "  Building test set..."
    python src/classification/build_expanded_dataset.py \
        --system ${SYSTEM} \
        --split test \
        --indices 9 \
        --no-balance

    echo -e "${GREEN}Dataset build complete!${NC}"
    echo ""

    # Step 2: Train classifier
    echo -e "${GREEN}Step 2: Training classifier${NC}"
    python src/classification/train.py \
        --config-name=train_${SYSTEM}_classification

    echo -e "${GREEN}Training complete!${NC}"
    echo ""
}

if [ "$SYSTEM" == "all" ]; then
    run_pipeline "quadrotor2d"
    run_pipeline "quadrotor3d"
else
    run_pipeline "$SYSTEM"
fi

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Pipeline complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Find best checkpoint in outputs/${SYSTEM}_classification/<timestamp>/checkpoints/"
echo "  2. Evaluate on eval_states.txt:"
echo "     python src/classification/evaluate_on_eval_states.py \\"
echo "         --checkpoint <checkpoint_path> \\"
echo "         --eval-file /common/users/shared/pracsys/genMoPlan/data_trajectories/<system>/eval_states.txt \\"
echo "         --system ${SYSTEM} \\"
echo "         --delta 0.15"
