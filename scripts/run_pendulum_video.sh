#!/bin/bash
#SBATCH --job-name=pend-video
#SBATCH --output=slurm_logs/pend_video_%j.out
#SBATCH --error=slurm_logs/pend_video_%j.err
#SBATCH --gres=gpu:4500_ada:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Usage:
#   sbatch scripts/run_pendulum_video.sh cache
#   sbatch scripts/run_pendulum_video.sh cache runc05
#   sbatch scripts/run_pendulum_video.sh cache runc0
#   sbatch scripts/run_pendulum_video.sh cache runc1
#   sbatch scripts/run_pendulum_video.sh render
#   sbatch scripts/run_pendulum_video.sh full

set -e

PROJECT_DIR="/common/home/dm1487/robotics_research/tripods/olympics-classifier"
PYTHON="/common/users/dm1487/envs/arcmg/bin/python"

cd "$PROJECT_DIR"
mkdir -p slurm_logs

MODE="${1:-full}"
METHOD="${2:-}"

METHOD_ARG=""
if [ -n "$METHOD" ]; then
    METHOD_ARG="--methods $METHOD"
fi

case "$MODE" in
    cache)
        echo "=== Building caches (GPU) ${METHOD:+[$METHOD]} ==="
        $PYTHON scripts/plot_qualitative_pendulum_video.py --cache_only --device cuda:0 --batch_size 1000000 $METHOD_ARG
        ;;
    render)
        echo "=== Rendering videos with sample overlay (CPU) ==="
        $PYTHON scripts/plot_qualitative_pendulum_video.py --show_samples --combined_only
        ;;
    full)
        echo "=== Building caches (GPU) ==="
        $PYTHON scripts/plot_qualitative_pendulum_video.py --cache_only --device cuda:0 --batch_size 1000000 $METHOD_ARG
        echo ""
        echo "=== Rendering videos with sample overlay ==="
        $PYTHON scripts/plot_qualitative_pendulum_video.py --show_samples $METHOD_ARG
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: sbatch $0 [cache|render|full] [runc0|runc05|runc1]"
        exit 1
        ;;
esac

echo ""
echo "Done! Outputs in: results/videos_and_images/pendulum/"
