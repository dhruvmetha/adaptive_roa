#!/usr/bin/env python3
"""Export per-point (query_state, gt_label, native probs) for train/val/cal/test,
per adaptive step (epoch), driven by a training run directory.

The set of probability arrays written is whatever the run's predictor declares
native: classifier -> p_success ; FM (generative) -> p_success, p_failure, p_invalid.

Usage:
  python scripts/export_probabilities.py --run-dir <run_dir> [--out-dir DIR] \
         [--device cuda] [--epochs 0 1 2]
"""
import argparse

from adaptive_roa.probabilistic_classifier.export import export_run

OUT_ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/exp_probabilities"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help=f"output dir (default: {OUT_ROOT}/<run basename>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        from pathlib import Path
        out_dir = str(Path(OUT_ROOT) / Path(args.run_dir.rstrip("/")).name)

    counts = export_run(args.run_dir, out_dir, device=args.device, epochs=args.epochs)
    print(f"done. wrote {len(counts)} epochs to {out_dir}")


if __name__ == "__main__":
    main()
