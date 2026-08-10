"""Campaign CLI: expand a manifest, report status, launch what is missing."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from adaptive_roa.benchmark.launcher import plan_launch, sbatch_command
from adaptive_roa.benchmark.manifest import expand_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--exp-root", type=Path, required=True)
    ap.add_argument("--launch", action="store_true",
                    help="actually submit; default is a dry run")
    args = ap.parse_args()

    specs = expand_manifest(yaml.safe_load(args.manifest.read_text()))
    to_launch, states = plan_launch(specs, args.exp_root)

    tally = {s: sum(1 for v in states.values() if v == s)
             for s in ("complete", "partial", "absent")}
    print(f"{len(specs)} runs: {tally['complete']} complete, "
          f"{tally['partial']} partial, {tally['absent']} absent")

    for spec in to_launch:
        cmd = sbatch_command(spec, args.exp_root)
        if args.launch:
            subprocess.run(cmd, check=True)
        else:
            print(" ".join(cmd))
    if not args.launch:
        print(f"\ndry run -- {len(to_launch)} would be submitted. Pass --launch.")


if __name__ == "__main__":
    main()
