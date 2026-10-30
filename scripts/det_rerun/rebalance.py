#!/usr/bin/env python
"""Move queued Amarel cells onto free local cards, longest first, and order the rest.

Amarel gives this account low fairshare, so a cell can wait there for days while a
direct-box or iLab card sits idle. Each run:
  1. ranks the cells PENDING on Amarel by expected runtime (EST_HOURS, measured on
     this campaign; longer first, manifest order breaks ties);
  2. reads the VIABLE rows of a fresh gpu-free.sh scan: direct boxes, and on iLab
     only the node that already hosts this account's jobs (the association allows
     one node at a time);
  3. for each free card, longest cell first: confirms the Amarel job is still
     PENDING, cancels it, marks its registry row ~amarel_moved, and relaunches the
     cell there with launch.sh from the same commit;
  4. sets `nice` on the cells left on Amarel so the long ones are considered first.
Prints the plan; acts only with --apply. Usage: rebalance.py [--apply] [--max-moves N]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

HERE = Path("/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun")
WT = Path("/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper")
EXP = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"
GPU_FREE = os.path.expanduser("~/.claude/skills/compute/scripts/gpu-free.sh")
AM = "amarel.rutgers.edu"
ENV = {**os.environ, "SSH_AUTH_SOCK": ""}

# Expected hours per cell on an A5000-class card (uniform runs measured 2026-10-30;
# adaptive rounds add acquisition time). Only the order matters here.
EST_HOURS = {
    "q3d110": {"bnnreg": 30, "fm": 20, "mlpreg": 18, "bnnclf": 6, "mlpclf": 3},
    "q3d60": {"bnnreg": 18, "fm": 12, "mlpreg": 10, "bnnclf": 4, "mlpclf": 2},
    "q3d30": {"bnnreg": 9, "fm": 7, "mlpreg": 5, "bnnclf": 2, "mlpclf": 1},
    "q2d": {"bnnreg": 2.5, "fm": 3.5, "mlpreg": 2, "bnnclf": 0.6, "mlpclf": 0.5},
    "cp": {"bnnreg": 0.6, "fm": 1.2, "mlpreg": 0.5, "bnnclf": 0.4, "mlpclf": 0.4},
    "pen": {"bnnreg": 0.5, "fm": 0.8, "mlpreg": 0.4, "bnnclf": 0.3, "mlpclf": 0.3},
}


def est_hours(name: str) -> float:
    m = re.match(r"dr_([a-z0-9]+)_([a-z]+)_", name)
    return EST_HOURS.get(m.group(1), {}).get(m.group(2), 0.0) if m else 0.0


def nice_for(hours: float) -> int:
    return 0 if hours >= 8 else 50 if hours >= 2 else 100


def sh(cmd: str, timeout: int = 300) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                          env=ENV, executable="/bin/bash").stdout


def amarel_pending() -> list[dict]:
    """Cells whose latest registry row is an Amarel GPU job that is still PENDING."""
    latest = {}
    for line in open(HERE / "launched.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 7:
            latest[f[0]] = f
    rows = [f for f in latest.values() if f[1] == "amarel" and f[2].startswith("--partition=gpu")]
    if not rows:
        return []
    state = {}
    for line in sh(f"ssh -n -o BatchMode=yes {AM} 'squeue -u st1122 -h -o \"%i %T %Q %y\"'").splitlines():
        p = line.split()
        if len(p) == 4:
            state[p[0]] = (p[1], int(p[2]), int(p[3]))
    manifest = {r["name"]: r for r in csv.DictReader(open(HERE / "manifest.tsv"), delimiter="\t")}
    out = []
    for f in rows:
        st = state.get(f[3])
        if st and st[0] == "PENDING" and f[0] in manifest:
            out.append({"name": f[0], "jid": f[3], "commit": f[6], "prio": st[1], "nice": st[2],
                        "mem": manifest[f[0]]["mem"], "order": int(manifest[f[0]]["priority"]),
                        "hours": est_hours(f[0])})
    return sorted(out, key=lambda c: (-c["hours"], c["order"], c["name"]))


def ilab_home_node() -> str | None:
    nodes = {n for n in sh("ssh -n -o BatchMode=yes ilab1.cs.rutgers.edu 'squeue -u st1122 -h -t R -o %N'").split() if n}
    return next(iter(nodes)) if len(nodes) == 1 else None


def free_cards(mem_gb: int) -> list[tuple[str, str, str]]:
    """(world, host_or_gres, gpu) per free card from the VIABLE block of a fresh scan."""
    txt = sh(f"{GPU_FREE} --fit 4 {mem_gb}", timeout=600)
    block = txt.split("VIABLE", 1)[1].split("REJECTED", 1)[0] if "VIABLE" in txt else ""
    home = ilab_home_node()
    cards = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        host = line.split()[0]
        if host.startswith("ilab/"):
            node = host.split("/", 1)[1]
            m = re.search(r"(\d+)/\d+ free.*--gres=gpu:([\w]+):1", line)
            if m and (home is None or node == home):
                cards += [("ilab", m.group(2), "")] * int(m.group(1))
        elif host in ("westeros", "arrakis", "perception"):
            for g in re.findall(r"gpu(\d+) \(", line):
                cards.append(("direct", "1spring-303-desktop-10" if host == "perception" else host, g))
    return cards


def move(c: dict, card: tuple[str, str, str]) -> str:
    world, where, gpu = card
    if sh(f"ssh -n -o BatchMode=yes {AM} 'squeue -h -j {c['jid']} -o %T'").strip() != "PENDING":
        return f"skip {c['name']}: no longer PENDING"
    sh(f"ssh -n -o BatchMode=yes {AM} 'scancel {c['jid']}'")
    st = sh(f"ssh -n -o BatchMode=yes {AM} 'sacct -j {c['jid']} -X -n -o State'").strip()
    if not st.startswith("CANCELLED"):
        return f"FAILED to cancel {c['name']} ({c['jid']}: {st}); left as is"
    lines = open(HERE / "launched.tsv").read().split("\n")
    lines = [("\t".join([l.split("\t")[0] + "~amarel_moved"] + l.split("\t")[1:])
              if l.split("\t")[0] == c["name"] else l) for l in lines]
    tmp = HERE / "launched.tsv.tmp"
    tmp.write_text("\n".join(lines)); os.replace(tmp, HERE / "launched.tsv")
    out = re.search(r"output_dir=(\S+)", next(r["overrides"] for r in csv.DictReader(
        open(HERE / "manifest.tsv"), delimiter="\t") if r["name"] == c["name"])).group(1)
    marker = Path(out) / ".amarel_mirror"
    if marker.exists() and len(list(Path(out).iterdir())) == 1:
        marker.unlink()
    args = f"{c['name']} ilab {where}" if world == "ilab" else f"{c['name']} direct {where} {gpu}"
    res = sh(f"cd {WT} && DET_REF={c['commit']} bash scripts/det_rerun/launch.sh {args} 2>&1 | tail -1",
             timeout=900).strip()
    return f"moved {c['name']} (Amarel {c['jid']}, ~{c['hours']}h) -> {res}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-moves", type=int, default=8)
    a = ap.parse_args()
    pend = amarel_pending()
    print(f"{len(pend)} cells PENDING on Amarel; longest first:")
    for c in pend[:10]:
        print(f"  {c['name']:36s} ~{c['hours']:>4}h  job {c['jid']}  nice {c['nice']}")
    cards = free_cards(max([int(c["mem"].rstrip("G")) for c in pend[:a.max_moves]] or [16]))
    print(f"free cards: {cards}")
    for c, card in list(zip(pend, cards))[:a.max_moves]:
        print(("" if a.apply else "would move: ") + (move(c, card) if a.apply else f"{c['name']} -> {card}"))
    moved = {c["name"] for c, _ in list(zip(pend, cards))[:a.max_moves]} if a.apply else set()
    renice = [c for c in pend if c["name"] not in moved and c["nice"] != nice_for(c["hours"])]
    for c in renice:
        if a.apply:
            sh(f"ssh -n -o BatchMode=yes {AM} 'scontrol update jobid={c['jid']} nice={nice_for(c['hours'])}'")
    print(f"{'set' if a.apply else 'would set'} nice on {len(renice)} Amarel cells (0 for >=8h, 50 for 2-8h, 100 below)")


if __name__ == "__main__":
    main()
