#!/usr/bin/env python
"""Resume ended slots and move PENDING det_v2 runs onto free direct cards.

Prints the plan; acts only with --apply. Each call:
  1. holds off acting while a launch loop appends to the ledger: the ledger changed in
     the last LEDGER_QUIET_S seconds, or a det_v2 launch.sh / resume_*.sh of this account
     is running (--force overrides). The plan is still printed;
  2. resumes each run of this account whose latest iLab or Amarel job ended TIMEOUT,
     PREEMPTED, NODE_FAIL, BOOT_FAIL or DEADLINE short of 11 epochs, from its banked
     epochs, in a fresh slot of the same pool. A run with nothing banked is launched again
     from the same commit (DET_V2_RELAUNCH=1); a Part-X run reruns its launch command,
     which continues from partx_state.pkl. FAILED and OUT_OF_MEMORY are reported only;
  3. reads the VIABLE direct cards of a fresh `gpu-free.sh --fit` scan, drops the hosts in
     HOLD_HOSTS and the cards a det_v2 run of either account holds (ledger pid alive in
     ps, or a claim in progress), and adds SHARED_CARDS that no run holds;
  4. gives each free card the first PENDING run of this account (iLab or Amarel, manifest
     rank order) that the placement rule allows: confirms the job is still PENDING, cancels
     it, appends a cancel row, and resumes the run there from its banked epochs, or
     launches it there when nothing is banked. A RUNNING job is never cancelled.
Placement rule (det_v2): q3d and q2d FM runs need an L40S, A100 or Ada card; MLP and BNN
classifier and final-state runs go on A4000, V100 or 2080 Ti cards; cp and pen FM runs
take any card. Part-X runs are CPU-only and never move. iLab resumes keep to the node
that already hosts this account's jobs (the association allows one node).

    rebalance.py [--apply] [--force] [--max-moves N]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_v2_lib as lib  # noqa: E402

HERE = Path(__file__).resolve().parent
LEDGER_QUIET_S = 300
ENDED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")
NEEDS_A_LOOK = ("FAILED", "OUT_OF_MEMORY")
# Card model per direct host (gpu-free.sh's fixed inventory).
DIRECT_CARDS = {"westeros": "RTX 2080 Ti", "arrakis": "RTX 6000 Ada", "1spring-303-desktop-10": "RTX 4500 Ada"}
SCAN_LABELS = {"westeros": "westeros", "arrakis": "arrakis", "perception": "1spring-303-desktop-10"}
BIG_SETTINGS = ("q3dlqr", "q3dppo", "q2dppo")
SMALL_MODELS = ("mlpclf", "bnnclf", "mlpreg", "bnnreg")
VRAM_GB = 8     # every det_v2 run fits an 11 GB card; the rule, not VRAM, picks the card

ENV: dict = {}


def sh(cmd: str, timeout: int = 300) -> str:
    env = dict(os.environ)
    if ENV.get("SSH_CLEAR_AGENT") == "1":
        env["SSH_AUTH_SOCK"] = ""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                              env=env, executable="/bin/bash").stdout
    except subprocess.TimeoutExpired:
        return ""


def ssh(host: str, cmd: str) -> str:
    return f"ssh -n -o BatchMode=yes {host} '{cmd}'"


# --- placement rule ---------------------------------------------------------------------

def card_class(card: str) -> str:
    """big (L40S, A100, Ada), small (A4000, V100, 2080 Ti) or other."""
    c = card.lower().replace(" ", "").replace("-", "").replace("_", "")
    if any(k in c for k in ("l40s", "a100", "ada")):
        return "big"
    if any(k in c for k in ("a4000", "v100", "volta", "2080")):
        return "small"
    return "other"


def allowed(name: str, card: str) -> bool:
    setting, model = lib.run_kind(name)
    cls = card_class(card)
    if model == "fm":
        return cls == "big" if setting in BIG_SETTINGS else True
    if model in SMALL_MODELS:
        return cls == "small"
    return False            # Part-X and anything unrecognized stay where they are


# --- ledger and queues ------------------------------------------------------------------

def my_placed() -> dict[str, dict]:
    """Latest row per name when it is a launch or resume by this account."""
    return {n: r for n, r in lib.latest_rows(ENV["LEDGER"]).items()
            if r["user"] == ENV["NET_ID"] and r["action"] in ("launch", "resume")}


def manifest() -> dict[str, dict]:
    return {r["name"]: r for r in lib.read_manifest(ENV["MANIFEST"])}


def launch_loop_active() -> str | None:
    led = Path(ENV["LEDGER"])
    if led.exists() and time.time() - led.stat().st_mtime < LEDGER_QUIET_S:
        return f"the ledger changed {int(time.time() - led.stat().st_mtime)} s ago"
    me = {os.getpid(), os.getppid()}
    for line in sh(f"ps -u {ENV['NET_ID']} -o pid=,args=").splitlines():
        pid, _, args = line.strip().partition(" ")
        if pid.isdigit() and int(pid) not in me and re.search(r"det_v2/(launch|resume_\w+)\.sh", args):
            return f"pid {pid} is running {args[:80]}"
    return None


def pending_runs() -> list[dict]:
    """This account's runs whose latest iLab or Amarel job is PENDING, in rank order."""
    placed = {n: r for n, r in my_placed().items() if r["where"] in ("ilab", "amarel")}
    if not placed:
        return []
    pend = set()
    for host in (ENV["ILAB_SUBMIT_HOST"], ENV["AMAREL_HOST"]):
        user = ENV["AMAREL_USER"] if host == ENV["AMAREL_HOST"] else ENV["NET_ID"]
        pend |= set(sh(ssh(host, f"squeue -u {user} -h -t PD -o %i")).split())
    man = manifest()
    out = [{**r, "rank": int(man[n]["rank"]), "est_hours": man[n]["est_hours"], "mem": man[n]["mem"]}
           for n, r in placed.items() if r["job"] in pend and n in man]
    return sorted(out, key=lambda c: c["rank"])


def ilab_home_node() -> str | None:
    nodes = {n for n in sh(ssh(ENV["ILAB_SUBMIT_HOST"], f"squeue -u {ENV['NET_ID']} -h -t R -o %N")).split() if n}
    return next(iter(nodes)) if len(nodes) == 1 else None


def ilab_exclude(home: str | None) -> str:
    """The known-bad nodes, plus every node but the home one while a job of ours runs there."""
    bad = {n for n in ENV["ILAB_EXCLUDE"].split(",") if n}
    if home:
        names = sh(ssh(ENV["ILAB_SUBMIT_HOST"], "sinfo -h -o %n")).split()
        bad |= {n for n in names if re.fullmatch(r"[a-z]+\d+", n) and n != home}
    return ",".join(sorted(bad))


def banked_on_amarel(run_dir: str) -> int:
    adir = run_dir.replace(ENV["EXP_DIR"], ENV.get("AMAREL_EXP_DIR", ""), 1)
    out = sh(ssh(ENV["AMAREL_HOST"], f"ls {adir}/epoch_*/artifacts_v2.json 2>/dev/null | wc -l")).strip()
    return int(out) if out.isdigit() else 0


def banked(r: dict) -> int:
    return banked_on_amarel(r["run_dir"]) if r["where"] == "amarel" else lib.epoch_count(r["run_dir"])


# --- direct cards -----------------------------------------------------------------------

def occupied_direct() -> set[tuple[str, str]]:
    """(host, gpu) of each direct card a det_v2 run of either account holds."""
    by_host: dict[str, list[tuple[str, str]]] = {}
    busy = set()
    for r in lib.latest_rows(ENV["LEDGER"]).values():
        if r["where"] != "direct" or ":" not in r["hostgpu"]:
            continue
        host, gpu = r["hostgpu"].split(":", 1)
        if r["action"] == "claim":                      # a launch is starting there
            busy.add((host, gpu))
        elif r["action"] in ("launch", "resume") and r["job"].isdigit():
            by_host.setdefault(host, []).append((r["job"], gpu))
    for host, cells in by_host.items():
        pids = ",".join(p for p, _ in cells)
        out = sh(ssh(host + ENV.get("DIRECT_HOST_SUFFIX", ""), f"ps -p {pids} -o pid= 2>/dev/null; echo DET_V2_OK")).split()
        # An unreachable host proves nothing: its cards stay busy.
        live = set(out) if "DET_V2_OK" in out else {p for p, _ in cells}
        busy |= {(host, gpu) for p, gpu in cells if p in live}
    return busy


def held_hosts() -> set[str]:
    f = Path(ENV["HOLD_HOSTS"])
    return {l.split("#")[0].strip() for l in f.read_text().splitlines()} - {""} if f.exists() else set()


def shared_cards() -> list[tuple[str, str]]:
    return [tuple(c.split(":", 1)) for c in ENV.get("SHARED_CARDS", "").split() if ":" in c]


def free_cards(mem_gb: int) -> list[tuple[str, str, str]]:
    """(host, gpu, card model) per free direct card."""
    txt = sh(f"{ENV['GPU_FREE']} --fit {VRAM_GB} {mem_gb}", timeout=600)
    block = txt.split("VIABLE", 1)[1].split("REJECTED", 1)[0] if "VIABLE" in txt else ""
    cards = []
    for line in block.splitlines():
        words = line.split()
        if words and words[0] in SCAN_LABELS:
            host = SCAN_LABELS[words[0]]
            cards += [(host, g, DIRECT_CARDS[host]) for g in re.findall(r"gpu(\d+) \(", line)]
    busy, held = occupied_direct(), held_hosts()
    cards = [c for c in cards if (c[0], c[1]) not in busy]
    cards += [(h, g, DIRECT_CARDS.get(h, "")) for h, g in shared_cards()
              if (h, g) not in busy and (h, g) not in {(c[0], c[1]) for c in cards}]
    return [c for c in cards if c[0] not in held]


def plan_moves(pend: list[dict], cards: list[tuple[str, str, str]], max_moves: int) -> list[tuple[dict, tuple]]:
    """Each card takes the first run, in rank order, that the placement rule allows on it."""
    plan, left = [], list(pend)
    for card in cards:
        for c in left:
            if allowed(c["name"], card[2]):
                plan.append((c, card))
                left.remove(c)
                break
        if len(plan) == max_moves:
            break
    return plan


# --- actions ----------------------------------------------------------------------------

def cancel_row(c: dict) -> None:
    lib.append_ledger(ENV["LEDGER"], {**c, "user": ENV["NET_ID"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "log": "-", "action": "cancel"})


def move(c: dict, card: tuple[str, str, str]) -> str:
    """Cancel a PENDING run and start it on a direct card: resumed from its banked epochs, or
    launched when nothing is banked. Never touches a job that is no longer PENDING."""
    host, gpu, _ = card
    qhost = ENV["AMAREL_HOST"] if c["where"] == "amarel" else ENV["ILAB_SUBMIT_HOST"]
    if sh(ssh(qhost, f"squeue -h -j {c['job']} -o %T")).strip() != "PENDING":
        return f"skip {c['name']}: no longer PENDING"
    n = banked(c)
    sh(ssh(qhost, f"scancel {c['job']}"))
    st = sh(ssh(qhost, f"sacct -j {c['job']} -X -n -o State")).strip()
    if not st.startswith("CANCELLED"):
        return f"FAILED to cancel {c['name']} ({c['where']} {c['job']}: {st}); left as is"
    cancel_row(c)
    if n:
        res = sh(f"bash {HERE}/resume_direct.sh {c['name']} {host} {gpu} 2>&1 | tail -1", timeout=900)
    else:
        res = sh(f"DET_V2_RELAUNCH=1 DET_V2_REF={c['commit']} bash {HERE}/launch.sh {c['name']} direct {host} {gpu} "
                 "2>&1 | tail -1", timeout=900)
    return f"moved {c['name']} ({c['where']} {c['job']}, {n} banked, rank {c['rank']}) -> {res.strip()}"


def execute(plan, cands) -> list[dict]:
    """Run the moves; a card whose run started meanwhile goes to the next candidate."""
    done, tried = [], {c["name"] for c, _ in plan}
    while plan:
        freed = []
        for c, card in plan:
            res = move(c, card)
            print(res)
            (freed.append(card) if res.startswith("skip") else done.append(c))
        rest = [c for c in cands if c["name"] not in tried]
        plan = plan_moves(rest, freed, len(freed)) if freed else []
        tried |= {c["name"] for c, _ in plan}
    return done


def ended_slots() -> list[tuple[dict, str]]:
    """(row, sacct state) for this account's iLab and Amarel jobs that are no longer running."""
    rows = {r["job"]: r for r in my_placed().values() if r["where"] in ("ilab", "ilabcpu", "amarel")}
    states: dict[str, str] = {}
    for world, host in (("amarel", ENV["AMAREL_HOST"]), ("ilab", ENV["ILAB_SUBMIT_HOST"])):
        ids = ",".join(j for j, r in rows.items() if (r["where"] == "amarel") == (world == "amarel"))
        if not ids:
            continue
        for line in sh(ssh(host, f"sacct -X -n -P -j {ids} -o JobID,State")).splitlines():
            jid, _, state = line.partition("|")
            if jid in rows:
                states[jid] = state.split()[0] if state else ""
    return [(rows[j], s) for j, s in states.items()]


def resume_ended(apply: bool) -> list[str]:
    out = []
    man = manifest()
    for r, state in ended_slots():
        if state in NEEDS_A_LOOK:
            out.append(f"NEEDS A LOOK {r['name']}: {r['where']} job {r['job']} {state}")
            continue
        if state not in ENDED:
            continue
        partx = man.get(r["name"], {}).get("script") == "scripts/run_partx_faithful.py"
        n = banked(r)
        if n >= lib.N_EPOCHS:
            continue
        if not apply:
            out.append(f"would resume {r['name']} ({r['where']} {state}, {n}/{lib.N_EPOCHS} epochs)")
            continue
        if partx:
            cmd = f"DET_V2_REF={r['commit']} bash {HERE}/launch.sh {r['name']} ilabcpu"
        elif n and r["where"] == "amarel":
            cmd = f"bash {HERE}/resume_amarel.sh {r['name']}"
        elif n:
            cmd = f"DET_V2_EXCLUDE={ilab_exclude(ilab_home_node())} bash {HERE}/resume_ilab.sh {r['name']} {ENV.get('ILAB_RESUME_WALL', '06:00:00')} {r['hostgpu']}"
        else:        # died inside epoch 0: nothing banked, so start it again
            where = "amarel" if r["where"] == "amarel" else f"ilab {r['hostgpu']} {ENV.get('ILAB_RESUME_WALL', '06:00:00')}"
            pre = "" if r["where"] == "amarel" else f"DET_V2_EXCLUDE={ilab_exclude(ilab_home_node())} "
            cmd = f"{pre}DET_V2_RELAUNCH=1 DET_V2_REF={r['commit']} bash {HERE}/launch.sh {r['name']} {where}"
        res = sh(f"{cmd} 2>&1 | tail -1", timeout=900)
        out.append(f"{state} {r['name']} at {n}/{lib.N_EPOCHS} epochs -> {res.strip()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true", help="act even while a launch loop appends to the ledger")
    ap.add_argument("--max-moves", type=int, default=8)
    a = ap.parse_args()
    ENV.update(lib.load_user_env())
    busy = launch_loop_active()
    apply = a.apply and (a.force or busy is None)
    if a.apply and not apply:
        print(f"holding off: {busy}; printing the plan only (--force overrides)")
    for line in resume_ended(apply):
        print(line)
    pend = pending_runs()
    print(f"{len(pend)} runs of {ENV['NET_ID']} PENDING on iLab or Amarel, in rank order:")
    for c in pend[:10]:
        print(f"  {c['rank']:>3} {c['name']:32s} ~{c['est_hours']}h  {c['where']} job {c['job']}")
    cards = free_cards(max([int(c["mem"].rstrip("G")) for c in pend] or [16]))
    print(f"free direct cards: {cards}")
    plan = plan_moves(pend, cards, a.max_moves)
    if apply:
        execute(plan, pend)
    else:
        for c, card in plan:
            print(f"would move: {c['name']} -> {card[0]} gpu{card[1]} ({card[2]})")


if __name__ == "__main__":
    main()
