# Ensemble Epistemic Acquisition — Experiment Log

Design: `docs/superpowers/specs/2026-08-04-ensemble-epistemic-acquisition-design.md`
Machine-readable run records: `runs.jsonl` (see `scripts/exp_log.py`).

Append newest entries at the top. Record what was launched, what broke, what was
decided, and why — the "why" is the part that is impossible to reconstruct later.

## 2026-08-04 — plan approved, implementation started

## 2026-08-04 03:00-04:00 — FM arms relocated to iLab; numpy shadow found and fixed

The five FM arms never started on Amarel. Their estimated start slipped from
03:19 to 11:21 the following day: five-GPU jobs on a contended cluster under the
preemptible `general` account simply do not schedule.

Freed capacity on both clusters by cancelling the previous stochastic-pendulum
campaign's still-running arms, after re-running `scripts/refresh_stoch_compare.sh`
so its deliverable was captured at maximum depth first. Kept the arms backing its
one unsettled question, xhigh-FM: `sc_fm_xhigh_dir00` + `sc_fm_xhigh_tb10` on
Amarel and `sc_fm_xhigh_ent10` on iLab. Cancelled 9 on Amarel and 7 on iLab, all
backing verdicts already computed (med-FM, high-FM, med/high/xhigh-CLF).

A canary arm on 4x a100 then exposed a bug that would have wasted the whole
allocation. Two numpy 2.2.6 installs coexist on iLab — the env's own and one in
`~/.local` — and the user-site copy shadows the env's. Every `mp.spawn` child
re-imports the main module, numpy's C extension initializes twice in one process,
and the child dies with "CPU dispatcher tracer already initlized". The failure is
silent: members created their checkpoint dirs and printed epoch-0 validation, then
died, leaving the GPUs at 0% and the parent blocked in `mp.spawn(join=True)` with
`squeue` still reporting R. Fixed with `PYTHONNOUSERSITE=1` in both job templates
(commit bd6a80f), and verified by relaunch: 0 dispatcher errors, 5 GPU processes
per arm, checkpoints written.

Relaunched at **2 GPUs per arm, not 5**. Members train concurrently and a pendulum
flow matcher only drives an a4500 to ~25%, so 2-3 members share a card cheaply.
All five arms then fit in 10 of the 12-GPU quota and advance together — which is
what the matched-epoch comparison needs, since its common depth is bounded by the
slowest arm. a4500 is requested explicitly so the arms do not scatter across
iLab's mixed pool at different speeds.

Running (iLab, seed 42, high): epi_var 199892, total 199894, dir00 199895,
epi_bald 199896, aleat 199897. The 25 CLF arms continue on Amarel and arrakis.
