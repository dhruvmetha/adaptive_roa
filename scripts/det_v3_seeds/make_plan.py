#!/usr/bin/env python
"""Build scripts/det_v3_seeds/run_plan.json from det_v3's plan (Sumanth, 2026-11-08).

Every row is a det_v3 row (its base, ``base_name`` / ``base_id``) with one stated change, at a
seed. A row keeps its det_v3 setting, model, owner and estimated hours. Seeds: 99512 is repeat
1 (_r1, det_v3's own seed), 33407 repeat 2 (_r2), 24475 repeat 3 (_r3). Blocks, by plan key:

  main       the 60 det_v3 main rows at _r2 and _r3 (ranks 1-120). Ids S + repeat + M number.
  abl        the 30 det_v3 A7 rows, U_H instead of q_A (X34-X63), at _r2 and _r3 (121-180), so
             every main row has both a q_A and a U_H run per seed. Ids U + repeat + X number.
  utau_main  U_tau main: the 30 q_A main rows with acquisition u_tau, at _r1, _r2 and _r3
             (181-270). Ids T + repeat + M number; names swap _qa for _utau.
  uh_abl     U_H ablations: the q_A rows of A6, A10 and A11 with acquisition u_h, at _r1, _r2
             and _r3. Ids H + repeat + X number; names swap _qa_ for _uh_.
  utau_abl   U_tau ablations: the q_A rows of A1, A2, A6, A10 and A11 with acquisition u_tau, at
             all three seeds. Ids V + repeat + X number; names swap _qa_ for _utau_.
  unif_a11   the 30 uniform main rows under each A11 schedule (the initial_train_size and
             samples_per_epoch of that setting's q_A A11 rows), at all three seeds: single-model
             arms warm-start, so the schedule changes the uniform run too. Ids N + repeat +
             (1 x0.5, 2 x2) + M number.
  k_abl      acquisition_mc_samples (K for probability, acquisition and threshold; eval K stays
             100) at 20 and 50, at all three seeds: q_A main rows of FM, MLP final-state, BNN
             final-state and BNN classifier; the A7 U_H rows of FM, MLP and BNN final-state; and
             the U_tau main rows of all five models. Ids K + repeat + (2 for K = 20, 5 for 50) +
             (1 q_A, 2 U_H, 3 U_tau) + M or X number.
  qa_abl     the det_v3 q_A ablation rows themselves (A1, A2, A6, A10, A11) at _r2 and _r3; _r1 is
             det_v3. Ids Q + repeat + X number; names add the repeat tag.
  partx      the six det_v2 Part-X rows (det_v3 has none) at _r2 and _r3 (ranks 3073-3084, after
             every det_v3 row): the det_v2 config with only the seed changed, as _r1 is det_v2's own
             run at 99512. base_campaign det_v2; owner, setting and hours are the det_v2 row's. CPU
             only (scripts/run_partx_faithful.py), launched with `launch.sh <name> ilabcpu`. Ids P +
             repeat + M number; names add the repeat tag.
  k_method   the matrix method row of each setting (U_H, A6 d2_ratio = 0.75: the uh_abl rows below) at
             acquisition_mc_samples 20 and 50, at all three seeds (ranks 3085-3120; Sumanth's GO via the
             driver, 2026-11-10). The uh_abl row with only that K added. Ids W + repeat + (2 for K = 20,
             5 for 50) + X number; names add _k20 / _k50 before the repeat tag.

Left out on purpose: A1/A2 under uniform and U_H (their acquisition never reads tau*); A6/A10
under uniform (no score to mix or select by); K under uniform, under the MLP classifier for
q_A and U_H, and under the BNN classifier for U_H (no K anywhere: BNN p_s uses its
n_marginal_samples). Ranks: utau_main seed-major in det_v3 order (longest first); then the
ablation blocks seed-major: every _r1 row of uh_abl, utau_abl, unif_a11 and k_abl, longest
first, then every _r2 row of those and qa_abl, then every _r3 row (ties: block order as listed,
then det_v3 rank, then variant order), so the 99512 ablations that pair with det_v3's q_A
ablations finish first. partx after them, seed-major, longest first (ties: det_v2 rank); k_method last,
seed-major, longest first, K = 20 before K = 50.

    make_plan.py [--out scripts/det_v3_seeds/run_plan.json]
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ((2, 33407), (3, 24475))        # (repeat number, seed): the other two paper seeds (tracker D04);
                                         # repeat 1 is det_v3 at 99512. Their pool files already exist.
R1 = (1, 99512)
KEEP = ("setting", "model", "arm", "hours", "owner")
KEEP_ABL = ("setting", "model", "ab", "change", "hours", "owner")
ABLATIONS = ("A7",)                      # det_v3 ablations repeated in "abl", by "ab"
UH_ABL = ("A6", "A10", "A11")
UTAU_ABL = ("A1", "A2", "A6", "A10", "A11")
A11_TAGS = {"initial batch x 0.5": (1, "a11half"), "initial batch x 2": (2, "a11double")}
K_VALUES = ((20, 2), (50, 5))            # (K, id digit)
K_QA_MODELS = ("FM", "MLP final-state", "BNN final-state", "BNN classifier")
# prediction_store settings for every row past rank 180 (the launched rows keep det_v3's: endpoint_epochs
# [0, 5, 10], endpoint_test_rows 100000). Endpoint dumps are ~73% of an endpoint-model run and the
# st1122 quota cannot hold them for the extension, so they default to none; set this from Sumanth's
# answer and regenerate. Audit options: {"endpoint_epochs": [10]}, {"endpoint_test_rows": 10000}, or both.
# Nothing reads endpoints_*.npz; store_check.py checks each run against its own endpoint_epochs.
NEW_ROW_STORE = {"endpoint_epochs": []}
K_UH_MODELS = ("FM", "MLP final-state", "BNN final-state")
PARTX_KEEP = ("setting", "model", "arm", "hours", "owner")
# The matrix method row of each setting (det_v3 base of its uh_abl row): U_H with d2_ratio = 0.75 (A6).
METHOD_BASES = ("x33_q3dlqr_fm_qa_a6d075", "x70_q3dppo_fm_qa_a6d075", "x31_q2dppo_fm_qa_a6d075",
                "x29_cplqr_fm_qa_a6d075", "x161_cpppo_fm_qa_a6d075", "x231_penlqr_fm_qa_a6d075")


def _num(r: dict) -> int:
    return int(r["id"][1:])


def _initial_budget():
    spec = importlib.util.spec_from_file_location("det_v3_make_run_configs", ROOT / "scripts/det_v3/make_run_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.initial_budget


def repeats(base: list[dict], prefix: str, keep: tuple, kind: str, first_rank: int) -> list[dict]:
    rows = []
    for rep, seed in SEEDS:
        for r in base:
            rows.append({"id": f"{prefix}{rep}{_num(r):02d}", "base_id": r["id"], "base_name": r["name"],
                         "seed": seed, "name": f"{r['name']}_r{rep}", **{k: r[k] for k in keep},
                         "kind": kind, "rank": first_rank + len(rows)})
    return rows


def _row(base: dict, rid: str, name: str, seed: int, kind: str, arm: str, change: str | None,
         ab: str | None, acquisition: str | None = None, overrides: dict | None = None) -> dict:
    row = {"id": rid, "base_id": base["id"], "base_name": base["name"], "seed": seed, "name": name,
           "setting": base["setting"], "model": base["model"], "arm": arm, "ab": ab, "change": change,
           "hours": base["hours"], "owner": base["owner"], "kind": kind}
    if acquisition:
        row["acquisition"] = acquisition
    if overrides:
        row["set"] = overrides
    return row


def utau_main(qa: list[dict], first_rank: int) -> list[dict]:
    rows = []
    for rep, seed in (R1,) + SEEDS:
        for r in qa:
            rows.append(_row(r, f"T{rep}{_num(r):02d}", f"{r['name'][:-3]}_utau_r{rep}", seed, "utau_main",
                             "U_tau", "U_tau instead of q_A", None, acquisition="u_tau"))
    return rows


ABL_BLOCKS = ("uh_abl", "utau_abl", "unif_a11", "k_abl", "qa_abl")


def ablation_blocks(v3: dict, rep: int, seed: int) -> dict[str, list[dict]]:
    """The ablation blocks at one seed, each in det_v3 rank order, then variant order (before ranking).
    qa_abl only at _r2 and _r3: at 99512 those rows are det_v3's own."""
    rank = lambda r: r["rank"]                                       # noqa: E731
    main = sorted(v3["main"], key=rank)
    qa = [r for r in main if r["arm"] == "q_A"]
    unif = [r for r in main if r["arm"] == "uniform"]
    abl = sorted(v3["abl"], key=rank)
    out = {k: [] for k in ABL_BLOCKS}
    for r in abl:
        if r["ab"] in UH_ABL:
            out["uh_abl"].append(_row(r, f"H{rep}{_num(r):03d}", r["name"].replace("_qa_", "_uh_") + f"_r{rep}",
                                      seed, "uh_abl", "U_H", r["change"], r["ab"], acquisition="u_h"))
        if r["ab"] in UTAU_ABL:
            out["utau_abl"].append(_row(r, f"V{rep}{_num(r):03d}", r["name"].replace("_qa_", "_utau_") + f"_r{rep}",
                                        seed, "utau_abl", "U_tau", r["change"], r["ab"], acquisition="u_tau"))
            if rep > 1:
                out["qa_abl"].append(_row(r, f"Q{rep}{_num(r):03d}", f"{r['name']}_r{rep}", seed, "qa_abl",
                                          "q_A", r["change"], r["ab"]))
    budget = _initial_budget()
    for r in unif:
        for change, (v, tag) in A11_TAGS.items():
            n0, k = budget({"setting": r["setting"], "change": change})
            out["unif_a11"].append(_row(r, f"N{rep}{v}{_num(r):02d}", f"{r['name']}_{tag}_r{rep}", seed, "unif_a11",
                                        "uniform", change, "A11",
                                        overrides={"initial_train_size": n0, "samples_per_epoch": k}))
    a7 = [r for r in abl if r["ab"] == "A7"]
    for kval, kd in K_VALUES:
        change, ov = f"K = {kval}", {"acquisition_mc_samples": kval}
        for r in qa:
            if r["model"] in K_QA_MODELS:
                out["k_abl"].append(_row(r, f"K{rep}{kd}1{_num(r):02d}", f"{r['name']}_k{kval}_r{rep}", seed, "k_abl",
                                         "q_A", change, "K", overrides=ov))
        for r in a7:
            if r["model"] in K_UH_MODELS:
                out["k_abl"].append(_row(r, f"K{rep}{kd}2{_num(r):02d}", f"{r['name']}_k{kval}_r{rep}", seed, "k_abl",
                                         "U_H", change, "K", overrides=ov))
        for r in qa:
            out["k_abl"].append(_row(r, f"K{rep}{kd}3{_num(r):02d}", f"{r['name'][:-3]}_utau_k{kval}_r{rep}", seed,
                                     "k_abl", "U_tau", change, "K", acquisition="u_tau", overrides=ov))
    return out


def partx(v2: dict, first_rank: int) -> list[dict]:
    """The det_v2 Part-X rows at _r2 and _r3, seed-major, longest first (ties: det_v2 rank)."""
    base = sorted((r for r in v2["main"] if r["model"] == "Part-X"), key=lambda r: (-r["hours"], r["rank"]))
    rows = []
    for rep, seed in SEEDS:
        for r in base:
            rows.append({"id": f"P{rep}{_num(r):02d}", "base_id": r["id"], "base_name": r["name"],
                         "base_campaign": "det_v2", "seed": seed, "name": f"{r['name']}_r{rep}",
                         **{k: r[k] for k in PARTX_KEEP}, "kind": "partx", "rank": first_rank + len(rows)})
    return rows


def k_method(uh_abl: list[dict], first_rank: int) -> list[dict]:
    """The method rows (uh_abl at METHOD_BASES) at K = 20 and 50, seed-major, longest first, K = 20 first."""
    rows = []
    for rep, seed in (R1,) + SEEDS:
        base = sorted((r for r in uh_abl if r["seed"] == seed and r["base_name"] in METHOD_BASES),
                      key=lambda r: (-r["hours"], METHOD_BASES.index(r["base_name"])))
        assert len(base) == len(METHOD_BASES), seed
        for r in base:
            for kval, kd in K_VALUES:
                rows.append({**{k: v for k, v in r.items() if k not in ("id", "name", "kind", "rank")},
                             "id": f"W{rep}{kd}{int(r['base_id'][1:]):03d}",
                             "name": f"{r['name'][:-3]}_k{kval}_r{rep}", "ab": "K", "change": f"K = {kval}",
                             "kind": "k_method", "set": {"acquisition_mc_samples": kval},
                             "rank": first_rank + len(rows)})
    return rows


def plan(v3: dict, v2: dict) -> dict:
    main = repeats(sorted(v3["main"], key=lambda r: r["rank"]), "S", KEEP, "main", 1)
    abl = sorted((r for r in v3["abl"] if r.get("ab") in ABLATIONS), key=lambda r: r["rank"])
    p = {"main": main, "abl": repeats(abl, "U", KEEP_ABL, "abl", len(main) + 1)}
    p["utau_main"] = utau_main([r for r in sorted(v3["main"], key=lambda r: r["rank"]) if r["arm"] == "q_A"], 0)
    nxt = len(main) + len(p["abl"]) + 1
    for i, r in enumerate(p["utau_main"]):
        r["rank"] = nxt + i
    nxt += len(p["utau_main"])
    merged = {k: [] for k in ABL_BLOCKS}
    for rep, seed in (R1,) + SEEDS:                              # seed-major
        blocks = ablation_blocks(v3, rep, seed)
        pooled = [(r, ABL_BLOCKS.index(k), i) for k, rows in blocks.items() for i, r in enumerate(rows)]
        pooled.sort(key=lambda t: (-t[0]["hours"], t[1], t[2]))  # longest first; then block, then det_v3 order
        for i, (r, _, _) in enumerate(pooled):
            r["rank"] = nxt + i
        nxt += len(pooled)
        for k, rows in blocks.items():
            merged[k] += rows
    for k, rows in merged.items():
        p[k] = sorted(rows, key=lambda r: r["rank"])
    if NEW_ROW_STORE:
        for k in ("utau_main",) + ABL_BLOCKS:
            for r in p[k]:
                r["store"] = dict(NEW_ROW_STORE)
    p["partx"] = partx(v2, nxt)                 # no store override: the Part-X engine writes no endpoints
    p["k_method"] = k_method(p["uh_abl"], nxt + len(p["partx"]))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "scripts/det_v3_seeds/run_plan.json")
    a = ap.parse_args()
    p = plan(json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text()),
             json.loads((ROOT / "scripts/det_v2/run_plan.json").read_text()))
    a.out.write_text(json.dumps(p, indent=1) + "\n")
    for kind, rows in p.items():
        seeds = sorted({r["seed"] for r in rows})
        tot = {(o, s): round(sum(r["hours"] for r in rows if r["owner"] == o and r["seed"] == s), 1)
               for o in ("Sumanth", "Dhruv") for s in seeds}
        print(f"{kind}: {len(rows)} runs, ranks {rows[0]['rank']}-{rows[-1]['rank']}; est GPU-h by (owner, seed) {tot}")


if __name__ == "__main__":
    main()
