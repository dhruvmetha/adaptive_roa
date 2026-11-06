"""Ablation scope (Sumanth's ruling, 2026-11-06): every ablation runs for all five models
in all six settings, q_A arm, seed 99512, at the main budgets.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {"q3d-ppo", "q3d-lqr", "q2d-ppo", "cp-ppo", "cp-lqr", "pen-lqr"}
MODELS = {"FM", "MLP final-state", "BNN final-state", "MLP classifier", "BNN classifier"}
VARIANTS = {"A1": {"J_ver, w = 0.1"},
            "A2": {"J_sel, w = 0.01", "J_sel, w = 0.5", "J_ver, w = 0.01", "J_ver, w = 0.5"},
            "A6": {"d2_ratio = 0.5", "d2_ratio = 0.75"},
            "A7": {"U_H instead of q_A"}}


def _plan():
    return json.loads((ROOT / "scripts/det_v2/run_plan.json").read_text())


def test_each_ablation_covers_every_model_setting_and_variant_once():
    cells = Counter((r["ab"], r["setting"], r["model"], r["change"]) for r in _plan()["abl"])
    for ab, changes in VARIANTS.items():
        want = {(ab, s, m, c) for s in SETTINGS for m in MODELS for c in changes}
        got = {k for k in cells if k[0] == ab}
        assert got == want, (ab, sorted(want - got)[:3], sorted(got - want)[:3])
    assert max(cells.values()) == 1


def test_new_rows_use_the_main_budget_and_unique_ids():
    plan = _plan()
    rows = plan["main"] + plan["abl"]
    assert len({r["id"] for r in rows}) == len(rows) and len({r["name"] for r in rows}) == len(rows)
    hours = {(r["setting"], r["model"]): r["hours"] for r in plan["main"] if r.get("arm") == "q_A"}
    for r in plan["abl"]:
        assert r["hours"] == hours[(r["setting"], r["model"])], r["id"]
        assert r["owner"] in ("Sumanth", "Dhruv") and not r["reused"]
