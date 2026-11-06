"""det_v2 core: the decision rule, threshold opt, acquisition scores, batch rules, run configs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from adaptive_roa.adaptive_v2.eval import det_v2_rule as rule
from adaptive_roa.adaptive_v2.strategy import verdict_scores as vs

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- rule
def test_rule_ties_follow_the_method():
    # S needs p_s > tau_s and p_f <= tau_f; F the mirror; both or neither is A.
    lab = rule.predict([0.5, 0.51, 0.51, 0.3, 0.2, 0.6], [0.3, 0.3, 0.31, 0.31, 0.2, 0.4], 0.5, 0.3)
    assert lab.tolist() == [rule.A_INV, rule.S, rule.A_OUT, rule.F, rule.A_INV, rule.A_OUT]
    # A final-state fraction n / K equal to a grid threshold compares equal.
    assert rule.predict([7 / 100], [0.0], 0.07, 0.5)[0] == rule.A_INV
    # Classifier: p_f = 1 - p_s.
    assert rule.predict([0.8, 0.5, 0.2], None, 0.6, 0.6).tolist() == [rule.S, rule.A_INV, rule.F]


def _brute(p_s, p_f, y, w, objective, classifier):
    succ = np.asarray(y) == 1
    best = None
    for a, ts in enumerate(rule.GRID):
        for b, tf in enumerate(rule.GRID):
            if classifier and (a + 1) + (b + 1) < 100:
                continue
            lab = rule.predict(p_s, p_f, ts, tf)
            s_, f_ = lab == rule.S, lab == rule.F
            tp, fp = np.sum(s_ & succ), np.sum(s_ & ~succ)
            fn, tn = np.sum(f_ & succ), np.sum(f_ & ~succ)
            ans = tp + fp + fn + tn
            mr = (fp + fn) / ans if ans else 0.0
            ar = (len(succ) - ans) / len(succ)
            fcr = fp / (tp + fp) if tp + fp else 0.0
            rec = tp / succ.sum() if succ.sum() else 0.0
            j_sel = (1 - w) * mr + w * ar
            j_ver = (1 - w) * fcr + w * (1 - rec)
            key = ((round(j_ver, 12), round(j_sel, 12)) if objective == "j_ver" else (round(j_sel, 12), 0.0)) + (a, b)
            if best is None or key < best[0]:
                best = (key, float(ts), float(tf))
    return best[1], best[2]


@pytest.mark.parametrize("objective,w", [("j_sel", 0.1), ("j_sel", 0.5), ("j_ver", 0.1), ("j_ver", 0.01)])
def test_jsel_matches_brute_force_final_state(objective, w):
    rng = np.random.default_rng(1)
    n = 400
    y = np.where(rng.random(n) < 0.4, 1, -1)
    base = np.where(y == 1, 0.7, 0.25)
    counts = np.array([rng.multinomial(100, [b * 0.9, (1 - b) * 0.9, 0.1]) for b in base])
    p_s, p_f = counts[:, 0] / 100, counts[:, 1] / 100
    got = rule.fit(p_s, p_f, y, w=w, objective=objective)
    assert (got["tau_s"], got["tau_f"]) == _brute(p_s, p_f, y, w, objective, False)


@pytest.mark.parametrize("objective", ["j_sel", "j_ver"])
def test_jsel_matches_brute_force_classifier(objective):
    rng = np.random.default_rng(2)
    n = 300
    y = np.where(rng.random(n) < 0.5, 1, -1)
    p = np.clip(np.where(y == 1, 0.65, 0.35) + 0.25 * rng.standard_normal(n), 0, 1)
    got = rule.fit(p, None, y, w=0.1, objective=objective, classifier=True)
    assert got["tau_s"] + got["tau_f"] >= 1.0 - 1e-12
    assert (got["tau_s"], got["tau_f"]) == _brute(p, None, y, 0.1, objective, True)


def test_jsel_all_abstain_corner_scores_w():
    # Nothing separable: answering anything costs more than abstaining, J = w.
    y = np.array([1, -1, 1, -1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    got = rule.fit(p, 1 - p, y, w=0.1)
    assert got["mr"] == 0.0 and got["ar"] == 1.0 and got["j"] == pytest.approx(0.1)


# ----------------------------------------------------------------- q_A
CASES = [((50, 30, 20), 0.4, 0.3), ((50, 30, 20), 0.6, 0.5), ((10, 85, 5), 0.2, 0.5),
         ((100, 0, 0), 0.95, 0.01), ((33, 33, 34), 0.25, 0.25), ((3, 2, 95), 0.05, 0.03)]


@pytest.mark.parametrize("n,ts,tf", CASES)
def test_qa_matches_dirichlet_mc(n, ts, tf):
    q = vs.verdict_probabilities([n[0]], [n[1]], [n[2]], ts, tf)
    d = np.random.default_rng(0).dirichlet(np.asarray(n) + 1.0, size=2_000_000)
    sp, fp = d[:, 0] > ts, d[:, 1] > tf
    mc = np.mean((sp & fp) | (~sp & ~fp))
    sigma = np.sqrt(max(mc * (1 - mc), 1e-12) / len(d))
    assert abs(q["q_a"][0] - mc) <= max(5 * sigma, 1e-4)
    assert q["q_s"][0] + q["q_f"][0] + q["q_a"][0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n,ts,tf", CASES)
def test_both_pass_integral_matches_adaptive_quadrature(n, ts, tf):
    ns, nf, ni = n
    N = ns + nf + ni
    if ts + tf >= 1:
        ref = 0.0
    else:
        f = lambda x: stats.beta.pdf(x, ns + 1, N - ns + 2) * stats.beta.sf(tf / (1 - x), nf + 1, ni + 1)
        ref, _ = integrate.quad(f, ts, 1 - tf, epsabs=1e-13, epsrel=1e-11, limit=500)
    assert vs.both_pass_probability([ns], [nf], [ni], ts, tf)[0] == pytest.approx(ref, abs=1e-9)


def test_verdict_probabilities_stay_in_the_unit_interval():
    rng = np.random.default_rng(3)
    c = rng.multinomial(100, [0.3, 0.3, 0.4], size=3000)
    c = np.vstack([c, [[100, 0, 0], [0, 100, 0], [0, 0, 100], [50, 50, 0]]])
    for ts, tf in ((0.3, 0.4), (0.01, 0.01), (0.99, 0.99), (0.5, 0.49)):
        q = vs.verdict_probabilities(c[:, 0], c[:, 1], c[:, 2], ts, tf)
        for v in q.values():
            assert v.min() >= 0.0 and v.max() <= 1.0


def test_qa_classifiers():
    # MLP: one draw, the indicator of A.
    q = vs.classifier_verdict_probabilities(np.array([[0.9, 0.5, 0.1]]), 0.6, 0.6)
    assert q["q_a"].tolist() == [0.0, 1.0, 0.0]
    # BNN: the fraction of draws whose verdict is A.
    draws = np.array([[0.9, 0.5], [0.5, 0.5], [0.5, 0.9], [0.1, 0.5]])
    q = vs.classifier_verdict_probabilities(draws, 0.6, 0.6)
    assert q["q_a"].tolist() == [0.5, 0.75]


def test_uh():
    assert vs.u_h([0.5]) == pytest.approx([1.0])                       # classifier H2(p_s)
    assert vs.u_h([0.0], [0.0], [1.0]) == pytest.approx([1.0])         # no valid draw
    assert vs.u_h([0.8], [0.2], [0.0]) == pytest.approx(vs.binary_entropy_bits([0.8]))
    assert vs.u_h([0.4], [0.4], [0.2]) == pytest.approx([0.2 + 0.8 * 1.0])
    assert vs.u_h([1.0], [0.0], [0.0]) == pytest.approx([0.0])


# ----------------------------------------------------------------- batch rules
def test_topk_ties_go_farthest_then_pool_order():
    u = np.array([0.9, 0.5, 0.5, 0.5, 0.2, 0.5])
    dist = {1: 1.0, 2: 3.0, 3: 3.0, 5: 2.0}
    seen = []

    def distance(pos):
        seen.append(list(pos))
        return np.array([dist[int(p)] for p in pos])

    picks, d = vs.select_topk(u, 3, distance)
    assert picks.tolist() == [0, 2, 3]                 # 0 by score; 2 and 3 tie on distance, pool order
    assert seen == [[1, 2, 3, 5]]                      # only the tie group is measured
    assert d["n_tie_group"] == 4 and d["n_tie_taken"] == 2


def test_topk_without_ties_never_measures():
    picks, _ = vs.select_topk(np.array([0.1, 0.9, 0.5]), 2, lambda pos: pytest.fail("measured"))
    assert picks.tolist() == [1, 2]


def test_proportional_draws_only_positive_scores_and_tracks_u():
    u = np.array([0.0, 1.0, 3.0, 0.0])
    rng = np.random.default_rng(0)
    picks, d = vs.select_proportional(u, 3, rng)
    assert sorted(picks.tolist()) == [1, 2] and d["n_positive"] == 2   # short: the engine fills uniform
    first = [vs.select_proportional(u, 1, rng)[0][0] for _ in range(4000)]
    assert np.mean(np.array(first) == 2) == pytest.approx(0.75, abs=0.03)


def test_nearest_distance():
    d = vs.nearest_distance(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 1.0], [10.0, 10.0]]))
    assert d.tolist() == pytest.approx([1.0, np.hypot(3, 3)])


# ----------------------------------------------------------------- configs
def test_run_configs_match_the_plan():
    out = subprocess.run([sys.executable, "scripts/det_v2/make_run_configs.py", "--check"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_every_run_config_composes():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts/run_partx_faithful.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                      # registers the resolvers the run scripts use
    plan = json.loads((ROOT / "scripts/det_v2/run_plan.json").read_text())
    runs = [r for r in plan["main"] + plan["abl"] if r.get("rank") is not None]
    assert len(runs) == 306
    with initialize_config_dir(config_dir=str(ROOT / "configs/adaptive_v2"), version_base=None):
        for r in runs:
            partx = r["model"] == "Part-X"
            cfg = compose(config_name="partx_faithful" if partx else "default",
                          overrides=[f"+experiment=det_v2/runs/{r['name']}"])
            c = OmegaConf.to_container(cfg, resolve=True)
            assert c["n_epochs"] == 11 and c["seed"] == 99512 and c["prediction_store"]["enabled"]
            assert c["output_dir"].endswith(f"det_v2/{r['setting']}/{r['name']}_s99512")
            if partx:
                continue
            assert c["threshold"]["_target_"].endswith("JSelThresholdBackend")
            for k in ("acquisition", "threshold", "probability"):
                assert c[k]["num_mc_samples"] == 100
            assert c["eval"]["num_mc_samples_eval"] == 100 and c["eval"]["binary_outcomes"] is False
            assert c["acquisition"]["selection"] == "topk" and c["acquisition"]["n_candidates"] == 50000
            change = r.get("change") or ""
            assert c["acquisition"]["score"] == ("u_h" if change.startswith("U_H") else "q_a")
            want_d2 = 0.0 if r.get("arm") == "uniform" else (float(change.split("= ")[1]) if change.startswith("d2_ratio") else 1.0)
            assert c["acquisition"]["d2_ratio"] == want_d2
            want_obj = "j_ver" if change.startswith("J_ver") else "j_sel"
            want_w = float(change.split("w = ")[1]) if "w = " in change else 0.1
            assert (c["threshold"]["objective"], c["threshold"]["w"]) == (want_obj, want_w)
            hyst = c["threshold"]["_target_"].endswith("HysteresisJSelThresholdBackend")
            assert hyst == (change == "tau* regime hysteresis")
            if hyst:
                assert (c["threshold"]["n_boot"], c["threshold"]["se_mult"], c["threshold"]["margin"],
                        c["threshold"]["seed"]) == (200, 2.0, None, 99512)
            if r["model"] == "FM":
                assert c["predictor"]["flow_matching"]["mae_val_frequency"] == 1000000
            if r["model"] in ("MLP final-state", "BNN final-state"):
                assert c["predictor"]["final_state"]["reuse_body_features"] is True
            # Bias-free speedups, on from epoch 0 of every run.
            assert c["predictor"]["batched_indexing"] is True
            assert c["predictor"]["lightning_trainer"]["enable_progress_bar"] is False
            assert c["prediction_store"]["reuse_cal_for_qhat"] is True
            assert c["adaptive_v2"]["endpoint_error"] is False
            assert c["eval"]["plot_epochs"] == [0, 5, 10]
