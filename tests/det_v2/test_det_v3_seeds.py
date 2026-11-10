"""det_v3_seeds (Sumanth, 2026-11-08): det_v3 rows, each with one stated change, at the paper seeds.

Blocks (scripts/det_v3_seeds/make_plan.py): the 60 main rows and the 30 A7 U_H rows at 33407 and
24475 (ranks 1-180, launched); U_tau main at 99512, 33407 and 24475; U_H and U_tau ablations,
uniform under A11 and the K ablation at all three seeds; the q_A ablation rows at 33407 and
24475; det_v2's six Part-X rows at 33407 and 24475 (ranks 3073-3084); the six matrix method rows at
K = 20 and 50, all three seeds (ranks 3085-3120). The plan, manifest and run
configs regenerate from det_v3's (det_v2's for Part-X); every config composes to its det_v3 (det_v2)
counterpart plus only the keys its change names; ranks 1-180 stay as launched; the
scripts/det_v3_seeds/ wrappers get their own root, ledger, logs, Amarel log dir and tmux tag."""
import csv
import hashlib
import importlib.util
import json
import multiprocessing
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import pytest

from det_v2_world import GPU_RUN, MANIFEST, PARTX_RUN, ROOT, RUN_YAML, SMALL_RUN, git

SEEDS = {2: 33407, 3: 24475}
ALL_SEEDS = {1: 99512, **SEEDS}
V3 = json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text())
V3_MAIN = sorted(V3["main"], key=lambda r: r["rank"])
V3_ABL = sorted(V3["abl"], key=lambda r: r["rank"])
V3_ROWS = {r["name"]: r for r in V3_MAIN + V3_ABL}
V3_QA = [r for r in V3_MAIN if r["arm"] == "q_A"]
V3_UNIF = [r for r in V3_MAIN if r["arm"] == "uniform"]
V3_A7 = [r for r in V3_ABL if r["ab"] == "A7"]
V2 = json.loads((ROOT / "scripts/det_v2/run_plan.json").read_text())
V2_PARTX = {r["name"]: r for r in V2["main"] if r["model"] == "Part-X"}
PLAN = json.loads((ROOT / "scripts/det_v3_seeds/run_plan.json").read_text())
RUNS = sorted((r for rows in PLAN.values() for r in rows), key=lambda r: r["rank"])
BY_NAME = {r["name"]: r for r in RUNS}
V3_RUNS = [r for r in RUNS if r["kind"] != "partx"]                  # the rows with a det_v3 counterpart
BLOCKS = {"main": 120, "abl": 60, "utau_main": 90, "uh_abl": 540, "utau_abl": 990, "unif_a11": 180,
          "k_abl": 432, "qa_abl": 660, "partx": 12, "k_method": 36}
# The matrix method row of each setting (uh_abl, U_H with d2_ratio = 0.75) and its K rows (k_method).
METHOD = ("x33_q3dlqr_fm_uh_a6d075", "x70_q3dppo_fm_uh_a6d075", "x31_q2dppo_fm_uh_a6d075",
          "x29_cplqr_fm_uh_a6d075", "x161_cpppo_fm_uh_a6d075", "x231_penlqr_fm_uh_a6d075")
# Ranks 1-180 as launched (e8476a7, f819778): manifest lines 1-181 and those 180 configs, in rank order.
LAUNCHED_MANIFEST_SHA = "72bd46cd3d2fc91d50f31df9cdf0e10258013e4ca40e953eea004a21b0c52865"
LAUNCHED_CONFIGS_SHA = "d520e98da798fe3f9923700c6fa01c876d7aaf3d9561a46a7a78e3c8fffd37ee"
WRAPPERS = ("launch.sh", "resume_amarel.sh", "resume_direct.sh", "resume_ilab.sh", "sync_amarel.sh",
            "preflight.py", "rebalance.py", "run_dir.py", "status.py")
# The keys the seed reaches: itself, the acquisition RNG and the seed's pool order and labels.
SEED_KEYS = {"seed", "acquisition.seed", "data_source.shuffled_indices_file", "data_source.shuffled_labels_file"}
# acquisition_mc_samples is K for probability, acquisition and threshold (the j_sel fit); eval K is separate.
K_KEYS = {"acquisition_mc_samples", "probability.num_mc_samples", "acquisition.num_mc_samples",
          "threshold.num_mc_samples"}


def _rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def _load(name, campaign="det_v3_seeds"):
    spec = importlib.util.spec_from_file_location(f"{campaign}_{name}", ROOT / "scripts" / campaign / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rep(seed):
    return {s: r for r, s in ALL_SEEDS.items()}[seed]


# ----------------------------------------------------------------- plan, manifest, configs
def test_plan_regenerates_byte_for_byte(tmp_path):
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/det_v3_seeds/make_plan.py"), "--out", str(out)],
                   check=True, capture_output=True)
    assert out.read_text() == (ROOT / "scripts/det_v3_seeds/run_plan.json").read_text()


@pytest.mark.parametrize("block,base,first,prefix,keep", [
    ("main", V3_MAIN, 1, "S", ("setting", "model", "arm", "hours", "owner")),
    ("abl", V3_A7, 121, "U", ("setting", "model", "ab", "change", "hours", "owner"))])
def test_plan_repeats_det_v3_rows_at_two_seeds_seed_major(block, base, first, prefix, keep):
    sys.path.insert(0, str(ROOT / "scripts/det_v2"))
    import det_v2_lib as lib
    rows, n = PLAN[block], len(base)
    assert n == {"main": 60, "abl": 30}[block] and len(rows) == 2 * n
    assert [r["rank"] for r in rows] == list(range(first, first + 2 * n))
    for i, r in enumerate(rows):
        rep, b = 2 + i // n, base[i % n]                    # all of _r2 first, each half in det_v3 order
        assert r["seed"] == SEEDS[rep] and r["base_id"] == b["id"] and r["base_name"] == b["name"]
        assert r["name"] == f"{b['name']}_r{rep}" and r["id"] == f"{prefix}{rep}{b['id'][1:]}"
        assert {k: r[k] for k in keep} == {k: b[k] for k in keep}
        assert re.fullmatch(prefix + r"[23]\d\d", r["id"]) and lib.run_kind(r["name"]) == lib.run_kind(b["name"])


def test_plan_blocks_ranks_ids_and_counterparts():
    sys.path.insert(0, str(ROOT / "scripts/det_v2"))
    import det_v2_lib as lib
    assert {k: len(v) for k, v in PLAN.items()} == BLOCKS
    assert [r["rank"] for r in RUNS] == list(range(1, 3121))
    assert len({r["id"] for r in RUNS}) == len({r["name"] for r in RUNS}) == 3120
    for r in RUNS:
        b = V2_PARTX[r["base_name"]] if r["kind"] == "partx" else V3_ROWS[r["base_name"]]
        assert r["base_id"] == b["id"] and re.fullmatch(r"[A-Z]\d+", r["id"]), r["name"]
        assert {k: r[k] for k in ("setting", "model", "hours", "owner")} == {k: b[k] for k in ("setting", "model", "hours", "owner")}
        assert r["name"].endswith(f"_r{_rep(r['seed'])}") and lib.run_kind(r["name"]) == lib.run_kind(b["name"])


def test_ranks_1_to_180_are_as_launched():
    lines = (ROOT / "scripts/det_v3_seeds/manifest.tsv").read_text().splitlines(keepends=True)
    assert hashlib.sha256("".join(lines[:181]).encode()).hexdigest() == LAUNCHED_MANIFEST_SHA
    h = hashlib.sha256()
    for line in lines[1:181]:
        h.update((ROOT / "configs/adaptive_v2/experiment/det_v3_seeds/runs" / f"{line.split(chr(9))[1]}.yaml").read_bytes())
    assert h.hexdigest() == LAUNCHED_CONFIGS_SHA


def _cell(r):
    """(det_v3 counterpart, acquisition group override, the A11 change or K, seed)."""
    k = r.get("set", {}).get("acquisition_mc_samples")
    a11 = r["change"] if "initial_train_size" in r.get("set", {}) else None
    return r["base_name"], r.get("acquisition"), a11 or k, r["seed"]


def test_blocks_are_the_full_crosses_and_nothing_else():
    all3, r23 = list(ALL_SEEDS.values()), list(SEEDS.values())
    ab = lambda *names: [r["name"] for r in V3_ABL if r["ab"] in names]      # noqa: E731
    k_cells = ([(m["name"], None, k) for m in V3_QA if m["model"] in ("FM", "MLP final-state", "BNN final-state", "BNN classifier")
                for k in (20, 50)]
               + [(x["name"], None, k) for x in V3_A7 if x["model"] in ("FM", "MLP final-state", "BNN final-state") for k in (20, 50)]
               + [(m["name"], "u_tau", k) for m in V3_QA for k in (20, 50)])
    want = {
        "utau_main": [(m["name"], "u_tau", None, s) for m in V3_QA for s in all3],
        "uh_abl": [(x, "u_h", None, s) for x in ab("A6", "A10", "A11") for s in all3],
        "utau_abl": [(x, "u_tau", None, s) for x in ab("A1", "A2", "A6", "A10", "A11") for s in all3],
        "unif_a11": [(u["name"], None, c, s) for u in V3_UNIF for c in ("initial batch x 0.5", "initial batch x 2") for s in all3],
        "k_abl": [(b, acq, k, s) for b, acq, k in k_cells for s in all3],
        "qa_abl": [(x, None, None, s) for x in ab("A1", "A2", "A6", "A10", "A11") for s in r23],
    }
    for block, cells in want.items():
        got = Counter(_cell(r) for r in PLAN[block])
        assert max(got.values()) == 1 and set(got) == set(cells) and len(cells) == BLOCKS[block], block
    # The exclusions, spelled out: no tau*-only ablation without a tau* reader, no K where nothing draws K.
    assert not {V3_ROWS[r["base_name"]]["ab"] for r in PLAN["uh_abl"]} & {"A1", "A2"}
    assert all(V3_ROWS[r["base_name"]].get("arm") != "uniform" for r in PLAN["k_abl"])
    assert not [r for r in PLAN["k_abl"] if r["model"] == "MLP classifier" and r["arm"] in ("q_A", "U_H")]
    assert not [r for r in PLAN["k_abl"] if r["model"] == "BNN classifier" and r["arm"] == "U_H"]


def test_ranks_are_seed_major_and_longest_first():
    a = PLAN["utau_main"]
    assert [r["rank"] for r in a] == list(range(181, 271))
    assert [(r["base_name"], r["seed"]) for r in a] == [(m["name"], s) for s in ALL_SEEDS.values() for m in V3_QA]
    rest = [r for r in V3_RUNS if 270 < r["rank"] <= 3072]
    assert [r["seed"] for r in rest] == [99512] * 714 + [33407] * 1044 + [24475] * 1044
    assert not {r["kind"] for r in rest[:714]} & {"qa_abl"}
    for lo, hi in ((0, 714), (714, 1758), (1758, 2802)):
        hours = [r["hours"] for r in rest[lo:hi]]
        assert hours == sorted(hours, reverse=True)


def test_manifest_regenerates_and_copies_the_det_v3_requests(tmp_path):
    out = tmp_path / "m.tsv"
    subprocess.run([sys.executable, str(ROOT / "scripts/det_v3_seeds/make_manifest.py"), "--out", str(out)],
                   check=True, capture_output=True)
    assert out.read_text() == (ROOT / "scripts/det_v3_seeds/manifest.tsv").read_text()
    rows = _rows(out)
    assert list(rows[0]) == ["id", "name", "rank", "owner", "config", "script", "mem", "est_hours", "base_id", "seed"]
    v3 = {r["id"]: r for r in _rows(ROOT / "scripts/det_v3/manifest.tsv")}
    v2 = {r["id"]: r for r in _rows(ROOT / "scripts/det_v2/manifest.tsv")}
    for r in rows:
        b = (v2 if BY_NAME[r["name"]]["kind"] == "partx" else v3)[r["base_id"]]
        assert (r["mem"], r["est_hours"], r["script"], r["owner"]) == (b["mem"], b["est_hours"], b["script"], b["owner"])
        assert r["config"] == f"+experiment=det_v3_seeds/runs/{r['name']}"
        assert (ROOT / "configs/adaptive_v2/experiment/det_v3_seeds/runs" / f"{r['name']}.yaml").exists()
    assert [r["name"] for r in rows] == [r["name"] for r in RUNS]
    assert {o: sum(r["owner"] == o for r in rows) for o in ("Sumanth", "Dhruv")} == {"Sumanth": 1042, "Dhruv": 2078}
    # det_v3's owners split by setting; the Part-X rows keep det_v2's owner (q3d-lqr Part-X was Sumanth's).
    assert {BY_NAME[r["name"]]["setting"] for r in rows if r["owner"] == "Sumanth"
            and BY_NAME[r["name"]]["kind"] != "partx"} == {"q3d-ppo", "q2d-ppo"}
    for block, expect in (("main", {"Sumanth": 37.2, "Dhruv": 43.0}), ("abl", {"Sumanth": 18.6, "Dhruv": 21.5})):
        hours = {(r["owner"], r["seed"]): 0.0 for r in rows if BY_NAME[r["name"]]["kind"] == block}
        for r in rows:
            if BY_NAME[r["name"]]["kind"] == block:
                hours[(r["owner"], r["seed"])] += float(r["est_hours"])
        assert {k: round(v, 1) for k, v in hours.items()} == {(o, s): h for o, h in expect.items()
                                                              for s in ("33407", "24475")}, block
    new = {o: round(sum(float(r["est_hours"]) for r in rows if int(r["rank"]) > 180 and r["owner"] == o), 1)
           for o in ("Sumanth", "Dhruv")}
    assert new == {"Sumanth": 1908.6, "Dhruv": 2218.2}
    partx = [r for r in rows if BY_NAME[r["name"]]["kind"] == "partx"]
    assert len(partx) == 12 and {r["script"] for r in partx} == {"scripts/run_partx_faithful.py"}
    assert all(r["script"] == "scripts/run_adaptive.py" for r in rows if BY_NAME[r["name"]]["kind"] != "partx")


def test_new_rows_take_the_plan_store_override_and_launched_rows_do_not():
    store = _load("make_plan").NEW_ROW_STORE
    assert store == {"endpoint_epochs": []}                       # no endpoint dumps until Sumanth says otherwise
    assert all("store" not in r for r in RUNS[:180])
    assert all(r.get("store") == store for r in RUNS[180:3072] + PLAN["k_method"])
    assert all("store" not in r for r in PLAN["partx"])           # the Part-X engine writes no endpoints


def test_run_configs_match_the_plan():
    out = subprocess.run([sys.executable, "scripts/det_v3_seeds/make_run_configs.py", "--check"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_make_run_configs_check_flags_a_config_the_plan_lacks(tmp_path, monkeypatch):
    mod = _load("make_run_configs")
    monkeypatch.setattr(mod, "OUT", tmp_path)
    run = PLAN["main"][0]
    (tmp_path / f"{run['name']}.yaml").write_text(mod.render(run))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"main": [run], "abl": []}))
    monkeypatch.setattr(sys, "argv", ["x", "--plan", str(plan), "--check"])
    mod.main()                                                       # matches
    (tmp_path / "stale_r2.yaml").write_text("# @package _global_\n")
    with pytest.raises(SystemExit, match="stale_r2.yaml"):
        mod.main()


def _flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flat(v, f"{p}.{k}" if p else str(k)))
        return out
    return {p: repr(d) if isinstance(d, list) else d}


PICKS = ("seed", "acquisition.seed", "det_v3_run", "output_dir", "sampling_mode", "acquisition.score",
         "initial_train_size", "samples_per_epoch", "eval.num_mc_samples_eval", "calibration.num_mc_samples",
         *sorted(K_KEYS), "data_source.shuffled_indices_file", "data_source.shuffled_labels_file",
         "prediction_store.enabled", "prediction_store.endpoint_epochs", "prediction_store.endpoint_test_rows")


def _compose_init(exp_dir):
    """Pool worker: the run scripts' resolvers and one Hydra context for the worker's life."""
    sys.path.insert(0, str(ROOT / "scripts/det_v2"))
    import det_v2_lib as lib
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    lib.register_resolvers({"NET_ID": "tester", "EXP_DIR": exp_dir, "DATA_DIR": "/data", "SHARED_DATA_BASE": "/data"})
    GlobalHydra.instance().clear()
    globals()["_HYDRA_CTX"] = initialize_config_dir(config_dir=str(ROOT / "configs/adaptive_v2"), version_base=None)


def _compose_rows(pairs):
    """[(name, base name)] -> [(name, keys that differ from the base, picked values of both)]."""
    from hydra import compose
    from omegaconf import OmegaConf

    def get(over):
        return _flat(OmegaConf.to_container(compose(config_name="default", overrides=[over]), resolve=True))
    base, out = {}, []
    for name, b in pairs:
        if b not in base:
            base[b] = get(f"+experiment=det_v3/runs/{b}")
        a, c = base[b], get(f"+experiment=det_v3_seeds/runs/{name}")
        diff = sorted(k for k in set(a) | set(c) if a.get(k) != c.get(k))
        out.append((name, diff, {k: c.get(k) for k in PICKS}, {k: a.get(k) for k in PICKS}))
    return out


def _intended(r, base):
    keys = {"det_v3_run", "output_dir"}
    keys |= {f"prediction_store.{k}" for k, v in r.get("store", {}).items()
             if base[f"prediction_store.{k}"] != (repr(v) if isinstance(v, list) else v)}
    if r["seed"] != ALL_SEEDS[1]:
        keys |= SEED_KEYS
    if r.get("acquisition"):
        keys |= {"sampling_mode", "acquisition.score"}
    if "acquisition_mc_samples" in r.get("set", {}):
        keys |= K_KEYS
    if "initial_train_size" in r.get("set", {}):
        keys |= {"initial_train_size", "samples_per_epoch"}
    return keys


def test_every_config_is_its_det_v3_counterpart_plus_only_its_change(tmp_path):
    groups = defaultdict(list)
    for r in V3_RUNS:
        groups[r["base_name"]].append((r["name"], r["base_name"]))
    chunks = [[] for _ in range(8)]
    for i, g in enumerate(sorted(groups.values(), key=len, reverse=True)):
        min(chunks, key=len).extend(g)
    exp = str(tmp_path / "exp")
    with multiprocessing.get_context("fork").Pool(8, initializer=_compose_init, initargs=(exp,)) as pool:
        results = [x for part in pool.map(_compose_rows, chunks) for x in part]
    assert sorted(n for n, *_ in results) == sorted(r["name"] for r in V3_RUNS)
    a11 = {(BY_NAME[n]["setting"], BY_NAME[n]["change"]): (a["initial_train_size"], a["samples_per_epoch"])
           for n, _, _, a in results if V3_ROWS[BY_NAME[n]["base_name"]].get("ab") == "A11"}
    for name, diff, c, a in results:
        r = BY_NAME[name]
        assert set(diff) == _intended(r, a), (name, diff)
        assert c["prediction_store.enabled"] is True
        for k, v in r.get("store", {}).items():
            assert c[f"prediction_store.{k}"] == (repr(v) if isinstance(v, list) else v)
        assert c["det_v3_run"] == name and c["seed"] == c["acquisition.seed"] == r["seed"]
        assert c["output_dir"] == f"{exp}/det_v3_seeds/{r['setting']}/{name}_s{r['seed']}"
        assert c["eval.num_mc_samples_eval"] == c["calibration.num_mc_samples"] == 100
        for k in ("data_source.shuffled_indices_file", "data_source.shuffled_labels_file"):
            assert c[k] == a[k].replace("_99512.txt", f"_{r['seed']}.txt")
        if r.get("acquisition"):
            assert c["sampling_mode"] == c["acquisition.score"] == r["acquisition"]
        if "acquisition_mc_samples" in r.get("set", {}):
            assert {c[k] for k in K_KEYS} == {r["set"]["acquisition_mc_samples"]}
        if r["kind"] == "unif_a11":                       # the schedule of that setting's q_A A11 rows
            assert (c["initial_train_size"], c["samples_per_epoch"]) == a11[(r["setting"], r["change"])]


# ----------------------------------------------------------------- Part-X at the other two seeds
def test_partx_rows_are_det_v2s_partx_rows_at_33407_and_24475():
    rows = PLAN["partx"]
    base = sorted(V2_PARTX.values(), key=lambda r: (-r["hours"], r["rank"]))   # longest first, then det_v2 rank
    assert len(V2_PARTX) == 6 and [r["rank"] for r in rows] == list(range(3073, 3085))
    assert [(r["base_name"], r["seed"]) for r in rows] == [(b["name"], s) for s in SEEDS.values() for b in base]
    for r in rows:
        b, rep = V2_PARTX[r["base_name"]], _rep(r["seed"])
        assert r["name"] == f"{b['name']}_r{rep}" and r["id"] == f"P{rep}{b['id'][1:]}" and r["base_id"] == b["id"]
        assert r["base_campaign"] == "det_v2" and r["kind"] == "partx"
        assert {k: r[k] for k in ("setting", "model", "arm", "hours", "owner")} == \
               {k: b[k] for k in ("setting", "model", "arm", "hours", "owner")}
        assert not {"acquisition", "set", "store", "change"} & set(r)          # the seed is the only change
    assert {r["setting"] for r in rows} == {"q3d-lqr", "q3d-ppo", "q2d-ppo", "cp-lqr", "cp-ppo", "pen-lqr"}


def _compose_partx(pairs):
    """[(name, det_v2 base name)] -> [(name, keys that differ from the base, values of both)], composed with
    run_partx_faithful.py's config_name."""
    from hydra import compose
    from omegaconf import OmegaConf

    def get(over):
        return _flat(OmegaConf.to_container(compose(config_name="partx_faithful", overrides=[over]), resolve=True))
    out = []
    for name, b in pairs:
        a, c = get(f"+experiment=det_v2/runs/{b}"), get(f"+experiment=det_v3_seeds/runs/{name}")
        out.append((name, sorted(k for k in set(a) | set(c) if a.get(k) != c.get(k)), c, a))
    return out


def test_partx_configs_are_their_det_v2_config_plus_only_the_seed(tmp_path):
    exp = str(tmp_path / "exp")
    with multiprocessing.get_context("fork").Pool(1, initializer=_compose_init, initargs=(exp,)) as pool:
        results = pool.apply(_compose_partx, ([(r["name"], r["base_name"]) for r in PLAN["partx"]],))
    assert sorted(n for n, *_ in results) == sorted(r["name"] for r in PLAN["partx"])
    for name, diff, c, a in results:
        r = BY_NAME[name]
        # The runner seeds Part-X from cfg.seed (partx_faithful.runner); it reads no acquisition group, and the
        # default one these configs compose does not interpolate the seed.
        assert set(diff) == SEED_KEYS - {"acquisition.seed"} | {"det_v3_run", "output_dir"}, (name, diff)
        assert c["seed"] == r["seed"] and a["seed"] == 99512
        assert c["det_v3_run"] == name and c["det_v2_run"] == r["base_name"]
        assert c["output_dir"] == f"{exp}/det_v3_seeds/{r['setting']}/{name}_s{r['seed']}"
        for k in ("data_source.shuffled_indices_file", "data_source.shuffled_labels_file"):
            assert c[k] == a[k].replace("_99512.txt", f"_{r['seed']}.txt") and c[k] != a[k]
        # D_val, D_cal and the test set are det_v2's own, so partx_recert.py reads every seed the same way.
        for k in ("data_source.val_set_file", "data_source.cal_set_file", "data_source.test_set_file"):
            assert c[k] == a[k] and c[k], (name, k)
        assert c["prediction_store.enabled"] is True and c["n_epochs"] == 11


# ----------------------------------------------------------------- the method rows at K = 20 and 50
def test_k_method_rows_are_the_matrix_method_rows_at_k20_and_k50():
    rows, uh = PLAN["k_method"], {r["name"]: r for r in PLAN["uh_abl"]}
    assert [r["rank"] for r in rows] == list(range(3085, 3121))
    want = []
    for seed in ALL_SEEDS.values():                           # seed-major; longest first, K = 20 before 50
        bases = sorted((uh[f"{m}_r{_rep(seed)}"] for m in METHOD), key=lambda b: -b["hours"])
        want += [(b["name"], k) for b in bases for k in (20, 50)]
    assert [(f"{r['name'].rsplit('_k', 1)[0]}_r{_rep(r['seed'])}", r["set"]["acquisition_mc_samples"]) for r in rows] == want
    for r in rows:
        b = uh[f"{r['name'].rsplit('_k', 1)[0]}_r{_rep(r['seed'])}"]
        k = r["set"]["acquisition_mc_samples"]
        assert r["name"] == f"{b['name'][:-3]}_k{k}_r{_rep(r['seed'])}" and r["id"] == f"W{_rep(r['seed'])}{k // 10}{int(b['base_id'][1:]):03d}"
        assert r["set"] == {"acquisition_mc_samples": k} and r["kind"] == "k_method" and r["change"] == f"K = {k}"
        same = ("base_id", "base_name", "seed", "setting", "model", "arm", "hours", "owner", "acquisition", "store")
        assert {f: r[f] for f in same} == {f: b[f] for f in same}
    owners = {o: sum(r["owner"] == o for r in rows) for o in ("Sumanth", "Dhruv")}
    assert owners == {"Sumanth": 12, "Dhruv": 24}
    assert {r["setting"] for r in rows if r["owner"] == "Sumanth"} == {"q3d-ppo", "q2d-ppo"}


def test_k_method_configs_differ_from_their_method_row_only_in_k(tmp_path):
    pairs = [(r["name"], f"{r['name'].rsplit('_k', 1)[0]}_r{_rep(r['seed'])}") for r in PLAN["k_method"]]
    exp = str(tmp_path / "exp")
    with multiprocessing.get_context("fork").Pool(1, initializer=_compose_init, initargs=(exp,)) as pool:
        results = pool.apply(_compose_seeds_pairs, (pairs,))
    assert len(results) == 36
    for name, diff, c, a in results:
        k = BY_NAME[name]["set"]["acquisition_mc_samples"]
        assert set(diff) == K_KEYS | {"det_v3_run", "output_dir"}, (name, diff)
        assert {c[key] for key in K_KEYS} == {k} and {a[key] for key in K_KEYS} == {100}
        assert c["eval.num_mc_samples_eval"] == c["calibration.num_mc_samples"] == 100     # eval K stays 100
        assert c["seed"] == a["seed"] == BY_NAME[name]["seed"] and c["det_v3_run"] == name


def _compose_seeds_pairs(pairs):
    """[(det_v3_seeds name, det_v3_seeds name)] -> [(name, keys that differ, values of name, values of the other)]."""
    from hydra import compose
    from omegaconf import OmegaConf

    def get(n):
        return _flat(OmegaConf.to_container(compose(config_name="default", overrides=[f"+experiment=det_v3_seeds/runs/{n}"]),
                                            resolve=True))
    out = []
    for name, other in pairs:
        c, a = get(name), get(other)
        out.append((name, sorted(k for k in set(a) | set(c) if a.get(k) != c.get(k)), c, a))
    return out


# ----------------------------------------------------------------- wrappers in a stub world
def test_every_det_v3_wrapper_has_a_det_v3_seeds_twin():
    for name in WRAPPERS:
        mine = (ROOT / "scripts/det_v3_seeds" / name).read_text()
        theirs = (ROOT / "scripts/det_v3" / name).read_text()
        assert mine.replace("det_v3_seeds", "det_v3").replace("the det_v3 ones", "det_v3's") == theirs, name
        assert (ROOT / "scripts/det_v3_seeds" / name).stat().st_mode == (ROOT / "scripts/det_v3" / name).stat().st_mode


def _add_campaigns(world):
    """The world's repo gains det_v3 (one run config per stub run) and det_v3_seeds as the real tools
    write it: the wrappers, a manifest with base_id and seed, and _r2 configs from render()."""
    r = world.repo
    render = _load("make_run_configs").render
    for camp in ("det_v3", "det_v3_seeds"):
        d = r / "scripts" / camp
        d.mkdir()
        for name in WRAPPERS:
            shutil.copy2(ROOT / "scripts" / camp / name, d / name)
        (r / "configs/adaptive_v2/experiment" / camp / "runs").mkdir(parents=True)
        (world.exp / camp).mkdir()
    (r / "scripts/det_v3/manifest.tsv").write_text(MANIFEST.replace("+experiment=det_v2/", "+experiment=det_v3/"))
    lines = MANIFEST.splitlines()
    seeds = [lines[0] + "\tbase_id\tseed"]
    for line in lines[1:]:
        f = line.split("\t")
        seeds.append("\t".join([f"S2{f[0][1:]}", f"{f[1]}_r2", f[2], f[3], f"+experiment=det_v3_seeds/runs/{f[1]}_r2",
                                *f[5:], f[0], "33407"]))
    (r / "scripts/det_v3_seeds/manifest.tsv").write_text("\n".join(seeds) + "\n")
    for name, setting in ((GPU_RUN, "q3d-lqr"), (SMALL_RUN, "cp-lqr"), (PARTX_RUN, "cp-lqr")):
        (r / "configs/adaptive_v2/experiment/det_v3/runs" / f"{name}.yaml").write_text(
            RUN_YAML.replace("det_v2/", "det_v3/").format(setting=setting, name=name))
        run = {"id": "S2xx", "base_id": "Mxx", "base_name": name, "seed": 33407, "rank": 1, "owner": "Sumanth",
               "name": f"{name}_r2", "setting": setting}
        (r / "configs/adaptive_v2/experiment/det_v3_seeds/runs" / f"{name}_r2.yaml").write_text(render(run))
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "det_v3 + det_v3_seeds")
    git(r, "push", "-q", "origin", "HEAD:main")


def _run(world, script, *args):
    r = subprocess.run(["bash", str(world.repo / "scripts/det_v3_seeds" / script), *args],
                       env=world.env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"det_v3_seeds/{script} {args} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}"
    return r


def test_det_v3_seeds_campaign_gets_its_own_paths_and_tag(world):
    r = subprocess.run(["bash", "-c", f"source {world.repo}/scripts/det_v2/common.sh && "
                        'echo "$DET_V2_ROOT|$LEDGER|$LOG_DIR|$MANIFEST|$AMAREL_LOG_DIR|$SESSION_TAG"'],
                       env={**world.env, "DET_CAMPAIGN": "det_v3_seeds"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    root, ledger, logs, manifest, alogs, tag = r.stdout.strip().split("|")
    assert root == f"{world.exp}/det_v3_seeds" and ledger == f"{root}/launched.tsv" and logs == f"{root}/logs"
    assert manifest == f"{world.repo}/scripts/det_v3_seeds/manifest.tsv"
    assert alogs == f"{world.amarel / 'logs'}/det_v3_seeds" and tag == "v3_seeds_"


def test_det_v3_seeds_launch_writes_only_its_own_ledger(world):
    _add_campaigns(world)
    v3 = world.exp / "det_v3" / "launched.tsv"
    v3.write_text("name\tuser\twhere\tjob\thostgpu\ttimestamp\tlog\tcommit\trun_dir\taction\n")
    before = v3.read_text()
    out = _run(world, "launch.sh", f"{GPU_RUN}_r2", "ilab", "a4000")
    assert "iLab job" in out.stdout
    rows = _rows(world.exp / "det_v3_seeds" / "launched.tsv")
    assert [r["action"] for r in rows] == ["claim", "launch"]
    assert rows[-1]["run_dir"] == str(world.exp / "det_v3_seeds" / "q3d-lqr" / f"{GPU_RUN}_r2_s33407")
    assert rows[-1]["log"].startswith(str(world.exp / "det_v3_seeds" / "logs"))
    assert v3.read_text() == before and not (world.exp / "det_v3" / "q3d-lqr").exists()


def test_det_v3_seeds_direct_launch_tags_its_tmux_session(world):
    _add_campaigns(world)
    out = _run(world, "launch.sh", f"{SMALL_RUN}_r2", "direct", "boxa", "2")
    assert f"session gpu_v3_seeds_{SMALL_RUN}_r2" in out.stdout
    assert f"-s gpu_v3_seeds_{SMALL_RUN}_r2" in (world.calls / "tmux").read_text()


def test_det_v3_seeds_python_wrappers_read_their_campaign(world):
    _add_campaigns(world)
    _run(world, "launch.sh", f"{GPU_RUN}_r2", "ilab", "a4000")
    st = subprocess.run([sys.executable, str(world.repo / "scripts/det_v3_seeds/status.py")],
                        env=world.env, capture_output=True, text=True, timeout=300)
    assert st.returncode == 0, st.stderr
    line = next(l for l in st.stdout.splitlines() if f"{GPU_RUN}_r2" in l)
    assert "not launched" not in line
    rd = subprocess.run([sys.executable, str(world.repo / "scripts/det_v3_seeds/run_dir.py"), f"{GPU_RUN}_r2"],
                        env=world.env, capture_output=True, text=True, timeout=300)
    assert rd.stdout.strip().endswith(f"det_v3_seeds/q3d-lqr/{GPU_RUN}_r2_s33407"), rd.stderr


def test_det_v3_seeds_partx_row_launches_on_ilab_cpu_only(world):
    _add_campaigns(world)
    r = world.repo
    run = {"id": "P303", "base_id": "M03", "base_name": PARTX_RUN, "base_campaign": "det_v2", "seed": 24475,
           "rank": 2, "owner": "Dhruv", "name": f"{PARTX_RUN}_r3", "setting": "cp-lqr"}
    (r / "configs/adaptive_v2/experiment/det_v3_seeds/runs" / f"{run['name']}.yaml").write_text(
        _load("make_run_configs").render(run))
    man = r / "scripts/det_v3_seeds/manifest.tsv"
    man.write_text(man.read_text() + "\t".join(["P303", run["name"], "2", "Dhruv",
                                                f"+experiment=det_v3_seeds/runs/{run['name']}",
                                                "scripts/run_partx_faithful.py", "16G", "0.1", "M03", "24475"]) + "\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "partx _r3")
    git(r, "push", "-q", "origin", "HEAD:main")
    bad = subprocess.run(["bash", str(r / "scripts/det_v3_seeds/launch.sh"), run["name"], "ilab", "a4000"],
                         env=world.env, capture_output=True, text=True, timeout=300)
    assert bad.returncode != 0 and "launch it with ilabcpu" in bad.stderr
    out = _run(world, "launch.sh", run["name"], "ilabcpu")
    assert "iLab CPU job" in out.stdout
    led = _rows(world.exp / "det_v3_seeds" / "launched.tsv")
    assert [(x["action"], x["where"]) for x in led] == [("claim", "ilabcpu"), ("launch", "ilabcpu")]
    assert led[-1]["run_dir"] == str(world.exp / "det_v3_seeds" / "cp-lqr" / f"{run['name']}_s24475")
    args = [d["ARGS"] for d in world.dumps_list()]
    assert len(args) == 1 and args[0].split()[:2] == ["scripts/run_partx_faithful.py",
                                                      f"+experiment=det_v3_seeds/runs/{run['name']}"], args
