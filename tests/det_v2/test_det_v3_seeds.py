"""det_v3_seeds (Sumanth, 2026-11-08): the 60 det_v3 main rows and the 30 A7 U_H rows (X34-X63) at the
paper seeds 33407 and 24475.

The plan, manifest and run configs regenerate from det_v3's; every config composes to its det_v3
run with only the seed and what follows from it changed; the scripts/det_v3_seeds/ wrappers get
their own root, ledger, logs, Amarel log dir and tmux tag."""
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys

import pytest

from det_v2_world import GPU_RUN, MANIFEST, PARTX_RUN, ROOT, RUN_YAML, SMALL_RUN, git

SEEDS = {2: 33407, 3: 24475}
V3 = json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text())
V3_A7 = sorted((r for r in V3["abl"] if r.get("ab") == "A7"), key=lambda r: r["rank"])
PLAN = json.loads((ROOT / "scripts/det_v3_seeds/run_plan.json").read_text())
RUNS = sorted(PLAN["main"] + PLAN["abl"], key=lambda r: r["rank"])
WRAPPERS = ("launch.sh", "resume_amarel.sh", "resume_direct.sh", "resume_ilab.sh", "sync_amarel.sh",
            "preflight.py", "rebalance.py", "run_dir.py", "status.py")
# The keys the seed reaches: itself, the acquisition RNG and the seed's pool order and labels.
SEED_KEYS = {"seed", "acquisition.seed", "data_source.shuffled_indices_file", "data_source.shuffled_labels_file"}


def _rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/det_v3_seeds" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- plan, manifest, configs
def test_plan_regenerates_byte_for_byte(tmp_path):
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/det_v3_seeds/make_plan.py"), "--out", str(out)],
                   check=True, capture_output=True)
    assert out.read_text() == (ROOT / "scripts/det_v3_seeds/run_plan.json").read_text()


@pytest.mark.parametrize("block,base,first,prefix,keep", [
    ("main", sorted(V3["main"], key=lambda r: r["rank"]), 1, "S", ("setting", "model", "arm", "hours", "owner")),
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


def test_plan_covers_both_blocks_once():
    assert [r["rank"] for r in RUNS] == list(range(1, 181))
    assert len({r["id"] for r in RUNS}) == len({r["name"] for r in RUNS}) == 180
    assert {r["base_id"] for r in PLAN["abl"]} == {f"X{i}" for i in range(34, 64)}
    assert {r["change"] for r in PLAN["abl"]} == {"U_H instead of q_A"}


def test_manifest_regenerates_and_copies_the_det_v3_requests(tmp_path):
    out = tmp_path / "m.tsv"
    subprocess.run([sys.executable, str(ROOT / "scripts/det_v3_seeds/make_manifest.py"), "--out", str(out)],
                   check=True, capture_output=True)
    assert out.read_text() == (ROOT / "scripts/det_v3_seeds/manifest.tsv").read_text()
    rows = _rows(out)
    assert list(rows[0]) == ["id", "name", "rank", "owner", "config", "script", "mem", "est_hours", "base_id", "seed"]
    v3 = {r["id"]: r for r in _rows(ROOT / "scripts/det_v3/manifest.tsv")}
    for r in rows:
        b = v3[r["base_id"]]
        assert (r["mem"], r["est_hours"], r["script"], r["owner"]) == (b["mem"], b["est_hours"], b["script"], b["owner"])
        assert r["config"] == f"+experiment=det_v3_seeds/runs/{r['name']}"
        assert (ROOT / "configs/adaptive_v2/experiment/det_v3_seeds/runs" / f"{r['name']}.yaml").exists()
    assert [r["name"] for r in rows] == [r["name"] for r in RUNS]
    assert {o: sum(r["owner"] == o for r in rows) for o in ("Sumanth", "Dhruv")} == {"Sumanth": 60, "Dhruv": 120}
    plan = {r["name"]: r for r in RUNS}
    assert {plan[r["name"]]["setting"] for r in rows if r["owner"] == "Sumanth"} == {"q3d-ppo", "q2d-ppo"}
    for block, expect in (("main", {"Sumanth": 37.2, "Dhruv": 43.0}), ("abl", {"Sumanth": 18.6, "Dhruv": 21.5})):
        hours = {(r["owner"], r["seed"]): 0.0 for r in rows if plan[r["name"]]["kind"] == block}
        for r in rows:
            if plan[r["name"]]["kind"] == block:
                hours[(r["owner"], r["seed"])] += float(r["est_hours"])
        assert {k: round(v, 1) for k, v in hours.items()} == {(o, s): h for o, h in expect.items()
                                                              for s in ("33407", "24475")}, block


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


def test_every_config_is_its_det_v3_run_with_only_the_seed_changed(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/det_v2"))
    import det_v2_lib as lib
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    lib.register_resolvers({"NET_ID": "tester", "EXP_DIR": str(tmp_path / "exp"), "DATA_DIR": "/data",
                            "SHARED_DATA_BASE": "/data"})
    GlobalHydra.instance().clear()

    def get(over):
        return _flat(OmegaConf.to_container(compose(config_name="default", overrides=[over]), resolve=True))

    with initialize_config_dir(config_dir=str(ROOT / "configs/adaptive_v2"), version_base=None):
        base = {r["name"]: get(f"+experiment=det_v3/runs/{r['name']}") for r in V3["main"] + V3_A7}
        for r in RUNS:
            a, b = base[r["base_name"]], get(f"+experiment=det_v3_seeds/runs/{r['name']}")
            diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            assert diff == SEED_KEYS | {"det_v3_run", "output_dir"}, (r["name"], diff)
            assert b["seed"] == b["acquisition.seed"] == r["seed"] and b["det_v3_run"] == r["name"]
            assert b["output_dir"] == f"{tmp_path}/exp/det_v3_seeds/{r['setting']}/{r['name']}_s{r['seed']}"
            for k in ("data_source.shuffled_indices_file", "data_source.shuffled_labels_file"):
                assert b[k] == a[k].replace("_99512.txt", f"_{r['seed']}.txt")
    GlobalHydra.instance().clear()


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
