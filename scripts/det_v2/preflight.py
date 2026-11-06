#!/usr/bin/env python
"""Check that every data file a det_v2 run reads is readable by the shared group.

Composes each run's Hydra config the way its runner does (configs/adaptive_v2, the
runner's config_name, `+experiment=det_v2/runs/<name>`) and collects the data paths:
every string under data_source that is an absolute path (pool, pool order and labels,
val/cal/test sets, split dirs), plus dataset_root, eval_root and system.dataset_dir.
A directory value also brings its immediate entries (split_manifest.json and the like),
not deeper ones. Each path must be readable (directories also searchable) by group
SHARED_GROUP, or by everyone, and every parent directory searchable the same way.

    preflight.py [name ...]              # default: every manifest row with a config
    preflight.py --overrides "system=pendulum predictor=det_mlp_clf acquisition=ranked +experiment=det_rerun"
    preflight.py <name> --list           # tab-separated kind (f, d, -) and path of the data_source paths
    preflight.py ... --fix               # chgrp SHARED_GROUP + chmod g+r (g+rx on dirs, g+x on a
                                         # parent) where the current user owns the entry and it sits
                                         # in DATA_DIR or SHARED_DATA_BASE; the rest is reported
Without --fix nothing changes; the report says what --fix would do. Exit 1 when a
problem is left.
"""
from __future__ import annotations

import argparse
import grp
import os
import pwd
import shlex
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_v2_lib as lib  # noqa: E402

TOP_KEYS = ("dataset_root", "eval_root", "system.dataset_dir")


def _walk(node, prefix: str, out: list[tuple[str, str]]) -> None:
    """Every resolvable absolute-path string under an OmegaConf node, with its key."""
    from omegaconf import DictConfig, ListConfig
    keys = node.keys() if isinstance(node, DictConfig) else range(len(node))
    for k in keys:
        key = f"{prefix}.{k}" if isinstance(node, DictConfig) else f"{prefix}[{k}]"
        try:
            v = node[k]
        except Exception:          # an interpolation that does not resolve here (oc.env etc.)
            continue
        if isinstance(v, (DictConfig, ListConfig)):
            _walk(v, key, out)
        elif isinstance(v, str) and v.startswith("/"):
            out.append((key, v))


def data_paths(cfg) -> list[tuple[str, str]]:
    from omegaconf import OmegaConf
    out: list[tuple[str, str]] = []
    if cfg.get("data_source") is not None:
        _walk(cfg.data_source, "data_source", out)
    for key in TOP_KEYS:
        try:
            v = OmegaConf.select(cfg, key)
        except Exception:
            v = None
        if isinstance(v, str) and v.startswith("/"):
            out.append((key, v))
    return out


def expand(paths: list[tuple[str, str]]) -> dict[str, list[str]]:
    """realpath -> the keys that reach it. data_source dirs add their immediate entries."""
    seen: dict[str, list[str]] = {}
    for key, p in paths:
        rp = os.path.realpath(p)
        seen.setdefault(rp, []).append(key)
        if key.startswith("data_source") and os.path.isdir(rp):
            for child in sorted(os.listdir(rp)):
                seen.setdefault(os.path.realpath(os.path.join(rp, child)), []).append(f"{key}/*")
    return seen


def _ok(st: os.stat_result, gid: int, need_r: bool, need_x: bool) -> bool:
    g = (not need_r or st.st_mode & stat.S_IRGRP) and (not need_x or st.st_mode & stat.S_IXGRP)
    o = (not need_r or st.st_mode & stat.S_IROTH) and (not need_x or st.st_mode & stat.S_IXOTH)
    return bool((st.st_gid == gid and g) or o)


def _owner(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _group(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


class Problem:
    def __init__(self, path: str, st: os.stat_result | None, why: str, keys: list[str], is_dir: bool):
        self.path, self.st, self.why, self.keys, self.is_dir = path, st, why, keys, is_dir

    def describe(self) -> str:
        if self.st is None:
            return f"MISSING {self.path}  ({', '.join(self.keys)})"
        mode = stat.filemode(self.st.st_mode)
        return (f"{mode} {_owner(self.st.st_uid)}:{_group(self.st.st_gid)} {self.path}"
                f"  ({self.why}; {', '.join(sorted(set(self.keys)))[:120]})")


# Paths a config lists but no det_v2 code path reads, so their mode does not matter.
# deterministic/quadrotor2D_rl/trajectory_lengths_cache.txt (dm1487:nogroup 660): the npz
# data source keeps trajectory lengths in memory (reviewed 2026-11-06).
NOT_READ = ("deterministic/quadrotor2D_rl/trajectory_lengths_cache.txt",)


def check(seen: dict[str, list[str]], gid: int) -> tuple[list[Problem], int]:
    """Problems with the entries and every parent dir, and how many paths were checked."""
    problems: dict[str, Problem] = {}
    checked = set()
    for rp, keys in seen.items():
        if rp.endswith(NOT_READ):
            continue
        if not os.path.lexists(rp):
            problems[rp] = Problem(rp, None, "missing", keys, False)
            continue
        st = os.stat(rp)
        is_dir = stat.S_ISDIR(st.st_mode)
        checked.add(rp)
        if not _ok(st, gid, True, is_dir):
            problems[rp] = Problem(rp, st, "group cannot read" + (" or search" if is_dir else ""), keys, is_dir)
        parent = os.path.dirname(rp)
        while parent and parent not in checked:
            checked.add(parent)
            pst = os.stat(parent)
            if not _ok(pst, gid, False, True):
                problems.setdefault(parent, Problem(parent, pst, "group cannot search (parent)", [f"parent of {rp}"], True))
            if parent == "/":
                break
            parent = os.path.dirname(parent)
    return list(problems.values()), len(checked)


def fixable(p: Problem, roots: list[str]) -> bool:
    return (p.st is not None and p.st.st_uid == os.getuid()
            and any(p.path == r or p.path.startswith(r + "/") for r in roots))


def fix_cmd(p: Problem, group: str) -> str:
    bits = "g+rx" if p.is_dir else "g+r"
    if p.why.endswith("(parent)"):
        bits = "g+x"
    return f"chgrp {group} {shlex.quote(p.path)} && chmod {bits} {shlex.quote(p.path)}"


def apply_fix(p: Problem, gid: int) -> None:
    if p.st.st_gid != gid:
        os.chown(p.path, -1, gid)
    add = stat.S_IXGRP if p.why.endswith("(parent)") else (stat.S_IRGRP | (stat.S_IXGRP if p.is_dir else 0))
    os.chmod(p.path, stat.S_IMODE(os.stat(p.path).st_mode) | add)


def report(label: str, cfg, env: dict, group: str, gid: int, do_fix: bool) -> int:
    seen = expand(data_paths(cfg))
    problems, n = check(seen, gid)
    roots = sorted({os.path.realpath(env["DATA_DIR"]), os.path.realpath(env["SHARED_DATA_BASE"])})
    print(f"{label}: {len(seen)} data paths, {n} entries checked with parents, {len(problems)} problem(s)")
    left = 0
    for p in sorted(problems, key=lambda q: q.path):
        if p.st is None:
            print(f"  {p.describe()}")
            left += 1
        elif fixable(p, roots):
            if do_fix:
                apply_fix(p, gid)
                print(f"  FIXED      {p.describe()}")
            else:
                print(f"  WOULD FIX  {p.describe()}\n             {fix_cmd(p, group)}")
                left += 1
        elif p.st.st_uid == os.getuid():
            print(f"  OUTSIDE    {p.describe()}\n             outside the data tree, left alone: {fix_cmd(p, group)}")
            left += 1
        else:
            print(f"  NOT MINE   {p.describe()}\n             owner {_owner(p.st.st_uid)} must run: {fix_cmd(p, group)}")
            left += 1
    return left


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--overrides", default=None, help="compose these Hydra overrides instead of a manifest row")
    ap.add_argument("--config-name", default="default", help="config_name for --overrides")
    ap.add_argument("--code", default=None, help="code tree whose configs to compose (default: this repo)")
    ap.add_argument("--group", default=None, help="group that must read the data (default: SHARED_GROUP)")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the data_source paths only")
    a = ap.parse_args()
    env = lib.load_user_env()
    code = a.code or env["DET_V2_REPO"]
    group = a.group or env["SHARED_GROUP"]
    gid = grp.getgrnam(group).gr_gid
    jobs = []
    if a.overrides is not None:
        jobs.append((a.overrides, lambda: lib.compose(env, code, a.overrides.split(), a.config_name)))
    else:
        rows = lib.read_manifest(env["MANIFEST"])
        if a.names:
            rows = [lib.manifest_row(env, n) for n in a.names]
        for r in rows:
            cfg_file = Path(code) / "configs/adaptive_v2/experiment" / (r["config"].split("=", 1)[1] + ".yaml")
            if not cfg_file.exists():
                print(f"{r['name']}: no config yet ({cfg_file})")
                continue
            jobs.append((r["name"], (lambda r=r: lib.compose_row(env, r, code))))
    left = 0
    for label, make in jobs:
        cfg = make()
        if a.list:
            for _, p in data_paths(cfg):
                if _.startswith("data_source"):
                    kind = "d" if os.path.isdir(p) else "f" if os.path.exists(p) else "-"
                    print(f"{kind}\t{p}")
            continue
        left += report(label, cfg, env, group, gid, a.fix)
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
