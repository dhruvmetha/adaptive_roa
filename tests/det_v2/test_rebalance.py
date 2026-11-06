"""rebalance.py: the det_v2 placement rule, the card scan, and moves that never strand a run."""
import os
import sys
import time

import pytest

from det_v2_world import SRC

sys.path.insert(0, str(SRC))
import det_v2_lib as lib  # noqa: E402
import rebalance  # noqa: E402

Q3D_FM, Q2D_FM, CP_FM = "m16_q3dppo_fm_qa", "x02_q2dppo_fm_qa_a1jver", "m43_cplqr_fm_qa"
CP_CLF, Q3D_BNNREG, PEN_PARTX = "m39_cplqr_mlpclf_qa", "m04_q3dlqr_bnnreg_qa", "m66_penlqr_partx"


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = {"NET_ID": "tester", "AMAREL_USER": "tester", "LEDGER": str(tmp_path / "launched.tsv"),
         "MANIFEST": str(tmp_path / "manifest.tsv"), "HOLD_HOSTS": str(tmp_path / "hold_hosts.txt"),
         "ILAB_SUBMIT_HOST": "ilab1", "AMAREL_HOST": "amarel", "EXP_DIR": str(tmp_path / "exp"),
         "AMAREL_EXP_DIR": "/scratch/x", "ILAB_EXCLUDE": "ilab3,ilab4", "GPU_FREE": "gpu-free.sh",
         "DIRECT_HOST_SUFFIX": "", "SHARED_CARDS": ""}
    rows = [("M16", Q3D_FM, 1, "16G", "scripts/run_adaptive.py"), ("M39", CP_CLF, 2, "8G", "scripts/run_adaptive.py"),
            ("M04", Q3D_BNNREG, 3, "24G", "scripts/run_adaptive.py"), ("M66", PEN_PARTX, 4, "16G", "scripts/run_partx_faithful.py")]
    (tmp_path / "manifest.tsv").write_text("id\tname\trank\towner\tconfig\tscript\tmem\test_hours\n" + "".join(
        f"{i}\t{n}\t{r}\tSumanth\t+experiment=det_v2/runs/{n}\t{s}\t{m}\t1.0\n" for i, n, r, m, s in rows))
    monkeypatch.setattr(rebalance, "ENV", e)
    return e


def _row(env, name, where, job, hostgpu, action="launch", user="tester", run_dir=None, commit="abc1234567"):
    lib.append_ledger(env["LEDGER"], {"name": name, "user": user, "where": where, "job": job, "hostgpu": hostgpu,
                                      "timestamp": "t", "log": "-", "commit": commit,
                                      "run_dir": run_dir or f"{env['EXP_DIR']}/det_v2/s/{name}_s1", "action": action})


# --- placement rule ------------------------------------------------------------------------

@pytest.mark.parametrize("card, cls", [
    ("RTX 2080 Ti", "small"), ("a4000", "small"), ("Tesla V100-SXM2-16GB", "small"), ("volta", "small"),
    ("RTX 6000 Ada", "big"), ("RTX 4500 Ada", "big"), ("4500_ada", "big"), ("a100", "big"), ("L40S", "big"),
    ("a4500", "other"), ("a6000", "other"),
])
def test_card_classes(card, cls):
    assert rebalance.card_class(card) == cls


@pytest.mark.parametrize("name, card, ok", [
    (Q3D_FM, "RTX 6000 Ada", True), (Q3D_FM, "a100", True), (Q3D_FM, "L40S", True),
    (Q3D_FM, "RTX 2080 Ti", False), (Q3D_FM, "a4000", False), (Q3D_FM, "a4500", False),
    (Q2D_FM, "RTX 4500 Ada", True), (Q2D_FM, "RTX 2080 Ti", False),
    (CP_FM, "RTX 2080 Ti", True), (CP_FM, "RTX 6000 Ada", True), (CP_FM, "a4500", True),
    (CP_CLF, "RTX 2080 Ti", True), (CP_CLF, "a4000", True), (CP_CLF, "V100", True),
    (CP_CLF, "RTX 6000 Ada", False), (Q3D_BNNREG, "a100", False), (Q3D_BNNREG, "a4500", False),
    (PEN_PARTX, "RTX 2080 Ti", False), (PEN_PARTX, "RTX 6000 Ada", False),
])
def test_placement_rule(name, card, ok):
    assert rebalance.allowed(name, card) is ok


def test_each_card_takes_the_first_allowed_run_in_rank_order():
    pend = [{"name": Q3D_FM, "rank": 1}, {"name": CP_CLF, "rank": 2}, {"name": Q3D_BNNREG, "rank": 3}]
    cards = [("westeros", "2", "RTX 2080 Ti"), ("arrakis", "1", "RTX 6000 Ada"), ("westeros", "3", "RTX 2080 Ti")]
    plan = rebalance.plan_moves(pend, cards, 8)
    assert [(c["name"], card[0], card[1]) for c, card in plan] == [
        (CP_CLF, "westeros", "2"), (Q3D_FM, "arrakis", "1"), (Q3D_BNNREG, "westeros", "3")]
    assert len(rebalance.plan_moves(pend, cards, 1)) == 1
    assert rebalance.plan_moves([{"name": Q3D_FM, "rank": 1}], [("westeros", "2", "RTX 2080 Ti")], 8) == []


# --- card scan -----------------------------------------------------------------------------

SCAN = ("VIABLE\n  westeros      gpu2 (11GB idle),gpu3 (11GB idle)   host RAM 300G; note\n"
        "  arrakis       gpu1 (48GB idle)   host RAM 400G\n  perception    gpu0 (24GB idle)   host RAM 60G\n"
        "  ilab/rlab4    a4000 16GB, 2/8 free\nREJECTED\n  amarel  queued\n")


def test_free_cards_drop_held_hosts_and_cards_a_run_holds(env, monkeypatch):
    _row(env, "a", "direct", "111", "westeros:3")                  # alive: holds westeros 3
    _row(env, "b", "direct", "222", "arrakis:4")                   # dead pid: frees nothing it was not scanned for
    _row(env, "c", "direct", "-", "arrakis:1", action="claim")     # another account is starting there
    _row(env, "d", "direct", "333", "1spring-303-desktop-10:0", user="other")
    _row(env, "d", "ilab", "9", "a4000", user="other")             # moved away since

    def sh(cmd, timeout=300):
        if "gpu-free.sh --fit" in cmd:
            return SCAN
        if "westeros" in cmd and "ps -p" in cmd:
            return "  111\nDET_V2_OK\n"
        if "arrakis" in cmd and "ps -p" in cmd:
            return "DET_V2_OK\n"
        return ""
    monkeypatch.setattr(rebalance, "sh", sh)
    cards = rebalance.free_cards(16)
    assert cards == [("westeros", "2", "RTX 2080 Ti"), ("1spring-303-desktop-10", "0", "RTX 4500 Ada")]
    open(env["HOLD_HOSTS"], "w").write("westeros   # CPU-starved\n")
    env["SHARED_CARDS"] = "arrakis:3 westeros:3"
    assert rebalance.free_cards(16) == [("1spring-303-desktop-10", "0", "RTX 4500 Ada"), ("arrakis", "3", "RTX 6000 Ada")]


def test_an_unreachable_host_keeps_its_cards_busy(env, monkeypatch):
    _row(env, "a", "direct", "111", "arrakis:3")
    env["SHARED_CARDS"] = "arrakis:3"
    monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300: "")
    assert rebalance.occupied_direct() == {("arrakis", "3")}
    assert rebalance.free_cards(16) == []


# --- moves -----------------------------------------------------------------------------------

@pytest.fixture
def fake_sh(env, monkeypatch):
    calls, state = [], {"queue": "PENDING", "banked": "2", "sacct": "CANCELLED by 1"}

    def sh(cmd, timeout=300):
        calls.append(cmd)
        if "squeue -h -j" in cmd:
            return state["queue"] + "\n"
        if "artifacts_v2.json" in cmd:
            return state["banked"] + "\n"
        if "sacct -j" in cmd:
            return state["sacct"] + "\n"
        return "ok\n"
    monkeypatch.setattr(rebalance, "sh", sh)
    return calls, state


def _cell(env, where="amarel"):
    _row(env, Q3D_FM, where, "777", "gpu" if where == "amarel" else "a4000")
    return {**lib.latest_rows(env["LEDGER"])[Q3D_FM], "rank": 1}


def test_a_running_job_is_never_cancelled(env, fake_sh):
    calls, state = fake_sh
    state["queue"] = "RUNNING"
    res = rebalance.move(_cell(env), ("arrakis", "1", "RTX 6000 Ada"))
    assert res.startswith("skip") and not any("scancel" in c for c in calls)
    assert lib.latest_rows(env["LEDGER"])[Q3D_FM]["action"] == "launch"


def test_a_banked_pending_run_resumes_on_the_card(env, fake_sh):
    calls, _ = fake_sh
    res = rebalance.move(_cell(env), ("arrakis", "1", "RTX 6000 Ada"))
    assert any("ssh -n -o BatchMode=yes amarel 'scancel 777'" in c for c in calls)
    assert any(f"resume_direct.sh {Q3D_FM} arrakis 1" in c for c in calls)
    assert not any("launch.sh" in c for c in calls) and "2 banked" in res
    last = lib.latest_rows(env["LEDGER"])[Q3D_FM]
    assert last["action"] == "cancel" and last["job"] == "777" and last["user"] == "tester"


def test_an_unbanked_pending_run_launches_on_the_card_from_its_commit(env, fake_sh):
    calls, state = fake_sh
    state["banked"] = "0"
    c = _cell(env, where="ilab")
    rebalance.move(c, ("westeros", "4", "RTX 2080 Ti"))
    cmd = next(x for x in calls if "launch.sh" in x)
    assert "DET_V2_RELAUNCH=1" in cmd and "DET_V2_REF=abc1234567" in cmd and f"launch.sh {Q3D_FM} direct westeros 4" in cmd
    assert any("ssh -n -o BatchMode=yes ilab1 'scancel 777'" in x for x in calls)


def test_a_cancel_that_did_not_take_leaves_the_run_alone(env, fake_sh):
    calls, state = fake_sh
    state["sacct"] = "RUNNING"
    res = rebalance.move(_cell(env), ("arrakis", "1", "RTX 6000 Ada"))
    assert res.startswith("FAILED to cancel") and not any("resume_direct" in c or "launch.sh" in c for c in calls)


def test_only_this_accounts_pending_jobs_are_candidates(env, monkeypatch):
    _row(env, Q3D_FM, "amarel", "1", "gpu")
    _row(env, CP_CLF, "ilab", "2", "a4000", user="other")
    _row(env, Q3D_BNNREG, "ilab", "3", "a4000")
    monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300: "1\n2\n3\n" if "-t PD" in cmd else "")
    assert [c["name"] for c in rebalance.pending_runs()] == [Q3D_FM, Q3D_BNNREG]


# --- ended slots ------------------------------------------------------------------------------

def test_ended_slots_resume_relaunch_or_get_reported(env, monkeypatch, tmp_path):
    run = lambda n: f"{env['EXP_DIR']}/det_v2/s/{n}_s1"
    _row(env, Q3D_FM, "amarel", "10", "gpu")
    _row(env, CP_CLF, "ilab", "11", "a4000")
    _row(env, Q3D_BNNREG, "ilab", "12", "a4000")
    _row(env, PEN_PARTX, "ilabcpu", "13", "cpu")
    for e in range(4):
        os.makedirs(f"{run(CP_CLF)}/epoch_{e:03d}")
        open(f"{run(CP_CLF)}/epoch_{e:03d}/artifacts_v2.json", "w").write("{}")
    calls = []

    def sh(cmd, timeout=300):
        calls.append(cmd)
        if "sacct -X" in cmd and "amarel" in cmd:
            return "10|TIMEOUT\n"
        if "sacct -X" in cmd:
            return "11|PREEMPTED\n12|TIMEOUT\n13|TIMEOUT\n"
        if "artifacts_v2.json" in cmd:
            return "3\n"
        if "-t R -o %N" in cmd:
            return "rlab4\n"
        if "sinfo" in cmd:
            return "ilab1\nrlab3\nrlab4\n"
        return "ok\n"
    monkeypatch.setattr(rebalance, "sh", sh)
    assert all(l.startswith("would resume") for l in rebalance.resume_ended(False))
    calls.clear()
    out = rebalance.resume_ended(True)
    assert len(out) == 4
    assert any(f"resume_amarel.sh {Q3D_FM}" in c for c in calls)
    assert any(f"DET_V2_EXCLUDE=ilab1,ilab3,ilab4,rlab3 bash" in c and f"resume_ilab.sh {CP_CLF} 06:00:00 a4000" in c for c in calls)
    assert any("DET_V2_RELAUNCH=1" in c and f"launch.sh {Q3D_BNNREG} ilab a4000" in c for c in calls)
    assert any(f"launch.sh {PEN_PARTX} ilabcpu" in c and "RELAUNCH" not in c for c in calls)


def test_failed_jobs_are_reported_not_retried(env, monkeypatch):
    _row(env, CP_CLF, "ilab", "11", "a4000")
    monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300: "11|OUT_OF_MEMORY\n" if "sacct" in cmd else "")
    assert rebalance.resume_ended(True) == [f"NEEDS A LOOK {CP_CLF}: ilab job 11 OUT_OF_MEMORY"]


# --- launch loop guard ----------------------------------------------------------------------

def test_holds_off_while_a_launch_loop_appends(env, monkeypatch):
    monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300: "")
    assert rebalance.launch_loop_active() is None                     # no ledger yet
    _row(env, Q3D_FM, "ilab", "1", "a4000")
    assert "ledger changed" in rebalance.launch_loop_active()
    old = time.time() - 2 * rebalance.LEDGER_QUIET_S
    os.utime(env["LEDGER"], (old, old))
    assert rebalance.launch_loop_active() is None
    monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300:
                        "  4242 bash scripts/det_v2/launch.sh m01 direct westeros 3\n" if "ps -u" in cmd else "")
    assert "pid 4242" in rebalance.launch_loop_active()


def test_main_prints_the_plan_only_while_held(env, monkeypatch, capsys):
    _row(env, Q3D_FM, "amarel", "1", "gpu")
    monkeypatch.setattr(rebalance.lib, "load_user_env", lambda: dict(env))
    moved = []
    monkeypatch.setattr(rebalance, "move", lambda c, card: moved.append(c) or "moved")
    monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300:
                        "1\n" if "-t PD" in cmd else SCAN if "gpu-free" in cmd else "DET_V2_OK\n")
    monkeypatch.setattr(sys, "argv", ["rebalance.py", "--apply"])
    rebalance.main()
    out = capsys.readouterr().out
    assert "holding off" in out and f"would move: {Q3D_FM} -> arrakis gpu1" in out and not moved
