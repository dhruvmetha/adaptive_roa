"""rebalance.move never strands a queued resume (dr_q3d60_fm_direct, 2026-10-30 23:0x).

The move cancelled a PENDING Amarel resume, then launch.sh refused the cell for its
banked epochs, leaving it with no job. A cell banked on Amarel now resumes on a direct
or iLab card through resume_direct.sh or resume_ilab.sh, never from 0.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "det_rerun"))
import rebalance  # noqa: E402

EXP = rebalance.EXP
NAME = "dr_q3d60_fm_direct_s99512"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "manifest.tsv").write_text(
        "name\tsystem\tarm\tmode\tseed\tmem\tpriority\toverrides\tscript\n"
        f"{NAME}\tq3d60\tfm_det\tdirect\t99512\t20G\t1\tseed=99512 output_dir={EXP}/q3d60/{NAME}\tscripts/run_adaptive.py\n")
    (tmp_path / "launched.tsv").write_text(f"{NAME}\tamarel\t--partition=gpu\t111\tts\tlog\t907fefc\n")
    monkeypatch.setattr(rebalance, "HERE", tmp_path)
    calls = []

    def sh(cmd, timeout=300):
        calls.append(cmd)
        if "squeue -h -j" in cmd:
            return "PENDING\n"
        if "artifacts_v2.json" in cmd:
            return f"{fake.banked}\n"
        if "sacct" in cmd:
            return "CANCELLED by 1\n"
        return "ok\n"

    monkeypatch.setattr(rebalance, "sh", sh)
    fake.calls = calls
    return fake


def _cell():
    return {"name": NAME, "jid": "111", "commit": "907fefc", "hours": 12}


def test_banked_cell_resumes_on_a_direct_card(fake):
    fake.banked = 2
    res = rebalance.move(_cell(), ("direct", "westeros", "1"))
    assert any("scancel 111" in c for c in fake.calls)
    assert any("resume_direct.sh dr_q3d60_fm_direct_s99512 westeros 1" in c for c in fake.calls)
    assert not any("launch.sh" in c for c in fake.calls)
    assert "2 banked" in res


def test_banked_cell_resumes_on_an_ilab_card(fake):
    fake.banked = 2
    rebalance.move(_cell(), ("ilab", "a4000", ""))
    assert any("resume_ilab.sh dr_q3d60_fm_direct_s99512 3-00:00:00 a4000" in c for c in fake.calls)
    assert not any("launch.sh" in c for c in fake.calls)


def test_fresh_cell_launches(fake):
    fake.banked = 0
    rebalance.move(_cell(), ("direct", "westeros", "1"))
    assert any("launch.sh dr_q3d60_fm_direct_s99512 direct westeros 1" in c for c in fake.calls)
    assert not any("resume_direct.sh" in c for c in fake.calls)


def _c(name, mem):
    return {"name": name, "mem": f"{mem}G"}


def test_ilab_cards_share_the_nodes_free_memory():
    # rlab4 2026-10-30 23:3x: 4 cards listed free, 80G unreserved; a 48G cell after
    # 12+12+8 queued on iLab with AssocGrpNodeLimit instead of starting.
    pend = [_c("a", 12), _c("b", 12), _c("c", 8), _c("big", 48), _c("d", 8)]
    plan = rebalance.plan_moves(pend, [("ilab", "a4000", "")] * 4, 40, 8)
    assert [c["name"] for c, _ in plan] == ["a", "b", "c", "d"]   # big would not fit the 8G left


def test_direct_cards_take_cells_in_order_and_max_moves_caps():
    pend = [_c("big", 48), _c("a", 12), _c("b", 8)]
    plan = rebalance.plan_moves(pend, [("direct", "westeros", "1"), ("direct", "westeros", "2")], 0, 1)
    assert [(c["name"], card[2]) for c, card in plan] == [("big", "1")]


def test_nice_puts_fm_first_then_longest():
    n = rebalance.nice_for
    fm, q3d110_bnnreg, q3d30_mlpreg, cp_mlpclf = (
        n("dr_cp_fm_direct_s99512", 1.2), n("dr_q3d110_bnnreg_direct_s99512", 30),
        n("dr_q3d30_mlpreg_direct_s99512", 5), n("dr_cp_mlpclf_direct_s99512", 0.4))
    assert fm < q3d110_bnnreg < q3d30_mlpreg < cp_mlpclf


def test_ilab_runs_keep_their_card_for_the_whole_run(fake):
    fake.banked = 0
    rebalance.move(_cell(), ("ilab", "a4000", ""))
    assert any("launch.sh dr_q3d60_fm_direct_s99512 ilab a4000 3-00:00:00" in c for c in fake.calls)
    assert rebalance.wall_for("amarel", "dr_q3d60_fm_direct_s99512") == "04:00:00"


def test_free_ilab_cards_discount_our_own_pending_jobs(monkeypatch):
    scan = ("VIABLE\n  ilab/rlab4    a4000 16GB, 2/8 free, 30G host; quota ...; --gres=gpu:a4000:1\n"
            "  westeros      gpu4 (11GB idle)   host RAM 399G\nREJECTED\n")
    def sh(cmd, timeout=300):
        if "gpu-free" in cmd:
            return scan
        if "-t R -o %N" in cmd:
            return "rlab4\n"
        if "-t PD -o %m" in cmd:
            return "12G\n"
        if "--test-only" in cmd:
            import datetime as dt
            return f"Job 1 to start at {dt.datetime.now().isoformat(timespec='seconds')} a using 2 processors on nodes rlab4"
        return ""
    monkeypatch.setattr(rebalance, "sh", sh)
    monkeypatch.setattr(rebalance, "shared_free", lambda: [])
    cards, mem = rebalance.free_cards(8)
    assert cards == [("direct", "westeros", "4"), ("ilab", "a4000", "")] and mem == 18


def test_ilab_card_used_only_when_slurm_would_start_it_now(monkeypatch):
    # rlab4 2026-10-31 08:2x: scan listed 1 card and 14G free, but a 4G test job was
    # predicted to start on rlab3 on Nov 4.
    import datetime as dt
    now = dt.datetime(2026, 10, 31, 8, 25)
    for txt, ok in [("Job 9 to start at 2026-11-04T11:32:49 a using 2 processors on nodes rlab3 in partition unlimited", False),
                    ("Job 9 to start at 2026-10-31T08:25:30 a using 2 processors on nodes rlab4 in partition unlimited", True),
                    ("Job 9 to start at 2026-10-31T08:25:30 a using 2 processors on nodes rlab3 in partition unlimited", False),
                    ("allocation failure", False)]:
        monkeypatch.setattr(rebalance, "sh", lambda cmd, timeout=300, t=txt: t)
        assert rebalance.ilab_starts_now("a4000", 12, "rlab4", now=now) is ok, txt


def test_cells_pending_on_ilab_go_only_to_direct_cards():
    pend = [{"name": "ilab_cell", "mem": "12G", "world": "ilab"}, {"name": "amarel_cell", "mem": "12G"}]
    plan = rebalance.plan_moves(pend, [("ilab", "a4000", ""), ("direct", "westeros", "4")], 40, 8)
    assert [(c["name"], card[0]) for c, card in plan] == [("amarel_cell", "ilab"), ("ilab_cell", "direct")]


def test_a_skipped_move_hands_its_card_to_the_next_candidate(monkeypatch):
    # 2026-10-31 12:3x: q3d110 mlpreg uniform started on Amarel between the plan and the
    # move, and arrakis gpu4 idled for a cycle.
    cands = [{"name": n, "mem": "20G"} for n in ("started_meanwhile", "next", "after")]
    calls = []

    def fake_move(c, card):
        calls.append((c["name"], card[2]))
        return f"skip {c['name']}: no longer PENDING" if c["name"] == "started_meanwhile" else f"moved {c['name']}"

    monkeypatch.setattr(rebalance, "move", fake_move)
    plan = rebalance.plan_moves(cands, [("direct", "arrakis", "4")], 0, 8)
    done = rebalance.execute(plan, cands, 0)
    assert calls == [("started_meanwhile", "4"), ("next", "4")] and [c["name"] for c in done] == ["next"]
