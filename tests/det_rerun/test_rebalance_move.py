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
    assert any("resume_ilab.sh dr_q3d60_fm_direct_s99512 04:00:00 a4000" in c for c in fake.calls)
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
