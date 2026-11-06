"""launch.sh and resume_*.sh refuse what would clobber or duplicate a run."""
import subprocess

from det_v2_world import GPU_RUN, PARTX_RUN, SMALL_RUN, git


def test_launch_refuses_a_run_with_banked_epochs(world):
    world.bank(world.run_dir(GPU_RUN), 1)
    r = world.run("launch.sh", GPU_RUN, "ilab", "a4000", check=False)
    assert r.returncode != 0 and "banked epoch" in r.stderr and "resume it instead" in r.stderr
    assert not (world.calls / "sbatch").exists() and world.ledger() == []


def test_launch_refuses_banked_epochs_on_amarel(world):
    arun = world.amarel / "exp" / world.run_dir(GPU_RUN).relative_to(world.exp)
    world.bank(arun, 1)
    r = world.run("launch.sh", GPU_RUN, "amarel", check=False)
    assert r.returncode != 0 and "banked epochs on Amarel" in r.stderr


def test_launch_refuses_a_name_already_claimed(world):
    world.run("launch.sh", SMALL_RUN, "ilab", "a4000")
    r = world.run("launch.sh", SMALL_RUN, "direct", "boxa", "1", check=False)
    assert r.returncode != 0 and "already in the ledger" in r.stderr
    # Nothing banked: the explicit relaunch override claims over it.
    world.run("launch.sh", SMALL_RUN, "direct", "boxa", "1", extra={"DET_V2_RELAUNCH": "1"})
    assert [row[9] for row in world.ledger()[1:]] == ["claim", "launch", "claim", "launch"]


def test_launch_refuses_while_the_last_placement_still_runs(world):
    world.ledger_row(SMALL_RUN, "ilab", "4242", "a4000", world.run_dir(SMALL_RUN))
    (world.calls / "squeue_out").write_text("RUNNING\n")
    r = world.run("launch.sh", SMALL_RUN, "ilab", "a4000", check=False, extra={"DET_V2_RELAUNCH": "1"})
    assert r.returncode != 0 and "still runs" in r.stderr


def test_a_failed_submit_releases_the_claim(world):
    (world.bin / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error' >&2\nexit 1\n")
    r = world.run("launch.sh", GPU_RUN, "ilab", "a4000", check=False)
    assert r.returncode != 0 and "sbatch failed" in r.stderr
    assert [row[9] for row in world.ledger()[1:]] == ["claim", "abandon"]
    (world.bin / "sbatch").unlink()
    from det_v2_world import STUBS, _exe
    _exe(world.bin / "sbatch", STUBS["sbatch"])
    world.run("launch.sh", GPU_RUN, "ilab", "a4000")            # an abandoned claim does not block


def test_launch_refuses_a_dirty_repo_and_wrong_worlds(world):
    (world.repo / "scripts" / "run_adaptive.py").write_text("# edited\n")
    r = world.run("launch.sh", GPU_RUN, "ilab", "a4000", check=False)
    assert r.returncode != 0 and "uncommitted changes" in r.stderr
    git(world.repo, "checkout", "--", "scripts/run_adaptive.py")
    assert "Part-X (CPU) row" in world.run("launch.sh", PARTX_RUN, "ilab", "a4000", check=False).stderr
    assert "needs a GPU" in world.run("launch.sh", GPU_RUN, "ilabcpu", check=False).stderr
    assert "no manifest row" in world.run("launch.sh", "m99_nope_fm", "ilab", "a4000", check=False).stderr


def test_launch_refuses_a_config_outside_the_det_v2_root(world):
    cfg = world.repo / "configs/adaptive_v2/experiment/det_v2/runs" / f"{GPU_RUN}.yaml"
    cfg.write_text("# @package _global_\noutput_dir: /tmp/elsewhere\n")
    git(world.repo, "commit", "-qam", "bad dir")
    r = world.run("launch.sh", GPU_RUN, "ilab", "a4000", check=False)
    assert r.returncode != 0 and "is not under" in r.stderr


def test_partx_reruns_its_launch_to_continue(world):
    world.run("launch.sh", PARTX_RUN, "ilabcpu")
    run_dir = world.run_dir(PARTX_RUN)
    world.bank(run_dir, 3)
    (run_dir / "partx_state.pkl").write_text("state")
    r = world.run("launch.sh", PARTX_RUN, "ilabcpu")
    assert "continues from" in r.stdout
    assert "Part-X row" in world.run("resume_ilab.sh", PARTX_RUN, check=False).stderr


def test_amarel_launch_refuses_unstaged_data(world):
    (world.amarel / "data" / "toy" / "splits" / "test_set.txt").unlink()
    r = world.run("launch.sh", GPU_RUN, "amarel", check=False)
    assert r.returncode != 0 and "not staged on Amarel" in r.stderr and "test_set.txt" in r.stderr
    assert world.ledger() == []


def test_amarel_launch_needs_the_commit_on_origin(world):
    (world.repo / "x.txt").write_text("x")
    git(world.repo, "add", "x.txt")
    git(world.repo, "commit", "-qm", "local only")
    r = world.run("launch.sh", GPU_RUN, "amarel", check=False)
    assert r.returncode != 0 and "not on origin" in r.stderr


def test_resume_refusals(world):
    run_dir = world.run_dir(GPU_RUN)
    assert "no ledger row" in world.run("resume_ilab.sh", GPU_RUN, check=False).stderr
    world.ledger_row(GPU_RUN, "ilab", "77", "a4000", run_dir)
    assert "no banked epoch" in world.run("resume_ilab.sh", GPU_RUN, check=False).stderr
    world.bank(run_dir, 2)
    (world.calls / "squeue_out").write_text("PENDING\n")
    assert "still runs" in world.run("resume_direct.sh", GPU_RUN, "boxa", "0", check=False).stderr
    (world.calls / "squeue_out").write_text("CANCELLED\n")       # still listed, but final
    world.run("resume_ilab.sh", GPU_RUN)                         # gres from the last iLab row
    sb = (world.calls / "sbatch").read_text().splitlines()[-1]
    assert sb.startswith(GPU_RUN)
    assert "--gres=gpu:a4000:1" in (world.calls / "ssh").read_text().splitlines()[-1]


def test_resume_fails_closed_when_the_host_cannot_be_asked(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(run_dir, 2)
    world.ledger_row(GPU_RUN, "ilab", "77", "a4000", run_dir)
    (world.bin / "ssh").write_text("#!/bin/bash\nexit 255\n")
    r = world.run("resume_direct.sh", GPU_RUN, "boxa", "0", check=False)
    assert r.returncode != 0 and "could not reach" in r.stderr


def test_resume_from_amarel_needs_the_account_that_ran_it(world):
    run_dir = world.run_dir(GPU_RUN)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir, user="someoneelse")
    r = world.run("resume_ilab.sh", GPU_RUN, "1:00:00", "a4000", check=False)
    assert r.returncode != 0 and "someoneelse's Amarel scratch" in r.stderr
    assert world.ledger()[-1][9] == "abandon"


def test_two_launches_racing_for_one_name_claim_it_once(world):
    procs = [subprocess.Popen(["bash", str(world.repo / "scripts/det_v2/launch.sh"), SMALL_RUN, "ilab", "a4000"],
                              env=world.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(3)]
    codes = [p.wait(timeout=300) for p in procs]
    assert sorted(codes) == [0, 1, 1]
    assert [row[9] for row in world.ledger()[1:]] == ["claim", "launch"]


def test_amarel_constraint_keeps_q3d_q2d_fm_on_ada_nodes():
    import subprocess
    from pathlib import Path
    common = Path(__file__).resolve().parents[2] / "scripts/det_v2/common.sh"
    def feat(name, override=None):
        env = "" if override is None else f"DET_V2_CONSTRAINT={override} "
        out = subprocess.run(["bash", "-c", f'eval "$(sed -n "/^amarel_constraint()/,/^}}/p" {common})"; {env}amarel_constraint {name}'],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    assert feat("m16_q3dppo_fm_qa") == "adalovelace"
    assert feat("x24_q3dlqr_fm_qa_a2jver_w001") == "adalovelace"
    assert feat("m27_q2dppo_fm_qa") == "adalovelace"
    assert feat("m15_q3dppo_bnnreg_qa") == ""
    assert feat("m38_cplqr_fm_qa") == ""            # cp and pen FM take any card
    assert feat("m16_q3dppo_fm_qa", override="ampere") == "ampere"
