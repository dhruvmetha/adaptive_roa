"""Members must stay inside the parent's CUDA_VISIBLE_DEVICES restriction.

The child used to write an ABSOLUTE index back into CUDA_VISIBLE_DEVICES, which
discards whatever the parent was given. Under SLURM that is invisible -- cgroup
device isolation makes the allocation's cards 0..n-1 inside the job, so absolute
indices are accidentally correct. On a direct box it is not: launching five arms
pinned to GPUs 0..4 on arrakis put all 25 members on physical GPU 0 at 99%
utilisation while four cards sat idle.
"""
import os
from unittest import mock

from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (
    _device_for_member, visible_devices,
)


def test_members_round_robin_over_the_parents_own_ids():
    devices = ["2", "5"]          # parent was given physical 2 and 5
    got = [_device_for_member(r, devices) for r in range(5)]
    assert got == ["2", "5", "2", "5", "2"], "must cycle the parent's ids, not 0..n"


def test_a_single_inherited_device_pins_every_member_to_it():
    """The arrakis case: one card per arm, five members sharing it."""
    assert [_device_for_member(r, ["3"]) for r in range(5)] == ["3"] * 5


def test_no_devices_returns_the_cpu_sentinel_not_a_crash():
    assert _device_for_member(0, []) == ""      # rank % 0 would raise


def test_visible_devices_prefers_the_inherited_restriction():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=False):
        assert visible_devices() == ["4"]
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3,6"}, clear=False):
        assert visible_devices() == ["1", "3", "6"]


def test_empty_restriction_means_no_gpus_not_all_gpus():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
        # CUDA_VISIBLE_DEVICES="" hides every device; falling back to a device
        # count here would hand members GPUs the parent was explicitly denied.
        assert _device_for_member(0, visible_devices()) in ("", "0")
