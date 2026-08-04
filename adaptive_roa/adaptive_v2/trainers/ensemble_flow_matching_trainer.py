"""Deep ensemble of flow matchers, trained one member per process in parallel.

Members are statistically independent, so training them is embarrassingly
parallel: M processes, no communication. Member m is round-robined onto
device `m % n_visible_gpus` rather than pinned to device m, because M can
exceed the GPU count available on a single node (e.g. M=5 members on a
4-GPU node): flow-matching models are small and the cards have plenty of
headroom, so doubling a member up on an already-used card costs wall-clock
on that card, not correctness. When exactly one member lands per GPU,
wall-clock equals a SINGLE member; the sequential loop used for the
classifier ensemble would be ~M x 50h per arm here, which is why this
exists.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import FlowMatchingTrainer


class EnsembleFlowMatcherHandle:
    """M flow matchers behind one object.

    `predict_endpoint` round-robins members EXACTLY rather than sampling one at
    random. EnsemblePosterior.predictive_logit_samples documents why: a seeded
    generator produced fixed weights [.125, .281, .109, .234, .250] against an
    exact .2, "enough to flip decisions near lambda*". A biased marginal is a
    systematic error, not noise that averages out.

    `predict_endpoint_member` forwards `x` to the member's own `predict_endpoint`
    UNCHANGED -- it does not embed or normalize. Every concrete flow matcher's
    `predict_endpoint` (adaptive_roa/flow_matching/base/flow_matcher.py:746)
    accepts RAW states and does normalization + embedding internally via
    `_prepare_model_inputs`; the existing single-model caller
    (adaptive_v2/eval/full_roa.py:740) passes raw batch tensors straight through
    for the same reason. Reaching past that and pre-embedding here would double
    -embed and crash with a shape mismatch -- the exact bug Task 2 had on the
    classifier path, where the ensemble backend skipped
    `system.embed_state_for_model(system.normalize_state(x))` because it called
    a raw net-forward instead of the module's own `predict_endpoint`.
    """

    def __init__(self, members: list):
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"EnsembleFlowMatcherHandle needs at least 2 members, got {len(members)}")
        self.members = members
        self._cursor = 0

    @property
    def n_members(self) -> int:
        return len(self.members)

    def predict_endpoint_member(self, m: int, x: torch.Tensor, **kw) -> torch.Tensor:
        if not 0 <= m < len(self.members):
            raise IndexError(f"member {m} out of range for {len(self.members)} members")
        return self.members[m].predict_endpoint(x, **kw)

    def predict_endpoint(self, x: torch.Tensor, **kw) -> torch.Tensor:
        m = self._cursor % len(self.members)
        self._cursor += 1
        return self.predict_endpoint_member(m, x, **kw)

    def eval(self):
        for mem in self.members:
            if hasattr(mem, "eval"):
                mem.eval()
        return self

    def to(self, device):
        """Required by AdaptiveEngine.run: `model_handle.to(self.device)` is
        called unconditionally right after `.eval()` for every predictor type
        (adaptive_v2/engine.py:137-138). Without this, every real epoch would
        crash with AttributeError before the first acquisition call.
        """
        for mem in self.members:
            if hasattr(mem, "to"):
                mem.to(device)
        return self

    # ------------------------------------------------------------------
    # Manifold-aware error reporting passthroughs.
    #
    # adaptive/endpoint_evaluation.py:100,116 (compute_endpoint_prediction_error,
    # called unconditionally from engine.py:179 for every non-classifier,
    # non-smoke, eval epoch) calls these two methods on the handle with no
    # hasattr guard. eval/full_roa.py:760,782,786 additionally checks
    # hasattr(flow_matcher, "distance_manifold") to decide whether to use
    # geodesic (circular-aware) distance or fall back to raw Euclidean --
    # skipping this attribute wouldn't crash there, but it would silently
    # degrade pendulum/cartpole error reporting to a wrong metric (raw
    # Euclidean over an angle that wraps at +-pi). All members share the same
    # system and were built from the same manifold structure, so delegating to
    # member 0 is exact, not an approximation.
    # ------------------------------------------------------------------

    @property
    def distance_manifold(self):
        return self.members[0].distance_manifold

    def get_manifold_component_names(self) -> list:
        return self.members[0].get_manifold_component_names()

    def compute_manifold_distance_per_component(
        self, predicted_endpoints: torch.Tensor, true_endpoints: torch.Tensor
    ) -> torch.Tensor:
        return self.members[0].compute_manifold_distance_per_component(
            predicted_endpoints, true_endpoints
        )


_CPU_SENTINEL = ""


def visible_devices() -> list[str]:
    """The device ids this process may use, as CUDA_VISIBLE_DEVICES understands them.

    Returns the INHERITED id strings when CUDA_VISIBLE_DEVICES is already set, not
    a 0..n-1 range. That distinction is the whole point: a child that rewrites the
    variable with an absolute index discards its parent's restriction and lands on
    physical device 0 regardless of what the parent was given.

    Under SLURM this is masked -- cgroup device isolation means the allocation's
    cards ARE 0..n-1 inside the job, so absolute indices happen to be right. On a
    direct box with no cgroup (arrakis), launching five arms with
    CUDA_VISIBLE_DEVICES=0..4 put all 25 members on physical GPU 0 at 99% while the
    other four cards sat idle.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and raw.strip() != "":
        return [d.strip() for d in raw.split(",") if d.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def _device_for_member(rank: int, devices: list[str]) -> str:
    """Map ensemble member `rank` onto one of the parent's visible devices.

    Members round-robin over the parent's OWN device ids: `devices[rank % len]`.
    With 5 members and 4 devices, members 0-3 each get a card and member 4 wraps
    onto the first -- balanced (no device gets more than ceil(M / n) members)
    rather than piling every overflow member onto one card.

    `devices` must be captured by the PARENT before any child narrows its own
    CUDA_VISIBLE_DEVICES. Re-deriving it inside a child, after this function's
    return value has been written to the environment, would see exactly one device
    and silently pin every later member to it.

    An empty list (CPU-only machine, or no GPUs visible) returns the CPU sentinel
    "" rather than raising ZeroDivisionError on `rank % 0`; CUDA_VISIBLE_DEVICES=""
    hides all devices, and FlowMatchingTrainer already falls back to the "cpu"
    accelerator when torch.cuda.is_available() is False.
    """
    # Accept an int as well as a list. Long-running parents started before this
    # signature changed still pass a device COUNT through the mp.spawn args tuple,
    # and every child re-imports this module fresh from disk -- so a list-only
    # signature makes those parents crash at their next epoch boundary with
    # "object of type 'int' has no len()". That killed fm_high_total 13.5 hours in.
    # Never let an edit here break a process already running against the old shape.
    if isinstance(devices, int):
        devices = [str(i) for i in range(max(devices, 0))]
    if not devices:
        return _CPU_SENTINEL
    return devices[rank % len(devices)]


def _train_one_member(rank: int, cfg_blob, system_name: str, dataset_files, out_dir,
                       resume, seed_base, devices: list[str]):
    """Child process: train member `rank` on its assigned device.

    `system_name` is passed explicitly and `system` is rebuilt from `cfg_blob`
    here rather than pickled across the process boundary: FlowMatchingTrainer
    requires a real `system_name` string (it looks it up in `_DATAMODULES` and
    raises otherwise -- see flow_matching_trainer.py:43-44), and every field the
    flow matcher needs from `system` (bounds, manifold structure, embedding) is
    fully determined by `cfg.system`, so reconstructing it with
    `hydra.utils.instantiate` is both correct and mirrors exactly what
    AdaptiveEngine.__init__ does for the parent process (engine.py:68).

    `devices` is captured by the PARENT (EnsembleFlowMatchingTrainer.fit, via
    visible_devices() before any child narrows its own CUDA_VISIBLE_DEVICES) and
    threaded through the mp.spawn args tuple -- see _device_for_member's docstring
    for why re-deriving it here would be wrong.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = _device_for_member(rank, devices)
    torch.manual_seed(seed_base + rank)
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_blob)
    # Each child sees at most one GPU (or none, on a CPU-only machine), so it
    # must address its device as device 0.
    OmegaConf.update(cfg, "predictor.lightning_trainer.devices", 1, force_add=True)
    system = hydra.utils.instantiate(cfg.system)
    trainer = FlowMatchingTrainer(cfg, system, system_name)
    member_dir = Path(out_dir) / f"member_{rank}"
    member_dir.mkdir(parents=True, exist_ok=True)
    trainer.fit(dataset_files, str(member_dir), resume_checkpoint=resume)


class EnsembleFlowMatchingTrainer:
    """Trains M flow matchers concurrently and assembles them into one handle."""

    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name
        self.n_members = int(cfg.predictor.ensemble.n_members)
        self.seed_base = int(cfg.get("seed", 42))

    def fit(self, dataset_files: dict, output_dir: str,
            resume_checkpoint: str | None = None):
        from omegaconf import OmegaConf
        blob = OmegaConf.to_container(self.cfg, resolve=True)
        # Capture the visible device IDS here, in the parent, before any child
        # narrows its own CUDA_VISIBLE_DEVICES. Ids, not a count: a child that
        # writes back an absolute index discards the parent's restriction.
        devices = visible_devices()
        mp.spawn(
            _train_one_member,
            args=(blob, self.system_name, dataset_files, output_dir,
                  resume_checkpoint, self.seed_base, devices),
            nprocs=self.n_members, join=True,
        )
        members = [self._load_member(output_dir, m) for m in range(self.n_members)]
        return EnsembleFlowMatcherHandle(members)

    def _load_member(self, output_dir: str, m: int):
        """Reload member m from disk.

        The child process already reloaded its own best checkpoint, but that
        object cannot cross the process boundary, so the parent reloads from
        disk using the same path FlowMatchingTrainer.fit uses: glob best*.ckpt
        and let Lightning reconstruct from the saved hyperparameters.
        """
        import glob

        from hydra.utils import get_class

        member_dir = Path(output_dir) / f"member_{m}"
        ckpts = sorted(glob.glob(str(member_dir / "**" / "best*.ckpt"), recursive=True))
        if not ckpts:
            raise FileNotFoundError(
                f"no best*.ckpt under {member_dir}; member {m} did not finish training. "
                "Assembling an ensemble from a partially-trained member would report a "
                "silently wrong epistemic estimate."
            )
        cls = get_class(self.cfg.flow_matcher._target_)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return cls.load_from_checkpoint(ckpts[0], device=device)
        except Exception as exc:  # mirrors the fallback in FlowMatchingTrainer.fit
            print(f"Warning: load_from_checkpoint failed for member {m} ({exc}); "
                  "loading state dict directly")
            import hydra
            from omegaconf import OmegaConf

            # Reuse FlowMatchingTrainer's own cfg-navigation (predictor.flow_matching
            # vs. top-level flow_matching, local vs. global) instead of duplicating
            # it, so this fallback cannot silently drift from the primary trainer.
            inner = FlowMatchingTrainer(self.cfg, self.system, self.system_name)
            flow_matching = inner._flow_matching_cfg

            model = hydra.utils.instantiate(self.cfg.model)
            flow_matcher_kwargs = {
                "system": self.system,
                "model": model,
                "optimizer": self.cfg.optimizer,
                "scheduler": self.cfg.scheduler,
                "model_config": OmegaConf.to_container(self.cfg.model, resolve=True),
                "latent_dim": flow_matching.latent_dim,
                "mae_val_frequency": flow_matching.mae_val_frequency,
                "use_loss_weights": flow_matching.get("use_loss_weights", False),
                "use_manifold": flow_matching.get("use_manifold", True),
                "use_log_loss_weights": flow_matching.get("use_log_loss_weights", False),
                "clamp_noise": flow_matching.get("clamp_noise", True),
                "zero_latent": flow_matching.get("zero_latent", False),
                "noise_scale": flow_matching.get("noise_scale", 1.0),
                "val_error_log_file": None,
                "_recursive_": False,
            }
            if self.system_name == "quadrotor3d":
                flow_matcher_kwargs["quat_loss_weight"] = flow_matching.get("quat_loss_weight", 1.0)
            if inner.is_local:
                flow_matcher_kwargs["sequence_length"] = flow_matching.get("sequence_length", 32)
                flow_matcher_kwargs["history_length"] = flow_matching.get("history_length", 1)

            flow_matcher = hydra.utils.instantiate(self.cfg.flow_matcher, **flow_matcher_kwargs)
            ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
            model_state_dict = {k.replace("model.", ""): v
                                 for k, v in ckpt["state_dict"].items()
                                 if k.startswith("model.")}
            flow_matcher.model.load_state_dict(model_state_dict)
            return flow_matcher
