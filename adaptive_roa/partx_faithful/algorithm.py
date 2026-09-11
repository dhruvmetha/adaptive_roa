"""Alg. 4 of Pedrielli et al. (Part-X) as a resumable state machine (spec §3, §6.4, §6.6).

The flow follows cpslab's singlereplication.py. The root gets SampleBO and a
classification. Each later iteration either branches every remaining region
(when the budget covers the n0 top-up plus nBO for every child) and then runs
continued sampling on classified regions, or runs the final volume-proportional
phase.

The budget counts NEW evaluations. The initial samples are shared with every
other arm and are free. `log` lists new sample ids in canonical order
(ascending region id within each phase). Every random draw comes from a
generator seeded by (seed, iteration, region id, stage), and every fit is
seeded by (seed, region id, number of samples), so a resumed run replays
exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_roa.partx_faithful.bo import sample_bo
from adaptive_roa.partx_faithful.mcstep import classify, falsification_indicator, mc_step
from adaptive_roa.partx_faithful.region import CLASSIFIED, REMAINING, Region, child_masks
from adaptive_roa.partx_faithful.sampling import SampleStore, lhs


@dataclass
class PartXConfig:
    n0: int
    n_bo: int = 10
    n_c: int = 100
    R: int = 20
    M: int = 500
    B: int = 2
    delta_c: float = 0.05
    delta_v: float = 0.001
    ei_cand_cap: int = 20000
    ei_f_star: str = "plugin"
    compat_ci_var: bool = False
    compat_eq4_times_volume: bool = False
    seed: int = 42


def derive_seed(*parts) -> int:
    return int(np.random.SeedSequence([int(p) % (2**63) for p in parts]).generate_state(1)[0])


def multinomial_counts(weights, total, rng) -> np.ndarray:
    """cpslab assign_budgets, drawn from a seeded generator (spec §4 row 7)."""
    w = np.asarray(weights, dtype=float)
    if total <= 0 or len(w) == 0 or w.sum() <= 0.0:
        return np.zeros(len(w), dtype=int)
    return rng.multinomial(int(total), w / w.sum())


_UNCLASSIFIED, _INDICATOR, _CONTINUED, _FINAL = 1, 2, 3, 4
_ITERATION = -1          # region-id slot for draws that belong to the whole iteration


class PartX:
    def __init__(self, cfg: PartXConfig, low, high, sampler, model_factory, budget: int,
                 init_X=None, init_y=None, init_pool_idx=None):
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        self.cfg, self.d, self.budget, self.used = cfg, len(low), int(budget), 0
        self.sampler, self.model_factory = sampler, model_factory
        self.on_sample = None
        self.model_cache: dict = {}
        self.store = SampleStore(self.d)
        root = Region(rid=0, parent=-1, low=low, high=high, branch_dir=0, depth=0)
        sampler.init_root(root)
        if init_X is not None and len(init_X):
            root.samples = self.store.add(init_X, init_y, init_pool_idx)
        self.n_init = len(root.samples)
        if self.budget < max(cfg.n0 - self.n_init, 0) + cfg.n_bo:
            raise ValueError("budget must cover the root's top-up and nBO "
                             "(cpslab singlereplication.py:58-60)")
        self.regions = {0: root}
        self.next_rid = 1
        self.root_volume = root.volume()
        self.min_volume = cfg.delta_v ** self.d * self.root_volume
        self.perm = np.random.default_rng(cfg.seed).permutation(self.d)
        self.remaining, self.classified, self.unclassified, self.infeasible = [], [], [], []
        self.log: list = []
        self.iteration, self.phase, self.done = 0, "root", False
        self.shortfall = self.n_fits = self.n_capped = self.n_at_bound = 0

    # ---- bookkeeping ------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(sampler=None, model_factory=None, on_sample=None, model_cache={})
        return state

    def attach(self, sampler, model_factory) -> None:
        """Reattach after unpickling. Rebuilds the sampler's used-mask from the store."""
        self.sampler, self.model_factory, self.model_cache = sampler, model_factory, {}
        pool = self.store.pool_idx(np.arange(self.store.n))
        sampler.mark_used(pool[pool >= 0])

    def leaves(self) -> list:
        return [self.regions[k] for k in sorted(self.regions) if not self.regions[k].children]

    def _rng(self, rid: int, stage: int):
        return np.random.default_rng(derive_seed(self.cfg.seed, self.iteration, rid, stage))

    def _take(self, region, handles) -> list:
        if len(handles) == 0:
            return []
        ids = self.store.add(*self.sampler.take(handles))
        region.samples.extend(ids)
        self.log.extend(ids)
        self.used += len(ids)
        if self.on_sample is not None:
            self.on_sample(self)
        return ids

    def fit(self, region):
        n = len(region.samples)
        cached = self.model_cache.get(region.rid)
        if cached is not None and cached[0] == n:
            return cached[1]
        model = self.model_factory(derive_seed(self.cfg.seed, region.rid, n))
        model.fit(self.store.X(region.samples), self.store.y(region.samples))
        self.n_fits += 1
        self.n_capped += int(bool(getattr(model, "capped", False)))
        self.n_at_bound += int(bool(getattr(model, "at_bound", False)))
        self.model_cache[region.rid] = (n, model)
        return model

    def _classify(self, region, rng) -> None:
        q = mc_step(self.fit(region), region.low, region.high, self.cfg.R, self.cfg.M,
                    self.cfg.delta_c, rng)
        region.rtype = classify(region.rtype, region.volume(), self.min_volume, q,
                                self.cfg.delta_c, self.cfg.compat_ci_var)

    def _file(self, region, remaining: list, classified: list) -> None:
        if region.rtype in REMAINING:
            remaining.append(region.rid)
        elif region.rtype in CLASSIFIED:
            classified.append(region.rid)
        elif region.rtype == "u":
            self.unclassified.append(region.rid)
        elif region.rtype == "i":
            self.infeasible.append(region.rid)
        else:
            raise ValueError(f"unknown region type {region.rtype!r}")

    def _sample_unclassified(self, region) -> None:
        """cpslab samples_management_unclassified: SampleBO, then MCstep and Classify."""
        rng = self._rng(region.rid, _UNCLASSIFIED)
        res = sample_bo(region, self.store, self.sampler, self.fit,
                        lambda h: self._take(region, h), self.cfg.n0, self.cfg.n_bo,
                        self.cfg.ei_cand_cap, rng, self.cfg.ei_f_star)
        self.shortfall += res["shortfall"]
        if res["infeasible"] or not region.samples:
            region.rtype = "i"
            return
        self._classify(region, rng)

    def _sample_more(self, region, count: int, rng) -> None:
        """cpslab samples_management_classified: `count` LHS points, then reclassify."""
        if self.sampler.n_available(region) == 0:
            self.shortfall += count
            region.rtype = "i"
            return
        handles = self.sampler.snap_design(lhs(count, region.low, region.high, rng), region)
        self.shortfall += count - len(handles)
        self._take(region, handles)
        self._classify(region, rng)

    # ---- Alg. 4 -------------------------------------------------------------
    def run_root(self) -> None:
        if self.phase != "root":
            raise RuntimeError("run_root was already called")
        root = self.regions[0]
        self._sample_unclassified(root)
        self._file(root, self.remaining, self.classified)
        self.phase = "main"

    def step(self) -> None:
        if self.done:
            return
        if self.phase == "root":
            raise RuntimeError("call run_root first")
        left = self.budget - self.used
        if left <= 0 or not (self.remaining or self.classified):
            self.done = True
            return
        self.iteration += 1
        cfg = self.cfg
        plan, need = [], 0
        for rid in sorted(self.remaining):
            parent = self.regions[rid]
            dim = int(self.perm[parent.branch_dir % self.d])
            bounds = parent.split_bounds(dim, cfg.B)
            masks = child_masks(self.store.X(parent.samples)[:, dim], bounds, dim)
            parts = [[s for s, keep in zip(parent.samples, m) if keep] for m in masks]
            need += sum(max(cfg.n0 - len(p), 0) + cfg.n_bo for p in parts)
            plan.append((parent, dim, bounds, parts))
        if self.remaining and need <= left:
            self._branch(plan)
            self._continued_sampling()
        else:
            self._final_phase()

    def _branch(self, plan) -> None:
        self.phase = "branch"
        children = []
        for parent, dim, bounds, parts in plan:
            kids = []
            for (lo, hi), ids in zip(bounds, parts):
                kid = Region(rid=self.next_rid, parent=parent.rid, low=lo, high=hi,
                             branch_dir=parent.branch_dir + 1, depth=parent.depth + 1,
                             rtype="r", samples=list(ids))
                self.regions[kid.rid] = kid
                self.next_rid += 1
                kids.append(kid)
            parent.children = [k.rid for k in kids]
            self.sampler.split(parent, kids, dim)
            children.extend(kids)
        remaining = []
        for kid in children:                       # ascending rid (spec §6.4)
            self._sample_unclassified(kid)
            self._file(kid, remaining, self.classified)
        self.remaining = remaining

    def _continued_sampling(self) -> None:
        if not self.classified:
            return
        self.phase = "continued"
        cfg = self.cfg
        rids = sorted(self.classified)
        weights = []
        for rid in rids:
            r = self.regions[rid]
            weights.append(falsification_indicator(self.fit(r), r.low, r.high, cfg.R, cfg.M,
                                                   self._rng(rid, _INDICATOR),
                                                   times_volume=cfg.compat_eq4_times_volume))
        counts = multinomial_counts(weights, min(cfg.n_c, self.budget - self.used),
                                    self._rng(_ITERATION, _CONTINUED))
        classified = []
        for rid, cnt in zip(rids, counts):
            r = self.regions[rid]
            if cnt > 0:
                self._sample_more(r, int(cnt), self._rng(rid, _CONTINUED))
            self._file(r, self.remaining, classified)
        self.classified = classified

    def _final_phase(self) -> None:
        self.phase = "final"
        rids = sorted(self.remaining + self.classified)
        counts = multinomial_counts([self.regions[r].volume() for r in rids],
                                    self.budget - self.used, self._rng(_ITERATION, _FINAL))
        remaining, classified = [], []
        for rid, cnt in zip(rids, counts):
            r = self.regions[rid]
            if cnt > 0:
                self._sample_more(r, int(cnt), self._rng(rid, _FINAL))
            self._file(r, remaining, classified)
        self.remaining, self.classified = remaining, classified
        # Spec §3: the final phase ends the loop. D8: anything the pool could not
        # supply stays unspent and is never topped up.
        self.done = True

    def run(self) -> None:
        if self.phase == "root":
            self.run_root()
        while not self.done:
            self.step()
