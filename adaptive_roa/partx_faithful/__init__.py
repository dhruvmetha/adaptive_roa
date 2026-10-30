"""Faithful Part-X (Pedrielli et al., arXiv 2110.10729) for the stochastic RoA benchmarks.

Spec: docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md.
The core modules (gp_laplace, backends, region, sampling, mcstep, bo, algorithm,
volume, state) import numpy/scipy only, so the cpslab reference check can run
them under numpy 1.x. readout and runner need the project env.
"""
