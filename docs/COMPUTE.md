# Compute — running this repo off iLab

iLab is the default and needs nothing documented here: `scripts/sbatch_run.sh`, the conda env in
`CLAUDE.md`, and `.env` cover it. This page exists for **Amarel**, which is a different filesystem
with different failure modes.

Recorded 2026-07 from st1122's Amarel setup, ported here from `CLAUDE.md` during the docs
restructure. Paths are that account's — substitute your own.

## Amarel is a separate filesystem

Code arrives via GitHub, results come back via `rsync`. **Nothing under `/common/` is mounted
there.** This is the fact everything below follows from.

- amarel_clone_path: `/home/st1122/Projects/adaptive_roa`
- amarel_conda_env: `/home/st1122/Projects/adaptive_roa/env`
- amarel_conda_base: `/home/st1122/miniforge3`
- amarel_log_dir: `/home/st1122/Projects/adaptive_roa/slurm_logs`
- amarel_data_dir: `/scratch/st1122/genMoPlan-exp/data_trajectories`
- amarel_exp_dir: `/scratch/st1122/adaptive_roa/experiments`
- amarel_account: `general`
- sbatch template: `scripts/sbatch_amarel.sh`; dataset staging: `scripts/stage_dataset_amarel.sh`

**Sibling dependency:** `flow_matching` (github.com/Ewerton-Vieira/flow_matching) is an editable
install and must be cloned separately at `/home/st1122/Projects/flow_matching`.

## `.env` does not travel with the clone

`.env` is gitignored, so a fresh Amarel clone has none. The copy there must set `DATA_DIR`,
`SHARED_DATA_BASE`, `EXP_DIR`, `USER_BASE`, `NET_ID`.

`SHARED_DATA_BASE` is the one that gets missed — quadrotor3d and humanoid read it instead of
`DATA_DIR`, and `get_shared_data_base()` falls back to a hardcoded iLab path rather than raising
(`adaptive_roa/utils/env_config.py`). On Amarel that path does not exist, so the failure surfaces
later and further away than the mistake.

## GPU partitions

All `3-00:00:00` limit, untyped `--gres=gpu:N`.

- `gpu-redhat` — the main pool: volta (sm_70), ampere (sm_80/86), adalovelace (sm_89)
- `cgpu-redhat` — **Camden nodes, do not submit**
- Torch 2.5.1/cu118 covers every arch present; there are no Blackwell cards on Amarel.
- There is no `legacy-gpu` partition — it no longer exists. `gpu-redhat` is the only usable GPU
  pool and it does queue; check `sbatch --test-only` before assuming a fast start.

## glibc split — the env does not run on the login node

The login node is CentOS 7 (glibc 2.17); the `*-redhat` compute partitions are RHEL 9.6
(glibc 2.34). The conda env is built against the latter, so importing torch on the login node fails
with `GLIBC_2.27 not found ... libcurand.so.10`. **This is expected, not a broken env.** Build and
test the env through SLURM:

```bash
srun --account=general --partition=main-redhat --time=00:10:00 --mem=8G --cpus-per-task=2 \
     ./env/bin/python -c "import torch; print(torch.__version__)"
```

`torch.cuda.get_arch_list()` returns `[]` on a CPU node (no CUDA driver) — that is not evidence of
a CPU-only build. Check `torch.version.cuda` instead.

## Datasets are staged per run

Not bulk-mirrored. `${DATA_DIR}` on Amarel holds only the regime roots (`deterministic/`, `noisy/`,
`partial_deterministic/`); a run copies in the subtree it needs. Reference sizes on iLab:
`deterministic/pendulum` 1.9G, `noisy/pendulum` 12G, `deterministic/humanoid_get_up_medium` 88G —
the tree totals ~119G.
