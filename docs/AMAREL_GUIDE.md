# Running on Amarel Cluster - Quick Guide

This guide shows you how to run experiments on the Amarel HPC cluster at Rutgers.

> **Generic walkthrough — placeholders throughout** (`NETID`, `PROJECT_DIR`, `ENV_NAME`). For this
> repo's *concrete* Amarel setup — clone/env/data paths, the `sbatch_amarel.sh` template, dataset
> staging, and the login-node glibc split that stops the env importing torch — see
> **`COMPUTE.md`**. Read that one first; this is the general how-to underneath it.
>
> Corrected 2026-07-28 against `COMPUTE.md`: `TMPDIR` used `/common/home/...`, which does not exist
> on Amarel (nothing under `/common/` is mounted there — that's an iLab path), and the SLURM
> examples named a `gpu` partition and omitted `--account=general`.

## Prerequisites

- Your Rutgers NetID (e.g., `abc123`)
- SSH access to Amarel
- A conda environment set up for your project

---

## 1. Logging In

From your local computer:

```bash
ssh NETID@amarel.rutgers.edu
```

Replace `NETID` with your actual Rutgers NetID.

---

## 2. Navigate to Project and Setup Environment

Once logged into Amarel:

```bash
# Navigate to your project directory
cd /home/NETID/PROJECT_DIR

# Load conda (if not already loaded)
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your project environment
conda activate ENV_NAME
```

Replace:
- `PROJECT_DIR` with your actual project directory name
- `ENV_NAME` with your conda environment name

**Important: Set up temporary directory**

Amarel's `/tmp` directory is shared and can cause permission errors. Use your home directory instead:

```bash
mkdir -p /home/NETID/tmp
export TMPDIR=/home/NETID/tmp
```

To make this permanent, add the `export` line to your `~/.bashrc`:

```bash
echo 'export TMPDIR=/home/NETID/tmp' >> ~/.bashrc
```

---

## 3. Quick Test (Login Node Only)

On the login node, you can only run very short tests. For example:

```bash
cd /home/NETID/PROJECT_DIR
conda activate ENV_NAME
export TMPDIR=/home/NETID/tmp

# Test that your environment works
python your_script.py --help
```

**Important:** Don't run long training jobs on the login node. Use compute nodes instead (see below).

---

## 4. Interactive GPU Session (For Debugging)

For debugging or testing, you can request an interactive GPU session:

```bash
salloc --account=general --partition=gpu-redhat --gres=gpu:1 --time=02:00:00 --mem=32G --constraint=ampere
```

Once you get a compute node, run your code:

```bash
cd /home/NETID/PROJECT_DIR
conda activate ENV_NAME
export TMPDIR=/home/NETID/tmp

# Example: run a short test
python your_script.py \
  --arg1 value1 \
  --arg2 value2
```

When done, type `exit` to release the node.

---

## 5. Submit Batch Jobs (Recommended for Real Runs)

Create a Slurm batch script. Example: `run_job.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=your_job_name
#SBATCH --account=general
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --constraint=ampere
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ENV_NAME

export TMPDIR=/home/NETID/tmp

cd /home/NETID/PROJECT_DIR

python your_script.py \
  --arg1 value1 \
  --arg2 value2
```

**Submit the job:**

```bash
cd /home/NETID/PROJECT_DIR
sbatch run_job.sbatch
```

---

## 6. Check Job Status

**See your running/pending jobs:**

```bash
squeue -u NETID
```

**See job history:**

```bash
sacct -u NETID --format=JobID,JobName,State,Elapsed
```

**Check logs after job completes:**

```bash
cd /home/NETID/PROJECT_DIR/logs
ls -lt
less your_job_name_<JOBID>.out
less your_job_name_<JOBID>.err
```

**Check output directories:**

```bash
cd /home/NETID/PROJECT_DIR/outputs
ls
```

---

## 7. Common Slurm Options

Here are useful options for your batch scripts:

- `--account=general`: Required; jobs are rejected without it
- `--partition=gpu-redhat`: The GPU partition. Do **not** submit to `cgpu-redhat` (Camden nodes),
  and there is no `legacy-gpu` — it no longer exists
- `--gres=gpu:1`: Request 1 GPU (untyped)
- `--constraint=ampere`: Request A100 GPUs (best performance)
- `--time=24:00:00`: Maximum runtime (24 hours)
- `--mem=64G`: Request 64GB RAM
- `--output=logs/%j.out`: Where to save stdout
- `--error=logs/%j.err`: Where to save stderr

---

## 8. Cancelling Jobs

To cancel a running job:

```bash
scancel JOBID
```

To cancel all your jobs:

```bash
scancel -u NETID
```

---

## Troubleshooting

**Permission denied errors with `/tmp`:**
- Make sure you set `export TMPDIR=/home/NETID/tmp` before running your code

**Job gets killed:**
- Check if you exceeded time limit (`--time` in your script)
- Check if you exceeded memory limit (`--mem` in your script)
- Look at the `.err` log file for error messages

**Can't find conda:**
- Make sure you run: `source ~/miniconda3/etc/profile.d/conda.sh`
- Or check where your conda is installed: `which conda`

---

## Quick Reference

Replace placeholders in all commands above:
- `NETID`: Your actual Rutgers NetID
- `PROJECT_DIR`: Your project directory name
- `ENV_NAME`: Your conda environment name

Common paths (customize as needed):
- Project directory: `/home/NETID/PROJECT_DIR`
- Temporary files: `/home/NETID/tmp`
- Logs: `/home/NETID/PROJECT_DIR/logs`
- Outputs: `/home/NETID/PROJECT_DIR/outputs`

