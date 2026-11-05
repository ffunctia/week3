#!/bin/bash
#SBATCH --job-name=tile-chol
#SBATCH --partition=khawarizmi--gpu-p
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd /home/khawarizmi/team1234
mkdir -p results

if [[ -z "${N:-}" || -z "${PRECISION:-}" ]]; then
    echo "Submit one case with --export=ALL,N=1000,PRECISION=Float32" >&2
    exit 2
fi

if ! command -v julia >/dev/null 2>&1; then
    echo "Julia is not available in PATH; load or install Julia before submitting." >&2
    exit 127
fi

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JULIA_PKG_OFFLINE=true

julia --project=. -O3 --startup-file=no bench_one.jl
