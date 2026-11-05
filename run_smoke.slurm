#!/bin/bash
#SBATCH --job-name=chol-smoke
#SBATCH --partition=khawarizmi--gpu-p
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=smoke-%j.out
#SBATCH --error=smoke-%j.err

set -euo pipefail
cd /home/khawarizmi/team1234

if ! command -v julia >/dev/null 2>&1; then
    echo "Julia is not available in PATH; load the cluster Julia module first." >&2
    exit 127
fi

echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Julia: $(julia --version)"
nvidia-smi

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JULIA_PKG_OFFLINE=true
julia --project=. --startup-file=no smoke_test.jl
