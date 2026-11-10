#!/bin/bash
#SBATCH --job-name=pairAblation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pair_ablation_%j.out
#SBATCH --error=logs/pair_ablation_%j.err

set -euo pipefail

cd "$HOME/ttHH"

export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1

source .venv/bin/activate

python -u scripts/compare_pair_groups.py
