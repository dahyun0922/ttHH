#!/bin/bash
#SBATCH --job-name=recoEDA
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reco_eda_%j.out
#SBATCH --error=logs/reco_eda_%j.err

set -euo pipefail

cd "$HOME/ttHH"

export OMP_NUM_THREADS=2
source .venv/bin/activate

python -u scripts/reco_candidate_eda.py
