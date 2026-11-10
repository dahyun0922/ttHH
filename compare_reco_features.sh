#!/bin/bash
#SBATCH --job-name=recoDNN
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reco_dnn_%j.out
#SBATCH --error=logs/reco_dnn_%j.err

set -euo pipefail

cd "$HOME/ttHH"

export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1

source .venv/bin/activate

python -u scripts/compare_reco_features.py
