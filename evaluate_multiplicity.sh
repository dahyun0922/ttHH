#!/bin/bash
#SBATCH --job-name=multAUC
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/multiplicity_%j.out
#SBATCH --error=logs/multiplicity_%j.err

set -euo pipefail

cd "$HOME/ttHH"

export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1

source .venv/bin/activate
python -u scripts/evaluate_multiplicity.py
