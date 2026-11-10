#!/bin/bash
#SBATCH --job-name=validateCache
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/validate_cache_%A_%a.out
#SBATCH --error=logs/validate_cache_%A_%a.err

set -euo pipefail

export GCC_HOME=/opt/ohpc/pub/compiler/gcc/12.2.0
export PATH="$GCC_HOME/bin:$PATH"
export CXX="$GCC_HOME/bin/g++"

cd "$HOME/ttHH"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        SAMPLE=ttHH
        FILES=220
        EVENTS=9987962
        SUMW=9987962
        ;;
    1)
        SAMPLE=HH
        FILES=61
        EVENTS=1199991
        SUMW=35550.602502387017
        ;;
    2)
        SAMPLE=ttbar
        FILES=780
        EVENTS=470123263
        SUMW=37820110815.09285
        ;;
esac

root -l -b -q \
  "scripts/validate_cache.C(\"${SAMPLE}\",${FILES},${EVENTS},${SUMW})"
