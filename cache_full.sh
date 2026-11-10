#!/bin/bash
#SBATCH --job-name=cacheFull
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cache_%A_%a.out
#SBATCH --error=logs/cache_%A_%a.err

set -euo pipefail

export GCC_HOME=/opt/ohpc/pub/compiler/gcc/12.2.0
export PATH="$GCC_HOME/bin:$PATH"
export CXX="$GCC_HOME/bin/g++"

cd "$HOME/ttHH"

MANIFEST="config/cache_chunks/manifest.tsv"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=$'\t' read -r SAMPLE CHUNK FILELIST <<< "$LINE"

case "$SAMPLE" in
    ttHH) LABEL=0 ;;
    HH) LABEL=1 ;;
    ttbar) LABEL=2 ;;
    *) echo "Unknown sample: $SAMPLE"; exit 1 ;;
esac

mkdir -p \
    "cache/objects/$SAMPLE" \
    "cache/features/$SAMPLE" \
    "cache/completed/$SAMPLE"

echo "Sample: $SAMPLE"
echo "Chunk: $CHUNK"

COUNT=0
LIMIT="${FILE_LIMIT:-5}"

while IFS=$'\t' read -r INDEX INPUT; do
    if (( COUNT >= LIMIT )); then
        break
    fi
    COUNT=$((COUNT + 1))

    OBJ="cache/objects/$SAMPLE/${INDEX}.root"
    FEAT="cache/features/$SAMPLE/${INDEX}.root"
    DONE="cache/completed/$SAMPLE/${INDEX}.done"

    if [[ -f "$DONE" && -s "$OBJ" && -s "$FEAT" ]]; then
        echo "SKIP completed: $SAMPLE $INDEX"
        continue
    fi

    echo "Processing: $SAMPLE $INDEX"

    rm -f "$DONE" "$OBJ" "$FEAT"

    root -l -b -q \
      "scripts/make_object_cache.C(\"${INPUT}\",\"${OBJ}\")"

    test -s "$OBJ"

    root -l -b -q \
      "scripts/make_feature_cache.C(\"${OBJ}\",\"${FEAT}\",${LABEL})"

    test -s "$FEAT"

    touch "$DONE"

    echo "SUCCESS: $SAMPLE $INDEX"
done < "$FILELIST"

echo "CHUNK COMPLETED: $SAMPLE $CHUNK"
