#!/bin/bash

#SBATCH --partition=gpu-single     # shared GPU partition; use gpu-multi for exclusive node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:H200:1               # 141 GB — fits Qwen3-30B-A3B BF16 (~60 GB) with big KV-cache headroom for batching
#SBATCH --time=120:00:00                 # gpu-single max walltime; resume-safe
#SBATCH --mem=80gb
#SBATCH --job-name=simplify
#SBATCH --output=simplify_%j.out
#SBATCH --error=simplify_%j.err

if [[ ! -f data/alignment/instance_alignments_raw.xlsx ]]; then
    echo "ERROR: must sbatch from repo root (missing data/alignment/instance_alignments_raw.xlsx)" >&2
    exit 1
fi

module load devel/cuda

# Point to a shared HuggingFace cache to avoid re-downloading the model:
# export HF_HOME=/path/to/shared/hf_cache

export OMP_NUM_THREADS=8

source env/bin/activate

LANGUAGE="${LANGUAGE:-de}"
TIERS="${TIERS:-4_tier 3_tier 2_tier}"
# Shared output dir across DE + EN runs — resume keys include language, so
# the two jobs won't collide. Submit each language as its own sbatch:
#   LANGUAGE=de sbatch simplification/run_simplify.sh
#   LANGUAGE=en sbatch simplification/run_simplify.sh
OUTDIR="${OUTDIR:-simplification/results/qwen3-30b}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Typer expects repeated --tiers flags (--tiers X --tiers Y), not --tiers X Y.
TIER_ARGS=()
for t in ${TIERS}; do TIER_ARGS+=(--tiers "$t"); done

python simplification/run_simplify.py \
    --language "${LANGUAGE}" \
    "${TIER_ARGS[@]}" \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "${OUTDIR}"
