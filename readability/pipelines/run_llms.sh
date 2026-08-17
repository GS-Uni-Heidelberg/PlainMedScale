#!/bin/bash

#SBATCH --partition=gpu-single        # single GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A40:1             # GPT-2 / BERT models fit easily on one GPU
#SBATCH --time=16:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=llm_metrics
#SBATCH --output=llm_metrics_%j.out
#SBATCH --error=llm_metrics_%j.err

# ============================================================================
# LLM-based readability metrics pipeline (ANSP, APPL, LMFM)
# ============================================================================
#
# Required input data files (relative to repo root):
#   data/corpus/*.json   (the Zenodo corpus release: one file per
#                         source/language, MSD split by subtree)
#
# Output directories (created automatically):
#   data/metrics_llm_strip/   (--table-mode strip)
#   data/metrics_llm_edit/    (--table-mode edit)
#
# Submit from repo root:
#   sbatch readability/pipelines/run_llms.sh
# ============================================================================

module load devel/cuda

# Point to a shared HuggingFace cache to avoid re-downloading models:
# export HF_HOME=/path/to/shared/hf_cache

export OMP_NUM_THREADS=8
export PYTHONPATH=readability:$PYTHONPATH

source env/bin/activate

# Create output dirs upfront
mkdir -p data/metrics_llm_strip
# mkdir -p data/metrics_llm_edit

# Run both table modes
python readability/pipelines/pipeline_llms.py --table-mode strip --output-dir data/metrics_llm_strip --overwrite
# python readability/pipelines/pipeline_llms.py --table-mode edit  --output-dir data/metrics_llm_edit --overwrite
