#!/bin/bash

#SBATCH --partition=cpu-single         # CPU partition (no GPU needed for spacy)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=06:00:00
#SBATCH --mem=32gb
#SBATCH --job-name=spacy_metrics
#SBATCH --output=spacy_metrics_%j.out
#SBATCH --error=spacy_metrics_%j.err

# ============================================================================
# Spacy readability metrics pipeline
# ============================================================================
#
# Required input data files (relative to repo root):
#   data/corpus/*.json   (the Zenodo corpus release: one file per
#                         source/language, MSD split by subtree)
#
# Output directories (created automatically):
#   data/metrics_v2_strip/   (--table-mode strip)
#   data/metrics_v2_edit/    (--table-mode edit)
#
# Submit from repo root:
#   sbatch readability/pipelines/run_spacy.sh
# ============================================================================

export OMP_NUM_THREADS=8
export PYTHONPATH=readability:$PYTHONPATH

source env/bin/activate

# Create output dirs upfront
mkdir -p data/metrics_v2_strip
mkdir -p data/metrics_v2_edit

# Run both table modes
python readability/pipelines/pipeline_spacy.py --table-mode strip --output-dir data/metrics_v2_strip --overwrite
python readability/pipelines/pipeline_spacy.py --table-mode edit  --output-dir data/metrics_v2_edit --overwrite
