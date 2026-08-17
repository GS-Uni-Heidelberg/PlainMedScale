#!/bin/bash

#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=3:00:00                # GPT-2/BERT base over ~7400 short simp rows
                                       # finishes well inside this; if SLURM kills it,
                                       # default-resume picks up where it left off.
#SBATCH --mem=40gb
#SBATCH --job-name=simp_llm_metrics
#SBATCH --output=simp_llm_metrics_%j.out
#SBATCH --error=simp_llm_metrics_%j.err

# ============================================================================
# LLM-based readability metrics (ANSP, APPL, LMFM) over LLM-simplified texts.
# Mirrors readability/pipelines/run_llms.sh but points pipeline_metrics.py at a run dir
# instead of pipeline_llms.py at the source corpora.
#
# Reads:
#   <RUN_DIR>/simplifications_*.tsv
# Writes:
#   <RUN_DIR>/metrics_llm_strip/<lang>_<source>_<subtree>.tsv
#   <RUN_DIR>/metrics_llm_strip/INFO.txt   (auto-stamped)
#
# Submit from repo root:
#   sbatch simplification/run_pipeline_metrics_llm.sh
#   RUN_DIR=simplification/results/other-run \
#     sbatch simplification/run_pipeline_metrics_llm.sh
# ============================================================================

module load devel/cuda

# export HF_HOME=/path/to/shared/hf_cache

export OMP_NUM_THREADS=8
export PYTHONPATH=readability:$PYTHONPATH

source env/bin/activate

RUN_DIR="${RUN_DIR:-simplification/results/qwen3-30b}"
TABLE_MODE="${TABLE_MODE:-strip}"

python simplification/pipeline_metrics.py \
    --run-dir "${RUN_DIR}" \
    --metrics llm \
    --table-mode "${TABLE_MODE}"
