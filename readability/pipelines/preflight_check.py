#!/usr/bin/env python3
"""
Preflight check for readability pipelines.

Run this before submitting sbatch jobs to verify that:
 - All input data files exist and are valid JSON
 - Python imports resolve correctly
 - Spacy models are installed
 - (Optional) LLM models can be loaded (pass --llm to test GPU + transformers)

Usage from repo root:
    PYTHONPATH=readability python readability/pipelines/preflight_check.py
    PYTHONPATH=readability python readability/pipelines/preflight_check.py --llm   # also test GPU + model loading
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

errors = []
warnings = []


def check(label: str, fn):
    try:
        fn()
        print(f"  [OK] {label}")
    except Exception as e:
        errors.append(f"{label}: {e}")
        print(f"  [FAIL] {label}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/corpus"),
                        help="Directory holding the Zenodo release JSONs.")
    parser.add_argument("--llm", action="store_true", help="Also check GPU and LLM model loading")
    args = parser.parse_args()

    # --- 1. Input files ---
    print("Checking corpus release files...")
    from lib import corpus

    for split in corpus.SPLITS:
        def _check_file(sp=split):
            path = sp.path(args.input_dir)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            data = json.loads(path.read_text())
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError(f"{path} is empty or not a JSON list")
            print(f"         ({len(data)} records)")

        check(split.filename, _check_file)

    # --- 2. Python imports ---
    print("Checking Python imports...")

    def _check_spacy_imports():
        from lib import metrics, spacy_nlp, markdown_cleaner, corpus  # noqa: F401

    check("readability.lib imports", _check_spacy_imports)

    # --- 3. Spacy models ---
    print("Checking spacy models...")

    def _check_de_model():
        from lib.spacy_nlp import ger_nlp  # noqa: F401

    def _check_en_model():
        from lib.spacy_nlp import eng_nlp  # noqa: F401

    check("German spacy model", _check_de_model)
    check("English spacy model", _check_en_model)

    # --- 4. Quick spacy smoke test ---
    print("Running spacy smoke test (1 article)...")

    def _spacy_smoke():
        from lib import metrics, spacy_nlp
        doc = spacy_nlp.ger_nlp("Dies ist ein kurzer Testtext mit zwei Sätzen. Er dient der Überprüfung.")
        _ = metrics.avg_sentence_length(doc)
        _ = metrics.fkgl(doc)

    check("spacy metric computation", _spacy_smoke)

    # --- 5. Optional LLM checks ---
    if args.llm:
        print("Checking GPU / LLM models...")

        def _check_torch_gpu():
            import torch
            if not torch.cuda.is_available():
                warnings.append("CUDA not available — LLM pipeline will run on CPU (very slow)")
                print(f"         WARNING: CUDA not available")
            else:
                print(f"         CUDA device: {torch.cuda.get_device_name(0)}")

        check("torch + CUDA", _check_torch_gpu)

        def _check_llm_imports():
            from lib import modelling  # noqa: F401

        check("readability.lib.modelling import", _check_llm_imports)

        def _check_llm_load():
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tok = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
            model.eval()
            print(f"         gpt2 loaded on {device}")

        check("GPT-2 model load", _check_llm_load)
    else:
        print("Skipping LLM checks (pass --llm to enable)")

    # --- Summary ---
    print()
    if errors:
        print(f"PREFLIGHT FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED — all checks OK")
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")
        sys.exit(0)


if __name__ == "__main__":
    main()
