"""Wraps `bfcl generate`/`bfcl evaluate` (github.com/ShishirPatil/gorilla,
`pip install bfcl-eval`, part of this project's `[eval]` extra) against a
trained LoRA adapter, targeting BFCL v4's multi-turn subsets as the primary
signal post-pivot -- see docs/plan for why single-turn categories are
tracked only as a regression check, not the headline metric.

Prerequisite the caller must confirm before this will work (not automated
here): the base model's handler_key must be registered in
`bfcl_eval/constants/model_config.py`, or a custom handler subclassing
`base_oss_handler.py` must be written -- `gemma-4-12b-it` was very new
(~2026-06) as of this project's planning and wasn't independently confirmed
to be registered. See scripts/run_bfcl.sh for the intended invocation
context (checks bfcl-eval is installed before calling into this module).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def run_bfcl_generate(
    *,
    handler_key: str,
    local_model_path: str,
    adapter_dir: str,
    max_lora_rank: int,
    test_categories: list[str],
    backend: str = "vllm",
) -> None:
    """`bfcl generate --enable-lora ...` -- produces model responses for the
    given test categories. `--enable-lora`/`--lora-modules` only work with
    `--backend vllm` (per BFCL's docs); this deliberately doesn't silently
    fall back to another backend if `vllm` is requested but unavailable.
    """
    cmd = [
        "bfcl",
        "generate",
        "--model",
        handler_key,
        "--backend",
        backend,
        "--local-model-path",
        local_model_path,
        "--enable-lora",
        "--max-lora-rank",
        str(max_lora_rank),
        "--lora-modules",
        f"agentforge={adapter_dir}",
        "--test-category",
        ",".join(test_categories),
    ]
    subprocess.run(cmd, check=True)


def run_bfcl_evaluate(*, handler_key: str, test_categories: list[str]) -> None:
    cmd = ["bfcl", "evaluate", "--model", handler_key, "--test-category", ",".join(test_categories)]
    subprocess.run(cmd, check=True)


def collect_bfcl_scores(*, handler_key: str, score_dir: str = "score") -> dict:
    """Parse BFCL's `score/<model>/*.csv` output into a single summary dict.
    BFCL's exact CSV filenames/columns weren't independently verified during
    this build (no local GPU to run a real `bfcl evaluate` pass) -- this
    reads whatever CSVs exist under score/<handler_key>/ generically (each
    file's rows keyed by its first column) rather than hardcoding exact
    filenames, so it degrades gracefully if the real layout differs
    slightly; verify against a real run before trusting the summary shape.
    """
    model_score_dir = Path(score_dir) / handler_key
    summary: dict = {}
    if not model_score_dir.exists():
        return summary
    for csv_path in sorted(model_score_dir.glob("*.csv")):
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        summary[csv_path.stem] = rows
    return summary


def run_bfcl_eval(
    *,
    handler_key: str,
    local_model_path: str,
    adapter_dir: str,
    max_lora_rank: int,
    test_categories: list[str],
    regression_categories: list[str],
    backend: str,
    run_name: str,
) -> dict:
    if shutil.which("bfcl") is None:
        raise RuntimeError(
            "bfcl CLI not found on PATH -- install the `eval` extra first: "
            "uv pip install -e '.[eval]' (see scripts/run_bfcl.sh)"
        )

    all_categories = list(dict.fromkeys([*test_categories, *regression_categories]))
    run_bfcl_generate(
        handler_key=handler_key,
        local_model_path=local_model_path,
        adapter_dir=adapter_dir,
        max_lora_rank=max_lora_rank,
        test_categories=all_categories,
        backend=backend,
    )
    run_bfcl_evaluate(handler_key=handler_key, test_categories=all_categories)
    scores = collect_bfcl_scores(handler_key=handler_key)

    report_dir = Path("reports/bfcl") / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--handler-key", default="gemma-4-12b-it")
    parser.add_argument("--local-model-path", default="google/gemma-4-12B-it")
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument(
        "--test-categories",
        nargs="+",
        default=[
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ],
    )
    parser.add_argument(
        "--regression-categories", nargs="+", default=["simple", "multiple", "parallel"]
    )
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--run-name", default="bfcl_eval")
    args = parser.parse_args()

    scores = run_bfcl_eval(
        handler_key=args.handler_key,
        local_model_path=args.local_model_path,
        adapter_dir=args.adapter_dir,
        max_lora_rank=args.max_lora_rank,
        test_categories=args.test_categories,
        regression_categories=args.regression_categories,
        backend=args.backend,
        run_name=args.run_name,
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
