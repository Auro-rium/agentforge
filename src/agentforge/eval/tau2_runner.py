"""Wraps τ²-bench (github.com/sierra-research/tau2-bench) -- the dual-control
tool-agent-user benchmark used as agentforge's second multi-turn reliability
signal alongside BFCL's multi-turn subsets.

UNVERIFIED, unlike bfcl_runner.py: tau2-bench's exact install method (source
checkout vs. a pip package), how it points at a local HF model + LoRA
adapter (likely via an OpenAI-compatible local server -- e.g. serving the
adapter through vLLM and pointing tau2-bench's client config at it, similar
to the BFCL vLLM path), and which domain subset to run by default were not
independently confirmed against the live repo during this build (no local
GPU to run a real pass). This module is intentionally a thin, honest
placeholder: it defines the shape the pipeline expects
(`run_tau2_eval(...) -> dict`, mirroring bfcl_runner's signature) and
documents exactly what needs a real spike before this can run for real,
rather than guessing at a specific CLI invocation and presenting it as
verified. Do this spike before flipping `eval.tau2_bench.enabled: true` in
a real training config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class Tau2BenchNotConfigured(RuntimeError):
    """Raised until the real tau2-bench integration is spiked and implemented."""


def run_tau2_eval(
    *,
    repo: str,
    adapter_dir: str,
    base_model: str,
    domains: list[str] | None,
    run_name: str,
) -> dict:
    raise Tau2BenchNotConfigured(
        "tau2-bench integration is not yet implemented -- this needs a real spike "
        f"against {repo} first: confirm the install method, how it points at a local "
        "model + LoRA adapter (likely an OpenAI-compatible server, e.g. vLLM serving "
        "the merged/adapter model), and which domain subset to run by default. "
        "See this module's docstring. Once spiked, implement the actual subprocess/"
        "API calls here, following bfcl_runner.py's shape "
        "(run_generate -> run_evaluate -> collect_scores -> write reports/tau2/<run_name>/)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--repo", default="https://github.com/sierra-research/tau2-bench")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--run-name", default="tau2_eval")
    args = parser.parse_args()

    scores = run_tau2_eval(
        repo=args.repo,
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        domains=args.domains,
        run_name=args.run_name,
    )
    report_dir = Path("reports/tau2") / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
