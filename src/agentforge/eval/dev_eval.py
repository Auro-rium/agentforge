"""Fast offline dev-loop scorer: for each held-out row, generate a
completion from everything but the final assistant turn, parse the model's
emitted tool call(s), and score against that turn as the reference. Cheap
enough to run between GPU-rental sessions, unlike a full BFCL/tau2-bench
run, at the cost of being a rougher signal.

`score_row` is a pure function (no model involved) -- fully unit-testable
against synthetic generated/reference strings. `run_dev_eval` is the CLI
path that actually loads a model and generates completions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge.data.schema import Row
from agentforge.eval.tool_call_parsers import parse_tool_calls


def _normalize_call(call: dict) -> tuple[str, dict]:
    """(name, arguments-as-dict) for comparison, tolerant of `arguments`
    being either a dict (as tool_call_parsers emits) or a JSON-encoded
    string (as the canonical schema's FunctionCall.arguments stores it)."""
    name = call.get("name", "")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_unparsed": arguments}
    return name, arguments


def _reference_calls(reference_message: dict) -> list[tuple[str, dict]]:
    tool_calls = reference_message.get("tool_calls") or []
    return [_normalize_call(tc["function"]) for tc in tool_calls]


def _arg_overlap_score(expected: dict, actual: dict) -> float:
    """Fraction of expected key/value pairs also present (same value) in
    actual. 1.0 for an exact match, 1.0 (vacuously) if expected is empty."""
    if not expected:
        return 1.0
    matched = sum(1 for k, v in expected.items() if actual.get(k) == v)
    return matched / len(expected)


def score_row(generated_text: str, row: Row, *, model_family: str = "gemma4") -> dict:
    """Score a single generation against `row`'s final message as ground
    truth. Assumes the caller generated `generated_text` from
    `row.messages[:-1]` (i.e. row.messages[-1] was the held-out target).
    """
    reference_message = row.messages[-1].model_dump()
    expected_calls = _reference_calls(reference_message)

    parsed_raw = parse_tool_calls(generated_text, model_family=model_family)
    parsed_calls = [_normalize_call(c) for c in parsed_raw]

    result: dict = {
        "row_id": row.id,
        "expected_call_count": len(expected_calls),
        "parsed_call_count": len(parsed_calls),
    }

    if not expected_calls:
        # Terminal free-text answer turn (common for ReAct trajectories) --
        # no structured call expected. Task-success proxy: does the
        # generated text contain the reference answer's content, as a loose
        # substring/overlap check (real semantic equivalence needs a judge
        # model or human eval; this is a fast, crude dev-loop signal only).
        reference_content = (reference_message.get("content") or "").strip().lower()
        generated_lower = generated_text.strip().lower()
        result["json_valid"] = len(parsed_calls) == 0  # correctly emitted no spurious call
        result["name_match"] = None
        result["arg_match_score"] = None
        result["task_success_proxy"] = (
            bool(reference_content) and reference_content in generated_lower
        )
        return result

    result["json_valid"] = len(parsed_calls) > 0
    name_matches = 0
    arg_scores: list[float] = []
    for i, (expected_name, expected_args) in enumerate(expected_calls):
        if i >= len(parsed_calls):
            arg_scores.append(0.0)
            continue
        actual_name, actual_args = parsed_calls[i]
        if actual_name == expected_name:
            name_matches += 1
        arg_scores.append(_arg_overlap_score(expected_args, actual_args))

    result["name_match"] = name_matches / len(expected_calls)
    result["arg_match_score"] = sum(arg_scores) / len(arg_scores) if arg_scores else 0.0
    result["task_success_proxy"] = (
        result["name_match"] == 1.0 and result["arg_match_score"] == 1.0
    )
    return result


def aggregate_scores(row_scores: list[dict]) -> dict:
    """Summarize a list of `score_row` outputs into overall rates."""
    if not row_scores:
        return {"count": 0}
    n = len(row_scores)
    with_calls = [s for s in row_scores if s["expected_call_count"] > 0]
    without_calls = [s for s in row_scores if s["expected_call_count"] == 0]
    return {
        "count": n,
        "json_valid_rate": sum(s["json_valid"] for s in row_scores) / n,
        "task_success_rate": sum(s["task_success_proxy"] for s in row_scores) / n,
        "tool_call_rows": {
            "count": len(with_calls),
            "avg_name_match": (
                sum(s["name_match"] for s in with_calls) / len(with_calls) if with_calls else None
            ),
            "avg_arg_match_score": (
                sum(s["arg_match_score"] for s in with_calls) / len(with_calls)
                if with_calls
                else None
            ),
        },
        "terminal_answer_rows": {
            "count": len(without_calls),
            "task_success_rate": (
                sum(s["task_success_proxy"] for s in without_calls) / len(without_calls)
                if without_calls
                else None
            ),
        },
    }


def run_dev_eval(
    *, base_model: str, adapter_dir: str, holdout_path: str, quantized: bool, run_name: str
) -> dict:
    """CLI path: loads a real model, generates over the real holdout set.
    Not unit-tested (needs a real model) -- see score_row/aggregate_scores
    above for the tested, model-free logic this wraps.
    """
    from agentforge.infer import generate_completion, load_adapter_model
    from agentforge.model_utils import load_tokenizer

    tokenizer = load_tokenizer(base_model=base_model)
    model = load_adapter_model(
        base_model=base_model, adapter_dir=adapter_dir, quantization_enabled=quantized
    )

    holdout_lines = Path(holdout_path).read_text().splitlines()
    rows = [Row.model_validate_json(line) for line in holdout_lines if line.strip()]

    row_scores = []
    for row in rows:
        preceding = [m.model_dump() for m in row.messages[:-1]]
        tools = [t.model_dump() for t in row.tools] or None
        generated = generate_completion(model, tokenizer, messages=preceding, tools=tools)
        row_scores.append(score_row(generated, row))

    summary = aggregate_scores(row_scores)
    report_dir = Path("reports/dev_eval") / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (report_dir / "per_row.json").write_text(json.dumps(row_scores, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--holdout-path", default="data/holdout.jsonl")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--run-name", default="dev_eval")
    args = parser.parse_args()

    summary = run_dev_eval(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        holdout_path=args.holdout_path,
        quantized=args.quantized,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
