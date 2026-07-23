"""CLI: normalize all 5 source datasets -> dedupe -> stratified holdout split
-> per-source sampling weights -> data/manifest.jsonl + data/holdout.jsonl +
data/manifest_stats.json.

Intended to run on the AWS training instance (see scripts/build_data.sh /
scripts/aws/bootstrap_and_train.sh), not the local dev machine -- it
downloads the real datasets, including the gated Salesforce/xlam-function-
calling-60k which needs HF_TOKEN set and its terms accepted on
huggingface.co first.

Usage:
    python -m agentforge.data.build_manifest --config configs/gemma4-12b-qlora.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agentforge.config import AgentForgeConfig
from agentforge.data.mix import apply_weights_by_source
from agentforge.data.normalizers.agent_flan import AgentFlanNormalizer
from agentforge.data.normalizers.base import NormalizationStats, Normalizer
from agentforge.data.normalizers.glaive import GlaiveNormalizer
from agentforge.data.normalizers.hermes import HermesNormalizer
from agentforge.data.normalizers.toolace import ToolACENormalizer
from agentforge.data.normalizers.xlam import XlamNormalizer
from agentforge.data.schema import Row
from agentforge.eval.holdout_manifest import stratified_holdout_split

NORMALIZER_CLASSES: dict[str, type[Normalizer]] = {
    "glaive": GlaiveNormalizer,
    "hermes": HermesNormalizer,
    "toolace": ToolACENormalizer,
    "xlam": XlamNormalizer,
    "agent_flan": AgentFlanNormalizer,
}


def _row_dedup_key(row: Row) -> str:
    canonical = json.dumps([m.model_dump() for m in row.messages], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _dedupe_across_sources(
    rows_by_source: dict[str, list[Row]],
) -> tuple[dict[str, list[Row]], int]:
    """Remove exact-duplicate rows (by message content) across all sources,
    keeping the first occurrence. Returns (deduped_rows_by_source, removed_count).
    """
    seen: set[str] = set()
    deduped: dict[str, list[Row]] = {}
    removed = 0
    for source, rows in rows_by_source.items():
        kept: list[Row] = []
        for row in rows:
            key = _row_dedup_key(row)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(row)
        deduped[source] = kept
    return deduped, removed


def _write_jsonl(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json())
            f.write("\n")


def _row_stats(rows: list[Row]) -> dict:
    if not rows:
        return {
            "count": 0,
            "avg_turns": 0.0,
            "pct_with_tool_calls": 0.0,
            "pct_parsed_from_text": 0.0,
            "pct_args_raw": 0.0,
            "pct_orphan_tool_response": 0.0,
        }
    total_turns = sum(len(r.messages) for r in rows)
    with_tool_calls = sum(1 for r in rows if any(m.tool_calls for m in r.messages))
    parsed_from_text = sum(1 for r in rows if r.meta.get("parsed_from_text"))
    args_raw = sum(1 for r in rows if r.meta.get("args_raw"))
    orphan = sum(1 for r in rows if r.meta.get("orphan_tool_response"))
    n = len(rows)
    return {
        "count": n,
        "avg_turns": round(total_turns / n, 2),
        "pct_with_tool_calls": round(100 * with_tool_calls / n, 1),
        "pct_parsed_from_text": round(100 * parsed_from_text / n, 1),
        "pct_args_raw": round(100 * args_raw / n, 1),
        "pct_orphan_tool_response": round(100 * orphan / n, 1),
    }


def run_normalizers(
    cfg: AgentForgeConfig,
) -> tuple[dict[str, list[Row]], dict[str, NormalizationStats]]:
    """Network-touching step: instantiate each configured source's real
    Normalizer (no injected raw_examples -> hits the live HF dataset) and run
    it. Kept separate from `_process_rows` so the rest of the pipeline is
    testable against synthetic Row data with no network access.
    """
    normalization_stats: dict[str, NormalizationStats] = {}
    raw_rows_by_source: dict[str, list[Row]] = {}

    normalized_dir = Path("data/normalized")
    for source in cfg.data.sources:
        if source not in NORMALIZER_CLASSES:
            raise ValueError(
                f"Unknown source {source!r} in config data.sources; "
                f"known sources: {sorted(NORMALIZER_CLASSES)}"
            )
        normalizer = NORMALIZER_CLASSES[source]()
        rows, stats = normalizer.run()
        max_n = cfg.data.max_examples_per_source
        if max_n is not None and len(rows) > max_n:
            rows = rows[:max_n]
        normalization_stats[source] = stats
        raw_rows_by_source[source] = rows
        _write_jsonl(rows, normalized_dir / f"{source}.jsonl")
    return raw_rows_by_source, normalization_stats


def process_rows(
    cfg: AgentForgeConfig,
    raw_rows_by_source: dict[str, list[Row]],
    normalization_stats: dict[str, NormalizationStats],
    *,
    holdout_size: int = 750,
) -> dict:
    """Pure pipeline step (dedupe -> holdout split -> weight -> write):
    no network access, fully testable against synthetic `raw_rows_by_source`.
    Writes manifest.jsonl/holdout.jsonl/manifest_stats.json per `cfg.data`
    and returns the stats dict.
    """
    deduped_rows_by_source, dedup_removed = _dedupe_across_sources(raw_rows_by_source)

    train_pool_by_source, holdout_by_source = stratified_holdout_split(
        deduped_rows_by_source, holdout_size=holdout_size, seed=cfg.seed
    )

    weighted_by_source = apply_weights_by_source(
        train_pool_by_source, cfg.data.sources, seed=cfg.seed
    )

    all_train_rows: list[Row] = [row for rows in weighted_by_source.values() for row in rows]
    all_holdout_rows: list[Row] = [row for rows in holdout_by_source.values() for row in rows]

    if cfg.data.shuffle:
        import random

        random.Random(cfg.seed).shuffle(all_train_rows)
        random.Random(cfg.seed + 1).shuffle(all_holdout_rows)

    manifest_path = Path(cfg.data.manifest_path)
    holdout_path = Path(cfg.data.holdout_path)
    _write_jsonl(all_train_rows, manifest_path)
    _write_jsonl(all_holdout_rows, holdout_path)

    stats = {
        "sources": {
            source: {
                "raw": normalization_stats[source].total_raw,
                "emitted": normalization_stats[source].emitted,
                "dropped": normalization_stats[source].dropped,
                "drop_reasons": dict(normalization_stats[source].drop_reasons),
                "post_dedup": len(deduped_rows_by_source.get(source, [])),
                "holdout": len(holdout_by_source.get(source, [])),
                "post_weight_in_manifest": len(weighted_by_source.get(source, [])),
                "configured_weight": cfg.data.sources.get(source, 0.0),
            }
            for source in cfg.data.sources
        },
        "dedup_removed_total": dedup_removed,
        "manifest_total_rows": len(all_train_rows),
        "holdout_total_rows": len(all_holdout_rows),
        "manifest": _row_stats(all_train_rows),
        "holdout": _row_stats(all_holdout_rows),
    }

    stats_path = Path("data/manifest_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def build_manifest(cfg: AgentForgeConfig) -> dict:
    """Full pipeline entrypoint: run the real (network-touching) normalizers,
    then process the results. See `run_normalizers`/`process_rows` for the
    testable split.
    """
    raw_rows_by_source, normalization_stats = run_normalizers(cfg)
    return process_rows(cfg, raw_rows_by_source, normalization_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a training YAML config")
    args = parser.parse_args()

    cfg = AgentForgeConfig.from_yaml(args.config)
    build_manifest(cfg)


if __name__ == "__main__":
    main()
