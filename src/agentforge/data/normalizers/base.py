"""Shared contract every per-dataset normalizer implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agentforge.data.schema import Row


@dataclass
class NormalizationStats:
    """Per-source counters written into manifest_stats.json by build_manifest.py."""

    source: str
    total_raw: int = 0
    emitted: int = 0
    dropped: int = 0
    drop_reasons: Counter = field(default_factory=Counter)

    def record_drop(self, reason: str) -> None:
        self.dropped += 1
        self.drop_reasons[reason] += 1


class Normalizer(ABC):
    """Base class for a per-dataset normalizer.

    Subclasses implement `iter_raw()` to yield native-format examples and
    `to_canonical(raw)` to map one such example to a `Row` (or `None` to drop
    it). `run()` wires the two together, schema-validates every emitted Row,
    and never lets an invalid Row through silently -- it's dropped and
    counted, not raised past this boundary, so one bad row can't take down a
    full manifest build.
    """

    source: str

    @abstractmethod
    def iter_raw(self) -> Iterator[Any]:
        """Yield native-format raw examples from this dataset's source files/split."""

    @abstractmethod
    def to_canonical(self, raw: Any) -> Row | None:
        """Map one raw example to a canonical Row, or None to drop it (log the reason)."""

    def run(self) -> tuple[list[Row], NormalizationStats]:
        stats = NormalizationStats(source=self.source)
        rows: list[Row] = []
        for raw in self.iter_raw():
            stats.total_raw += 1
            try:
                row = self.to_canonical(raw)
            except Exception as exc:  # noqa: BLE001 - a malformed source row must never abort the build
                stats.record_drop(f"to_canonical_exception:{type(exc).__name__}")
                continue
            if row is None:
                stats.record_drop("to_canonical_returned_none")
                continue
            try:
                # Re-validate: to_canonical may have constructed the Row without
                # pydantic re-checking cross-field invariants in some code paths.
                Row.model_validate(row.model_dump())
            except ValidationError:
                stats.record_drop("schema_validation_failed")
                continue
            rows.append(row)
            stats.emitted += 1
        return rows, stats
