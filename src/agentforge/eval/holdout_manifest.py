"""Stratified held-out split, carved off *before* per-source sampling weights
are applied -- so weighting (which can oversample with replacement) can never
cause a row that's in the holdout set to also appear in the train manifest.
"""

from __future__ import annotations

import random

from agentforge.data.schema import Row


def stratified_holdout_split(
    rows_by_source: dict[str, list[Row]],
    *,
    holdout_size: int,
    seed: int,
) -> tuple[dict[str, list[Row]], dict[str, list[Row]]]:
    """Split each source's row pool into (remaining_for_train, holdout),
    proportional to that source's share of the total deduped pool, targeting
    `holdout_size` total holdout rows (best-effort -- small sources may
    contribute fewer than their exact proportional share if they don't have
    enough rows).

    Every source with at least one row contributes at least one holdout row
    (if `holdout_size > 0`), so no source is entirely absent from eval.
    """
    rng = random.Random(seed)
    total_rows = sum(len(rows) for rows in rows_by_source.values())
    if total_rows == 0 or holdout_size <= 0:
        return rows_by_source, {source: [] for source in rows_by_source}

    remaining: dict[str, list[Row]] = {}
    holdout: dict[str, list[Row]] = {}
    for source, rows in rows_by_source.items():
        if not rows:
            remaining[source] = []
            holdout[source] = []
            continue
        share = len(rows) / total_rows
        n_holdout = min(len(rows), max(1, round(holdout_size * share)))
        shuffled = rows[:]
        rng.shuffle(shuffled)
        holdout[source] = shuffled[:n_holdout]
        remaining[source] = shuffled[n_holdout:]
    return remaining, holdout
