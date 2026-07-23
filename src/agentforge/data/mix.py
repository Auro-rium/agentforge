"""Per-source sampling-weight application for the combined training manifest.

Weights are relative-oversampling factors, applied per source: a weight of
1.0 keeps a source's rows as-is (once each); a weight of 3.0 means "make this
source's contribution to the final manifest ~3x its own row count" (sampled
with replacement if that exceeds the source's actual row count, since e.g.
agent_flan is deliberately upweighted well past its natural size to make it
the dominant training signal post-pivot -- see docs/plan). This is applied
per-source *after* each source's own holdout carve-out (see
eval/holdout_manifest.py), so weighting never causes a holdout row to also
appear (via replacement) in the train manifest.
"""

from __future__ import annotations

import random

from agentforge.data.schema import Row


def apply_source_weight(rows: list[Row], weight: float, *, rng: random.Random) -> list[Row]:
    """Return a list of `rows` resampled to approximately `weight * len(rows)`
    items. `weight <= 0` returns an empty list (source excluded entirely).
    Sampling is without replacement while the target count doesn't exceed the
    pool size, and with replacement (oversampling) once it does.
    """
    if weight <= 0 or not rows:
        return []
    target_count = round(weight * len(rows))
    if target_count <= 0:
        return []
    if target_count <= len(rows):
        return rng.sample(rows, target_count)
    # Oversampling: every row at least once, then top up with replacement so
    # the achieved multiplier is as close to `weight` as possible rather than
    # skewed toward whichever rows random.choices happens to favor early on.
    full_copies, remainder = divmod(target_count, len(rows))
    resampled = list(rows) * full_copies
    if remainder:
        resampled += rng.sample(rows, remainder)
    rng.shuffle(resampled)
    return resampled


def apply_weights_by_source(
    rows_by_source: dict[str, list[Row]],
    weights: dict[str, float],
    *,
    seed: int,
) -> dict[str, list[Row]]:
    """Apply `apply_source_weight` per source. Sources present in `rows_by_source`
    but absent from `weights` are excluded (treated as weight 0) rather than
    silently defaulting to 1.0 -- an omitted source in the config is assumed
    deliberate, not an oversight to paper over.
    """
    rng = random.Random(seed)
    return {
        source: apply_source_weight(rows, weights.get(source, 0.0), rng=rng)
        for source, rows in rows_by_source.items()
    }
