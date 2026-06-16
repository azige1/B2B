from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def allocate_integer_plan(
    plan_qty: float,
    item_ids: Sequence[str],
    demand_scores: Sequence[float],
) -> dict[str, int]:
    if len(item_ids) != len(demand_scores):
        raise ValueError("item_ids and demand_scores must have the same length.")
    if not item_ids:
        return {}

    total_qty = max(int(round(float(plan_qty))), 0)
    if total_qty == 0:
        return {str(item_id): 0 for item_id in item_ids}

    scores = np.asarray(demand_scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, 0.0, None)
    if float(scores.sum()) <= 0:
        scores = np.ones(len(item_ids), dtype=float)

    raw = scores / scores.sum() * total_qty
    allocated = np.floor(raw).astype(int)
    remainder = total_qty - int(allocated.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocated), kind="stable")
        allocated[order[:remainder]] += 1

    return {
        str(item_id): int(qty)
        for item_id, qty in zip(item_ids, allocated)
    }
