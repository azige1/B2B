from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupDemandOutput:
    probability_positive: float
    conditional_qty: float
    expected_qty: float


def aggregate_hurdle_outputs(
    probabilities: Sequence[float],
    conditional_quantities: Sequence[float],
) -> GroupDemandOutput:
    """Aggregate SKU hurdle outputs into one SKC-level demand output.

    The positive-probability aggregation assumes conditional independence between
    SKU demand events. Expected quantity remains additive across SKUs.
    """
    if len(probabilities) != len(conditional_quantities):
        raise ValueError(
            "probabilities and conditional_quantities must have the same length."
        )
    if len(probabilities) == 0:
        return GroupDemandOutput(0.0, 0.0, 0.0)

    probability = np.asarray(probabilities, dtype=float)
    quantity = np.asarray(conditional_quantities, dtype=float)
    probability = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    quantity = np.nan_to_num(quantity, nan=0.0, posinf=0.0, neginf=0.0)
    probability = np.clip(probability, 0.0, 1.0)
    quantity = np.clip(quantity, 0.0, None)

    group_probability = float(1.0 - np.prod(1.0 - probability))
    expected_qty = float(np.sum(probability * quantity))
    conditional_qty = (
        expected_qty / group_probability
        if group_probability > 1e-12
        else float(quantity.sum())
    )
    return GroupDemandOutput(
        probability_positive=group_probability,
        conditional_qty=conditional_qty,
        expected_qty=expected_qty,
    )


def build_item_demand_gap_scores(
    probabilities: Sequence[float],
    conditional_quantities: Sequence[float],
    current_inventory: Sequence[float],
    positive_multiplier_mean: float = 1.0,
) -> list[float]:
    if not (
        len(probabilities)
        == len(conditional_quantities)
        == len(current_inventory)
    ):
        raise ValueError(
            "probabilities, conditional_quantities, and current_inventory "
            "must have the same length."
        )

    probability = np.clip(
        np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.0),
        0.0,
        1.0,
    )
    quantity = np.clip(
        np.nan_to_num(np.asarray(conditional_quantities, dtype=float), nan=0.0),
        0.0,
        None,
    )
    inventory = np.clip(
        np.nan_to_num(np.asarray(current_inventory, dtype=float), nan=0.0),
        0.0,
        None,
    )
    multiplier = max(float(positive_multiplier_mean), 0.0)
    return np.maximum(probability * quantity * multiplier - inventory, 0.0).tolist()
