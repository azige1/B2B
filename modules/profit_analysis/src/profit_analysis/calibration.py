from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .core import DemandScenario, ModelOutput, build_default_demand_scenarios


@dataclass(frozen=True)
class DemandScenarioCalibration:
    probability_x: tuple[float, ...]
    probability_y: tuple[float, ...]
    positive_multipliers: tuple[float, ...]
    positive_weights: tuple[float, ...]
    calibration_rows: int
    positive_calibration_rows: int

    @classmethod
    def from_dict(cls, payload: dict) -> "DemandScenarioCalibration":
        if "calibration" in payload:
            payload = payload["calibration"]
        required = [
            "probability_x",
            "probability_y",
            "positive_multipliers",
            "positive_weights",
            "calibration_rows",
            "positive_calibration_rows",
        ]
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"calibration payload missing required fields: {missing}")
        if len(payload["probability_x"]) != len(payload["probability_y"]):
            raise ValueError("probability_x and probability_y must have the same length.")
        if len(payload["positive_multipliers"]) != len(payload["positive_weights"]):
            raise ValueError(
                "positive_multipliers and positive_weights must have the same length."
            )
        return cls(
            probability_x=tuple(float(value) for value in payload["probability_x"]),
            probability_y=tuple(float(value) for value in payload["probability_y"]),
            positive_multipliers=tuple(
                float(value) for value in payload["positive_multipliers"]
            ),
            positive_weights=tuple(float(value) for value in payload["positive_weights"]),
            calibration_rows=int(payload["calibration_rows"]),
            positive_calibration_rows=int(payload["positive_calibration_rows"]),
        )

    def calibrate_probability(self, probability: float) -> float:
        value = min(max(float(probability), 0.0), 1.0)
        return float(np.interp(value, self.probability_x, self.probability_y))

    def build_scenarios(
        self,
        model_output: ModelOutput,
        horizon_days: int,
    ) -> list[DemandScenario]:
        normalized = model_output.normalized()
        calibrated = ModelOutput(
            sku_id=normalized.sku_id,
            snapshot_date=normalized.snapshot_date,
            pred_prob_positive=self.calibrate_probability(normalized.pred_prob_positive),
            pred_qty_30d=normalized.pred_qty_30d,
            prediction_version=normalized.prediction_version,
        )
        return build_default_demand_scenarios(
            calibrated,
            positive_multipliers=self.positive_multipliers,
            positive_weights=self.positive_weights,
            horizon_days=horizon_days,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        for field in [
            "probability_x",
            "probability_y",
            "positive_multipliers",
            "positive_weights",
        ]:
            payload[field] = list(payload[field])
        return payload


def load_demand_scenario_calibration(
    path: str | Path,
) -> DemandScenarioCalibration:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DemandScenarioCalibration.from_dict(payload)


def probability_calibration_metrics(
    calibration_df: pd.DataFrame,
    calibration: DemandScenarioCalibration,
    actual_col: str = "true_replenish_qty",
    probability_col: str = "ai_pred_prob",
) -> dict[str, float | int]:
    required = [actual_col, probability_col]
    missing = [col for col in required if col not in calibration_df.columns]
    if missing:
        raise ValueError(f"calibration metric input missing required columns: {missing}")

    work = calibration_df.loc[:, required].copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=required)
    if work.empty:
        raise ValueError("calibration metric input has no valid rows.")

    actual = (work[actual_col].to_numpy(dtype=float) > 0).astype(float)
    raw = work[probability_col].clip(0.0, 1.0).to_numpy(dtype=float)
    calibrated = np.asarray(
        [calibration.calibrate_probability(value) for value in raw],
        dtype=float,
    )
    return {
        "rows": int(len(work)),
        "actual_positive_rate": float(actual.mean()),
        "raw_mean_probability": float(raw.mean()),
        "calibrated_mean_probability": float(calibrated.mean()),
        "raw_brier_score": float(np.mean((raw - actual) ** 2)),
        "calibrated_brier_score": float(np.mean((calibrated - actual) ** 2)),
    }


def fit_demand_scenario_calibration(
    calibration_df: pd.DataFrame,
    actual_col: str = "true_replenish_qty",
    probability_col: str = "ai_pred_prob",
    conditional_qty_col: str = "ai_pred_qty_open",
    multiplier_quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    positive_weights: Sequence[float] = (0.25, 0.50, 0.25),
    multiplier_floor: float = 0.10,
    multiplier_cap: float = 5.00,
) -> DemandScenarioCalibration:
    required = [actual_col, probability_col, conditional_qty_col]
    missing = [col for col in required if col not in calibration_df.columns]
    if missing:
        raise ValueError(f"calibration input missing required columns: {missing}")
    if len(multiplier_quantiles) != len(positive_weights):
        raise ValueError("multiplier_quantiles and positive_weights must have the same length.")

    work = calibration_df.loc[:, required].copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=required)
    if work.empty:
        raise ValueError("calibration input has no valid rows.")

    raw_probability = work[probability_col].clip(0.0, 1.0).to_numpy(dtype=float)
    actual_positive = (work[actual_col].to_numpy(dtype=float) > 0).astype(int)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(raw_probability, actual_positive)

    positive = work[
        (work[actual_col] > 0)
        & (work[conditional_qty_col] > 0)
    ].copy()
    if positive.empty:
        raise ValueError("calibration input has no positive rows with positive conditional quantity.")
    ratio = (
        positive[actual_col] / positive[conditional_qty_col]
    ).clip(lower=float(multiplier_floor), upper=float(multiplier_cap))
    multipliers = tuple(float(value) for value in ratio.quantile(multiplier_quantiles).tolist())
    weights = tuple(float(value) for value in positive_weights)

    return DemandScenarioCalibration(
        probability_x=tuple(float(value) for value in isotonic.X_thresholds_),
        probability_y=tuple(float(value) for value in isotonic.y_thresholds_),
        positive_multipliers=multipliers,
        positive_weights=weights,
        calibration_rows=int(len(work)),
        positive_calibration_rows=int(len(positive)),
    )
