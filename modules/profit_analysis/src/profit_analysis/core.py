from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite, sqrt
from typing import Iterable, Sequence


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    text = str(value).strip()
    if text in {"", "<NA>", "nan", "NaT", "None"}:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(text).date()


def _clip_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _non_negative(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(numeric):
        return 0.0
    return max(numeric, 0.0)


def _maybe_non_negative(value: float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "<NA>", "nan", "NaT", "None"}:
        return None
    numeric = _non_negative(value)
    return numeric


def _round_to_batch(qty: float, batch_qty: float | None) -> float:
    qty = _non_negative(qty)
    if batch_qty is None or float(batch_qty) <= 0:
        return float(round(qty))
    batch = float(batch_qty)
    return float(round(qty / batch) * batch)


@dataclass(frozen=True)
class ModelOutput:
    sku_id: str
    snapshot_date: date | str
    pred_prob_positive: float
    pred_qty_30d: float
    prediction_version: str | None = None

    def normalized(self) -> "ModelOutput":
        return ModelOutput(
            sku_id=str(self.sku_id),
            snapshot_date=_coerce_date(self.snapshot_date) or date.today(),
            pred_prob_positive=_clip_probability(self.pred_prob_positive),
            pred_qty_30d=_non_negative(self.pred_qty_30d),
            prediction_version=self.prediction_version,
        )


@dataclass(frozen=True)
class InventoryState:
    sku_id: str
    snapshot_date: date | str
    current_inventory: float
    inbound_within_30d: float = 0.0
    lead_time_days: int = 0
    min_batch_qty: float | None = None
    max_replenish_qty: float | None = None
    safety_stock_qty: float | None = None
    last_decision_date: date | str | None = None

    def normalized(self) -> "InventoryState":
        return InventoryState(
            sku_id=str(self.sku_id),
            snapshot_date=_coerce_date(self.snapshot_date) or date.today(),
            current_inventory=_non_negative(self.current_inventory),
            inbound_within_30d=_non_negative(self.inbound_within_30d),
            lead_time_days=max(int(self.lead_time_days), 0),
            min_batch_qty=_maybe_non_negative(self.min_batch_qty),
            max_replenish_qty=_maybe_non_negative(self.max_replenish_qty),
            safety_stock_qty=_maybe_non_negative(self.safety_stock_qty),
            last_decision_date=_coerce_date(self.last_decision_date),
        )


@dataclass(frozen=True)
class Economics:
    sku_id: str
    unit_cost: float
    unit_price: float
    holding_cost_per_unit_per_day: float
    salvage_value_per_unit: float
    stockout_penalty_per_unit: float = 0.0
    other_fixed_cost: float = 0.0
    lifecycle_end_date: date | str | None = None

    def normalized(self) -> "Economics":
        return Economics(
            sku_id=str(self.sku_id),
            unit_cost=_non_negative(self.unit_cost),
            unit_price=_non_negative(self.unit_price),
            holding_cost_per_unit_per_day=_non_negative(self.holding_cost_per_unit_per_day),
            salvage_value_per_unit=_non_negative(self.salvage_value_per_unit),
            stockout_penalty_per_unit=_non_negative(self.stockout_penalty_per_unit),
            other_fixed_cost=_non_negative(self.other_fixed_cost),
            lifecycle_end_date=_coerce_date(self.lifecycle_end_date),
        )


@dataclass(frozen=True)
class CandidatePlan:
    plan_qty: float
    arrival_day: date | str | None = None
    policy: str | None = None

    def normalized(self, default_arrival_day: date) -> "CandidatePlan":
        arrival_day = _coerce_date(self.arrival_day) or default_arrival_day
        return CandidatePlan(
            plan_qty=_non_negative(self.plan_qty),
            arrival_day=arrival_day,
            policy=self.policy,
        )


@dataclass(frozen=True)
class DemandScenario:
    name: str
    demand_qty: float
    probability: float

    def normalized(self) -> "DemandScenario":
        return DemandScenario(
            name=str(self.name),
            demand_qty=_non_negative(self.demand_qty),
            probability=_clip_probability(self.probability),
        )


@dataclass(frozen=True)
class ProfitAssessment:
    sku_id: str
    plan_qty: float
    expected_profit: float
    profit_variance: float
    stockout_rate: float
    expected_sold_qty: float
    expected_leftover_qty: float
    expected_lost_sales_qty: float
    sell_through_rate: float
    scenario_breakdown: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def build_default_demand_scenarios(
    model_output: ModelOutput,
    positive_multipliers: Sequence[float] = (0.6, 1.0, 1.5),
    positive_weights: Sequence[float] = (0.25, 0.50, 0.25),
) -> list[DemandScenario]:
    model_output = model_output.normalized()
    if len(positive_multipliers) != len(positive_weights):
        raise ValueError("positive_multipliers and positive_weights must have the same length.")

    positive_weight_sum = float(sum(positive_weights))
    if positive_weight_sum <= 0:
        raise ValueError("positive_weights must sum to a positive value.")

    scenarios = [
        DemandScenario(name="zero", demand_qty=0.0, probability=1.0 - model_output.pred_prob_positive)
    ]
    for idx, (multiplier, weight) in enumerate(zip(positive_multipliers, positive_weights), start=1):
        scenarios.append(
            DemandScenario(
                name=f"positive_{idx}",
                demand_qty=model_output.pred_qty_30d * float(multiplier),
                probability=model_output.pred_prob_positive * (float(weight) / positive_weight_sum),
            )
        )
    return _normalize_scenarios(scenarios)


def _normalize_scenarios(scenarios: Iterable[DemandScenario]) -> list[DemandScenario]:
    rows = [scenario.normalized() for scenario in scenarios]
    total_prob = sum(row.probability for row in rows)
    if total_prob <= 0:
        raise ValueError("Scenario probabilities must sum to a positive value.")
    return [
        DemandScenario(name=row.name, demand_qty=row.demand_qty, probability=row.probability / total_prob)
        for row in rows
    ]


def _estimate_arrival_day(snapshot_date: date, lead_time_days: int) -> date:
    return date.fromordinal(snapshot_date.toordinal() + max(int(lead_time_days), 0))


def _simulate_scenario(
    demand_qty: float,
    inventory_state: InventoryState,
    economics: Economics,
    plan: CandidatePlan,
    horizon_days: int,
) -> dict:
    snapshot_date = _coerce_date(inventory_state.snapshot_date) or date.today()
    arrival_day = _coerce_date(plan.arrival_day) or snapshot_date
    arrival_offset_days = max((arrival_day - snapshot_date).days, 0)
    arrival_offset_days = min(arrival_offset_days, horizon_days)

    demand_before_arrival = demand_qty * (arrival_offset_days / max(horizon_days, 1))
    demand_after_arrival = demand_qty - demand_before_arrival

    current_inventory = inventory_state.current_inventory + inventory_state.inbound_within_30d
    sold_before_arrival = min(current_inventory, demand_before_arrival)
    inventory_after_arrival = max(current_inventory - demand_before_arrival, 0.0) + plan.plan_qty
    sold_after_arrival = min(inventory_after_arrival, demand_after_arrival)

    sold_qty = sold_before_arrival + sold_after_arrival
    leftover_qty = max(inventory_after_arrival - sold_after_arrival, 0.0)
    lost_sales_qty = max(demand_qty - sold_qty, 0.0)

    active_days_after_arrival = max(horizon_days - arrival_offset_days, 0)
    avg_days_held = active_days_after_arrival / 2.0

    sales_revenue = sold_qty * economics.unit_price
    salvage_revenue = leftover_qty * economics.salvage_value_per_unit
    replenish_cost = plan.plan_qty * economics.unit_cost
    holding_cost = leftover_qty * economics.holding_cost_per_unit_per_day * avg_days_held
    stockout_cost = lost_sales_qty * economics.stockout_penalty_per_unit
    profit = (
        sales_revenue
        + salvage_revenue
        - replenish_cost
        - holding_cost
        - stockout_cost
        - economics.other_fixed_cost
    )

    return {
        "arrival_offset_days": float(arrival_offset_days),
        "demand_qty": float(demand_qty),
        "sold_qty": float(sold_qty),
        "leftover_qty": float(leftover_qty),
        "lost_sales_qty": float(lost_sales_qty),
        "sales_revenue": float(sales_revenue),
        "salvage_revenue": float(salvage_revenue),
        "replenish_cost": float(replenish_cost),
        "holding_cost": float(holding_cost),
        "stockout_cost": float(stockout_cost),
        "profit": float(profit),
    }


def assess_replenishment_plan(
    model_output: ModelOutput,
    inventory_state: InventoryState,
    economics: Economics,
    plan: CandidatePlan,
    demand_scenarios: Sequence[DemandScenario] | None = None,
    horizon_days: int = 30,
) -> ProfitAssessment:
    model_output = model_output.normalized()
    inventory_state = inventory_state.normalized()
    economics = economics.normalized()

    arrival_day = _estimate_arrival_day(
        snapshot_date=_coerce_date(inventory_state.snapshot_date) or date.today(),
        lead_time_days=inventory_state.lead_time_days,
    )
    plan_qty = _round_to_batch(plan.plan_qty, inventory_state.min_batch_qty)
    if inventory_state.max_replenish_qty is not None:
        plan_qty = min(plan_qty, inventory_state.max_replenish_qty)
    normalized_plan = CandidatePlan(
        plan_qty=plan_qty,
        arrival_day=plan.arrival_day or arrival_day,
        policy=plan.policy,
    ).normalized(default_arrival_day=arrival_day)

    scenarios = _normalize_scenarios(
        demand_scenarios if demand_scenarios is not None else build_default_demand_scenarios(model_output)
    )

    scenario_breakdown: list[dict] = []
    expected_profit = 0.0
    expected_profit_sq = 0.0
    expected_sold_qty = 0.0
    expected_leftover_qty = 0.0
    expected_lost_sales_qty = 0.0
    stockout_rate = 0.0

    for scenario in scenarios:
        result = _simulate_scenario(
            demand_qty=scenario.demand_qty,
            inventory_state=inventory_state,
            economics=economics,
            plan=normalized_plan,
            horizon_days=horizon_days,
        )
        prob = scenario.probability
        expected_profit += prob * result["profit"]
        expected_profit_sq += prob * (result["profit"] ** 2)
        expected_sold_qty += prob * result["sold_qty"]
        expected_leftover_qty += prob * result["leftover_qty"]
        expected_lost_sales_qty += prob * result["lost_sales_qty"]
        stockout_rate += prob * float(result["lost_sales_qty"] > 0)
        scenario_breakdown.append(
            {
                "name": scenario.name,
                "probability": prob,
                **result,
            }
        )

    total_available = inventory_state.current_inventory + inventory_state.inbound_within_30d + normalized_plan.plan_qty
    sell_through_rate = expected_sold_qty / max(total_available, 1e-9)
    profit_variance = max(expected_profit_sq - (expected_profit ** 2), 0.0)

    return ProfitAssessment(
        sku_id=model_output.sku_id,
        plan_qty=float(normalized_plan.plan_qty),
        expected_profit=float(expected_profit),
        profit_variance=float(profit_variance),
        stockout_rate=float(stockout_rate),
        expected_sold_qty=float(expected_sold_qty),
        expected_leftover_qty=float(expected_leftover_qty),
        expected_lost_sales_qty=float(expected_lost_sales_qty),
        sell_through_rate=float(sell_through_rate),
        scenario_breakdown=scenario_breakdown,
    )


def build_default_candidate_plans(
    model_output: ModelOutput,
    inventory_state: InventoryState,
    policy: str = "balanced",
) -> list[CandidatePlan]:
    model_output = model_output.normalized()
    inventory_state = inventory_state.normalized()

    gap_qty = max(model_output.pred_qty_30d - inventory_state.current_inventory - inventory_state.inbound_within_30d, 0.0)
    raw_candidates = [
        0.0,
        inventory_state.min_batch_qty or 0.0,
        gap_qty,
        0.8 * model_output.pred_qty_30d,
        1.0 * model_output.pred_qty_30d,
        1.2 * model_output.pred_qty_30d,
        1.5 * model_output.pred_qty_30d,
    ]

    if inventory_state.safety_stock_qty is not None:
        raw_candidates.append(max(gap_qty, inventory_state.safety_stock_qty))

    arrival_day = _estimate_arrival_day(
        snapshot_date=_coerce_date(inventory_state.snapshot_date) or date.today(),
        lead_time_days=inventory_state.lead_time_days,
    )
    deduped = []
    seen = set()
    for qty in raw_candidates:
        normalized_qty = _round_to_batch(qty, inventory_state.min_batch_qty)
        if inventory_state.max_replenish_qty is not None:
            normalized_qty = min(normalized_qty, inventory_state.max_replenish_qty)
        key = round(float(normalized_qty), 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(CandidatePlan(plan_qty=float(normalized_qty), arrival_day=arrival_day, policy=policy))
    return deduped


def recommend_replenishment_plans(
    model_output: ModelOutput,
    inventory_state: InventoryState,
    economics: Economics,
    policy: str = "balanced",
    demand_scenarios: Sequence[DemandScenario] | None = None,
) -> dict:
    candidates = build_default_candidate_plans(model_output, inventory_state, policy=policy)
    assessments = [
        assess_replenishment_plan(
            model_output=model_output,
            inventory_state=inventory_state,
            economics=economics,
            plan=plan,
            demand_scenarios=demand_scenarios,
        )
        for plan in candidates
    ]

    def _score(assessment: ProfitAssessment) -> float:
        risk_penalty = sqrt(max(assessment.profit_variance, 0.0))
        if policy == "conservative":
            return assessment.expected_profit - 0.50 * risk_penalty - 0.50 * assessment.expected_leftover_qty
        if policy == "aggressive":
            return assessment.expected_profit - 0.10 * risk_penalty
        return assessment.expected_profit - 0.25 * risk_penalty - 0.15 * assessment.expected_leftover_qty

    ranked = sorted(assessments, key=_score, reverse=True)
    lowest_risk = min(ranked, key=lambda item: (item.profit_variance, item.stockout_rate, item.expected_leftover_qty))
    best_profit = max(ranked, key=lambda item: item.expected_profit)

    return {
        "sku_id": model_output.normalized().sku_id,
        "policy": policy,
        "ranked_candidates": [assessment.to_dict() for assessment in ranked],
        "best_balanced_plan": ranked[0].to_dict() if ranked else None,
        "best_profit_plan": best_profit.to_dict() if ranked else None,
        "lowest_risk_plan": lowest_risk.to_dict() if ranked else None,
    }
