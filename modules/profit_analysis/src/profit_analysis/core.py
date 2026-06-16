from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import ceil, isfinite, sqrt
from typing import Iterable, Sequence


DEFAULT_HORIZON_DAYS = 45
DEFAULT_QTY_HORIZON_DAYS = 30
DEFAULT_TARGET_SELL_THROUGH_RATE = 0.85


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


def _coerce_int(value: int | float | str | None, default: int = 0, minimum: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"", "<NA>", "nan", "NaT", "None"}:
        return default
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(numeric, minimum)


def _round_to_batch(qty: float, min_batch_qty: float | None, increment_batch_qty: float | None = None) -> float:
    qty = _non_negative(qty)
    if qty <= 0:
        return 0.0

    min_qty = _maybe_non_negative(min_batch_qty) or 0.0
    increment_qty = _maybe_non_negative(increment_batch_qty)
    if increment_qty is None or increment_qty <= 0:
        increment_qty = min_qty if min_qty > 0 else None

    qty = max(qty, min_qty)
    if increment_qty is None or increment_qty <= 0:
        return float(ceil(qty))
    return float(ceil(qty / increment_qty) * increment_qty)


def _scale_30d_qty_to_horizon(pred_qty_30d: float, horizon_days: int) -> float:
    horizon = max(int(horizon_days), 0)
    if horizon <= 0:
        return 0.0
    return _non_negative(pred_qty_30d) * (horizon / DEFAULT_QTY_HORIZON_DAYS)


def _remaining_lifecycle_days(snapshot_date: date, lifecycle_end_date: date | None) -> int | None:
    if lifecycle_end_date is None:
        return None
    return max((lifecycle_end_date - snapshot_date).days + 1, 0)


def _resolve_effective_horizon_days(
    snapshot_date: date,
    economics: "Economics",
    requested_horizon_days: int | float | str | None,
) -> tuple[int, int | None]:
    base_horizon_days = _coerce_int(
        requested_horizon_days,
        default=economics.lifecycle_days,
        minimum=0,
    )
    remaining_days = _remaining_lifecycle_days(snapshot_date, economics.lifecycle_end_date)
    if remaining_days is None:
        return max(base_horizon_days, 1), None
    return min(base_horizon_days, remaining_days), remaining_days


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
    increment_batch_qty: float | None = None
    max_replenish_qty: float | None = None
    safety_stock_qty: float | None = None
    last_decision_date: date | str | None = None

    def normalized(self) -> "InventoryState":
        return InventoryState(
            sku_id=str(self.sku_id),
            snapshot_date=_coerce_date(self.snapshot_date) or date.today(),
            current_inventory=_non_negative(self.current_inventory),
            inbound_within_30d=_non_negative(self.inbound_within_30d),
            lead_time_days=_coerce_int(self.lead_time_days, default=0, minimum=0),
            min_batch_qty=_maybe_non_negative(self.min_batch_qty),
            increment_batch_qty=_maybe_non_negative(self.increment_batch_qty),
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
    target_sell_through_rate: float = DEFAULT_TARGET_SELL_THROUGH_RATE
    lifecycle_days: int = DEFAULT_HORIZON_DAYS

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
            target_sell_through_rate=_clip_probability(self.target_sell_through_rate),
            lifecycle_days=_coerce_int(self.lifecycle_days, default=DEFAULT_HORIZON_DAYS, minimum=1),
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
    expected_supply_qty: float
    expected_sales_revenue: float
    expected_terminal_value: float
    expected_replenish_cost: float
    expected_holding_cost: float
    expected_stockout_cost: float
    expected_total_cost: float
    profit_positive_probability: float
    sell_through_target_probability: float
    effective_horizon_days: int
    remaining_lifecycle_days: int | None
    late_arrival_risk: int
    scenario_breakdown: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RealizedPlanResult:
    sku_id: str
    plan_qty: float
    actual_demand_qty: float
    realized_profit: float
    sold_qty: float
    leftover_qty: float
    lost_sales_qty: float
    stockout_flag: int
    sell_through_rate: float
    arrival_offset_days: float
    sales_revenue: float
    terminal_value: float
    replenish_cost: float
    holding_cost: float
    stockout_cost: float
    total_cost: float
    effective_horizon_days: int
    remaining_lifecycle_days: int | None
    late_arrival_risk: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_default_demand_scenarios(
    model_output: ModelOutput,
    positive_multipliers: Sequence[float] = (0.6, 1.0, 1.5),
    positive_weights: Sequence[float] = (0.25, 0.50, 0.25),
    horizon_days: int = DEFAULT_HORIZON_DAYS,
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
    base_demand_qty = _scale_30d_qty_to_horizon(model_output.pred_qty_30d, horizon_days)
    for idx, (multiplier, weight) in enumerate(zip(positive_multipliers, positive_weights), start=1):
        scenarios.append(
            DemandScenario(
                name=f"positive_{idx}",
                demand_qty=base_demand_qty * float(multiplier),
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
    daily_demand_curve: Sequence[float] | None = None,
) -> dict:
    snapshot_date = _coerce_date(inventory_state.snapshot_date) or date.today()
    arrival_day = _coerce_date(plan.arrival_day) or snapshot_date
    horizon_days = max(int(horizon_days), 0)
    arrival_offset_days = max((arrival_day - snapshot_date).days, 0)
    arrival_offset_days = min(arrival_offset_days, horizon_days)

    inventory_qty = inventory_state.current_inventory + inventory_state.inbound_within_30d
    demand_qty = _non_negative(demand_qty)
    if daily_demand_curve is None:
        daily_demands = [
            demand_qty / horizon_days if horizon_days > 0 else 0.0
            for _ in range(horizon_days)
        ]
    else:
        daily_demands = [_non_negative(value) for value in daily_demand_curve[:horizon_days]]
        if len(daily_demands) < horizon_days:
            daily_demands.extend([0.0] * (horizon_days - len(daily_demands)))
        curve_total = sum(daily_demands)
        if curve_total > 0:
            scale = demand_qty / curve_total
            daily_demands = [value * scale for value in daily_demands]
        elif demand_qty > 0 and horizon_days > 0:
            daily_demands = [demand_qty / horizon_days for _ in range(horizon_days)]
    sold_qty = 0.0
    lost_sales_qty = 0.0
    holding_cost = 0.0
    production_arrived_flag = 0

    for day_idx in range(horizon_days):
        if day_idx == arrival_offset_days:
            inventory_qty += plan.plan_qty
            production_arrived_flag = 1

        day_demand_qty = daily_demands[day_idx]
        day_sold_qty = min(inventory_qty, day_demand_qty)
        sold_qty += day_sold_qty
        inventory_qty -= day_sold_qty
        lost_sales_qty += max(day_demand_qty - day_sold_qty, 0.0)
        holding_cost += inventory_qty * economics.holding_cost_per_unit_per_day

    leftover_qty = max(inventory_qty, 0.0)
    lost_sales_qty = max(demand_qty - sold_qty, 0.0)

    sales_revenue = sold_qty * economics.unit_price
    terminal_value = leftover_qty * economics.salvage_value_per_unit
    replenish_cost = plan.plan_qty * economics.unit_cost
    stockout_cost = lost_sales_qty * economics.stockout_penalty_per_unit
    total_cost = replenish_cost + holding_cost + stockout_cost + economics.other_fixed_cost
    profit = (
        sales_revenue
        + terminal_value
        - total_cost
    )

    return {
        "arrival_offset_days": float(arrival_offset_days),
        "production_arrived_flag": int(production_arrived_flag),
        "demand_qty": float(demand_qty),
        "sold_qty": float(sold_qty),
        "leftover_qty": float(leftover_qty),
        "lost_sales_qty": float(lost_sales_qty),
        "sales_revenue": float(sales_revenue),
        "terminal_value": float(terminal_value),
        "replenish_cost": float(replenish_cost),
        "holding_cost": float(holding_cost),
        "stockout_cost": float(stockout_cost),
        "total_cost": float(total_cost),
        "profit": float(profit),
    }


def assess_replenishment_plan(
    model_output: ModelOutput,
    inventory_state: InventoryState,
    economics: Economics,
    plan: CandidatePlan,
    demand_scenarios: Sequence[DemandScenario] | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> ProfitAssessment:
    model_output = model_output.normalized()
    inventory_state = inventory_state.normalized()
    economics = economics.normalized()
    snapshot_date = _coerce_date(inventory_state.snapshot_date) or date.today()
    horizon_days, remaining_lifecycle_days = _resolve_effective_horizon_days(
        snapshot_date=snapshot_date,
        economics=economics,
        requested_horizon_days=horizon_days,
    )

    arrival_day = _estimate_arrival_day(
        snapshot_date=snapshot_date,
        lead_time_days=inventory_state.lead_time_days,
    )
    plan_qty = _round_to_batch(plan.plan_qty, inventory_state.min_batch_qty, inventory_state.increment_batch_qty)
    if inventory_state.max_replenish_qty is not None:
        plan_qty = min(plan_qty, inventory_state.max_replenish_qty)
    normalized_plan = CandidatePlan(
        plan_qty=plan_qty,
        arrival_day=plan.arrival_day or arrival_day,
        policy=plan.policy,
    ).normalized(default_arrival_day=arrival_day)

    scenarios = _normalize_scenarios(
        demand_scenarios
        if demand_scenarios is not None
        else build_default_demand_scenarios(model_output, horizon_days=horizon_days)
    )

    scenario_breakdown: list[dict] = []
    expected_profit = 0.0
    expected_profit_sq = 0.0
    expected_sold_qty = 0.0
    expected_leftover_qty = 0.0
    expected_lost_sales_qty = 0.0
    expected_sales_revenue = 0.0
    expected_terminal_value = 0.0
    expected_replenish_cost = 0.0
    expected_holding_cost = 0.0
    expected_stockout_cost = 0.0
    expected_total_cost = 0.0
    profit_positive_probability = 0.0
    sell_through_target_probability = 0.0
    stockout_rate = 0.0

    total_available = inventory_state.current_inventory + inventory_state.inbound_within_30d + normalized_plan.plan_qty
    target_sell_through_rate = economics.target_sell_through_rate
    late_arrival_risk = int(
        normalized_plan.plan_qty > 0
        and economics.lifecycle_end_date is not None
        and normalized_plan.arrival_day > economics.lifecycle_end_date
    )

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
        expected_sales_revenue += prob * result["sales_revenue"]
        expected_terminal_value += prob * result["terminal_value"]
        expected_replenish_cost += prob * result["replenish_cost"]
        expected_holding_cost += prob * result["holding_cost"]
        expected_stockout_cost += prob * result["stockout_cost"]
        expected_total_cost += prob * result["total_cost"]
        stockout_rate += prob * float(result["lost_sales_qty"] > 0)
        scenario_sell_through_rate = result["sold_qty"] / max(total_available, 1e-9)
        profit_positive_probability += prob * float(result["profit"] > 0)
        sell_through_target_probability += prob * float(scenario_sell_through_rate >= target_sell_through_rate)
        scenario_breakdown.append(
            {
                "name": scenario.name,
                "probability": prob,
                "sell_through_rate": float(scenario_sell_through_rate),
                "stockout_flag": int(result["lost_sales_qty"] > 0),
                "profit_positive_flag": int(result["profit"] > 0),
                "sell_through_target_flag": int(scenario_sell_through_rate >= target_sell_through_rate),
                **result,
            }
        )

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
        expected_supply_qty=float(total_available),
        expected_sales_revenue=float(expected_sales_revenue),
        expected_terminal_value=float(expected_terminal_value),
        expected_replenish_cost=float(expected_replenish_cost),
        expected_holding_cost=float(expected_holding_cost),
        expected_stockout_cost=float(expected_stockout_cost),
        expected_total_cost=float(expected_total_cost),
        profit_positive_probability=float(profit_positive_probability),
        sell_through_target_probability=float(sell_through_target_probability),
        effective_horizon_days=int(horizon_days),
        remaining_lifecycle_days=remaining_lifecycle_days,
        late_arrival_risk=late_arrival_risk,
        scenario_breakdown=scenario_breakdown,
    )


def realize_replenishment_plan(
    model_output: ModelOutput,
    inventory_state: InventoryState,
    economics: Economics,
    plan: CandidatePlan,
    actual_demand_qty: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    actual_daily_demand_curve: Sequence[float] | None = None,
) -> RealizedPlanResult:
    model_output = model_output.normalized()
    inventory_state = inventory_state.normalized()
    economics = economics.normalized()
    snapshot_date = _coerce_date(inventory_state.snapshot_date) or date.today()
    horizon_days, remaining_lifecycle_days = _resolve_effective_horizon_days(
        snapshot_date=snapshot_date,
        economics=economics,
        requested_horizon_days=horizon_days,
    )

    arrival_day = _estimate_arrival_day(
        snapshot_date=snapshot_date,
        lead_time_days=inventory_state.lead_time_days,
    )
    plan_qty = _round_to_batch(plan.plan_qty, inventory_state.min_batch_qty, inventory_state.increment_batch_qty)
    if inventory_state.max_replenish_qty is not None:
        plan_qty = min(plan_qty, inventory_state.max_replenish_qty)
    normalized_plan = CandidatePlan(
        plan_qty=plan_qty,
        arrival_day=plan.arrival_day or arrival_day,
        policy=plan.policy,
    ).normalized(default_arrival_day=arrival_day)
    late_arrival_risk = int(
        normalized_plan.plan_qty > 0
        and economics.lifecycle_end_date is not None
        and normalized_plan.arrival_day > economics.lifecycle_end_date
    )

    result = _simulate_scenario(
        demand_qty=_non_negative(actual_demand_qty),
        inventory_state=inventory_state,
        economics=economics,
        plan=normalized_plan,
        horizon_days=horizon_days,
        daily_demand_curve=actual_daily_demand_curve,
    )
    total_available = inventory_state.current_inventory + inventory_state.inbound_within_30d + normalized_plan.plan_qty
    sell_through_rate = result["sold_qty"] / max(total_available, 1e-9)
    return RealizedPlanResult(
        sku_id=model_output.sku_id,
        plan_qty=float(normalized_plan.plan_qty),
        actual_demand_qty=float(_non_negative(actual_demand_qty)),
        realized_profit=float(result["profit"]),
        sold_qty=float(result["sold_qty"]),
        leftover_qty=float(result["leftover_qty"]),
        lost_sales_qty=float(result["lost_sales_qty"]),
        stockout_flag=int(result["lost_sales_qty"] > 0),
        sell_through_rate=float(sell_through_rate),
        arrival_offset_days=float(result["arrival_offset_days"]),
        sales_revenue=float(result["sales_revenue"]),
        terminal_value=float(result["terminal_value"]),
        replenish_cost=float(result["replenish_cost"]),
        holding_cost=float(result["holding_cost"]),
        stockout_cost=float(result["stockout_cost"]),
        total_cost=float(result["total_cost"]),
        effective_horizon_days=int(horizon_days),
        remaining_lifecycle_days=remaining_lifecycle_days,
        late_arrival_risk=late_arrival_risk,
    )


def build_default_candidate_plans(
    model_output: ModelOutput,
    inventory_state: InventoryState,
    policy: str = "balanced",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    target_sell_through_rate: float = DEFAULT_TARGET_SELL_THROUGH_RATE,
) -> list[CandidatePlan]:
    model_output = model_output.normalized()
    inventory_state = inventory_state.normalized()
    horizon_demand_qty = _scale_30d_qty_to_horizon(model_output.pred_qty_30d, horizon_days)
    available_before_plan = inventory_state.current_inventory + inventory_state.inbound_within_30d
    gap_qty = max(horizon_demand_qty - available_before_plan, 0.0)
    target_supply_cap = horizon_demand_qty / max(_clip_probability(target_sell_through_rate), 1e-9)
    target_sell_through_plan_qty = max(target_supply_cap - available_before_plan, 0.0)

    raw_candidates = [
        0.0,
        inventory_state.min_batch_qty or 0.0,
        0.5 * gap_qty,
        0.8 * gap_qty,
        gap_qty,
        target_sell_through_plan_qty,
        max(1.2 * horizon_demand_qty - available_before_plan, 0.0),
        max(1.5 * horizon_demand_qty - available_before_plan, 0.0),
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
        normalized_qty = _round_to_batch(qty, inventory_state.min_batch_qty, inventory_state.increment_batch_qty)
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
    horizon_days: int | None = None,
) -> dict:
    normalized_model_output = model_output.normalized()
    normalized_inventory_state = inventory_state.normalized()
    normalized_economics = economics.normalized()
    requested_horizon_days = _coerce_int(
        horizon_days,
        default=normalized_economics.lifecycle_days,
        minimum=0,
    )
    snapshot_date = _coerce_date(normalized_inventory_state.snapshot_date) or normalized_model_output.snapshot_date
    horizon_days, remaining_lifecycle_days = _resolve_effective_horizon_days(
        snapshot_date=snapshot_date,
        economics=normalized_economics,
        requested_horizon_days=requested_horizon_days,
    )
    candidates = build_default_candidate_plans(
        normalized_model_output,
        normalized_inventory_state,
        policy=policy,
        horizon_days=horizon_days,
        target_sell_through_rate=normalized_economics.target_sell_through_rate,
    )
    assessments = [
        assess_replenishment_plan(
            model_output=normalized_model_output,
            inventory_state=normalized_inventory_state,
            economics=normalized_economics,
            plan=plan,
            demand_scenarios=demand_scenarios,
            horizon_days=horizon_days,
        )
        for plan in candidates
    ]

    def _score(assessment: ProfitAssessment) -> float:
        risk_penalty = sqrt(max(assessment.profit_variance, 0.0))
        leftover_penalty = assessment.expected_leftover_qty * max(normalized_economics.unit_cost - normalized_economics.salvage_value_per_unit, 0.0)
        stockout_penalty = assessment.expected_lost_sales_qty * normalized_economics.unit_price
        sell_through_gap = max(normalized_economics.target_sell_through_rate - assessment.sell_through_rate, 0.0)
        sell_through_penalty = sell_through_gap * assessment.expected_supply_qty * normalized_economics.unit_cost
        if policy == "conservative":
            return (
                assessment.expected_profit
                - 0.50 * risk_penalty
                - 0.70 * leftover_penalty
                - 0.10 * stockout_penalty
                - 0.30 * sell_through_penalty
            )
        if policy == "aggressive":
            return (
                assessment.expected_profit
                - 0.15 * risk_penalty
                - 0.10 * leftover_penalty
                - 0.45 * stockout_penalty
                - 0.10 * sell_through_penalty
            )
        return (
            assessment.expected_profit
            - 0.25 * risk_penalty
            - 0.35 * leftover_penalty
            - 0.20 * stockout_penalty
            - 0.20 * sell_through_penalty
        )

    ranked = sorted(assessments, key=_score, reverse=True)
    lowest_risk = min(ranked, key=lambda item: (item.profit_variance, item.stockout_rate, item.expected_leftover_qty))
    best_profit = max(ranked, key=lambda item: item.expected_profit)

    def _assessment_to_dict(assessment: ProfitAssessment) -> dict:
        out = assessment.to_dict()
        out["recommendation_score"] = float(_score(assessment))
        out["target_sell_through_rate"] = float(normalized_economics.target_sell_through_rate)
        out["horizon_days"] = int(horizon_days)
        out["remaining_lifecycle_days"] = remaining_lifecycle_days
        out["lifecycle_end_date"] = (
            normalized_economics.lifecycle_end_date.isoformat()
            if normalized_economics.lifecycle_end_date is not None
            else None
        )
        return out

    return {
        "sku_id": normalized_model_output.sku_id,
        "policy": policy,
        "horizon_days": int(horizon_days),
        "requested_horizon_days": int(requested_horizon_days),
        "remaining_lifecycle_days": remaining_lifecycle_days,
        "lifecycle_end_date": (
            normalized_economics.lifecycle_end_date.isoformat()
            if normalized_economics.lifecycle_end_date is not None
            else None
        ),
        "target_sell_through_rate": float(normalized_economics.target_sell_through_rate),
        "ranked_candidates": [_assessment_to_dict(assessment) for assessment in ranked],
        "best_balanced_plan": _assessment_to_dict(ranked[0]) if ranked else None,
        "best_recommended_plan": _assessment_to_dict(ranked[0]) if ranked else None,
        "best_profit_plan": _assessment_to_dict(best_profit) if ranked else None,
        "lowest_risk_plan": _assessment_to_dict(lowest_risk) if ranked else None,
    }
