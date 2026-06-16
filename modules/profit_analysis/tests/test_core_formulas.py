from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis.core import (  # noqa: E402
    CandidatePlan,
    DemandScenario,
    Economics,
    InventoryState,
    ModelOutput,
    _round_to_batch,
    assess_replenishment_plan,
    build_default_demand_scenarios,
    recommend_replenishment_plans,
    realize_replenishment_plan,
)


class ProfitFormulaTests(unittest.TestCase):
    def test_batch_rounding_uses_client_feedback_floor_and_increment(self) -> None:
        self.assertEqual(_round_to_batch(1, min_batch_qty=100, increment_batch_qty=10), 100)
        self.assertEqual(_round_to_batch(101, min_batch_qty=100, increment_batch_qty=10), 110)
        self.assertEqual(_round_to_batch(0, min_batch_qty=100, increment_batch_qty=10), 0)

    def test_30_day_model_quantity_is_scaled_to_45_day_lifecycle(self) -> None:
        scenarios = build_default_demand_scenarios(
            ModelOutput("SKU", "2026-04-10", pred_prob_positive=1.0, pred_qty_30d=30.0),
            horizon_days=45,
        )
        positive_demands = [row.demand_qty for row in scenarios if row.name != "zero"]
        self.assertEqual(positive_demands, [27.0, 45.0, 67.5])

    def test_profit_components_follow_revenue_minus_cost_plus_zero_terminal_value(self) -> None:
        assessment = assess_replenishment_plan(
            model_output=ModelOutput("SKU", "2026-04-10", 1.0, 100.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-10",
                current_inventory=0.0,
                lead_time_days=0,
                min_batch_qty=100,
                increment_batch_qty=10,
            ),
            economics=Economics(
                "SKU",
                unit_cost=10.0,
                unit_price=20.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
                target_sell_through_rate=0.85,
            ),
            plan=CandidatePlan(plan_qty=100.0),
            demand_scenarios=[DemandScenario("certain", 100.0, 1.0)],
            horizon_days=45,
        )

        self.assertEqual(assessment.expected_sales_revenue, 2000.0)
        self.assertEqual(assessment.expected_replenish_cost, 1000.0)
        self.assertEqual(assessment.expected_terminal_value, 0.0)
        self.assertEqual(assessment.expected_profit, 1000.0)
        self.assertEqual(assessment.sell_through_rate, 1.0)
        self.assertEqual(assessment.sell_through_target_probability, 1.0)

    def test_lifecycle_residual_value_is_zero_for_leftover_inventory(self) -> None:
        assessment = assess_replenishment_plan(
            model_output=ModelOutput("SKU", "2026-04-10", 0.0, 0.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-10",
                current_inventory=0.0,
                lead_time_days=0,
                min_batch_qty=100,
                increment_batch_qty=10,
            ),
            economics=Economics(
                "SKU",
                unit_cost=10.0,
                unit_price=20.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
            ),
            plan=CandidatePlan(plan_qty=100.0),
            demand_scenarios=[DemandScenario("zero", 0.0, 1.0)],
            horizon_days=45,
        )

        self.assertEqual(assessment.expected_leftover_qty, 100.0)
        self.assertEqual(assessment.expected_terminal_value, 0.0)
        self.assertEqual(assessment.expected_profit, -1000.0)

    def test_lead_time_creates_pre_arrival_lost_sales_in_daily_simulation(self) -> None:
        assessment = assess_replenishment_plan(
            model_output=ModelOutput("SKU", "2026-04-10", 1.0, 45.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-10",
                current_inventory=0.0,
                lead_time_days=10,
                min_batch_qty=1,
                increment_batch_qty=1,
            ),
            economics=Economics(
                "SKU",
                unit_cost=0.0,
                unit_price=1.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
            ),
            plan=CandidatePlan(plan_qty=45.0),
            demand_scenarios=[DemandScenario("flat", 45.0, 1.0)],
            horizon_days=45,
        )

        self.assertEqual(assessment.expected_lost_sales_qty, 10.0)
        self.assertEqual(assessment.expected_sold_qty, 35.0)
        self.assertEqual(assessment.stockout_rate, 1.0)

    def test_lifecycle_end_date_caps_effective_horizon(self) -> None:
        assessment = assess_replenishment_plan(
            model_output=ModelOutput("SKU", "2026-04-10", 1.0, 30.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-10",
                current_inventory=100.0,
                lead_time_days=0,
                min_batch_qty=1,
                increment_batch_qty=1,
            ),
            economics=Economics(
                "SKU",
                unit_cost=1.0,
                unit_price=10.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
                lifecycle_end_date="2026-04-19",
            ),
            plan=CandidatePlan(plan_qty=0.0),
            horizon_days=45,
        )

        self.assertEqual(assessment.effective_horizon_days, 10)
        self.assertEqual(assessment.remaining_lifecycle_days, 10)
        mid_scenario = next(row for row in assessment.scenario_breakdown if row["name"] == "positive_2")
        self.assertEqual(mid_scenario["demand_qty"], 10.0)

    def test_late_arrival_after_lifecycle_end_is_flagged(self) -> None:
        assessment = assess_replenishment_plan(
            model_output=ModelOutput("SKU", "2026-04-10", 1.0, 30.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-10",
                current_inventory=0.0,
                lead_time_days=12,
                min_batch_qty=100,
                increment_batch_qty=10,
            ),
            economics=Economics(
                "SKU",
                unit_cost=1.0,
                unit_price=10.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
                lifecycle_end_date="2026-04-19",
            ),
            plan=CandidatePlan(plan_qty=100.0),
            horizon_days=45,
        )

        self.assertEqual(assessment.effective_horizon_days, 10)
        self.assertEqual(assessment.late_arrival_risk, 1)

    def test_post_lifecycle_recommendation_prefers_zero_plan(self) -> None:
        recommendation = recommend_replenishment_plans(
            model_output=ModelOutput("SKU", "2026-04-20", 1.0, 300.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-20",
                current_inventory=0.0,
                lead_time_days=0,
                min_batch_qty=100,
                increment_batch_qty=10,
            ),
            economics=Economics(
                "SKU",
                unit_cost=10.0,
                unit_price=20.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
                lifecycle_end_date="2026-04-19",
            ),
            horizon_days=45,
        )

        self.assertEqual(recommendation["horizon_days"], 0)
        self.assertEqual(recommendation["best_recommended_plan"]["plan_qty"], 0.0)

    def test_realized_plan_uses_actual_daily_demand_timing(self) -> None:
        result = realize_replenishment_plan(
            model_output=ModelOutput("SKU", "2026-04-10", 1.0, 10.0),
            inventory_state=InventoryState(
                "SKU",
                "2026-04-10",
                current_inventory=0.0,
                lead_time_days=2,
                min_batch_qty=10,
                increment_batch_qty=10,
            ),
            economics=Economics(
                "SKU",
                unit_cost=1.0,
                unit_price=10.0,
                holding_cost_per_unit_per_day=0.0,
                salvage_value_per_unit=0.0,
            ),
            plan=CandidatePlan(plan_qty=10.0),
            actual_demand_qty=10.0,
            horizon_days=4,
            actual_daily_demand_curve=[5.0, 5.0, 0.0, 0.0],
        )

        self.assertEqual(result.sold_qty, 0.0)
        self.assertEqual(result.lost_sales_qty, 10.0)
        self.assertEqual(result.leftover_qty, 10.0)


if __name__ == "__main__":
    unittest.main()
