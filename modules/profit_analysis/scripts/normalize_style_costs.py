from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the client SKC cost workbook into a style_id cost table."
    )
    parser.add_argument("--source-xlsx", required=True)
    parser.add_argument(
        "--output-csv",
        default=str(
            PROJECT_ROOT
            / "data"
            / "incoming"
            / "profit_analysis"
            / "style_costs_2024_2026.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "current"
            / "profit_analysis_style_cost_audit_20260612.json"
        ),
    )
    parser.add_argument(
        "--audit-md",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "current"
            / "profit_analysis_style_cost_audit_20260612.md"
        ),
    )
    return parser.parse_args()


def _find_cost_sheet(source_path: Path) -> tuple[str, pd.DataFrame]:
    workbook = pd.ExcelFile(source_path)
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(source_path, sheet_name=sheet_name)
        normalized = {str(col).strip(): col for col in frame.columns}
        if "款号" in normalized and "成本价" in normalized:
            return sheet_name, frame.rename(
                columns={
                    normalized["款号"]: "style_id",
                    normalized["成本价"]: "unit_cost",
                }
            )
    raise ValueError("No worksheet containing 款号 and 成本价 was found.")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_xlsx)
    output_path = Path(args.output_csv)
    audit_json_path = Path(args.audit_json)
    audit_md_path = Path(args.audit_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_name, source = _find_cost_sheet(source_path)
    work = source.loc[:, ["style_id", "unit_cost"]].copy()
    work["style_id"] = work["style_id"].astype("string").str.strip().str.upper()
    work["unit_cost"] = pd.to_numeric(work["unit_cost"], errors="coerce")
    source_rows = int(len(work))
    invalid_rows = int(
        (work["style_id"].isna() | work["unit_cost"].isna() | (work["unit_cost"] <= 0)).sum()
    )
    valid = work[
        work["style_id"].notna()
        & work["unit_cost"].notna()
        & (work["unit_cost"] > 0)
    ].copy()

    normalized = (
        valid.groupby("style_id", as_index=False)
        .agg(
            unit_cost=("unit_cost", "max"),
            unit_cost_min=("unit_cost", "min"),
            unit_cost_max=("unit_cost", "max"),
            cost_record_count=("unit_cost", "size"),
            distinct_cost_count=("unit_cost", "nunique"),
        )
        .sort_values("style_id")
        .reset_index(drop=True)
    )
    normalized["cost_conflict_flag"] = (
        normalized["distinct_cost_count"] > 1
    ).astype(int)
    normalized["cost_source"] = "client_style_cost_2024_2026_max"
    normalized.to_csv(output_path, index=False, encoding="utf-8-sig")

    products = pd.read_csv(
        PROJECT_ROOT / "data" / "silver" / "clean_products.csv",
        usecols=["style_id"],
    )
    project_styles = set(
        products["style_id"].astype("string").str.strip().str.upper().dropna()
    )
    cost_styles = set(normalized["style_id"])
    matched_styles = project_styles & cost_styles

    audit = {
        "source_xlsx": str(source_path.resolve()),
        "source_sheet": sheet_name,
        "duplicate_resolution": "maximum_positive_cost_per_style_id",
        "source_rows": source_rows,
        "invalid_or_nonpositive_rows": invalid_rows,
        "valid_rows": int(len(valid)),
        "unique_cost_style_ids": int(len(normalized)),
        "duplicate_style_ids": int((normalized["cost_record_count"] > 1).sum()),
        "conflicting_cost_style_ids": int(normalized["cost_conflict_flag"].sum()),
        "project_style_ids": int(len(project_styles)),
        "matched_project_style_ids": int(len(matched_styles)),
        "project_style_coverage": float(
            len(matched_styles) / len(project_styles) if project_styles else 0.0
        ),
        "output_csv": str(output_path.resolve()),
    }
    audit_json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_md_path.write_text(
        "\n".join(
            [
                "# 盈亏分析 SKC 成本表审计",
                "",
                f"- 源文件：`{source_path.resolve()}`",
                f"- 成本明细工作表：`{sheet_name}`",
                "- 关联主键：`style_id = 款号`",
                "- 重复成本处理：同一 style_id 有多个正成本时取最大值，避免低估成本。",
                f"- 原始行数：`{source_rows:,}`",
                f"- 有效正成本行数：`{len(valid):,}`",
                f"- 唯一成本 SKC：`{len(normalized):,}`",
                f"- 重复 SKC：`{(normalized['cost_record_count'] > 1).sum():,}`",
                f"- 成本冲突 SKC：`{normalized['cost_conflict_flag'].sum():,}`",
                f"- 项目 SKC：`{len(project_styles):,}`",
                f"- 精确匹配项目 SKC：`{len(matched_styles):,}`",
                f"- 项目 SKC 覆盖率：`{audit['project_style_coverage']:.2%}`",
                "",
                "未匹配 SKC 继续使用甲方确认的 `price_tag / 7` 兜底，并输出成本来源标记。",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] normalized cost table -> {output_path}")
    print(f"[OK] unique cost styles -> {len(normalized):,}")
    print(f"[OK] project style coverage -> {audit['project_style_coverage']:.2%}")


if __name__ == "__main__":
    main()
