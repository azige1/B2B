#!/usr/bin/env python3
"""Inventory large workspace artifacts and remaining Git working-tree changes.

This is intentionally read-only. It helps separate source code and curated
reports from generated local artifacts before any cleanup or archiving step.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "current" / "workspace_artifact_inventory_20260616.md"
DEFAULT_CSV = PROJECT_ROOT / "data" / "manifests" / "workspace_artifact_inventory_20260616.csv"
DEFAULT_JSON = PROJECT_ROOT / "data" / "manifests" / "workspace_artifact_inventory_20260616.json"


@dataclass
class DirRecord:
    path: str
    files: int
    size_mb: float
    category: str
    recommendation: str


def run_git_status() -> list[tuple[str, str]]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:].strip()
        rows.append((status, path))
    return rows


def dir_size(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return files, total


def classify_dir(path: str) -> tuple[str, str]:
    name = Path(path).name
    if name.startswith("models_phase"):
        return "generated_model_artifact", "Keep local or archive by manifest; do not commit binaries."
    if path == "models":
        return "historical_model_artifact", "Keep tracked references only; generated binaries stay ignored."
    if path == "data":
        return "data_assets", "Use data/DATA_INDEX.md and manifests; do not bulk commit generated data."
    if path == "data_warehouse":
        return "raw_snapshot_warehouse", "Keep local snapshots with manifests; do not commit CSV extracts."
    if path == "reports":
        return "reports_and_experiments", "Commit curated current docs; leave generated exports ignored."
    if path == "modules":
        return "active_profit_analysis_module", "Review and commit as a separate module-focused change."
    if path in {"src", "scripts", "tests", "config", "docs", "utils"}:
        return "source_or_test_code", "Review normally; code changes should be committed intentionally."
    if path.startswith("."):
        return "tooling_or_cache", "Usually local; commit only config files."
    return "other", "Review manually."


def top_level_inventory() -> list[DirRecord]:
    records: list[DirRecord] = []
    for item in sorted(PROJECT_ROOT.iterdir()):
        if not item.is_dir():
            continue
        files, total = dir_size(item)
        category, recommendation = classify_dir(item.name)
        records.append(
            DirRecord(
                path=item.name,
                files=files,
                size_mb=round(total / 1024 / 1024, 2),
                category=category,
                recommendation=recommendation,
            )
        )
    return sorted(records, key=lambda row: row.size_mb, reverse=True)


def classify_status_path(path: str) -> str:
    clean = path.strip('"')
    if clean.startswith("models_phase") or clean.startswith("data/incoming") or clean.startswith("data/phase8_"):
        return "generated_local_artifact"
    if clean.startswith("reports/profit_analysis_") or clean.startswith("reports/phase8_") or clean.startswith("reports/phase7_"):
        return "generated_or_historical_report"
    if clean.startswith("reports/current/"):
        return "curated_current_report_candidate"
    if clean.startswith("modules/profit_analysis/"):
        return "profit_analysis_module_pending"
    if clean.startswith(("src/", "scripts/", "tests/", "evaluate")):
        return "source_code_pending"
    if clean.startswith(("data/manifests/", "data/current_assets.json")):
        return "data_manifest_or_registry"
    if clean.startswith("data_warehouse/"):
        return "raw_snapshot_local"
    return "other_pending"


def grouped_status(rows: list[tuple[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for status, path in rows:
        category = classify_status_path(path)
        groups.setdefault(category, []).append({"status": status, "path": path})
    return groups


def render_markdown(records: list[DirRecord], status_groups: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "# Workspace Artifact Inventory",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Purpose",
        "",
        "This report separates source code and curated handoff documents from generated local artifacts. It is a cleanup aid, not a deletion list.",
        "",
        "## Top-Level Directory Inventory",
        "",
        "| path | files | size_mb | category | recommendation |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in records:
        lines.append(
            f"| `{row.path}` | {row.files} | {row.size_mb:.2f} | "
            f"{row.category} | {row.recommendation} |"
        )

    lines.extend(
        [
            "",
            "## Remaining Git Working-Tree Groups",
            "",
        ]
    )
    for category in sorted(status_groups):
        items = status_groups[category]
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"- Count: `{len(items)}`")
        lines.append("")
        for item in items[:40]:
            lines.append(f"- `{item['status']}` `{item['path']}`")
        if len(items) > 40:
            lines.append(f"- ... {len(items) - 40} more")
        lines.append("")

    lines.extend(
        [
            "## Cleanup Recommendations",
            "",
            "- Keep `reports/current/`, root indexes, and small manifest files as the canonical handoff layer.",
            "- Keep raw snapshots and generated feature/model artifacts local or in external storage; do not bulk commit them.",
            "- Review `modules/profit_analysis/` separately and commit it as its own module-focused change.",
            "- Review remaining `src/` and `scripts/` changes separately before committing; they may belong to older Phase8 work.",
            "- Do not delete generated artifacts until a path-level archive manifest has been written.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    records = top_level_inventory()
    status_groups = grouped_status(run_git_status())

    report_path = Path(args.report)
    csv_path = Path(args.csv)
    json_path = Path(args.json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(render_markdown(records, status_groups), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "directories": [asdict(row) for row in records],
                "git_status_groups": status_groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] report={report_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] csv={csv_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] json={json_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
