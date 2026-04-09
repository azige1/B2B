from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# These locations are intentionally kept out of normal Git tracking.
FORBIDDEN_PREFIXES = (
    "data/raw/",
    "data/silver/",
    "data/gold/",
    "data/processed_",
    "data/artifacts_",
    "data_warehouse/",
    "models/",
    "models_phase8_event_shadow/",
    "models_phase8_event_inventory_shadow_2026/",
    ".vendor_tree_backends/",
)

FORBIDDEN_SUFFIXES = (
    ".bin",
    ".pth",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".dll",
    ".pyd",
)

WARN_SIZE_BYTES = 10 * 1024 * 1024
FAIL_SIZE_BYTES = 50 * 1024 * 1024


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def main() -> int:
    tracked = [line.strip() for line in git("ls-files").splitlines() if line.strip()]
    violations: list[str] = []
    warnings: list[str] = []

    for rel_path in tracked:
        posix_path = rel_path.replace("\\", "/")
        abs_path = REPO_ROOT / rel_path

        for prefix in FORBIDDEN_PREFIXES:
            if posix_path.startswith(prefix):
                violations.append(f"tracked forbidden path: {posix_path}")
                break

        for suffix in FORBIDDEN_SUFFIXES:
            if posix_path.endswith(suffix):
                violations.append(f"tracked forbidden binary artifact: {posix_path}")
                break

        if posix_path.startswith("reports/") and (
            posix_path.endswith(".csv") or posix_path.endswith(".html")
        ):
            violations.append(f"tracked rendered report export: {posix_path}")

        if abs_path.is_file():
            size = abs_path.stat().st_size
            if size >= FAIL_SIZE_BYTES:
                violations.append(
                    f"tracked file exceeds hard size limit ({format_size(size)}): {posix_path}"
                )
            elif size >= WARN_SIZE_BYTES:
                warnings.append(
                    f"tracked file exceeds soft size limit ({format_size(size)}): {posix_path}"
                )

    print("Git hygiene check")
    print(f"repo: {REPO_ROOT}")
    print(f"tracked files: {len(tracked)}")

    if warnings:
        print("\nWarnings")
        for item in warnings:
            print(f"- {item}")

    if violations:
        print("\nViolations")
        for item in violations:
            print(f"- {item}")
        return 1

    print("\nOK: no forbidden tracked files detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
