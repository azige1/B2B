#!/usr/bin/env python3
"""Export the client Oracle source tables into an auditable CSV snapshot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import oracledb


DEFAULT_TABLES = [
    "V_IRS_ORDERFTP",
    "V_IRS_PRODUCT",
    "V_IRS_STORAGE",
    "V_IRS_B2BSTORAGE",
    "V_IRS_EVENT",
    "V_IRS_STORE",
    "V_IRS_CUS_PROFILE",
    "V_IRS_PREORDER",
    "V_IRS_ORDER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema", default="BOSNDS3")
    parser.add_argument("--tables", nargs="+", default=DEFAULT_TABLES)
    parser.add_argument("--oracle-client", default="/root/instantclient_19_8")
    parser.add_argument(
        "--legacy-config",
        help="Read DB_CONFIG from an existing trusted Python script.",
    )
    parser.add_argument("--fetch-size", type=int, default=10_000)
    return parser.parse_args()


def load_legacy_config(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "DB_CONFIG"
                for target in node.targets
            )
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError(f"DB_CONFIG not found in {path}")


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.legacy_config:
        return load_legacy_config(Path(args.legacy_config))
    return {
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
        "host": os.environ["ORACLE_HOST"],
        "port": int(os.environ.get("ORACLE_PORT", "1521")),
        "sid": os.environ["ORACLE_SID"],
    }


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "read"):
        return value.read()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_table(
    conn: oracledb.Connection,
    schema: str,
    table: str,
    output_dir: Path,
    fetch_size: int,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    output_path = output_dir / f"{table}.csv"
    partial_path = output_path.with_suffix(".csv.part")
    cursor = conn.cursor()
    cursor.arraysize = fetch_size
    row_count = 0

    try:
        cursor.execute(f'SELECT * FROM "{schema}"."{table}"')
        columns = [item[0] for item in cursor.description]
        with partial_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            while True:
                rows = cursor.fetchmany(fetch_size)
                if not rows:
                    break
                writer.writerows(
                    [normalize_value(value) for value in row] for row in rows
                )
                row_count += len(rows)
                print(f"[PROGRESS] {table}: {row_count:,}", flush=True)
        partial_path.replace(output_path)
        result = {
            "table": f"{schema}.{table}",
            "status": "ok",
            "rows": row_count,
            "columns": columns,
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
            "file": output_path.name,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(
            f"[OK] {table}: rows={row_count:,} bytes={result['bytes']:,}",
            flush=True,
        )
        return result
    except Exception as exc:
        partial_path.unlink(missing_ok=True)
        result = {
            "table": f"{schema}.{table}",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(f"[ERROR] {table}: {result['error']}", flush=True)
        return result
    finally:
        cursor.close()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args)

    oracledb.init_oracle_client(lib_dir=args.oracle_client)
    dsn = oracledb.makedsn(
        config["host"],
        int(config["port"]),
        sid=config["sid"],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": args.schema,
        "tables_requested": args.tables,
        "exports": [],
    }

    with oracledb.connect(
        user=config["user"],
        password=config["password"],
        dsn=dsn,
    ) as conn:
        for table in args.tables:
            manifest["exports"].append(
                export_table(
                    conn,
                    args.schema,
                    table.upper(),
                    output_dir,
                    args.fetch_size,
                )
            )
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    failures = [item for item in manifest["exports"] if item["status"] != "ok"]
    print(f"[DONE] exported={len(manifest['exports']) - len(failures)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
