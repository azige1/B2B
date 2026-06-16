import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dated_files(folder, suffix, cutoff):
    selected = []
    for path in sorted(folder.glob(f"*_{suffix}.csv")):
        match = re.match(r"^(\d{8})_", path.name)
        if match and match.group(1) <= cutoff:
            selected.append(path)
    return selected


def copy_checked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    status = "copied"
    if destination.exists():
        destination_hash = sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"Refusing to overwrite conflicting file: {destination}\n"
                f"source={source_hash}\ndestination={destination_hash}"
            )
        status = "already_identical"
    else:
        shutil.copy2(source, destination)
    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "destination": str(destination.relative_to(PROJECT_ROOT)),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
        "status": status,
    }


def source_spec(source_root, cutoff):
    return [
        {
            "name": "order_flow",
            "table": "V_IRS_ORDERFTP",
            "role": "canonical_label_and_order_history",
            "files": [source_root.parent / f"V_IRS_ORDERFTP_{cutoff}.csv"],
            "destination": PROJECT_ROOT / "data_warehouse" / "fact_orders",
            "rename": lambda _: (
                f"V_IRS_ORDERFTP_{int(cutoff[4:6])}_{int(cutoff[6:])}.csv"
            ),
        },
        {
            "name": "storage_inventory",
            "table": "V_IRS_STORAGE",
            "role": "phase8_main_feature",
            "files": dated_files(source_root / "snapshot_inventory", "storage_stock", cutoff),
            "destination": PROJECT_ROOT / "data_warehouse" / "snapshot_inventory",
        },
        {
            "name": "b2b_inventory",
            "table": "V_IRS_B2BSTORAGE",
            "role": "phase8_main_feature",
            "files": dated_files(source_root / "snapshot_inventory", "b2b_stock", cutoff),
            "destination": PROJECT_ROOT / "data_warehouse" / "snapshot_inventory",
        },
        {
            "name": "event_full",
            "table": "V_IRS_EVENT",
            "role": "phase8_main_feature",
            "files": [source_root.parent / f"V_IRS_EVENT_full_{cutoff}.csv"],
            "destination": PROJECT_ROOT / "data_warehouse" / "fact_events",
            "rename": lambda _: f"V_IRS_EVENT_{cutoff}.csv",
        },
        {
            "name": "event_daily_extracts",
            "table": "V_IRS_EVENT_DAILY_AUDIT",
            "role": "coverage_audit_only",
            "files": dated_files(source_root / "fact_events", "user_events", cutoff),
            "destination": PROJECT_ROOT / "data_warehouse" / "fact_events",
        },
        {
            "name": "product",
            "table": "V_IRS_PRODUCT",
            "role": "phase8_main_dimension_and_listing_date",
            "files": [source_root / "dim_product" / "product_info_latest.csv"],
            "destination": PROJECT_ROOT / "data_warehouse" / "dim_product",
            "rename": lambda _: f"product_info_{cutoff}.csv",
        },
        {
            "name": "store",
            "table": "V_IRS_STORE",
            "role": "mapping_and_audit_only",
            "files": [source_root / "dim_store" / "store_info_latest.csv"],
            "destination": PROJECT_ROOT / "data_warehouse" / "dim_store",
            "rename": lambda _: f"store_info_{cutoff}.csv",
        },
        {
            "name": "customer_profile",
            "table": "V_IRS_CUS_PROFILE",
            "role": "shadow_only_not_phase8_mainline",
            "files": dated_files(source_root / "snapshot_metrics", "customer_profile", cutoff),
            "destination": PROJECT_ROOT / "data_warehouse" / "snapshot_metrics",
        },
        {
            "name": "preorder",
            "table": "V_IRS_PREORDER",
            "role": "diagnostic_only",
            "files": dated_files(source_root / "fact_orders", "b2b_preorder", cutoff),
            "destination": PROJECT_ROOT / "data_warehouse" / "fact_orders",
        },
    ]


def validate_sources(specs):
    missing = []
    for spec in specs:
        if not spec["files"]:
            missing.append(f"{spec['table']}: no files selected")
            continue
        missing.extend(str(path) for path in spec["files"] if not path.exists())
    if missing:
        raise FileNotFoundError("Missing Phase8 snapshot inputs:\n- " + "\n- ".join(missing))


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize the audited Phase8 server snapshot into the local warehouse."
    )
    parser.add_argument(
        "--source",
        default="data/incoming/server_20260614/data_warehouse",
        help="Server warehouse snapshot directory, relative to the project root.",
    )
    parser.add_argument("--cutoff", default="20260614", help="Inclusive YYYYMMDD cutoff.")
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}", args.cutoff):
        raise ValueError("--cutoff must be YYYYMMDD")

    source_root = (PROJECT_ROOT / args.source).resolve()
    specs = source_spec(source_root, args.cutoff)
    validate_sources(specs)

    copied = []
    sources = []
    for spec in specs:
        records = []
        for source in spec["files"]:
            name = spec.get("rename", lambda path: path.name)(source)
            records.append(copy_checked(source, spec["destination"] / name))
        copied.extend(records)
        sources.append(
            {
                "name": spec["name"],
                "table": spec["table"],
                "role": spec["role"],
                "file_count": len(records),
                "total_bytes": sum(record["bytes"] for record in records),
                "copied": sum(record["status"] == "copied" for record in records),
                "already_identical": sum(
                    record["status"] == "already_identical" for record in records
                ),
            }
        )

    manifest = {
        "snapshot_id": f"phase8_{args.cutoff}",
        "cutoff_date": datetime.strptime(args.cutoff, "%Y%m%d").date().isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.relative_to(PROJECT_ROOT)),
        "policy": {
            "canonical_order_table": "V_IRS_ORDERFTP",
            "excluded_order_sources": ["V_IRS_ORDER", "*_order_history.csv"],
            "mainline_tables": [
                "V_IRS_ORDERFTP",
                "V_IRS_STORAGE",
                "V_IRS_B2BSTORAGE",
                "V_IRS_EVENT",
                "V_IRS_PRODUCT",
            ],
            "shadow_only_tables": ["V_IRS_CUS_PROFILE"],
            "mapping_only_tables": ["V_IRS_STORE"],
            "diagnostic_only_tables": ["V_IRS_PREORDER"],
        },
        "sources": sources,
        "files": copied,
    }
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"phase8_data_snapshot_{args.cutoff}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Phase8 snapshot synchronized: {manifest_path}")
    for source in sources:
        print(
            f"- {source['table']}: files={source['file_count']} "
            f"copied={source['copied']} identical={source['already_identical']} "
            f"role={source['role']}"
        )


if __name__ == "__main__":
    main()
