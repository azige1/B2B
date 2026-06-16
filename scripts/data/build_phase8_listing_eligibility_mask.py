import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mask(keys_path, launch_map, chunk_size):
    parts = []
    missing_count = 0
    prelaunch_count = 0
    row_count = 0
    for chunk in pd.read_csv(
        keys_path,
        usecols=["sku_id", "date"],
        dtype={"sku_id": str},
        chunksize=chunk_size,
    ):
        row_dates = pd.to_datetime(chunk["date"], errors="coerce")
        launch_dates = chunk["sku_id"].map(launch_map)
        missing = launch_dates.isna()
        invalid_dates = row_dates.isna()
        eligible = missing | (row_dates >= launch_dates)
        eligible &= ~invalid_dates
        parts.append(eligible.to_numpy(dtype=bool))
        missing_count += int(missing.sum())
        prelaunch_count += int((~missing & (row_dates < launch_dates)).sum())
        row_count += len(chunk)
    mask = np.concatenate(parts) if parts else np.empty(0, dtype=bool)
    if len(mask) != row_count:
        raise ValueError(
            f"Mask row mismatch for {keys_path}: {len(mask)} != {row_count}"
        )
    return mask, {
        "rows": row_count,
        "eligible_rows": int(mask.sum()),
        "excluded_rows": int((~mask).sum()),
        "prelaunch_rows": prelaunch_count,
        "missing_launch_rows_kept": missing_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-keys", required=True, type=Path)
    parser.add_argument("--val-keys", required=True, type=Path)
    parser.add_argument("--listing", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    listing = pd.read_csv(
        args.listing,
        usecols=["sku_id", "launch_date"],
        dtype={"sku_id": str},
    )
    listing["launch_date"] = pd.to_datetime(
        listing["launch_date"],
        errors="coerce",
    )
    listing = (
        listing.sort_values("launch_date")
        .drop_duplicates("sku_id", keep="first")
    )
    launch_map = listing.set_index("sku_id")["launch_date"]

    train_mask, train_audit = build_mask(
        args.train_keys,
        launch_map,
        args.chunk_size,
    )
    val_mask, val_audit = build_mask(
        args.val_keys,
        launch_map,
        args.chunk_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output = args.output_dir / "train_listing_eligible.npy"
    val_output = args.output_dir / "val_listing_eligible.npy"
    meta_output = args.output_dir / "listing_eligibility_meta.json"
    np.save(train_output, train_mask)
    np.save(val_output, val_mask)

    payload = {
        "policy": (
            "Keep rows with anchor_date >= launch_date. Missing launch dates are "
            "kept and counted; invalid anchor dates are excluded."
        ),
        "sources": {
            "train_keys": str(args.train_keys),
            "train_keys_sha256": sha256(args.train_keys),
            "val_keys": str(args.val_keys),
            "val_keys_sha256": sha256(args.val_keys),
            "listing": str(args.listing),
            "listing_sha256": sha256(args.listing),
        },
        "train": train_audit,
        "validation": val_audit,
        "outputs": {
            "train_mask": str(train_output),
            "val_mask": str(val_output),
        },
    }
    meta_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
