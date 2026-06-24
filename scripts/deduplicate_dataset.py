#!/usr/bin/env python
"""
deduplicate_dataset.py
======================
Remove duplicate soil-sample records from a feature dataset CSV.

Background
----------
A post-publication audit of the dataset found that field "19-20" (farm
Агро Парасат, sampled 2023-04-25) had been entered a second time under the
name "19-20 (1)". Its 14 grid points therefore appear twice with identical
coordinates, geometry and identical six lab values (pH, SOC, NO3, P2O5, K2O, S),
differing only in the `id` column. The raw build thus reports 1085 records /
81 field names, but the number of UNIQUE samples is 1071 (80 unique fields).

A duplicate is defined as a row sharing the same physical location and sampling
date as an earlier row: identical (centroid_lon, centroid_lat, sampling_date).
The earliest record (lowest `id`) of each such group is kept; the rest are
dropped. This removes the "19-20 (1)" copies (ids 379-392) while keeping the
original "19-20" rows.

Note
----
The published paper (Agriculture 2026, 16, 1239) reports n = 1085 / 81 fields,
and the canonical build (data/features/master_dataset.csv, 1085 x 530) is kept
UNCHANGED so the published metrics remain reproducible. This script writes the
de-duplicated build to a SEPARATE output file; it never overwrites the input.

Usage
-----
    python scripts/deduplicate_dataset.py \
        --input  data/features/master_dataset.csv \
        --output data/features/master_dataset_dedup.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

DEDUP_KEY = ["centroid_lon", "centroid_lat", "sampling_date"]


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean_df, dropped_df). Keeps the lowest-`id` row per location+date."""
    missing = [c for c in DEDUP_KEY if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns for de-duplication: {missing}")

    # Deterministic: keep the earliest record (lowest id) of each duplicate group.
    order = df.sort_values("id") if "id" in df.columns else df
    keep_idx = order.drop_duplicates(subset=DEDUP_KEY, keep="first").index
    clean = df.loc[df.index.isin(keep_idx)].reset_index(drop=True)
    dropped = df.loc[~df.index.isin(keep_idx)].reset_index(drop=True)
    return clean, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/features/master_dataset.csv",
                    help="Input dataset CSV (default: data/features/master_dataset.csv)")
    ap.add_argument("--output", default="data/features/master_dataset_dedup.csv",
                    help="Output de-duplicated CSV (default: data/features/master_dataset_dedup.csv)")
    args = ap.parse_args()

    src, dst = Path(args.input), Path(args.output)
    if not src.exists():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 1
    if dst.resolve() == src.resolve():
        print("ERROR: output must differ from input (canonical build is kept unchanged).", file=sys.stderr)
        return 1

    df = pd.read_csv(src, low_memory=False)
    n0 = len(df)
    f0 = df["field_name"].nunique() if "field_name" in df.columns else None

    clean, dropped = deduplicate(df)
    n1, f1 = len(clean), (clean["field_name"].nunique() if "field_name" in clean.columns else None)

    print(f"input : {src}  ->  {n0} records, {f0} field names")
    print(f"dropped: {len(dropped)} duplicate record(s)")
    if "id" in dropped.columns and len(dropped):
        ids = sorted(dropped["id"].tolist())
        print(f"        dropped ids: {ids}")
        if "field_name" in dropped.columns:
            print(f"        dropped fields: {sorted(dropped['field_name'].unique().tolist())}")
    print(f"output: {dst}  ->  {n1} unique records, {f1} unique fields")

    clean.to_csv(dst, index=False)
    print(f"\nWrote de-duplicated build to {dst} (input left unchanged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
