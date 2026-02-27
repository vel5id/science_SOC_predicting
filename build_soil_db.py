"""
build_soil_db.py — ETL: Shapefiles → SQLite + Excel + merge с метаданными
=========================================================================
Reads all .shp files from raw_data/shp/, normalizes to Schema-3
(K, P, HU, S, ZN, MO, FE, MG, MN, CU, PH, NO3), merges with protocol
metadata (dates, protocol numbers) from Excel protocols, and exports
a unified SQLite database + Excel mirror.

Usage:
    uv run python build_soil_db.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ──────────────────────────── CONFIG ────────────────────────────

ROOT = Path(__file__).parent
SHP_DIR = ROOT / "raw_data" / "shp"
OUT_DIR = ROOT / "data"

DB_PATH = OUT_DIR / "soil_analysis.db"
EXCEL_PATH = OUT_DIR / "soil_analysis.xlsx"

PROTOCOL_METADATA_CSV = ROOT / "raw_data" / "pdf_excel_data" / "protocol_metadata.csv"

# Schema-3 (reference) — order matters for output
CORE_INDICATORS: list[str] = ["PH", "K", "P", "HU", "S", "NO3"]
MICRO_ELEMENTS: list[str] = ["ZN", "MO", "FE", "MG", "MN", "CU"]
ALL_INDICATORS: list[str] = CORE_INDICATORS + MICRO_ELEMENTS

META_COLUMNS: list[str] = [
    "year", "farm", "field_name", "grid_id",
]
GEO_COLUMNS: list[str] = [
    "centroid_lon", "centroid_lat", "geometry_wkt",
]

PROTOCOL_COLUMNS: list[str] = [
    "protocol_number", "analysis_date", "sampling_date",
]

FINAL_COLUMNS: list[str] = (
    META_COLUMNS
    + [c.lower() for c in ALL_INDICATORS]
    + GEO_COLUMNS
    + PROTOCOL_COLUMNS
)


# ──────────────────────────── ETL ───────────────────────────────

def extract_metadata(shp_path: Path) -> dict[str, str | int]:
    """Parse year/farm/field from directory structure.

    Expected: raw_data/shp/{year}/{farm}/{field}/{field}.shp
    """
    rel = shp_path.relative_to(SHP_DIR)
    parts = rel.parts  # (year, farm, field_dir, filename)
    return {
        "year": int(parts[0]),
        "farm": parts[1],
        "field_name": parts[2],
    }


def process_shapefile(shp_path: Path) -> pd.DataFrame | None:
    """Read a single shapefile and normalize to unified schema."""
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"  ✗ ERROR reading {shp_path.name}: {e}")
        return None

    if gdf.empty:
        print(f"  ⚠ EMPTY: {shp_path.name}")
        return None

    meta = extract_metadata(shp_path)
    n = len(gdf)

    # Build output dataframe
    out = pd.DataFrame()

    # Metadata
    out["year"] = [meta["year"]] * n
    out["farm"] = [meta["farm"]] * n
    out["field_name"] = [meta["field_name"]] * n
    out["grid_id"] = gdf["GRID"].values if "GRID" in gdf.columns else range(1, n + 1)

    # Indicators — map available columns, leave missing as NaN
    for col in ALL_INDICATORS:
        col_lower = col.lower()
        if col in gdf.columns:
            out[col_lower] = gdf[col].values
        else:
            out[col_lower] = pd.NA

    # Geometry — centroid + WKT polygon
    centroids = gdf.geometry.centroid
    out["centroid_lon"] = centroids.x.values
    out["centroid_lat"] = centroids.y.values
    out["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt).values

    return out


def _normalize_field_name(name: str) -> str:
    """Normalize field_name for fuzzy matching between SHP and Excel."""
    import re
    s = str(name).strip()
    s = re.sub(r"\s*\(\d+\)$", "", s)   # remove trailing " (1)"
    s = re.sub(r"^Поле[- ]*", "", s)     # remove "Поле-" prefix
    s = re.sub(r"^поле[- ]*", "", s)     # remove "поле-" prefix (lowercase)
    s = re.sub(r"\s+", "", s)            # remove all whitespace
    s = re.sub(r";", "; ", s).strip()    # normalize semicolons
    s = re.sub(r"\s+", " ", s)           # collapse remaining whitespace
    return s


def merge_with_protocol_metadata(shp_df: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN SHP data with protocol metadata on (year, farm, field_name)."""
    if not PROTOCOL_METADATA_CSV.exists():
        print(f"\n⚠ Protocol metadata not found: {PROTOCOL_METADATA_CSV}")
        print("  Run: python raw_data/pdf_excel_data/build_database.py first")
        for col in PROTOCOL_COLUMNS:
            shp_df[col] = pd.NA
        return shp_df

    meta = pd.read_csv(str(PROTOCOL_METADATA_CSV), encoding="utf-8-sig")
    print(f"\n── Merge with protocol metadata ──")
    print(f"Protocol metadata rows: {len(meta)}")

    # Ensure year types match
    meta["year"] = meta["year"].astype(int)
    shp_df["year"] = shp_df["year"].astype(int)

    # Normalize field_name for fuzzy matching
    shp_df["_fn_norm"] = shp_df["field_name"].apply(_normalize_field_name)
    meta["_fn_norm"] = meta["field_name"].apply(_normalize_field_name)

    # Merge on (year, farm, normalized field_name)
    merged = shp_df.merge(
        meta[["year", "farm", "_fn_norm", "protocol_number", "analysis_date", "sampling_date"]],
        on=["year", "farm", "_fn_norm"],
        how="left",
    )
    merged.drop(columns=["_fn_norm"], inplace=True)

    matched = merged["protocol_number"].notna().sum()
    total = len(merged)
    print(f"Matched:  {matched}/{total} rows ({matched/total*100:.1f}%)")

    unique_fields = shp_df[["year", "farm", "field_name"]].drop_duplicates()
    matched_fields = merged.loc[merged["protocol_number"].notna(), ["year", "farm", "field_name"]].drop_duplicates()
    print(f"Fields:   {len(matched_fields)}/{len(unique_fields)} matched")

    return merged


REPORT_PATH = OUT_DIR / "merge_report.xlsx"


def generate_merge_report(
    shp_df: pd.DataFrame,
    result: pd.DataFrame,
    shp_errors: list[str] | None = None,
) -> None:
    """Create a multi-sheet Excel report on merge coverage.

    Sheets:
      1) Сводка              — overall statistics
      2) Успешный merge      — matched fields
      3) SHP без метаданных  — SHP fields that got NO protocol match
      4) Excel без SHP       — protocol metadata that matched NO SHP field
      5) SHP ошибки чтения   — shapefiles that could not be read
    """
    meta = pd.read_csv(str(PROTOCOL_METADATA_CSV), encoding="utf-8-sig")
    meta["year"] = meta["year"].astype(int)

    # Normalize for comparison
    shp_fields = (
        shp_df[["year", "farm", "field_name"]]
        .drop_duplicates()
        .copy()
    )
    shp_fields["_fn_norm"] = shp_fields["field_name"].apply(_normalize_field_name)

    meta_fields = (
        meta[["year", "farm", "field_name", "protocol_number",
              "analysis_date", "sampling_date", "source_file"]]
        .copy()
    )
    meta_fields["_fn_norm"] = meta_fields["field_name"].apply(_normalize_field_name)

    # ── Sheet 1: SHP fields without metadata ──
    merged_check = shp_fields.merge(
        meta_fields[["year", "farm", "_fn_norm"]].drop_duplicates(),
        on=["year", "farm", "_fn_norm"],
        how="left",
        indicator=True,
    )
    shp_unmatched = (
        merged_check[merged_check["_merge"] == "left_only"]
        [["year", "farm", "field_name"]]
        .sort_values(["year", "farm", "field_name"])
        .reset_index(drop=True)
    )
    shp_unmatched["grid_count"] = shp_unmatched.apply(
        lambda r: len(
            shp_df[
                (shp_df["year"] == r["year"])
                & (shp_df["farm"] == r["farm"])
                & (shp_df["field_name"] == r["field_name"])
            ]
        ),
        axis=1,
    )
    shp_unmatched.rename(columns={
        "year": "Год",
        "farm": "Ферма",
        "field_name": "Поле (SHP)",
        "grid_count": "Кол-во grid в SHP",
    }, inplace=True)

    # ── Sheet 2: Excel metadata without SHP match ──
    meta_check = meta_fields.merge(
        shp_fields[["year", "farm", "_fn_norm"]].drop_duplicates(),
        on=["year", "farm", "_fn_norm"],
        how="left",
        indicator=True,
    )
    excel_unmatched = (
        meta_check[meta_check["_merge"] == "left_only"]
        [["year", "farm", "field_name", "protocol_number",
          "analysis_date", "sampling_date", "source_file"]]
        .sort_values(["year", "farm", "field_name"])
        .reset_index(drop=True)
    )
    excel_unmatched.rename(columns={
        "year": "Год",
        "farm": "Ферма",
        "field_name": "Поле (Excel)",
        "protocol_number": "Протокол №",
        "analysis_date": "Дата анализа",
        "sampling_date": "Дата отбора",
        "source_file": "Файл-источник",
    }, inplace=True)

    # ── Sheet 3: Successfully matched ──
    matched_rows = (
        result[result["protocol_number"].notna()]
        [["year", "farm", "field_name", "protocol_number",
          "analysis_date", "sampling_date"]]
        .drop_duplicates(subset=["year", "farm", "field_name"])
        .sort_values(["year", "farm", "field_name"])
        .reset_index(drop=True)
    )
    grid_counts = (
        result[result["protocol_number"].notna()]
        .groupby(["year", "farm", "field_name"])
        .size()
        .reset_index(name="grid_count")
    )
    matched_rows = matched_rows.merge(
        grid_counts, on=["year", "farm", "field_name"], how="left"
    )
    matched_rows.rename(columns={
        "year": "Год",
        "farm": "Ферма",
        "field_name": "Поле",
        "protocol_number": "Протокол №",
        "analysis_date": "Дата анализа",
        "sampling_date": "Дата отбора",
        "grid_count": "Кол-во grid",
    }, inplace=True)

    # ── Sheet 4: Summary stats ──
    total_shp_fields = len(shp_fields)
    total_meta_fields = len(meta_fields)
    matched_count = len(matched_rows)
    shp_unmatched_count = len(shp_unmatched)
    excel_unmatched_count = len(excel_unmatched)

    # ── Sheet 5: SHP read errors ──
    _errors = shp_errors or []
    shp_errors_df = pd.DataFrame(
        [{"Путь шейпфайла": e} for e in _errors]
    ) if _errors else pd.DataFrame(columns=["Путь шейпфайла"])

    # ── Sheet 1: Summary stats ──
    summary_data = [
        ("SHP полей (уникальных)", total_shp_fields),
        ("SHP шейпфайлов с ошибкой чтения", len(_errors)),
        ("Excel протоколов (уникальных полей)", total_meta_fields),
        ("", ""),
        ("Успешно привязано полей", matched_count),
        ("SHP полей без метаданных", shp_unmatched_count),
        ("Excel полей без SHP", excel_unmatched_count),
        ("", ""),
        ("SHP строк всего", len(result)),
        ("SHP строк с метаданными", int(result["protocol_number"].notna().sum())),
        ("SHP строк без метаданных", int(result["protocol_number"].isna().sum())),
        ("Покрытие (строк)", f"{result['protocol_number'].notna().sum() / len(result) * 100:.1f}%"),
        ("Покрытие (полей)", f"{matched_count / total_shp_fields * 100:.1f}%"),
    ]

    # Add per-year breakdown
    summary_data.append(("", ""))
    summary_data.append(("По годам:", ""))
    for year in sorted(shp_fields["year"].unique()):
        shp_y = len(shp_fields[shp_fields["year"] == year])
        matched_y = len(matched_rows[matched_rows["Год"] == year])
        unmatched_y = len(shp_unmatched[shp_unmatched["Год"] == year])
        summary_data.append((f"  {year}: SHP полей", shp_y))
        summary_data.append((f"  {year}: привязано", matched_y))
        summary_data.append((f"  {year}: без метаданных", unmatched_y))

    summary_df = pd.DataFrame(summary_data, columns=["Показатель", "Значение"])

    # ── Write Excel ──
    with pd.ExcelWriter(str(REPORT_PATH), engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Сводка", index=False)
        matched_rows.to_excel(writer, sheet_name="Успешный merge", index=False)
        shp_unmatched.to_excel(writer, sheet_name="SHP без метаданных", index=False)
        excel_unmatched.to_excel(writer, sheet_name="Excel без SHP", index=False)
        shp_errors_df.to_excel(writer, sheet_name="SHP ошибки чтения", index=False)

    print(f"✓ Report: {REPORT_PATH}  ({REPORT_PATH.stat().st_size / 1024:.0f} KB)")


def build_database() -> pd.DataFrame:
    """Main pipeline: scan → extract → normalize → merge → load."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shp_files = sorted(SHP_DIR.rglob("*.shp"))
    print(f"Found {len(shp_files)} shapefiles\n")

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for i, shp in enumerate(shp_files, 1):
        rel = shp.relative_to(SHP_DIR)
        print(f"[{i:3d}/{len(shp_files)}] {rel}")
        df = process_shapefile(shp)
        if df is not None:
            frames.append(df)
        else:
            errors.append(str(rel))

    if not frames:
        print("No data extracted. Aborting.")
        return pd.DataFrame()

    # Concatenate all SHP data
    result = pd.concat(frames, ignore_index=True)

    # Merge with protocol metadata (dates, protocol numbers)
    result = merge_with_protocol_metadata(result)
    result = result[FINAL_COLUMNS]  # enforce column order

    # ── SQLite ──
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    result.to_sql("soil_samples", conn, index=True, index_label="id")

    # Add useful indices
    conn.execute("CREATE INDEX idx_year ON soil_samples(year)")
    conn.execute("CREATE INDEX idx_farm ON soil_samples(farm)")
    conn.execute("CREATE INDEX idx_coords ON soil_samples(centroid_lon, centroid_lat)")
    conn.execute("CREATE INDEX idx_field ON soil_samples(field_name)")
    conn.commit()
    conn.close()
    print(f"\n✓ SQLite: {DB_PATH}  ({DB_PATH.stat().st_size / 1024:.0f} KB)")

    # ── Excel ──
    excel_df = result.copy()
    excel_df.to_excel(str(EXCEL_PATH), index=True, index_label="id", engine="openpyxl")
    print(f"✓ Excel:  {EXCEL_PATH}  ({EXCEL_PATH.stat().st_size / 1024:.0f} KB)")

    # ── Merge report ──
    shp_only = pd.concat(frames, ignore_index=True)
    generate_merge_report(shp_only, result, shp_errors=errors)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Total rows:       {len(result)}")
    print(f"Files processed:  {len(frames)} / {len(shp_files)}")
    if errors:
        print(f"Errors:           {len(errors)}")
        for e in errors:
            print(f"  - {e}")

    print(f"\nNULL distribution (indicators):")
    for col in [c.lower() for c in ALL_INDICATORS]:
        null_count = result[col].isna().sum()
        pct = null_count / len(result) * 100
        bar = "█" * int(pct / 5) if pct > 0 else "—"
        print(f"  {col:>6s}: {null_count:4d} / {len(result):4d}  ({pct:5.1f}%)  {bar}")

    print(f"\nProtocol metadata coverage:")
    for col in PROTOCOL_COLUMNS:
        filled = result[col].notna().sum()
        pct = filled / len(result) * 100
        print(f"  {col:>16s}: {filled:4d} / {len(result):4d}  ({pct:5.1f}%)")

    print(f"\nRows per year:")
    for year, count in result.groupby("year").size().items():
        print(f"  {year}: {count}")

    return result


if __name__ == "__main__":
    build_database()
