"""Canonical feature counts for Article 1.

Source of truth: data/features/master_dataset_old.csv (530 columns).
Composites: computed by math_statistics/composite_features.py (110 total).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER_OLD = ROOT / "data" / "features" / "master_dataset_old.csv"

META_COLS = {"id", "year", "farm", "field_name", "grid_id",
             "centroid_lon", "centroid_lat", "geometry_wkt",
             "protocol_number", "analysis_date", "sampling_date"}
SOIL_COLS = {"ph", "soc", "no3", "p", "k", "s", "hu"}


def get_canonical_counts() -> dict:
    df = pd.read_csv(MASTER_OLD, low_memory=False)
    cols = list(df.columns)

    meta = [c for c in cols if c in META_COLS]
    soil = [c for c in cols if c in SOIL_COLS]
    features = [c for c in cols if c not in META_COLS and c not in SOIL_COLS]

    def count(prefix): return sum(1 for c in features if c.startswith(prefix))

    return {
        "total_columns": len(cols),
        "meta": len(meta),
        "soil_targets": len(soil),
        "features_total": len(features),
        "s2_base": count("s2_"),
        "l8_base": count("l8_"),
        "spectral_eng": count("spectral_"),
        "glcm": count("glcm_"),
        "ts_stats": count("ts_"),
        "delta_multiseason": count("delta_"),
        "range_stats": count("range_"),
        "cs_cross_sensor": count("cs_"),
        "topo": count("topo_"),
        "climate": count("climate_"),
        # Composites (from composite_features.py, verified separately)
        "comp_inter_index": 48,
        "comp_multiseason": 42,
        "comp_ndi": 20,
        "comp_total": 110,
        # Derived totals
        "base_spectral_total": count("s2_") + count("l8_"),
        "spectral_flavored_total": count("s2_") + count("l8_") + count("spectral_")
                                   + count("ts_") + count("delta_") + count("cs_")
                                   + count("range_") + 110,
    }


if __name__ == "__main__":
    import json
    counts = get_canonical_counts()
    print(json.dumps(counts, indent=2, ensure_ascii=False))
