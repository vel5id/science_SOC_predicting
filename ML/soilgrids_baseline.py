"""
soilgrids_baseline.py
----------------------
Download SoilGrids v2.0 predictions for each sample point via ISRIC REST API
and compare with ground-truth measurements from master_dataset.csv.

Targets that have SoilGrids equivalents:
  ph  → SoilGrids property: phh2o  (pH in H₂O, ×10 encoding)
  soc → SoilGrids property: soc    (Soil Organic Carbon g/kg, ×10 encoding)

Depths queried: "0-5cm" (surface layer, closest to agronomic measurements).
The script caches API responses to a JSON file so interrupted runs resume.

Outputs:
  ML/results/soilgrids/soilgrids_raw_responses.json   (cache)
  ML/results/soilgrids/soilgrids_predictions.csv       (per-sample predictions)
  ML/results/soilgrids/soilgrids_metrics.json          (rho, R², RMSE, MAE)

Usage:
  python ML/soilgrids_baseline.py

Notes:
  - Requires internet access from WSL (test: curl https://rest.isric.org)
  - Rate limit: ~1 request/s.  The script sleeps 1.2s between requests.
  - All ~1071 unique centroid points are queried (de-duplicated by lon/lat).
  - SoilGrids covers 60°S–82°N at 250m resolution; Kazakhstan is within range.
"""
import os, sys, time, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
OUT_DIR  = ROOT / "ML" / "results" / "soilgrids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = OUT_DIR / "soilgrids_raw_responses.json"

# ─── SoilGrids API Config ──────────────────────────────────────────────────────
ISRIC_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
DEPTH     = "0-5cm"
VALUE_TYPE = "mean"

# Property → local target column, scale factor (divide raw integer by factor)
PROPERTIES = {
    "phh2o": {"target": "ph",  "scale": 10.0},
    "soc":   {"target": "soc", "scale": 10.0},  # g/kg
}

REQUEST_DELAY = 1.2   # seconds between API calls (ISRIC rate limit)
TIMEOUT       = 30    # seconds per request
MAX_RETRIES   = 3


# ─── API call ─────────────────────────────────────────────────────────────────
def query_soilgrids(lon: float, lat: float, prop: str) -> float | None:
    """
    Query SoilGrids v2.0 for a single (lon, lat) point.
    Returns the mean value for DEPTH layer, scaled to real units.
    Returns None on failure.
    """
    params = {
        "lon":      round(lon, 6),
        "lat":      round(lat, 6),
        "property": prop,
        "depth":    DEPTH,
        "value":    VALUE_TYPE,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(ISRIC_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            # Navigate response: data["properties"]["layers"][0]["depths"][0]["values"]["mean"]
            layers = data.get("properties", {}).get("layers", [])
            if not layers:
                return None
            layer = layers[0]
            depths = layer.get("depths", [])
            for d in depths:
                if d.get("label") == DEPTH:
                    raw = d.get("values", {}).get(VALUE_TYPE)
                    if raw is None:
                        return None
                    scale = PROPERTIES[prop]["scale"]
                    return raw / scale
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:          # rate-limited
                time.sleep(5 * attempt)
            else:
                print(f"[WARN] HTTP {r.status_code} for ({lon},{lat}) prop={prop}: {e}")
                return None
        except Exception as e:
            print(f"[WARN] Request error attempt {attempt} for ({lon},{lat}): {e}")
            time.sleep(2)
    return None


# ─── Load cache ────────────────────────────────────────────────────────────────
def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ─── Round coordinate key ──────────────────────────────────────────────────────
def coord_key(lon: float, lat: float, prop: str) -> str:
    return f"{prop}_{round(lon, 5)}_{round(lat, 5)}"


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading master dataset...")
    df = pd.read_csv(DATA_CSV)

    required_cols = ["centroid_lon", "centroid_lat", "ph", "soc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"master_dataset.csv is missing columns: {missing}")

    # De-duplicate coordinates (one API call per unique lat/lon point)
    coord_cols = ["centroid_lon", "centroid_lat"]
    unique_coords = df[coord_cols].dropna().drop_duplicates().reset_index(drop=True)
    print(f"[INFO] {len(unique_coords)} unique (lon, lat) points to query.")

    cache = load_cache()
    print(f"[INFO] Cache: {len(cache)} entries already stored.")

    # ── Download all needed responses ──────────────────────────────────────────
    n_total     = len(unique_coords) * len(PROPERTIES)
    n_cached    = sum(1 for _, row in unique_coords.iterrows()
                      for prop in PROPERTIES
                      if coord_key(row["centroid_lon"], row["centroid_lat"], prop) in cache)
    n_remaining = n_total - n_cached
    print(f"[INFO] {n_remaining} points still need to be queried (ETA ~{n_remaining*REQUEST_DELAY/60:.1f} min).")

    with tqdm(total=n_remaining, desc="SoilGrids queries") as pbar:
        for _, row in unique_coords.iterrows():
            lon, lat = row["centroid_lon"], row["centroid_lat"]
            for prop in PROPERTIES:
                key = coord_key(lon, lat, prop)
                if key in cache:
                    continue
                val = query_soilgrids(lon, lat, prop)
                cache[key] = val
                save_cache(cache)          # persist after every call
                pbar.update(1)
                time.sleep(REQUEST_DELAY)

    print(f"[INFO] Download complete. Cache has {len(cache)} entries.")

    # ── Attach SoilGrids predictions to each sample row ───────────────────────
    records = []
    for df_idx, row in df.iterrows():
        lon = row.get("centroid_lon")
        lat = row.get("centroid_lat")
        rec = {"df_idx": df_idx}

        for prop, cfg in PROPERTIES.items():
            tgt = cfg["target"]
            gt  = row.get(tgt)
            if pd.isna(lon) or pd.isna(lat):
                sg_val = None
            else:
                key    = coord_key(lon, lat, prop)
                sg_val = cache.get(key)

            rec[f"gt_{tgt}"]  = gt
            rec[f"sg_{tgt}"]  = sg_val
            rec["lon"]        = lon
            rec["lat"]        = lat

        records.append(rec)

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(OUT_DIR / "soilgrids_predictions.csv", index=False)
    print(f"[INFO] Predictions saved to {OUT_DIR / 'soilgrids_predictions.csv'}")

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = {}
    for prop, cfg in PROPERTIES.items():
        tgt = cfg["target"]
        sub = pred_df[["gt_" + tgt, "sg_" + tgt]].dropna()
        if len(sub) < 10:
            print(f"[WARN] Too few valid pairs for {tgt}: {len(sub)}")
            continue

        gt  = sub[f"gt_{tgt}"].values
        sg  = sub[f"sg_{tgt}"].values
        rho, _ = spearmanr(gt, sg)
        r2     = r2_score(gt, sg)
        rmse   = math.sqrt(mean_squared_error(gt, sg))
        mae    = mean_absolute_error(gt, sg)

        metrics[tgt] = {
            "rho":  round(rho,  4),
            "r2":   round(r2,   4),
            "rmse": round(rmse, 4),
            "mae":  round(mae,  4),
            "n":    len(sub),
            "soilgrids_property": prop,
            "depth": DEPTH,
        }
        print(f"\n[{tgt.upper()}]  N={len(sub)}")
        print(f"  Spearman ρ : {rho:.4f}")
        print(f"  R²         : {r2:.4f}")
        print(f"  RMSE       : {rmse:.4f}")
        print(f"  MAE        : {mae:.4f}")

    with open(OUT_DIR / "soilgrids_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Metrics saved to {OUT_DIR / 'soilgrids_metrics.json'}")


if __name__ == "__main__":
    main()
