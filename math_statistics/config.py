"""Shared configuration for statistical analysis modules."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_CSV = DATA_DIR / "features" / "full_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ── Soil target properties ─────────────────────────────────────────
# Article analyses 6 properties; column names in full_dataset.csv
SOIL_TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

# Display names for plots / tables (matching article notation)
SOIL_LABELS = {
    "ph": "pH (KCl)",
    "soc": "SOC, %",
    "no3": "NO₃, mg/kg",
    "p": "P₂O₅, mg/kg",
    "k": "K₂O, mg/kg",
    "s": "S, mg/kg",
}

# ── Feature groups (prefix-based) ─────────────────────────────────
# Sentinel-2 spectral indices by season
S2_INDEX_PREFIXES = [
    "s2_NDVI", "s2_NDRE", "s2_GNDVI", "s2_EVI",
    "s2_SAVI", "s2_BSI", "s2_Cl_Red_Edge",
]
# Sentinel-2 bands by season
S2_BAND_PREFIXES = [
    "s2_B2", "s2_B3", "s2_B4", "s2_B5", "s2_B6",
    "s2_B7", "s2_B8", "s2_B8A", "s2_B11", "s2_B12",
]
# Landsat-8
L8_PREFIXES = [
    "l8_GNDVI", "l8_NDVI", "l8_SAVI",
    "l8_SR_B2", "l8_SR_B3", "l8_SR_B4",
    "l8_SR_B5", "l8_SR_B6", "l8_SR_B7",
]
# Topographic
TOPO_COLS = [
    "topo_DEM", "topo_slope", "topo_aspect_sin", "topo_aspect_cos",
    "topo_TWI", "topo_TPI", "topo_plan_curvature", "topo_profile_curvature",
]
# Climate
CLIMATE_COLS = ["climate_MAT", "climate_MAP", "climate_GS_temp", "climate_GS_precip"]

SEASONS = ["spring", "summer", "late_summer", "autumn"]
SEASON_LABELS = {
    "spring": "Spring",
    "summer": "Summer",
    "late_summer": "Late summer",
    "autumn": "Autumn",
}

# ── Key article claims to verify ──────────────────────────────────
ARTICLE_CLAIMS = {
    "ph_gndvi_l8_spring": {"target": "ph", "feature": "l8_GNDVI_spring", "rho": -0.670},
    "ph_map": {"target": "ph", "feature": "climate_MAP", "rho": 0.66},
    "ph_slope": {"target": "ph", "feature": "topo_slope", "rho": 0.609},
    "soc_s_corr": {"target": "soc", "feature": "s", "rho": 0.100},
    "soc_no3_corr": {"target": "soc", "feature": "no3", "rho": 0.093},
    "ph_soc_corr": {"target": "ph", "feature": "soc", "rho": -0.178},
    "k_bsi_spring": {"target": "k", "feature": "s2_BSI_spring", "rho": -0.48},
    "p_gs_temp": {"target": "p", "feature": "climate_GS_temp", "rho": 0.48},
    "p_aspect_cos": {"target": "p", "feature": "topo_aspect_cos", "rho": 0.47},
    "ph_sin_aspect": {"target": "ph", "feature": "topo_aspect_sin", "rho": -0.475},
}

# ── Statistical parameters ─────────────────────────────────────────
ALPHA = 0.05
SEED = 42
