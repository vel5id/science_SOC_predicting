"""
Configuration for satellite data extraction pipeline.

Note on S2 band values:
    COPERNICUS/S2_SR_HARMONIZED provides Bottom-of-Atmosphere (BOA)
    reflectance scaled by 10000 (uint16). So B8 = 3000 means ρ = 0.30.
    Spectral indices (NDVI, NDRE, etc.) are computed on these integer
    values; normalized ratios cancel the scale factor, but absolute
    formulas (SAVI, EVI) use L/C1/C2 constants calibrated for 0–1
    reflectance — they still work because the additive constants are
    negligible relative to the band magnitudes.

Note on L8 band values:
    LANDSAT/LC08/C02/T1_L2 stores SR as scaled integers:
    ρ = DN × 0.0000275 − 0.2  (must apply before using absolute indices).
"""
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "soil_analysis.db"

# Output directories
TEMP_DIR = DATA_DIR / "temperature"
S2_DIR = DATA_DIR / "sentinel2"
L8_DIR = DATA_DIR / "landsat8"
S1_DIR = DATA_DIR / "sentinel1"
TOPO_DIR = DATA_DIR / "topography"
SOIL_DIR = DATA_DIR / "soil_maps"
HYPER_DIR = DATA_DIR / "hyperspectral"
FEATURES_DIR = DATA_DIR / "features"
CLIMATE_DIR = DATA_DIR / "climate"
SEMIVARIOGRAM_DIR = DATA_DIR / "semivariograms"
SPECTRAL_ENG_DIR = DATA_DIR / "spectral_eng"
GLCM_DIR = DATA_DIR / "glcm"

# ─── Temporal ────────────────────────────────────────────────────
YEARS = [2022, 2023]

# Seasonal composites (month ranges)
SEASONS = {
    "spring": (4, 5),      # April-May
    "summer": (6, 7),      # June-July
    "late_summer": (8, 9), # August-September
    "autumn": (10, 10),    # October
}

# ─── Spatial ─────────────────────────────────────────────────────
# Will be computed from DB at runtime
REGION_BBOX = None  # (min_lon, min_lat, max_lon, max_lat)
CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32641"  # UTM 41N for Kazakhstan

# ─── Vegetation Index Constants ──────────────────────────────────
# SAVI soil-brightness correction factor (Huete, 1988)
# L=0.5 is standard for intermediate vegetation cover.
# For sparse cover (bare soil) L→1; for dense cover L→0.
SAVI_L_FACTOR = 0.5

# Landsat 8 Collection 2 Level 2 scale factors
L8_SCALE_FACTOR = 0.0000275
L8_OFFSET = -0.2

# ─── Sentinel-2 ──────────────────────────────────────────────────
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_THRESHOLD = 20  # CLOUDY_PIXEL_PERCENTAGE

S2_BANDS = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "red_edge_1",
    "B6": "red_edge_2",
    "B7": "red_edge_3",
    "B8": "nir",
    "B8A": "nir_narrow",
    "B11": "swir1",
    "B12": "swir2",
}

S2_INDICES = {
    "NDVI": "(B8 - B4) / (B8 + B4)",
    "NDRE": "(B8 - B5) / (B8 + B5)",
    "GNDVI": "(B8 - B3) / (B8 + B3)",
    "Cl_Red_Edge": "(B7 / B5) - 1",
    "SAVI": "((B8 - B4) / (B8 + B4 + 0.5)) * 1.5",
    "EVI": "2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))",
    "BSI": "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
}

# Band ratios for soil analysis
S2_BAND_RATIOS = {
    "B3_B4": "B3 / B4",
    "B8_B4": "B8 / B4",
    "B11_B8": "B11 / B8",
}

# ─── Landsat 8 ───────────────────────────────────────────────────
L8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

L8_BANDS = {
    "SR_B2": "blue",
    "SR_B3": "green",
    "SR_B4": "red",
    "SR_B5": "nir",
    "SR_B6": "swir1",
    "SR_B7": "swir2",
}

L8_INDICES = {
    "NDVI": "(SR_B5 - SR_B4) / (SR_B5 + SR_B4)",
    "GNDVI": "(SR_B5 - SR_B3) / (SR_B5 + SR_B3)",
    "SAVI": "((SR_B5 - SR_B4) / (SR_B5 + SR_B4 + 0.5)) * 1.5",
}

# ─── Sentinel-1 ──────────────────────────────────────────────────
S1_COLLECTION = "COPERNICUS/S1_GRD"
S1_POLARIZATIONS = ["VV", "VH"]

# ─── Topography ──────────────────────────────────────────────────
DEM_COLLECTION = "COPERNICUS/DEM/GLO30"

# ─── Soil Maps ───────────────────────────────────────────────────
SOILGRIDS = {
    "sand": "projects/soilgrids-isric/sand_mean",    # g/kg  (div 10 -> %)
    "silt": "projects/soilgrids-isric/silt_mean",    # g/kg  (div 10 -> %)
    "clay": "projects/soilgrids-isric/clay_mean",    # g/kg  (div 10 -> %)
    "soc": "projects/soilgrids-isric/soc_mean",      # dg/kg (div 10 -> g/kg)
    "ph": "projects/soilgrids-isric/phh2o_mean",     # pH*10 (div 10 -> pH)
    "cec": "projects/soilgrids-isric/cec_mean",      # mmol(c)/kg
    "bdod": "projects/soilgrids-isric/bdod_mean",    # cg/cm3 (div 100 -> g/cm3)
    "nitrogen": "projects/soilgrids-isric/nitrogen_mean",  # cg/kg (div 100 -> g/kg)
}
SOILGRIDS_DEPTHS = [
    "0-5cm_mean",
    "5-15cm_mean",
    "15-30cm_mean",
    "30-60cm_mean",
    "60-100cm_mean",
    "100-200cm_mean",
]
SOILGRIDS_PRIMARY_DEPTH = "0-5cm_mean"
SOILGRIDS_SCALE = 250  # native resolution in meters

# ─── ERA5 ────────────────────────────────────────────────────────
ERA5_COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
TEMP_THRESHOLD_C = 0.0  # Growing season: T > 0°C

# ─── GEE Settings ────────────────────────────────────────────────
GEE_SCALE = 10  # meters (S2 resolution)
GEE_MAX_PIXELS = 1e9
