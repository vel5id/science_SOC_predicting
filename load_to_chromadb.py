"""
load_to_chromadb.py — Load science_article codebase and data into ChromaDB.

Creates two collections:
  - code_functions : one document per Python function / method / class
  - data_schema    : CSV column metadata, SQLite table schemas, config constants

Usage:
    python load_to_chromadb.py                 # full load (skips if already populated)
    python load_to_chromadb.py --force         # reload even if collection exists
    python load_to_chromadb.py --verify        # run sample queries after loading
    python load_to_chromadb.py --skip-code     # skip Python source code
    python load_to_chromadb.py --help
"""

import argparse
import importlib
import io
import re
import sqlite3
import sys
from pathlib import Path

# Ensure stdout handles Unicode (needed on Windows with cp1252 default encoding)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import chromadb

# ── Project paths ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()
CHROMA_DIR = ROOT / "chroma_db"
DB_PATH = ROOT / "data" / "soil_analysis.db"
FEATURES_CSV = ROOT / "data" / "features" / "full_dataset.csv"

CODE_COLLECTION = "code_functions"
SCHEMA_COLLECTION = "data_schema"

# Reuse ast_map.py from the project root
sys.path.insert(0, str(ROOT))
from ast_map import build_file_map, DEFAULT_DIRS  # noqa: E402  (after sys.path.insert)

# ── Soil target columns (for _classify_column) ────────────────────────────────

SOIL_TARGETS = {"ph", "k", "p", "hu", "soc", "s", "no3", "zn", "mo", "fe", "mg", "mn", "cu"}
IDENTITY_COLS = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon", "centroid_lat",
                 "geometry_wkt", "protocol_number", "analysis_date", "sampling_date"}
SEASONS = {"spring", "summer", "late_summer", "autumn"}

# Known topographic and climate feature descriptions
TOPO_DESCRIPTIONS: dict[str, str] = {
    "dem": "Digital Elevation Model — elevation above sea level (metres). SRTM/GLO-30.",
    "slope": "Terrain slope in degrees. Derived from DEM via GEE terrain analysis.",
    "aspect_sin": "Sine of terrain aspect angle. Encodes north-south gradient.",
    "aspect_cos": "Cosine of terrain aspect angle. Encodes east-west gradient.",
    "twi": "Topographic Wetness Index. Proxy for soil moisture accumulation.",
    "tpi": "Topographic Position Index. Relative relief within neighbourhood.",
    "plan_curvature": "Plan curvature of terrain surface (lateral water flow).",
    "profile_curvature": "Profile curvature of terrain surface (downslope water flow).",
}

CLIMATE_DESCRIPTIONS: dict[str, str] = {
    "mat": "Mean Annual Temperature (°C). ERA5 Land, 2020-2023 average.",
    "map": "Mean Annual Precipitation (mm). ERA5 Land, 2020-2023 total.",
    "gs_temp": "Growing Season Mean Temperature (°C). ERA5 Land.",
    "gs_precip": "Growing Season Precipitation (mm). ERA5 Land.",
}

SOIL_DESCRIPTIONS: dict[str, str] = {
    "ph":  "Soil pH measured in KCl solution. Primary target variable.",
    "k":   "Exchangeable potassium as K₂O (mg/kg). Target variable.",
    "p":   "Available phosphorus as P₂O₅ (mg/kg). Target variable.",
    "hu":  "Humus content (%). Raw measurement; used to derive SOC.",
    "soc": "Soil Organic Carbon (%). Derived from humus via van Bemmelen factor (hu × 0.58).",
    "s":   "Sulfur content (mg/kg). Target variable.",
    "no3": "Mobile nitrogen as nitrate NO₃ (mg/kg). Target variable.",
    "zn":  "Zinc (mg/kg). Trace element.",
    "mo":  "Molybdenum (mg/kg). Trace element.",
    "fe":  "Iron (mg/kg). Trace element.",
    "mg":  "Magnesium (mg/kg). Trace element.",
    "mn":  "Manganese (mg/kg). Trace element.",
    "cu":  "Copper (mg/kg). Trace element.",
}

S2_INDEX_FORMULAS: dict[str, str] = {
    "NDVI":         "(B8 - B4) / (B8 + B4)",
    "NDRE":         "(B8 - B5) / (B8 + B5)",
    "GNDVI":        "(B8 - B3) / (B8 + B3)",
    "SAVI":         "((B8 - B4) / (B8 + B4 + 0.5)) × 1.5",
    "EVI":          "2.5 × (B8 - B4) / (B8 + 6×B4 - 7.5×B2 + 1)",
    "BSI":          "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
    "Cl_Red_Edge":  "(B7 / B5) - 1",
}

SEASON_LABELS: dict[str, str] = {
    "spring":      "spring (April–May)",
    "summer":      "summer (June–July)",
    "late_summer": "late summer (August–September)",
    "autumn":      "autumn (September–November)",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — ChromaDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_client(chroma_dir: str | Path) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at chroma_dir."""
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_or_create_collection(client: chromadb.PersistentClient, name: str):
    """Get or create a ChromaDB collection (default embedding function)."""
    return client.get_or_create_collection(name=name)


def is_collection_populated(collection) -> bool:
    """Return True if collection already has at least one document."""
    return collection.count() > 0


def _upsert_in_batches(collection, ids, documents, metadatas,
                       batch_size: int = 100, label: str = "items") -> int:
    """Upsert documents into ChromaDB in batches; returns total count."""
    total = len(ids)
    for i in range(0, total, batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        done = min(i + batch_size, total)
        print(f"    {done}/{total} {label}", end="\r")
    print()  # newline after \r progress
    return total


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Python code loading  (reuses ast_map.py)
# ─────────────────────────────────────────────────────────────────────────────

def _path_to_module(rel_path: Path) -> str:
    """Convert a relative file path to a dotted module name.

    Examples:
        src/db_utils.py             → src.db_utils
        math_statistics/plots.py    → math_statistics.plots
        load_to_chromadb.py         → load_to_chromadb
    """
    parts = list(rel_path.with_suffix("").parts)
    return ".".join(parts)


def _safe_id(s: str) -> str:
    """Make a ChromaDB-safe document ID (no special chars, max 512 chars)."""
    s = re.sub(r"[^\w]", "_", s)
    return s[:512]


def _make_doc_id(module: str, name: str, line: int) -> str:
    return _safe_id(f"code_{module}_{name}_{line}")


def _make_function_doc(entry: dict, rel_path: Path, module: str,
                       parent_class: str = "") -> tuple[str, str, dict]:
    """Build (id, document_text, metadata) for one function/method entry."""
    kind = "Method" if parent_class else "Function"
    lines = [
        f"{kind}: {entry['name']}",
        f"File: {rel_path.as_posix()} (line {entry['line']})",
        f"Module: {module}",
        f"Signature: {entry['signature']}",
    ]
    if entry.get("doc"):
        lines.append(f"Docstring: {entry['doc']}")
    if parent_class:
        lines.append(f"Class: {parent_class}")
    if entry.get("decorators"):
        lines.append(f"Decorators: {', '.join(entry['decorators'])}")
    lines.append(f"Type: {entry['type']}")

    doc_text = "\n".join(lines)
    doc_id = _make_doc_id(module, entry["name"], entry["line"])
    meta = {
        "type": entry["type"],          # "function" | "method" | "async_function"
        "file": rel_path.as_posix(),
        "function": entry["name"],
        "module": module,
        "line": entry["line"],
        "has_docstring": bool(entry.get("doc")),
        "is_private": entry["name"].startswith("_"),
        "is_test": "tests" in rel_path.parts or entry["name"].startswith("test_"),
        "parent_class": parent_class,
    }
    return doc_id, doc_text, meta


def _make_class_doc(entry: dict, rel_path: Path, module: str) -> tuple[str, str, dict]:
    """Build (id, document_text, metadata) for one class entry."""
    method_names = [m["name"] for m in entry.get("methods", [])]
    lines = [
        f"Class: {entry['name']}",
        f"File: {rel_path.as_posix()} (line {entry['line']})",
        f"Module: {module}",
    ]
    if entry.get("bases"):
        lines.append(f"Bases: {', '.join(entry['bases'])}")
    if entry.get("doc"):
        lines.append(f"Docstring: {entry['doc']}")
    if method_names:
        lines.append(f"Methods: {', '.join(method_names)}")
    lines.append("Type: class")

    doc_text = "\n".join(lines)
    doc_id = _make_doc_id(module, entry["name"], entry["line"])
    meta = {
        "type": "class",
        "file": rel_path.as_posix(),
        "function": entry["name"],
        "module": module,
        "line": entry["line"],
        "has_docstring": bool(entry.get("doc")),
        "is_private": entry["name"].startswith("_"),
        "is_test": "tests" in rel_path.parts or entry["name"].startswith("Test"),
        "parent_class": "",
    }
    return doc_id, doc_text, meta


def collect_code_documents(target_dirs=None) -> tuple[list, list, list]:
    """Iterate file_map from ast_map.build_file_map and build ChromaDB documents."""
    if target_dirs is None:
        target_dirs = DEFAULT_DIRS

    file_map = build_file_map(target_dirs)
    ids, documents, metadatas = [], [], []

    for file_path, entries in file_map.items():
        rel_path = file_path.relative_to(ROOT)
        module = _path_to_module(rel_path)

        for entry in entries:
            etype = entry.get("type", "")

            if etype in ("error", "syntax_error"):
                print(f"  [WARN] Skipping {rel_path}: {entry.get('error', '')}")
                continue

            if etype in ("function", "async_function"):
                doc_id, doc_text, meta = _make_function_doc(entry, rel_path, module)
                ids.append(doc_id)
                documents.append(doc_text)
                metadatas.append(meta)

            elif etype == "class":
                # One document for the class itself
                doc_id, doc_text, meta = _make_class_doc(entry, rel_path, module)
                ids.append(doc_id)
                documents.append(doc_text)
                metadatas.append(meta)

                # Individual documents for each method
                for method in entry.get("methods", []):
                    m_id, m_text, m_meta = _make_function_doc(
                        method, rel_path, module, parent_class=entry["name"]
                    )
                    # Override type to "method" for consistent filtering
                    m_meta["type"] = "method"
                    ids.append(m_id)
                    documents.append(m_text)
                    metadatas.append(m_meta)

    return ids, documents, metadatas


def load_code_to_chroma(collection) -> int:
    """Load all Python source code into the code_functions collection."""
    ids, documents, metadatas = collect_code_documents()
    return _upsert_in_batches(collection, ids, documents, metadatas,
                              batch_size=100, label="code documents")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CSV column schema loading
# ─────────────────────────────────────────────────────────────────────────────

def _detect_season(col_name: str) -> str:
    """Return season suffix if found at the end of col_name, else ''."""
    for season in ("late_summer", "spring", "summer", "autumn"):
        if col_name.endswith(f"_{season}"):
            return season
    return ""


def _classify_column(col_name: str) -> dict:
    """Classify a full_dataset column into group/sensor/season/is_soil_target."""
    col = col_name.lower()
    season = _detect_season(col)

    if col in SOIL_TARGETS:
        return {"group": "soil_target", "sensor": "none", "season": "", "is_soil_target": True}
    if col in IDENTITY_COLS:
        return {"group": "identity", "sensor": "none", "season": "", "is_soil_target": False}
    if col.startswith("s2_"):
        return {"group": "s2", "sensor": "sentinel2", "season": season, "is_soil_target": False}
    if col.startswith("l8_"):
        return {"group": "l8", "sensor": "landsat8", "season": season, "is_soil_target": False}
    if col.startswith("spectral_"):
        return {"group": "spectral", "sensor": "sentinel2", "season": season, "is_soil_target": False}
    if col.startswith("glcm_"):
        return {"group": "glcm", "sensor": "sentinel2", "season": season, "is_soil_target": False}
    if col.startswith("topo_"):
        return {"group": "topo", "sensor": "none", "season": "", "is_soil_target": False}
    if col.startswith("climate_"):
        return {"group": "climate", "sensor": "none", "season": "", "is_soil_target": False}
    if col.startswith("soilgrids_"):
        return {"group": "soilgrids", "sensor": "none", "season": "", "is_soil_target": False}
    if col.startswith("s1_"):
        return {"group": "s1", "sensor": "sentinel1", "season": season, "is_soil_target": False}
    return {"group": "other", "sensor": "none", "season": "", "is_soil_target": False}


def _describe_column(col_name: str, group: str, sensor: str) -> str:
    """Return a human-readable description for a column."""
    col = col_name.lower()

    # Soil targets
    if group == "soil_target":
        return SOIL_DESCRIPTIONS.get(col, f"Soil property: {col_name}")

    # Identity
    if group == "identity":
        identity_desc = {
            "id": "Row identifier.",
            "year": "Sampling year (2020–2023).",
            "farm": "Farm name (Ukrainian/Russian).",
            "field_name": "Field identifier within the farm.",
            "grid_id": "Grid point index within the field polygon.",
            "centroid_lon": "Longitude of grid cell centroid (WGS-84).",
            "centroid_lat": "Latitude of grid cell centroid (WGS-84).",
            "geometry_wkt": "Field polygon geometry in WKT format.",
            "protocol_number": "Agrochemical protocol number.",
            "analysis_date": "Date of lab analysis.",
            "sampling_date": "Date of soil sample collection.",
        }
        return identity_desc.get(col, f"Identifier column: {col_name}")

    # Sentinel-2
    if group == "s2":
        stem = col[3:]  # strip "s2_"
        season = _detect_season(stem)
        if season:
            index_or_band = stem[: -(len(season) + 1)]
        else:
            index_or_band = stem
        index_upper = index_or_band.upper()
        formula = S2_INDEX_FORMULAS.get(index_upper, "")
        season_label = SEASON_LABELS.get(season, season)
        formula_part = f" Formula: {formula}." if formula else ""
        return (f"Sentinel-2 {index_or_band} {season_label} composite "
                f"(COPERNICUS/S2_SR_HARMONIZED, 10 m).{formula_part}")

    # Landsat-8
    if group == "l8":
        stem = col[3:]  # strip "l8_"
        season = _detect_season(stem)
        if season:
            index_or_band = stem[: -(len(season) + 1)]
        else:
            index_or_band = stem
        season_label = SEASON_LABELS.get(season, season)
        return (f"Landsat-8 {index_or_band} {season_label} composite "
                f"(LANDSAT/LC08/C02/T1_L2, 30 m). Scale factor applied: DN×0.0000275−0.2.")

    # Spectral engineering
    if group == "spectral":
        stem = col[9:]  # strip "spectral_"
        season = _detect_season(stem)
        if season:
            feat = stem[: -(len(season) + 1)]
        else:
            feat = stem
        season_label = SEASON_LABELS.get(season, season)
        return f"Spectral engineering feature derived from Sentinel-2 bands: {feat}, {season_label}."

    # GLCM textures
    if group == "glcm":
        return (f"GLCM texture feature from Sentinel-2 imagery: {col_name}. "
                f"Grey-Level Co-occurrence Matrix computed in GEE.")

    # Topography
    if group == "topo":
        stem = col[5:]  # strip "topo_"
        return TOPO_DESCRIPTIONS.get(stem, f"Topographic feature: {col_name} (SRTM DEM).")

    # Climate
    if group == "climate":
        stem = col[8:]  # strip "climate_"
        return CLIMATE_DESCRIPTIONS.get(stem, f"Climate variable: {col_name} (ERA5 Land).")

    # SoilGrids
    if group == "soilgrids":
        return (f"SoilGrids global soil property: {col_name}. "
                f"Depth-weighted average (0–30 cm, ISDA/SoilGrids250m v2.0).")

    # Sentinel-1
    if group == "s1":
        return (f"Sentinel-1 SAR backscatter feature: {col_name} "
                f"(GRD, IW, VV+VH polarizations, dB).")

    return f"Dataset column: {col_name}"


def collect_csv_schema_documents(csv_path: Path) -> tuple[list, list, list]:
    """Build one document per column of full_dataset.csv."""
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    ids, documents, metadatas = [], [], []

    for col in df.columns:
        series = df[col]
        clf = _classify_column(col)
        description = _describe_column(col, clf["group"], clf["sensor"])

        n_total = len(series)
        n_valid = int(series.notna().sum())
        pct_missing = round((1 - n_valid / n_total) * 100, 1)

        stat_parts = [f"n_valid={n_valid}/{n_total}, pct_missing={pct_missing}%"]
        if pd.api.types.is_numeric_dtype(series) and n_valid > 0:
            stat_parts += [
                f"min={series.min():.4g}",
                f"max={series.max():.4g}",
                f"mean={series.mean():.4g}",
                f"std={series.std():.4g}",
            ]

        doc_text = (
            f"Column: {col}\n"
            f"Dataset: full_dataset.csv\n"
            f"Group: {clf['group']}\n"
            f"Sensor: {clf['sensor']}\n"
            f"Season: {clf['season']}\n"
            f"Description: {description}\n"
            f"Stats: {', '.join(stat_parts)}\n"
            f"Is soil target: {clf['is_soil_target']}\n"
        )

        doc_id = _safe_id(f"csv_full_dataset_{col}")
        meta = {
            "type": "csv_column",
            "dataset": "full_dataset",
            "column": col,
            "column_group": clf["group"],
            "sensor": clf["sensor"],
            "season": clf["season"],
            "is_soil_target": clf["is_soil_target"],
            "pct_missing": pct_missing,
        }
        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)

    return ids, documents, metadatas


def load_csv_schema_to_chroma(collection, csv_path: Path) -> int:
    """Load CSV column schema documents into the data_schema collection."""
    ids, documents, metadatas = collect_csv_schema_documents(csv_path)
    return _upsert_in_batches(collection, ids, documents, metadatas,
                              batch_size=50, label="CSV column documents")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SQLite schema loading
# ─────────────────────────────────────────────────────────────────────────────

def _make_table_doc(table_name: str, columns: list, row_count: int,
                    sample_rows: list) -> tuple[str, str, dict]:
    """Build (id, document_text, metadata) for one SQLite table."""
    col_block = "\n".join(f"  {c[1]} {c[2]}" for c in columns)

    # Format sample rows — repr() handles Cyrillic and special chars safely
    sample_lines = []
    col_names = [c[1] for c in columns]
    for i, row in enumerate(sample_rows[:2]):
        pairs = [f"{k}={repr(v)}" for k, v in zip(col_names, row) if v is not None]
        sample_lines.append(f"  row{i + 1}: {', '.join(pairs[:8])}")

    doc_text = (
        f"Table: {table_name}\n"
        f"Database: data/soil_analysis.db\n"
        f"Row count: {row_count}\n"
        f"Column count: {len(columns)}\n"
        f"Columns:\n{col_block}\n"
        + (f"Sample data:\n" + "\n".join(sample_lines) if sample_lines else "")
    )

    doc_id = _safe_id(f"db_soil_analysis_{table_name}")
    meta = {
        "type": "db_table",
        "table": table_name,
        "database": "soil_analysis.db",
        "row_count": row_count,
        "col_count": len(columns),
    }
    return doc_id, doc_text, meta


def collect_db_schema_documents(db_path: Path) -> tuple[list, list, list]:
    """Build one document per table in the SQLite database."""
    ids, documents, metadatas = [], [], []
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]

        for table in tables:
            columns = cur.execute(f"PRAGMA table_info({table})").fetchall()
            row_count = cur.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            try:
                sample_rows = cur.execute(f"SELECT * FROM [{table}] LIMIT 2").fetchall()
            except Exception:
                sample_rows = []

            doc_id, doc_text, meta = _make_table_doc(table, columns, row_count, sample_rows)
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(meta)
    finally:
        con.close()
    return ids, documents, metadatas


def load_db_schema_to_chroma(collection, db_path: Path) -> int:
    """Load SQLite table schema documents into the data_schema collection."""
    ids, documents, metadatas = collect_db_schema_documents(db_path)
    return _upsert_in_batches(collection, ids, documents, metadatas,
                              batch_size=50, label="DB table documents")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Config constants loading
# ─────────────────────────────────────────────────────────────────────────────

# Config blocks to extract: (module_path, attribute_name, description)
CONFIG_BLOCKS = [
    ("src.config",              "S2_INDICES",    "Sentinel-2 spectral index band formulas"),
    ("src.config",              "L8_INDICES",    "Landsat-8 spectral index band formulas"),
    ("src.config",              "SEASONS",       "Seasonal composite definitions (month ranges)"),
    ("src.config",              "SOILGRIDS",     "SoilGrids dataset band names and descriptions"),
    ("src.config",              "S2_BANDS",      "Sentinel-2 band name mappings used in pipeline"),
    ("math_statistics.config",  "SOIL_TARGETS",  "Soil property target columns for statistical analysis"),
    ("math_statistics.config",  "SOIL_LABELS",   "Human-readable display names for soil properties"),
    ("math_statistics.config",  "ARTICLE_CLAIMS","Key article claims to verify (expected rho values)"),
    ("math_statistics.config",  "TOPO_COLS",     "Topographic feature column names"),
    ("math_statistics.config",  "CLIMATE_COLS",  "Climate feature column names"),
]


def collect_config_documents() -> tuple[list, list, list]:
    """Import config modules safely and build documents for key constants."""
    # Ensure src/ and project root are importable
    for extra in [str(ROOT / "src"), str(ROOT)]:
        if extra not in sys.path:
            sys.path.insert(0, extra)

    ids, documents, metadatas = [], [], []

    for module_path, attr_name, description in CONFIG_BLOCKS:
        try:
            mod = importlib.import_module(module_path)
            value = getattr(mod, attr_name)
        except Exception as exc:
            print(f"  [WARN] Cannot load {module_path}.{attr_name}: {exc}")
            continue

        # Format value as readable text
        if isinstance(value, dict):
            content_lines = [f"  {k}: {v}" for k, v in value.items()]
        elif isinstance(value, (list, tuple)):
            content_lines = [f"  - {item}" for item in value]
        else:
            content_lines = [f"  {value}"]

        source_file = "src/config.py" if module_path.startswith("src") else "math_statistics/config.py"
        doc_text = (
            f"Config: {attr_name}\n"
            f"Source: {source_file}\n"
            f"Description: {description}\n"
            f"Values:\n" + "\n".join(content_lines)
        )
        doc_id = _safe_id(f"config_{attr_name.lower()}")
        meta = {
            "type": "config_block",
            "config_name": attr_name,
            "source_file": source_file,
        }
        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)

    return ids, documents, metadatas


def load_config_to_chroma(collection) -> int:
    """Load config constant documents into the data_schema collection."""
    ids, documents, metadatas = collect_config_documents()
    return _upsert_in_batches(collection, ids, documents, metadatas,
                              batch_size=50, label="config documents")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Verification queries
# ─────────────────────────────────────────────────────────────────────────────

def verify_load(code_col, schema_col) -> None:
    """Run sample semantic queries and print top results."""
    print("\n" + "=" * 60)
    print("VERIFICATION QUERIES")
    print("=" * 60)


    queries = [
        (code_col,   "compute Spearman correlation between soil properties and remote sensing",
         {"type": "function"}, "code_functions: Spearman correlation"),
        (schema_col, "NDVI normalized difference vegetation index spring seasonal composite",
         {"type": "csv_column"}, "data_schema: NDVI spring column"),
        (schema_col, "soil samples database table schema columns year farm field",
         {"type": "db_table"}, "data_schema: soil_samples table"),
        (code_col,   "test database connection sqlite",
         {"is_test": True}, "code_functions (tests only): DB connection"),
        (schema_col, "Sentinel-2 spectral index band formula config",
         {"type": "config_block"}, "data_schema: S2 index config"),
    ]

    for collection, query_text, where_filter, label in queries:
        print(f"\n>> {label}")
        print(f"  Query: \"{query_text}\"")
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=3,
                where=where_filter,
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            for doc, meta, dist in zip(docs, metas, distances):
                first_line = doc.split("\n")[0]
                print(f"  [{dist:.4f}] {first_line}  |  meta: {meta}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load science_article codebase and data into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--force",       action="store_true",
                        help="Re-load even if collection already has documents.")
    parser.add_argument("--skip-code",   action="store_true",
                        help="Skip Python source code loading.")
    parser.add_argument("--skip-csv",    action="store_true",
                        help="Skip CSV column schema loading.")
    parser.add_argument("--skip-db",     action="store_true",
                        help="Skip SQLite table schema loading.")
    parser.add_argument("--skip-config", action="store_true",
                        help="Skip config constants loading.")
    parser.add_argument("--verify",      action="store_true",
                        help="Run sample queries after loading to verify embeddings.")
    parser.add_argument("--chroma-dir",  default=str(CHROMA_DIR),
                        help=f"ChromaDB persistence directory (default: {CHROMA_DIR})")
    args = parser.parse_args()

    print(f"ChromaDB storage : {args.chroma_dir}")
    print(f"Project root     : {ROOT}")

    client     = get_chroma_client(args.chroma_dir)
    code_col   = get_or_create_collection(client, CODE_COLLECTION)
    schema_col = get_or_create_collection(client, SCHEMA_COLLECTION)

    totals = {"code": 0, "csv": 0, "db": 0, "config": 0}

    # ── 1. Python code ────────────────────────────────────────────────────────
    if not args.skip_code:
        if is_collection_populated(code_col) and not args.force:
            print(f"\n[SKIP] '{CODE_COLLECTION}' already has {code_col.count()} docs."
                  f" Use --force to reload.")
        else:
            print(f"\n[1/4] Loading Python source code ({CODE_COLLECTION}) …")
            totals["code"] = load_code_to_chroma(code_col)
            print(f"      OK {totals['code']} code documents loaded.")

    # ── 2. CSV column schema ──────────────────────────────────────────────────
    if not args.skip_csv:
        if not FEATURES_CSV.exists():
            print(f"\n[SKIP] CSV not found: {FEATURES_CSV}")
        elif is_collection_populated(schema_col) and not args.force:
            print(f"\n[SKIP] '{SCHEMA_COLLECTION}' already has {schema_col.count()} docs."
                  f" Use --force to reload.")
        else:
            print(f"\n[2/4] Loading CSV column schema ({FEATURES_CSV.name}) …")
            totals["csv"] = load_csv_schema_to_chroma(schema_col, FEATURES_CSV)
            print(f"      OK {totals['csv']} CSV column documents loaded.")

    # ── 3. SQLite table schemas ───────────────────────────────────────────────
    if not args.skip_db:
        if not DB_PATH.exists():
            print(f"\n[SKIP] Database not found: {DB_PATH}")
        else:
            print(f"\n[3/4] Loading SQLite table schemas ({DB_PATH.name}) …")
            totals["db"] = load_db_schema_to_chroma(schema_col, DB_PATH)
            print(f"      OK {totals['db']} DB table documents loaded.")

    # ── 4. Config constants ───────────────────────────────────────────────────
    if not args.skip_config:
        print(f"\n[4/4] Loading config constants …")
        totals["config"] = load_config_to_chroma(schema_col)
        print(f"      OK {totals['config']} config documents loaded.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print(f"  code_functions : {code_col.count():>5} documents")
    print(f"  data_schema    : {schema_col.count():>5} documents")
    loaded = sum(totals.values())
    if loaded:
        print(f"  Loaded this run: {loaded} new/updated documents")
    print(f"{'-' * 60}")

    # ── Verify ────────────────────────────────────────────────────────────────
    if args.verify:
        verify_load(code_col, schema_col)


if __name__ == "__main__":
    main()
