"""
Cross-Module Pipeline Integrity Tests
=======================================
Validation tests that check consistency ACROSS modules.
No GEE or database connection required — pure static analysis.
"""
import pytest
import sqlite3
import re
import inspect
from pathlib import Path

from src import config


# ─── Real DB Schema (ground truth) ──────────────────────────────

SOIL_SAMPLES_COLUMNS = {
    "id", "year", "farm", "field_name", "grid_id",
    "ph", "k", "p", "hu", "s", "no3",
    "zn", "mo", "fe", "mg", "mn", "cu",
    "centroid_lon", "centroid_lat", "geometry_wkt",
    "protocol_number", "analysis_date", "sampling_date",
}


# ─── Config Completeness ─────────────────────────────────────────

def test_all_output_dirs_are_path_objects():
    """All *_DIR config constants should be pathlib.Path objects."""
    dir_attrs = [
        attr for attr in dir(config)
        if attr.endswith("_DIR") and not attr.startswith("_")
    ]

    for attr in dir_attrs:
        value = getattr(config, attr)
        assert isinstance(value, Path), (
            f"config.{attr} = {value!r} is not a Path object"
        )


def test_years_are_sorted_and_continuous():
    """YEARS should be a sorted list of consecutive integers."""
    years = config.YEARS
    assert years == sorted(years), "YEARS should be sorted"
    for i in range(1, len(years)):
        assert years[i] - years[i - 1] == 1, (
            f"Gap between {years[i - 1]} and {years[i]}"
        )


def test_seasons_month_ranges_valid():
    """Season month ranges should be valid (1-12) and non-overlapping."""
    all_months = set()
    for season, (start, end) in config.SEASONS.items():
        assert 1 <= start <= 12, f"{season} start month {start} invalid"
        assert 1 <= end <= 12, f"{season} end month {end} invalid"
        assert start <= end, f"{season} start > end: {start} > {end}"

        months = set(range(start, end + 1))
        overlap = all_months & months
        assert not overlap, (
            f"{season} months {months} overlap with previously defined months: {overlap}"
        )
        all_months |= months


# ─── S2 Indices vs Code ─────────────────────────────────────────

def test_all_s2_indices_have_implementation():
    """Every index in config.S2_INDICES must have a branch in compute_s2_indices."""
    from src import s02_sentinel2

    source = inspect.getsource(s02_sentinel2.compute_s2_indices)

    for idx_name in config.S2_INDICES:
        # Check that the index name appears as a string literal in the function
        assert f'"{idx_name}"' in source or f"'{idx_name}'" in source, (
            f"Index '{idx_name}' is defined in config.S2_INDICES but has NO "
            f"implementation branch in compute_s2_indices(). "
            f"This will cause a GEE 'Band not found' error."
        )


def test_all_l8_indices_have_implementation():
    """Every index in config.L8_INDICES must have a branch in compute_l8_indices."""
    from src import s03_landsat8

    source = inspect.getsource(s03_landsat8.compute_l8_indices)

    for idx_name in config.L8_INDICES:
        assert f'"{idx_name}"' in source or f"'{idx_name}'" in source, (
            f"Index '{idx_name}' is defined in config.L8_INDICES but has NO "
            f"implementation in compute_l8_indices()."
        )


# ─── DB Schema Alignment ─────────────────────────────────────────

def _extract_sql_columns(source_code: str) -> list[str]:
    """Extract column names from SELECT ... FROM blocks in source code."""
    columns = []
    select_match = re.search(r"SELECT\s+(.+?)\s+FROM", source_code, re.DOTALL)
    if not select_match:
        return columns

    block = select_match.group(1)
    for line in block.split("\n"):
        line = line.strip().rstrip(",")
        if not line or line == "DISTINCT":
            continue
        # Handle aliases like MIN(centroid_lon) as centroid_lon
        alias_match = re.search(r"as\s+(\w+)", line, re.IGNORECASE)
        if alias_match:
            columns.append(alias_match.group(1))
        elif re.match(r"^\w+$", line):
            columns.append(line)

    return columns


def test_s10_semivariogram_sql_columns():
    """s10_semivariogram.load_soil_data() SQL columns must match DB schema."""
    from src import s10_semivariogram

    source = inspect.getsource(s10_semivariogram.load_soil_data)
    columns = _extract_sql_columns(source)

    for col in columns:
        assert col in SOIL_SAMPLES_COLUMNS, (
            f"s10_semivariogram uses column '{col}' which does NOT exist in soil_samples. "
            f"Available: {sorted(SOIL_SAMPLES_COLUMNS)}"
        )


def test_db_utils_sql_columns():
    """db_utils.get_field_polygons() SQL columns must match DB schema."""
    from src import db_utils

    source = inspect.getsource(db_utils.get_field_polygons)
    columns = _extract_sql_columns(source)

    for col in columns:
        assert col in SOIL_SAMPLES_COLUMNS, (
            f"db_utils uses column '{col}' which does NOT exist in soil_samples. "
            f"Available: {sorted(SOIL_SAMPLES_COLUMNS)}"
        )


# ─── Merge CSV Glob Patterns ─────────────────────────────────────

def test_merge_glob_patterns_match_output_filenames():
    """s08_merge loader glob patterns must match actual script output filenames."""
    # These are the patterns used in s08_merge_features.py loaders
    # vs the actual filename templates in each extraction script
    expected_patterns = {
        "era5_temperature_*.csv": "era5_temperature_{year}.csv",          # s01
        "s2_features_*.csv": "s2_features_{year}_{season}.csv",           # s02
        "l8_features_*.csv": "l8_features_{year}_{season}.csv",           # s03
        "s1_features_*.csv": "s1_features_{year}.csv",                    # s04
        "climate_features_*.csv": "climate_features_{year}.csv",          # s09
        "spectral_eng_*.csv": "spectral_eng_{year}_{season}.csv",         # s11
        "glcm_features_*.csv": "glcm_features_{year}_{season}.csv",      # s12
    }

    from src import s08_merge_features
    source = inspect.getsource(s08_merge_features)

    for glob_pattern in expected_patterns:
        assert glob_pattern in source, (
            f"Glob pattern '{glob_pattern}' not found in s08_merge_features.py. "
            f"The merge step will not pick up files matching template "
            f"'{expected_patterns[glob_pattern]}'."
        )


def test_topo_filename_matches_merge_loader():
    """Topography loader uses exact filename, not glob — must match."""
    from src import s08_merge_features

    source = inspect.getsource(s08_merge_features.load_topography_data)
    assert "topo_features.csv" in source


# ─── Import Chain ────────────────────────────────────────────────

def test_all_modules_import_successfully():
    """All pipeline modules should import without error."""
    modules = [
        "src.config",
        "src.db_utils",
        "src.file_utils",
        "src.gee_auth",
        "src.s01_temperature",
        "src.s02_sentinel2",
        "src.s03_landsat8",
        "src.s04_sentinel1",
        "src.s05_topography",
        "src.s06_soil_maps",
        "src.s07_hyperspectral",
        "src.s08_merge_features",
        "src.s09_climate",
        "src.s10_semivariogram",
        "src.s11_spectral_eng",
        "src.s12_glcm",
    ]

    import importlib
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            pytest.fail(f"Failed to import {mod_name}: {e}")


# ─── SoilGrids Config ───────────────────────────────────────────

def test_soilgrids_keys_complete():
    """SOILGRIDS config should have entries for all expected soil properties."""
    expected = {"sand", "silt", "clay", "soc", "ph", "cec", "bdod", "nitrogen"}
    actual = set(config.SOILGRIDS.keys())

    missing = expected - actual
    assert not missing, f"Missing SoilGrids properties: {missing}"


def test_soilgrids_depths_valid():
    """SOILGRIDS_DEPTHS should contain valid depth range strings."""
    for depth in config.SOILGRIDS_DEPTHS:
        assert "_mean" in depth, f"Depth '{depth}' missing '_mean' suffix"
        assert "cm" in depth, f"Depth '{depth}' missing 'cm' unit"


def test_soilgrids_primary_depth_in_depths():
    """Primary depth must be listed in SOILGRIDS_DEPTHS."""
    assert config.SOILGRIDS_PRIMARY_DEPTH in config.SOILGRIDS_DEPTHS


# ─── File Utils ──────────────────────────────────────────────────

def test_should_skip_file_nonexistent(tmp_path):
    """should_skip_file returns False for nonexistent files."""
    from src.file_utils import should_skip_file
    assert should_skip_file(tmp_path / "nonexistent.csv") is False


def test_should_skip_file_empty(tmp_path):
    """should_skip_file returns False for empty files."""
    from src.file_utils import should_skip_file
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert should_skip_file(empty) is False


def test_should_skip_file_with_content(tmp_path):
    """should_skip_file returns True for files with content."""
    from src.file_utils import should_skip_file
    filled = tmp_path / "data.csv"
    filled.write_text("a,b\n1,2\n")
    assert should_skip_file(filled) is True
