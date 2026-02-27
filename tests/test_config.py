"""Tests for config.py"""
import pytest
from pathlib import Path
from src import config


def test_paths_exist():
    """Test that all path constants are defined."""
    assert config.ROOT.exists()
    assert config.DATA_DIR == config.ROOT / "data"
    assert config.DB_PATH == config.DATA_DIR / "soil_analysis.db"


def test_years_configuration():
    """Test years configuration."""
    assert config.YEARS == [2022, 2023]
    assert all(isinstance(y, int) for y in config.YEARS)
    assert all(2022 <= y <= 2023 for y in config.YEARS)


def test_seasons_configuration():
    """Test seasonal composites configuration."""
    assert len(config.SEASONS) == 4
    assert "spring" in config.SEASONS
    assert "summer" in config.SEASONS
    assert "late_summer" in config.SEASONS
    assert "autumn" in config.SEASONS
    
    # Check month ranges
    assert config.SEASONS["spring"] == (4, 5)
    assert config.SEASONS["summer"] == (6, 7)
    assert config.SEASONS["late_summer"] == (8, 9)
    assert config.SEASONS["autumn"] == (10, 10)


def test_s2_configuration():
    """Test Sentinel-2 configuration."""
    assert config.S2_COLLECTION == "COPERNICUS/S2_SR_HARMONIZED"
    assert config.S2_CLOUD_THRESHOLD == 20
    assert isinstance(config.S2_BANDS, dict)
    assert len(config.S2_BANDS) == 10
    assert "B2" in config.S2_BANDS
    assert "B8" in config.S2_BANDS


def test_s2_indices():
    """Test Sentinel-2 indices formulas."""
    assert isinstance(config.S2_INDICES, dict)
    assert len(config.S2_INDICES) == 7
    assert "NDVI" in config.S2_INDICES
    assert "NDRE" in config.S2_INDICES
    assert "GNDVI" in config.S2_INDICES
    assert "Cl_Red_Edge" in config.S2_INDICES
    assert "SAVI" in config.S2_INDICES
    assert "BSI" in config.S2_INDICES


def test_l8_configuration():
    """Test Landsat 8 configuration."""
    assert config.L8_COLLECTION == "LANDSAT/LC08/C02/T1_L2"
    assert isinstance(config.L8_BANDS, dict)
    assert len(config.L8_BANDS) == 6
    assert "SR_B2" in config.L8_BANDS


def test_l8_indices():
    """Test Landsat 8 indices formulas."""
    assert isinstance(config.L8_INDICES, dict)
    assert len(config.L8_INDICES) == 3
    assert "NDVI" in config.L8_INDICES
    assert "GNDVI" in config.L8_INDICES
    assert "SAVI" in config.L8_INDICES


def test_s1_configuration():
    """Test Sentinel-1 configuration."""
    assert config.S1_COLLECTION == "COPERNICUS/S1_GRD"
    assert config.S1_POLARIZATIONS == ["VV", "VH"]


def test_dem_configuration():
    """Test DEM configuration."""
    assert config.DEM_COLLECTION == "COPERNICUS/DEM/GLO30"


def test_era5_configuration():
    """Test ERA5 configuration."""
    assert config.ERA5_COLLECTION == "ECMWF/ERA5_LAND/MONTHLY_AGGR"
    assert config.TEMP_THRESHOLD_C == 0.0


def test_gee_settings():
    """Test GEE settings."""
    assert config.GEE_SCALE == 10
    assert config.GEE_MAX_PIXELS == 1e9


def test_crs_configuration():
    """Test CRS configuration."""
    assert config.CRS_WGS84 == "EPSG:4326"
    assert config.CRS_UTM == "EPSG:32641"
