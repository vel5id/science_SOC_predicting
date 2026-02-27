"""Tests for s01_temperature.py"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src import s01_temperature, config


def test_kelvin_to_celsius():
    """Test Kelvin to Celsius conversion."""
    assert s01_temperature.kelvin_to_celsius(273.15) == 0.0
    assert s01_temperature.kelvin_to_celsius(300.0) == pytest.approx(26.85, rel=0.01)
    assert s01_temperature.kelvin_to_celsius(0.0) == -273.15


@patch("src.s01_temperature.ee")
def test_extract_temperature_for_year(mock_ee):
    """Test temperature extraction for a single year."""
    # Mock GEE objects
    mock_geometry = MagicMock()
    mock_ee.Geometry.Rectangle.return_value = mock_geometry
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.select.return_value = mock_collection
    mock_collection.mean.return_value = MagicMock()
    
    # Mock reduceRegion to return temperature data
    mock_stats = {"temperature_2m": 280.0}  # 6.85°C
    mock_collection.mean.return_value.reduceRegion.return_value.getInfo.return_value = mock_stats
    
    bbox = (50.0, 60.0, 51.0, 61.0)
    bbox = (50.0, 60.0, 51.0, 61.0)
    result = s01_temperature.extract_temperature_for_year(2021, bbox)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12  # 12 months
    assert "year" in result.columns
    assert "month" in result.columns
    assert "mean_temp_c" in result.columns
    assert "is_growing_season" in result.columns
    assert all(result["year"] == 2021)


@patch("src.s01_temperature.ee")
def test_extract_temperature_no_data(mock_ee):
    """Test temperature extraction when no data available."""
    mock_geometry = MagicMock()
    mock_ee.Geometry.Rectangle.return_value = mock_geometry
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.select.return_value = mock_collection
    mock_collection.mean.return_value = MagicMock()
    
    # Mock no data
    mock_stats = {}
    mock_collection.mean.return_value.reduceRegion.return_value.getInfo.return_value = mock_stats
    
    bbox = (50.0, 60.0, 51.0, 61.0)
    bbox = (50.0, 60.0, 51.0, 61.0)
    result = s01_temperature.extract_temperature_for_year(2021, bbox)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0  # No data


def test_determine_seasonal_windows():
    """Test seasonal window determination."""
    # Create test temperature DataFrame
    temp_df = pd.DataFrame({
        "year": [2021] * 12,
        "month": list(range(1, 13)),
        "mean_temp_c": [-5, -3, 0, 5, 10, 15, 20, 18, 12, 8, 2, -2],
        "is_growing_season": [False, False, False, True, True, True, True, True, True, True, True, False],
    })
    
    windows = s01_temperature.determine_seasonal_windows(temp_df)
    
    assert isinstance(windows, dict)
    assert "spring" in windows
    assert "summer" in windows
    
    # Check that windows are tuples of (start_date, end_date)
    for season, (start, end) in windows.items():
        assert isinstance(start, str)
        assert isinstance(end, str)
        assert start.startswith("2021-")
        assert end.startswith("2021-")


def test_determine_seasonal_windows_no_growing_season():
    """Test seasonal windows when no growing season detected."""
    temp_df = pd.DataFrame({
        "year": [2020] * 12,
        "month": list(range(1, 13)),
        "mean_temp_c": [-10] * 12,
        "is_growing_season": [False] * 12,
    })
    
    windows = s01_temperature.determine_seasonal_windows(temp_df)
    
    assert isinstance(windows, dict)
    assert len(windows) == 0


def test_determine_seasonal_windows_partial_overlap():
    """Test seasonal windows with partial season overlap."""
    # Only summer months have growing season
    temp_df = pd.DataFrame({
        "year": [2021] * 12,
        "month": list(range(1, 13)),
        "mean_temp_c": [-5, -3, -1, -0.5, 2, 10, 15, 12, 5, 1, -2, -5],
        "is_growing_season": [False, False, False, False, True, True, True, True, True, True, False, False],
    })
    
    windows = s01_temperature.determine_seasonal_windows(temp_df)
    
    # Should have summer and late_summer
    assert "summer" in windows
    assert "late_summer" in windows
    # Spring might be partial or missing
