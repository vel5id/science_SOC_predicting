"""Tests for s09_climate.py"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


from src import s09_climate, config


def test_kelvin_to_celsius():
    """Test Kelvin to Celsius conversion."""
    assert s09_climate.kelvin_to_celsius(273.15) == 0.0
    assert s09_climate.kelvin_to_celsius(300.0) == pytest.approx(26.85)
    assert s09_climate.kelvin_to_celsius(0.0) == -273.15


@patch("src.s09_climate.ee")
def test_reduce_with_fallback_primary(mock_ee):
    """Test _reduce_with_fallback when primary reduction succeeds."""
    mock_polygon = MagicMock()
    mock_image = MagicMock()
    mock_ee.Reducer.mean.return_value = MagicMock()

    mock_image.reduceRegion.return_value.getInfo.return_value = {
        "temperature_2m": 280.0
    }

    val = s09_climate._reduce_with_fallback(
        mock_image, mock_polygon, "temperature_2m"
    )

    assert val == 280.0
    # centroid should NOT be called when primary succeeds
    mock_polygon.centroid.assert_not_called()


@patch("src.s09_climate.ee")
def test_reduce_with_fallback_centroid(mock_ee):
    """Test _reduce_with_fallback centroid fallback when polygon returns None."""
    mock_polygon = MagicMock()
    mock_image = MagicMock()
    mock_ee.Reducer.mean.return_value = MagicMock()

    # Primary returns None, fallback returns value
    mock_image.reduceRegion.return_value.getInfo.side_effect = [
        {"temperature_2m": None},
        {"temperature_2m": 285.0},
    ]
    mock_polygon.centroid.return_value = MagicMock()

    val = s09_climate._reduce_with_fallback(
        mock_image, mock_polygon, "temperature_2m"
    )

    assert val == 285.0
    mock_polygon.centroid.assert_called_once()


@patch("src.s09_climate._reduce_with_fallback")
@patch("src.s09_climate.ee")
def test_extract_climate_features(mock_ee, mock_reduce):
    """Test extract_climate_features returns MAT, MAP, GS_temp, GS_precip."""
    mock_polygon = MagicMock()
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.select.return_value = mock_collection
    mock_collection.mean.return_value = MagicMock()
    mock_collection.sum.return_value = MagicMock()

    # MAT_K=280 → 6.85°C, MAP_m=0.5 → 500mm, GS_temp_K=290 → 16.85°C, GS_precip_m=0.3 → 300mm
    mock_reduce.side_effect = [280.0, 0.5, 290.0, 0.3]

    result = s09_climate.extract_climate_features(
        mock_polygon, 2021, "2021-04-01", "2021-10-01"
    )

    assert "MAT" in result
    assert "MAP" in result
    assert "GS_temp" in result
    assert "GS_precip" in result
    assert result["MAT"] == pytest.approx(6.85, abs=0.01)
    assert result["MAP"] == pytest.approx(500.0, abs=0.1)
    assert result["GS_temp"] == pytest.approx(16.85, abs=0.01)
    assert result["GS_precip"] == pytest.approx(300.0, abs=0.1)


@patch("src.s09_climate._reduce_with_fallback")
@patch("src.s09_climate.ee")
def test_extract_climate_features_null_handling(mock_ee, mock_reduce):
    """Test that None values from GEE are handled gracefully."""
    mock_polygon = MagicMock()
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.select.return_value = mock_collection
    mock_collection.mean.return_value = MagicMock()
    mock_collection.sum.return_value = MagicMock()

    # All None
    mock_reduce.side_effect = [None, None, None, None]

    result = s09_climate.extract_climate_features(
        mock_polygon, 2021, "2021-04-01", "2021-10-01"
    )

    assert result["MAT"] is None
    assert result["MAP"] is None
    assert result["GS_temp"] is None
    assert result["GS_precip"] is None
