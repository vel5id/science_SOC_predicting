"""Tests for s02_sentinel2.py"""
import pytest
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from shapely.geometry import Polygon

from src import s02_sentinel2, config


@pytest.fixture
def mock_seasonal_windows(tmp_path):
    """Create mock seasonal windows file."""
    windows_file = tmp_path / "seasonal_windows.txt"
    windows_content = """
2021:
  spring      : 2021-04-01 to 2021-06-01
  summer      : 2021-06-01 to 2021-08-01
  late_summer : 2021-08-01 to 2021-10-01
  autumn      : 2021-10-01 to 2021-11-01

2022:
  spring      : 2022-04-01 to 2022-06-01
  summer      : 2022-06-01 to 2022-08-01
"""
    windows_file.write_text(windows_content)
    return windows_file


def test_load_seasonal_windows(mock_seasonal_windows, monkeypatch):
    """Test loading seasonal windows from file."""
    monkeypatch.setattr(config, "TEMP_DIR", mock_seasonal_windows.parent)
    
    windows = s02_sentinel2.load_seasonal_windows()
    
    assert isinstance(windows, dict)
    assert 2021 in windows
    assert 2022 in windows
    assert "spring" in windows[2021]
    assert windows[2021]["spring"] == ("2021-04-01", "2021-06-01")
    assert windows[2021]["summer"] == ("2021-06-01", "2021-08-01")


def test_load_seasonal_windows_missing_file(tmp_path, monkeypatch):
    """Test loading seasonal windows when file doesn't exist."""
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    
    with pytest.raises(FileNotFoundError):
        s02_sentinel2.load_seasonal_windows()


@patch("src.s02_sentinel2.ee")
def test_cloud_mask_s2(mock_ee):
    """Test Sentinel-2 cloud masking."""
    mock_image = MagicMock()
    mock_scl = MagicMock()
    mock_image.select.return_value = mock_scl
    
    # Mock SCL band operations
    mock_scl.neq.return_value = mock_scl
    mock_scl.And.return_value = mock_scl
    
    result = s02_sentinel2.cloud_mask_s2(mock_image)
    
    # Should call select for SCL band
    mock_image.select.assert_called_with("SCL")
    # Should call updateMask
    mock_image.updateMask.assert_called_once()


@patch("src.s02_sentinel2.ee")
def test_compute_s2_indices(mock_ee):
    """Test Sentinel-2 index computation."""
    mock_image = MagicMock()
    
    # Mock band selection and operations
    mock_image.select.return_value = MagicMock()
    mock_image.normalizedDifference.return_value = MagicMock()
    mock_image.addBands.return_value = mock_image
    
    result = s02_sentinel2.compute_s2_indices(mock_image)
    
    # Should compute NDVI, NDRE, GNDVI
    assert mock_image.normalizedDifference.call_count >= 3
    # Should add bands
    assert mock_image.addBands.called


@patch("src.s02_sentinel2.ee")
def test_extract_s2_features_success(mock_ee):
    """Test successful S2 feature extraction."""
    mock_polygon = MagicMock()
    
    # Mock ImageCollection
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.filter.return_value = mock_collection
    mock_collection.map.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 5  # 5 images
    mock_collection.median.return_value = MagicMock()
    
    # Mock feature extraction
    mock_stats = {
        "B2": 0.1,
        "B3": 0.15,
        "B4": 0.2,
        "B8": 0.5,
        "NDVI": 0.6,
    }
    # The source code chains: median() -> select(bands) -> compute_s2_indices (addBands x6) -> select(all_bands) -> reduceRegion -> getInfo
    mock_composite = mock_collection.median.return_value
    mock_composite.select.return_value = mock_composite
    mock_composite.normalizedDifference.return_value = MagicMock()
    mock_composite.addBands.return_value = mock_composite
    mock_composite.reduceRegion.return_value.getInfo.return_value = mock_stats
    
    result = s02_sentinel2.extract_s2_features(
        mock_polygon, "2021-04-01", "2021-06-01"
    )
    
    assert isinstance(result, dict)
    assert "B2" in result
    assert "NDVI" in result
    assert "image_count" in result
    assert result["image_count"] == 5


@patch("src.s02_sentinel2.ee")
def test_extract_s2_features_no_data(mock_ee):
    """Test S2 feature extraction when no images available."""
    mock_polygon = MagicMock()
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.filter.return_value = mock_collection
    mock_collection.map.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 0  # No images
    
    result = s02_sentinel2.extract_s2_features(
        mock_polygon, "2021-04-01", "2021-06-01"
    )
    
    assert result is None


@patch("src.s02_sentinel2.ee")
@patch("src.s02_sentinel2.json")
@patch("src.s02_sentinel2.gpd")
def test_process_fields_for_season(mock_gpd, mock_json, mock_ee):
    """Test processing fields for a season."""
    # Create mock GeoDataFrame
    mock_gdf = MagicMock()
    mock_gdf.__getitem__.return_value = mock_gdf
    mock_gdf.copy.return_value = mock_gdf
    mock_gdf.__len__.return_value = 2
    mock_gdf.iterrows.return_value = [
        (0, {"year": 2021, "farm": "Farm1", "field_name": "Field1", 
             "centroid_lon": 50.0, "centroid_lat": 60.0, "geometry": Polygon()}),
        (1, {"year": 2021, "farm": "Farm1", "field_name": "Field2",
             "centroid_lon": 50.1, "centroid_lat": 60.1, "geometry": Polygon()}),
    ]
    
    # Mock GEE geometry conversion
    mock_json.loads.return_value = {"features": [{"geometry": {}}]}
    mock_ee.Geometry.return_value = MagicMock()
    
    # Mock extract_s2_features
    with patch("src.s02_sentinel2.extract_s2_features") as mock_extract:
        mock_extract.return_value = {"B2": 0.1, "NDVI": 0.6}
        
        result = s02_sentinel2.process_fields_for_season(
            mock_gdf, 2021, "spring", "2021-04-01", "2021-06-01"
        )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "year" in result.columns
    assert "season" in result.columns
    assert "farm" in result.columns
