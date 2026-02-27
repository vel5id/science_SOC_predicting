"""Tests for s03_landsat8.py"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src import s03_landsat8


@patch("src.s03_landsat8.ee")
def test_cloud_mask_l8(mock_ee):
    """Test Landsat 8 cloud masking."""
    mock_image = MagicMock()
    mock_qa = MagicMock()
    mock_image.select.return_value = mock_qa
    
    # Mock QA_PIXEL operations
    mock_qa.bitwiseAnd.return_value = mock_qa
    mock_qa.eq.return_value = mock_qa
    mock_qa.And.return_value = mock_qa
    
    result = s03_landsat8.cloud_mask_l8(mock_image)
    
    mock_image.select.assert_called_with("QA_PIXEL")
    mock_image.updateMask.assert_called_once()


@patch("src.s03_landsat8.ee")
def test_compute_l8_indices(mock_ee):
    """Test Landsat 8 index computation."""
    mock_image = MagicMock()
    
    # Mock band operations
    mock_image.normalizedDifference.return_value = MagicMock()
    mock_image.select.return_value = MagicMock()
    mock_image.addBands.return_value = mock_image
    
    result = s03_landsat8.compute_l8_indices(mock_image)
    
    # Should compute NDVI, GNDVI
    assert mock_image.normalizedDifference.call_count >= 2
    # Should add bands
    mock_image.addBands.assert_called_once()


@patch("src.s03_landsat8.ee")
def test_extract_l8_features_success(mock_ee):
    """Test successful Landsat 8 feature extraction."""
    mock_polygon = MagicMock()
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.map.return_value = mock_collection  # handles both .map() calls
    mock_collection.size.return_value.getInfo.return_value = 3
    mock_collection.median.return_value = MagicMock()
    
    mock_stats = {
        "SR_B2": 0.1,
        "SR_B3": 0.15,
        "SR_B4": 0.2,
        "SR_B5": 0.5,
        "NDVI": 0.6,
    }
    # The source code chains: median() -> select(bands) -> compute_l8_indices (addBands) -> select(all_bands) -> reduceRegion -> getInfo
    mock_composite = mock_collection.median.return_value
    mock_composite.select.return_value = mock_composite
    mock_composite.normalizedDifference.return_value = MagicMock()
    mock_composite.addBands.return_value = mock_composite
    mock_composite.reduceRegion.return_value.getInfo.return_value = mock_stats
    
    result = s03_landsat8.extract_l8_features(
        mock_polygon, "2021-04-01", "2021-06-01"
    )
    
    assert isinstance(result, dict)
    assert "SR_B2" in result
    assert "NDVI" in result
    assert "image_count" in result
    assert result["image_count"] == 3


@patch("src.s03_landsat8.ee")
def test_extract_l8_features_no_data(mock_ee):
    """Test Landsat 8 extraction when no images available."""
    mock_polygon = MagicMock()
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.map.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 0
    
    result = s03_landsat8.extract_l8_features(
        mock_polygon, "2021-04-01", "2021-06-01"
    )
    
    assert result is None
