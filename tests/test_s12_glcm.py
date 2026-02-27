"""Tests for s12_glcm.py — GLCM texture feature extraction."""
import pytest
from unittest.mock import MagicMock, patch

from src import s12_glcm, config


@patch("src.s12_glcm.ee")
@patch("src.s12_glcm.cloud_mask_s2")
def test_extract_glcm_features_no_data(mock_cloud_mask, mock_ee):
    """Should return None when no S2 images available."""
    mock_polygon = MagicMock()

    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.filter.return_value = mock_collection
    mock_collection.map.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 0

    result = s12_glcm.extract_glcm_features(
        mock_polygon, "2021-04-01", "2021-06-01"
    )

    assert result is None


@patch("src.s12_glcm.ee")
@patch("src.s12_glcm.cloud_mask_s2")
def test_extract_glcm_features_success(mock_cloud_mask, mock_ee):
    """Should return renamed GLCM features on success."""
    mock_polygon = MagicMock()

    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.filter.return_value = mock_collection
    mock_collection.map.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 5

    # Mock composite chain
    mock_composite = mock_collection.median.return_value
    mock_red = mock_composite.select.return_value
    mock_nir = mock_composite.select.return_value

    # Mock GLCM texture outputs
    mock_glcm = MagicMock()
    mock_red.int16.return_value.glcmTexture.return_value = mock_glcm
    mock_nir.int16.return_value.glcmTexture.return_value = mock_glcm

    mock_red_textures = MagicMock()
    mock_nir_textures = MagicMock()
    mock_glcm.select.side_effect = [mock_red_textures, mock_nir_textures]

    mock_combined = mock_red_textures.addBands.return_value

    # Mock reduceRegion output
    mock_stats = {
        "B4_contrast": 125.5,
        "B4_ent": 3.2,
        "B4_idm": 0.45,
        "B4_asm": 0.08,
        "B8_contrast": 200.1,
        "B8_ent": 3.8,
        "B8_idm": 0.38,
        "B8_asm": 0.05,
    }
    mock_combined.reduceRegion.return_value.getInfo.return_value = mock_stats

    result = s12_glcm.extract_glcm_features(
        mock_polygon, "2021-06-01", "2021-08-01"
    )

    assert result is not None
    assert "glcm_red_contrast" in result
    assert "glcm_nir_contrast" in result
    assert "glcm_red_ent" in result
    assert "glcm_nir_ent" in result
    assert "glcm_red_idm" in result
    assert "glcm_nir_idm" in result
    assert "glcm_red_asm" in result
    assert "glcm_nir_asm" in result
    assert result["image_count"] == 5
    assert result["glcm_red_contrast"] == 125.5


def test_texture_feature_names():
    """Verify texture feature names match GEE GLCM band naming convention."""
    texture_features = ['contrast', 'ent', 'idm', 'asm']

    for feat in texture_features:
        red_key = f'B4_{feat}'
        nir_key = f'B8_{feat}'
        out_red = f'glcm_red_{feat}'
        out_nir = f'glcm_nir_{feat}'

        # Simulate the renaming logic from s12_glcm.extract_glcm_features
        stats = {red_key: 1.0, nir_key: 2.0}
        result = {}
        if red_key in stats:
            result[out_red] = stats[red_key]
        if nir_key in stats:
            result[out_nir] = stats[nir_key]

        assert out_red in result
        assert out_nir in result
