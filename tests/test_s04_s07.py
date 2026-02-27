"""Tests for s04_sentinel1.py, s05_topography.py, s06_soil_maps.py, s07_hyperspectral.py"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src import s04_sentinel1, s05_topography, s06_soil_maps, s07_hyperspectral


# ─── Sentinel-1 Tests ────────────────────────────────────────────

@patch("src.s04_sentinel1.ee")
def test_extract_s1_features_success(mock_ee):
    """Test successful Sentinel-1 feature extraction."""
    mock_polygon = MagicMock()
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.filter.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 10
    mock_collection.median.return_value = MagicMock()
    
    # Mock VV, VH extraction
    mock_stats = {
        "VV": -12.5,
        "VH": -18.3,
        "VV_VH_ratio": 0.68,
    }
    # The source code chains: median() -> select("VV")/select("VH") -> divide -> rename -> addBands -> select([...]) -> reduceRegion -> getInfo
    mock_composite = mock_collection.median.return_value
    mock_composite.select.return_value = mock_composite
    mock_composite.divide.return_value.rename.return_value = MagicMock()
    mock_composite.addBands.return_value = mock_composite
    mock_composite.reduceRegion.return_value.getInfo.return_value = mock_stats
    
    result = s04_sentinel1.extract_s1_features(
        mock_polygon, "2021-04-01", "2021-10-01"
    )
    
    assert isinstance(result, dict)
    assert "VV" in result
    assert "VH" in result
    assert "VV_VH_ratio" in result
    assert "image_count" in result
    assert result["image_count"] == 10


@patch("src.s04_sentinel1.ee")
def test_extract_s1_features_no_data(mock_ee):
    """Test Sentinel-1 extraction when no images available."""
    mock_polygon = MagicMock()
    
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.filter.return_value = mock_collection
    mock_collection.size.return_value.getInfo.return_value = 0
    
    result = s04_sentinel1.extract_s1_features(
        mock_polygon, "2021-04-01", "2021-10-01"
    )
    
    assert result is None


# ─── Topography Tests ────────────────────────────────────────────

@patch("src.s05_topography.ee")
def test_extract_topo_features(mock_ee):
    """Test topographic feature extraction."""
    mock_polygon = MagicMock()

    mock_stats = {
        "DEM": 450.5,
        "slope": 5.2,
        "aspect_sin": 0.0,
        "aspect_cos": -1.0,
        "TWI": 6.5,
        "plan_curvature": 0.1,
        "profile_curvature": -0.2,
        "TPI": 0.05,
    }

    # The code chains: ee.ImageCollection() -> .select('DEM') -> .mosaic()
    # then terrain ops, then .addBands() -> .select([...]) -> .reduceRegion() -> .getInfo()
    mock_dem = MagicMock()
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.select.return_value = mock_collection
    mock_collection.mosaic.return_value = mock_dem

    # Terrain products
    mock_ee.Terrain.slope.return_value = MagicMock()
    mock_ee.Terrain.aspect.return_value = MagicMock()
    mock_ee.Terrain.products.return_value = MagicMock()

    # Kernel creation
    mock_ee.Kernel.fixed.return_value = MagicMock()

    # All arithmetic ops on images return MagicMock (supporting chained calls)
    mock_dem.multiply.return_value = MagicMock()
    mock_dem.convolve.return_value = MagicMock()
    mock_dem.focal_mean.return_value = MagicMock()
    mock_dem.subtract.return_value = MagicMock()

    # Final chain: addBands -> select -> reduceRegion -> getInfo
    mock_dem.addBands.return_value = mock_dem
    mock_dem.select.return_value = mock_dem
    mock_dem.reduceRegion.return_value.getInfo.return_value = mock_stats

    result = s05_topography.extract_topo_features(mock_polygon)

    assert isinstance(result, dict)
    assert "DEM" in result
    assert "slope" in result
    assert "aspect_sin" in result
    assert "aspect_cos" in result
    assert result["DEM"] == 450.5


# ─── SoilGrids Tests ─────────────────────────────────────────────

def test_unit_conversions():
    """Test that UNIT_CONVERSIONS dict is complete and reasonable."""
    from src.s06_soil_maps import UNIT_CONVERSIONS, OUTPUT_NAMES

    expected_keys = ["sand", "silt", "clay", "soc", "ph", "cec", "bdod", "nitrogen"]
    for key in expected_keys:
        assert key in UNIT_CONVERSIONS, f"Missing conversion for {key}"
        assert key in OUTPUT_NAMES, f"Missing output name for {key}"
        assert UNIT_CONVERSIONS[key] > 0


def test_depth_weights_sum_to_one():
    """Depth weights for 0-30cm should sum to 1.0."""
    from src.s06_soil_maps import DEPTH_WEIGHTS_0_30CM
    total = sum(DEPTH_WEIGHTS_0_30CM.values())
    assert abs(total - 1.0) < 1e-10, f"Weights sum to {total}, expected 1.0"


@patch("src.s06_soil_maps.ee")
def test_extract_soilgrids_features(mock_ee):
    """Test SoilGrids feature extraction with mocked GEE."""
    mock_polygon = MagicMock()

    # Mock ee.Image: each property asset returns an image
    mock_image = MagicMock()
    mock_ee.Image.return_value = mock_image
    mock_ee.Reducer.mean.return_value = MagicMock()

    # Mock reduceRegion — returns raw SoilGrids value (g/kg)
    mock_selected = MagicMock()
    mock_image.select.return_value = mock_selected
    mock_selected.reduceRegion.return_value.getInfo.return_value = {"0-5cm_mean": 450.0}

    mock_polygon.centroid.return_value = MagicMock()

    result = s06_soil_maps.extract_soilgrids_features(
        mock_polygon, properties=["sand"]
    )

    assert isinstance(result, dict)
    assert "sand_pct_0_5cm" in result
    assert "sand_pct_0_30cm" in result
    # 450 g/kg * 0.1 = 45.0%
    assert result["sand_pct_0_5cm"] == 45.0


@patch("src.s06_soil_maps.ee")
def test_reduce_soilgrids_band_fallback(mock_ee):
    """Test centroid fallback when polygon reduction returns None."""
    mock_polygon = MagicMock()
    mock_image = MagicMock()
    mock_ee.Reducer.mean.return_value = MagicMock()

    mock_selected = MagicMock()
    mock_image.select.return_value = mock_selected

    # First call returns None (polygon too small), second returns value
    mock_selected.reduceRegion.return_value.getInfo.side_effect = [
        {"0-5cm_mean": None},
        {"0-5cm_mean": 500.0},
    ]
    mock_polygon.centroid.return_value = MagicMock()

    val = s06_soil_maps._reduce_soilgrids_band(
        mock_image, mock_polygon, "0-5cm_mean"
    )

    assert val == 500.0


# ─── Hyperspectral Tests ─────────────────────────────────────────

def test_check_prisma_availability():
    """Test PRISMA availability check."""
    bbox = (50.0, 60.0, 51.0, 61.0)
    result = s07_hyperspectral.check_prisma_availability(bbox)
    
    assert isinstance(result, dict)
    assert "source" in result
    assert "available" in result
    assert result["source"] == "PRISMA (ASI/ESA)"
    assert result["available"] is False


def test_check_enmap_availability():
    """Test EnMAP availability check."""
    bbox = (50.0, 60.0, 51.0, 61.0)
    result = s07_hyperspectral.check_enmap_availability(bbox)
    
    assert isinstance(result, dict)
    assert result["source"] == "EnMAP (DLR)"
    assert result["available"] is False


def test_check_lucas_availability():
    """Test LUCAS availability check."""
    result = s07_hyperspectral.check_lucas_availability()
    
    assert isinstance(result, dict)
    assert result["source"] == "LUCAS Soil Spectral Library (JRC)"
    assert result["available"] is True
    assert "EU countries only" in result["coverage"]
