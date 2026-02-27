"""Tests for s08_merge_features.py"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src import s08_merge_features, config


@pytest.fixture
def mock_data_files(tmp_path):
    """Create mock CSV files for testing."""
    # Create directories
    (tmp_path / "temperature").mkdir()
    (tmp_path / "sentinel2").mkdir()
    (tmp_path / "landsat8").mkdir()
    (tmp_path / "sentinel1").mkdir()
    (tmp_path / "topography").mkdir()
    (tmp_path / "climate").mkdir()

    # Temperature data
    temp_df = pd.DataFrame({
        "year": [2021, 2021],
        "month": [4, 5],
        "mean_temp_c": [5.0, 10.0],
        "is_growing_season": [True, True],
    })
    temp_df.to_csv(tmp_path / "temperature" / "era5_temperature_2021.csv", index=False)

    # Sentinel-2 data
    s2_df = pd.DataFrame({
        "year": [2021, 2021],
        "season": ["spring", "summer"],
        "farm": ["Farm1", "Farm1"],
        "field_name": ["Field1", "Field1"],
        "centroid_lon": [50.0, 50.0],
        "centroid_lat": [60.0, 60.0],
        "B2": [0.1, 0.12],
        "NDVI": [0.6, 0.7],
        "image_count": [5, 8],
    })
    s2_df.to_csv(tmp_path / "sentinel2" / "s2_features_2021_spring.csv", index=False)

    # Landsat 8 data
    l8_df = pd.DataFrame({
        "year": [2021],
        "season": ["spring"],
        "farm": ["Farm1"],
        "field_name": ["Field1"],
        "centroid_lon": [50.0],
        "centroid_lat": [60.0],
        "SR_B2": [0.11],
        "NDVI": [0.65],
    })
    l8_df.to_csv(tmp_path / "landsat8" / "l8_features_2021_spring.csv", index=False)

    # Sentinel-1 data (multiple rows per field = pixel-level)
    s1_df = pd.DataFrame({
        "year": [2021, 2021, 2021],
        "farm": ["Farm1", "Farm1", "Farm1"],
        "field_name": ["Field1", "Field1", "Field1"],
        "centroid_lon": [50.0, 50.01, 50.02],
        "centroid_lat": [60.0, 60.01, 60.02],
        "VV": [-12.5, -13.0, -12.0],
        "VH": [-18.3, -19.0, -17.5],
        "VV_VH_ratio": [0.68, 0.68, 0.69],
        "image_count": [9, 9, 9],
    })
    s1_df.to_csv(tmp_path / "sentinel1" / "s1_features_2021.csv", index=False)

    # Topography data (multiple rows per field = pixel-level)
    topo_df = pd.DataFrame({
        "year": [2021, 2021, 2021],
        "farm": ["Farm1", "Farm1", "Farm1"],
        "field_name": ["Field1", "Field1", "Field1"],
        "centroid_lon": [50.0, 50.01, 50.02],
        "centroid_lat": [60.0, 60.01, 60.02],
        "DEM": [450.5, 451.0, 449.5],
        "slope": [5.2, 6.0, 4.5],
        "TWI": [6.5, 6.3, 6.8],
    })
    topo_df.to_csv(tmp_path / "topography" / "topo_features.csv", index=False)

    # Climate data
    climate_df = pd.DataFrame({
        "year": [2021, 2021],
        "farm": ["Farm1", "Farm1"],
        "field_name": ["Field1", "Field1"],
        "centroid_lon": [50.0, 50.01],
        "centroid_lat": [60.0, 60.01],
        "MAT": [5.64, 5.64],
        "MAP": [521.6, 521.6],
        "GS_temp": [14.27, 14.27],
        "GS_precip": [398.5, 398.5],
    })
    climate_df.to_csv(tmp_path / "climate" / "climate_features_2021.csv", index=False)

    return tmp_path


def test_load_temperature_data(mock_data_files, monkeypatch):
    """Test loading temperature data."""
    monkeypatch.setattr(config, "TEMP_DIR", mock_data_files / "temperature")

    df = s08_merge_features.load_temperature_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "year" in df.columns
    assert "month" in df.columns


def test_load_sentinel2_data(mock_data_files, monkeypatch):
    """Test loading Sentinel-2 data."""
    monkeypatch.setattr(config, "S2_DIR", mock_data_files / "sentinel2")

    df = s08_merge_features.load_sentinel2_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "season" in df.columns
    assert "NDVI" in df.columns


def test_load_landsat8_data(mock_data_files, monkeypatch):
    """Test loading Landsat 8 data."""
    monkeypatch.setattr(config, "L8_DIR", mock_data_files / "landsat8")

    df = s08_merge_features.load_landsat8_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "SR_B2" in df.columns


def test_load_sentinel1_data(mock_data_files, monkeypatch):
    """Test loading Sentinel-1 data."""
    monkeypatch.setattr(config, "S1_DIR", mock_data_files / "sentinel1")

    df = s08_merge_features.load_sentinel1_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "VV" in df.columns


def test_load_topography_data(mock_data_files, monkeypatch):
    """Test loading topography data."""
    monkeypatch.setattr(config, "TOPO_DIR", mock_data_files / "topography")

    df = s08_merge_features.load_topography_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "DEM" in df.columns


@patch("src.s08_merge_features.get_connection")
def test_load_soil_samples(mock_get_connection):
    """Test loading soil samples from database."""
    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn

    with patch("pandas.read_sql_query") as mock_read_sql:
        mock_df = pd.DataFrame({
            "year": [2021],
            "farm": ["Farm1"],
            "field_name": ["Field1"],
            "ph": [7.2],
        })
        mock_read_sql.return_value = mock_df

        df = s08_merge_features.load_soil_samples()

        assert isinstance(df, pd.DataFrame)
        assert "ph" in df.columns


class TestAddSocColumn:
    """Tests for SOC conversion."""

    def test_soc_conversion_basic(self):
        """SOC = hu * 0.58."""
        df = pd.DataFrame({"hu": [3.5, 4.0, 2.1]})
        result = s08_merge_features.add_soc_column(df)

        assert "soc" in result.columns
        assert result["soc"].iloc[0] == pytest.approx(3.5 * 0.58, abs=0.001)
        assert result["soc"].iloc[1] == pytest.approx(4.0 * 0.58, abs=0.001)
        assert result["soc"].iloc[2] == pytest.approx(2.1 * 0.58, abs=0.001)

    def test_soc_conversion_with_nulls(self):
        """SOC should be NaN where hu is NaN."""
        df = pd.DataFrame({"hu": [3.5, None, 2.1]})
        result = s08_merge_features.add_soc_column(df)

        assert result["soc"].iloc[0] == pytest.approx(3.5 * 0.58, abs=0.001)
        assert pd.isna(result["soc"].iloc[1])
        assert result["soc"].iloc[2] == pytest.approx(2.1 * 0.58, abs=0.001)

    def test_soc_conversion_no_hu_column(self):
        """Should warn but not crash if hu is missing."""
        df = pd.DataFrame({"ph": [7.2]})
        result = s08_merge_features.add_soc_column(df)
        assert "soc" not in result.columns


class TestAggregateToFieldLevel:
    """Tests for pixel-to-field aggregation."""

    def test_aggregation_mean(self):
        """Multiple pixels per field should be averaged."""
        df = pd.DataFrame({
            "farm": ["F1", "F1", "F1"],
            "field_name": ["A", "A", "A"],
            "centroid_lon": [50.0, 50.01, 50.02],
            "centroid_lat": [60.0, 60.01, 60.02],
            "DEM": [450.0, 452.0, 454.0],
            "slope": [5.0, 6.0, 7.0],
        })

        result = s08_merge_features._aggregate_to_field_level(
            df, ["farm", "field_name"], "topo_"
        )

        assert len(result) == 1
        assert result["topo_DEM"].iloc[0] == pytest.approx(452.0, abs=0.01)
        assert result["topo_slope"].iloc[0] == pytest.approx(6.0, abs=0.01)

    def test_aggregation_preserves_groups(self):
        """Multiple fields should remain separate after aggregation."""
        df = pd.DataFrame({
            "year": [2021, 2021, 2021, 2021],
            "farm": ["F1", "F1", "F2", "F2"],
            "field_name": ["A", "A", "B", "B"],
            "VV": [-12.0, -13.0, -15.0, -16.0],
            "VH": [-18.0, -19.0, -20.0, -21.0],
        })

        result = s08_merge_features._aggregate_to_field_level(
            df, ["year", "farm", "field_name"], "s1_"
        )

        assert len(result) == 2
        assert result.loc[result["farm"] == "F1", "s1_VV"].iloc[0] == pytest.approx(-12.5)
        assert result.loc[result["farm"] == "F2", "s1_VV"].iloc[0] == pytest.approx(-15.5)

    def test_aggregation_drops_coords(self):
        """centroid_lon/lat should be dropped before aggregation."""
        df = pd.DataFrame({
            "farm": ["F1", "F1"],
            "field_name": ["A", "A"],
            "centroid_lon": [50.0, 50.01],
            "centroid_lat": [60.0, 60.01],
            "DEM": [450.0, 452.0],
        })

        result = s08_merge_features._aggregate_to_field_level(
            df, ["farm", "field_name"], "topo_"
        )

        assert "centroid_lon" not in result.columns
        assert "centroid_lat" not in result.columns
        assert "topo_DEM" in result.columns


@patch("src.s08_merge_features.load_soil_samples")
@patch("src.s08_merge_features.load_sentinel2_data")
@patch("src.s08_merge_features.load_landsat8_data")
@patch("src.s08_merge_features.load_sentinel1_data")
@patch("src.s08_merge_features.load_topography_data")
@patch("src.s08_merge_features.load_climate_data")
@patch("src.s08_merge_features.load_spectral_eng_data")
@patch("src.s08_merge_features.load_glcm_data")
def test_merge_all_features_with_static(
    mock_glcm, mock_spectral, mock_climate, mock_topo,
    mock_s1, mock_l8, mock_s2, mock_soil,
):
    """Test that merge_all_features includes S1, topo, and climate."""
    mock_soil.return_value = pd.DataFrame({
        "year": [2021],
        "farm": ["Farm1"],
        "field_name": ["Field1"],
        "ph": [7.2],
        "hu": [3.5],
    })

    mock_s2.return_value = pd.DataFrame({
        "year": [2021, 2021],
        "season": ["spring", "summer"],
        "farm": ["Farm1", "Farm1"],
        "field_name": ["Field1", "Field1"],
        "NDVI": [0.6, 0.7],
    })

    mock_l8.return_value = pd.DataFrame()
    mock_spectral.return_value = pd.DataFrame()
    mock_glcm.return_value = pd.DataFrame()

    # S1 with multiple pixels
    mock_s1.return_value = pd.DataFrame({
        "year": [2021, 2021],
        "farm": ["Farm1", "Farm1"],
        "field_name": ["Field1", "Field1"],
        "VV": [-12.0, -13.0],
        "VH": [-18.0, -19.0],
    })

    # Topo with multiple pixels
    mock_topo.return_value = pd.DataFrame({
        "year": [2021, 2021],
        "farm": ["Farm1", "Farm1"],
        "field_name": ["Field1", "Field1"],
        "DEM": [450.0, 452.0],
        "slope": [5.0, 6.0],
    })

    # Climate
    mock_climate.return_value = pd.DataFrame({
        "year": [2021],
        "farm": ["Farm1"],
        "field_name": ["Field1"],
        "MAT": [5.64],
        "MAP": [521.6],
    })

    merged = s08_merge_features.merge_all_features()

    assert isinstance(merged, pd.DataFrame)
    assert len(merged) == 1

    # SOC should be computed
    assert "soc" in merged.columns
    assert merged["soc"].iloc[0] == pytest.approx(3.5 * 0.58, abs=0.001)

    # S1 should be aggregated and merged
    assert "s1_VV" in merged.columns
    assert merged["s1_VV"].iloc[0] == pytest.approx(-12.5)

    # Topo should be aggregated and merged
    assert "topo_DEM" in merged.columns
    assert merged["topo_DEM"].iloc[0] == pytest.approx(451.0)

    # Climate should be merged
    assert "climate_MAT" in merged.columns
    assert merged["climate_MAT"].iloc[0] == pytest.approx(5.64)

    # S2 should be pivoted
    assert "s2_NDVI_spring" in merged.columns


@patch("src.s08_merge_features.load_soil_samples")
@patch("src.s08_merge_features.load_sentinel2_data")
@patch("src.s08_merge_features.load_landsat8_data")
@patch("src.s08_merge_features.load_sentinel1_data")
@patch("src.s08_merge_features.load_topography_data")
@patch("src.s08_merge_features.load_climate_data")
@patch("src.s08_merge_features.load_spectral_eng_data")
@patch("src.s08_merge_features.load_glcm_data")
def test_merge_preserves_zeros(
    mock_glcm, mock_spectral, mock_climate, mock_topo,
    mock_s1, mock_l8, mock_s2, mock_soil,
):
    """Verify that legitimate 0.0 values are NOT replaced with NaN."""
    mock_soil.return_value = pd.DataFrame({
        "year": [2021],
        "farm": ["Farm1"],
        "field_name": ["Field1"],
        "ph": [7.0],
        "hu": [0.0],  # legitimate zero
    })

    mock_s2.return_value = pd.DataFrame()
    mock_l8.return_value = pd.DataFrame()
    mock_s1.return_value = pd.DataFrame()
    mock_topo.return_value = pd.DataFrame()
    mock_climate.return_value = pd.DataFrame()
    mock_spectral.return_value = pd.DataFrame()
    mock_glcm.return_value = pd.DataFrame()

    merged = s08_merge_features.merge_all_features()

    # hu=0.0 should remain 0.0, NOT be replaced with NaN
    assert merged["hu"].iloc[0] == 0.0
    # soc should be 0.0 * 0.58 = 0.0
    assert merged["soc"].iloc[0] == 0.0


def test_van_bemmelen_factor():
    """Test that VAN_BEMMELEN_FACTOR is the standard 0.58."""
    assert s08_merge_features.VAN_BEMMELEN_FACTOR == 0.58
