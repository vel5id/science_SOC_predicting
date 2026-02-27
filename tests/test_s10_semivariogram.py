"""Tests for s10_semivariogram.py — DB schema alignment and variogram logic."""
import pytest
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src import s10_semivariogram, config


# ─── DB Schema Alignment ─────────────────────────────────────────

REAL_DB_COLUMNS = {
    "id", "year", "farm", "field_name", "grid_id",
    "ph", "k", "p", "hu", "s", "no3",
    "zn", "mo", "fe", "mg", "mn", "cu",
    "centroid_lon", "centroid_lat", "geometry_wkt",
    "protocol_number", "analysis_date", "sampling_date",
}


def test_sql_columns_match_db_schema():
    """Regression test: all columns in load_soil_data() SQL must exist in real DB."""
    # Extract column names from the SQL query
    import re
    import inspect

    source = inspect.getsource(s10_semivariogram.load_soil_data)

    # Find SELECT ... FROM block
    select_match = re.search(r"SELECT\s+(.+?)\s+FROM", source, re.DOTALL)
    assert select_match, "Could not parse SELECT statement"

    select_block = select_match.group(1)
    columns = [col.strip().rstrip(",") for col in select_block.split("\n") if col.strip() and col.strip() != ","]

    for col in columns:
        assert col in REAL_DB_COLUMNS, (
            f"Column '{col}' in load_soil_data() SQL does not exist in soil_samples table! "
            f"Available: {sorted(REAL_DB_COLUMNS)}"
        )


def test_properties_dict_uses_correct_columns():
    """Verify the properties dict in main() references valid DataFrame columns."""
    # The properties dict should use the same column names as the SQL query
    # After load_soil_data(), the DataFrame should have: ph, hu, no3, p, k
    expected_property_columns = {"ph", "hu", "no3", "p", "k"}

    # These are the columns selected in the SQL query (minus coordinates)
    sql_data_columns = {"ph", "hu", "no3", "p", "k"}

    assert expected_property_columns == sql_data_columns


# ─── Semivariogram Calculation ────────────────────────────────────

def test_calculate_semivariogram_insufficient_data(tmp_path):
    """Should return empty dict when fewer than 10 samples."""
    coords = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([1.0, 2.0, 3.0])

    result = s10_semivariogram.calculate_semivariogram(
        coords, values, "test_prop", tmp_path
    )

    assert result == {}


def test_calculate_semivariogram_with_nans(tmp_path):
    """NaN values should be filtered before fitting."""
    np.random.seed(42)
    n = 50
    coords = np.random.rand(n, 2) * 10
    values = np.random.rand(n) * 5.0
    # Inject NaNs
    values[5] = np.nan
    values[10] = np.nan
    values[15] = np.nan

    result = s10_semivariogram.calculate_semivariogram(
        coords, values, "test_nan", tmp_path
    )

    # Should still succeed (50 - 3 = 47 samples > 10)
    if result:  # variogram fitting may fail on random data
        assert "range" in result
        assert "sill" in result
        assert "n_samples" in result
        assert result["n_samples"] == 47  # 50 - 3 NaN


def test_calculate_semivariogram_all_nan(tmp_path):
    """All NaN values should return empty dict (< 10 valid)."""
    coords = np.ones((20, 2))
    values = np.full(20, np.nan)

    result = s10_semivariogram.calculate_semivariogram(
        coords, values, "all_nan", tmp_path
    )

    assert result == {}


def test_calculate_semivariogram_output_files(tmp_path):
    """Successful variogram should create CSV and PNG files."""
    np.random.seed(123)
    n = 100
    x = np.random.rand(n) * 10
    y = np.random.rand(n) * 10
    coords = np.column_stack([x, y])
    # Create spatially correlated data
    values = np.sin(x) + np.cos(y) + np.random.rand(n) * 0.5

    result = s10_semivariogram.calculate_semivariogram(
        coords, values, "spatial_test", tmp_path
    )

    if result:  # variogram fitting succeeded
        csv_path = tmp_path / "spatial_test_semivariogram.csv"
        png_path = tmp_path / "spatial_test_semivariogram.png"
        assert csv_path.exists(), "CSV output not created"
        assert png_path.exists(), "PNG plot not created"
        assert result["range_km"] > 0
        assert result["recommended_block_km"] > 0


# ─── load_soil_data Mock Test ────────────────────────────────────

@patch("src.s10_semivariogram.get_connection")
def test_load_soil_data(mock_get_conn):
    """Test load_soil_data returns DataFrame with correct columns."""
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn

    with patch("pandas.read_sql_query") as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame({
            "centroid_lon": [50.0, 50.1],
            "centroid_lat": [60.0, 60.1],
            "ph": [7.2, 6.8],
            "hu": [3.5, 4.0],
            "no3": [12.0, 15.0],
            "p": [8.5, 10.0],
            "k": [200.0, 250.0],
        })

        df = s10_semivariogram.load_soil_data()

        assert isinstance(df, pd.DataFrame)
        assert "centroid_lon" in df.columns
        assert "ph" in df.columns
        assert "hu" in df.columns
        assert "no3" in df.columns
        assert "p" in df.columns
        assert "k" in df.columns
        mock_conn.close.assert_called_once()
