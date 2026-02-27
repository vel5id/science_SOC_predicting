"""Tests for db_utils.py"""
import pytest
import sqlite3
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from shapely.geometry import Polygon

from src import db_utils, config


@pytest.fixture
def mock_db_connection(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    
    # Create test table
    conn.execute("""
        CREATE TABLE soil_samples (
            id INTEGER PRIMARY KEY,
            year INTEGER,
            farm TEXT,
            field_name TEXT,
            centroid_lon REAL,
            centroid_lat REAL,
            geometry_wkt TEXT,
            sampling_date TEXT,
            ph REAL
        )
    """)
    
    # Insert test data
    test_data = [
        (1, 2020, "Farm1", "Field1", 50.0, 60.0, "POLYGON((50 60, 51 60, 51 61, 50 61, 50 60))", "2020-05-15", 7.2),
        (2, 2020, "Farm1", "Field2", 50.1, 60.1, "POLYGON((50.1 60.1, 51.1 60.1, 51.1 61.1, 50.1 61.1, 50.1 60.1))", "2020-06-10", 6.8),
        (3, 2021, "Farm2", "Field1", 50.2, 60.2, "POLYGON((50.2 60.2, 51.2 60.2, 51.2 61.2, 50.2 61.2, 50.2 60.2))", "2021-05-20", 7.0),
    ]
    
    conn.executemany(
        "INSERT INTO soil_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        test_data
    )
    conn.commit()
    conn.close()
    
    return db_path


def test_get_connection(mock_db_connection, monkeypatch):
    """Test database connection."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    conn = db_utils.get_connection()
    assert isinstance(conn, sqlite3.Connection)
    
    # Test that we can query
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM soil_samples")
    count = cursor.fetchone()[0]
    assert count == 3
    
    conn.close()


def test_get_connection_missing_db(tmp_path, monkeypatch):
    """Test connection with missing database."""
    missing_db = tmp_path / "missing.db"
    monkeypatch.setattr(config, "DB_PATH", missing_db)
    
    with pytest.raises(FileNotFoundError):
        db_utils.get_connection()


def test_get_field_polygons(mock_db_connection, monkeypatch):
    """Test field polygon extraction."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    gdf = db_utils.get_field_polygons()
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 3
    assert "year" in gdf.columns
    assert "farm" in gdf.columns
    assert "field_name" in gdf.columns
    assert "geometry" in gdf.columns
    assert "centroid_lon" in gdf.columns
    assert "centroid_lat" in gdf.columns
    
    # Check geometry types
    assert all(isinstance(geom, Polygon) for geom in gdf.geometry)
    
    # Check CRS
    assert gdf.crs == config.CRS_WGS84


def test_get_sampling_dates(mock_db_connection, monkeypatch):
    """Test sampling dates extraction."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    dates_by_year = db_utils.get_sampling_dates()
    
    assert isinstance(dates_by_year, dict)
    assert 2020 in dates_by_year
    assert 2021 in dates_by_year
    assert len(dates_by_year[2020]) == 2
    assert len(dates_by_year[2021]) == 1
    assert "2020-05-15" in dates_by_year[2020]
    assert "2021-05-20" in dates_by_year[2021]


def test_get_region_bbox(mock_db_connection, monkeypatch):
    """Test bounding box calculation."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    bbox = db_utils.get_region_bbox()
    
    assert isinstance(bbox, tuple)
    assert len(bbox) == 4
    min_lon, min_lat, max_lon, max_lat = bbox
    
    assert min_lon == 50.0
    assert min_lat == 60.0
    assert max_lon == 50.2
    assert max_lat == 60.2


def test_save_features_to_db(mock_db_connection, monkeypatch):
    """Test saving features to database."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        "year": [2020, 2021],
        "farm": ["Farm1", "Farm2"],
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0],
    })
    
    # Save to database
    db_utils.save_features_to_db("test_features", test_df)
    
    # Verify
    conn = db_utils.get_connection()
    result = pd.read_sql_query("SELECT * FROM test_features", conn)
    conn.close()
    
    assert len(result) == 2
    assert "year" in result.columns
    assert "feature1" in result.columns
    assert result["feature1"].tolist() == [1.0, 2.0]


def test_save_features_to_db_replace(mock_db_connection, monkeypatch):
    """Test replacing existing table."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    # Save first version
    df1 = pd.DataFrame({"col1": [1, 2]})
    db_utils.save_features_to_db("test_table", df1)
    
    # Save second version (replace)
    df2 = pd.DataFrame({"col1": [3, 4, 5]})
    db_utils.save_features_to_db("test_table", df2, if_exists="replace")
    
    # Verify
    conn = db_utils.get_connection()
    result = pd.read_sql_query("SELECT * FROM test_table", conn)
    conn.close()
    
    assert len(result) == 3
    assert result["col1"].tolist() == [3, 4, 5]


def test_table_exists(mock_db_connection, monkeypatch):
    """Test table existence check."""
    monkeypatch.setattr(config, "DB_PATH", mock_db_connection)
    
    assert db_utils.table_exists("soil_samples") is True
    assert db_utils.table_exists("nonexistent_table") is False
