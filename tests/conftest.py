"""Pytest configuration and shared fixtures."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_bbox():
    """Sample bounding box for testing."""
    return (50.0, 60.0, 51.0, 61.0)


@pytest.fixture
def sample_polygon():
    """Sample polygon geometry for testing."""
    from shapely.geometry import Polygon
    return Polygon([(50, 60), (51, 60), (51, 61), (50, 61)])


@pytest.fixture
def sample_dates():
    """Sample date range for testing."""
    return ("2020-04-01", "2020-06-01")
