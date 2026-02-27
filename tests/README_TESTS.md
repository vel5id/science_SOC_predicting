# Test Suite Documentation

## Overview

Comprehensive test suite for the satellite data extraction pipeline covering all 8 extraction scripts and utility modules.

## Test Coverage

### Core Utilities

#### `test_config.py` — 15 tests
- Path configuration
- Years and seasons
- S2/L8/S1/DEM/ERA5 collection settings
- Band mappings and index formulas
- CRS configuration

#### `test_db_utils.py` — 10 tests
- Database connection
- Field polygon extraction
- Sampling dates retrieval
- Bounding box calculation
- Feature saving (replace/append modes)
- Table existence checks

**Fixtures**: Mock SQLite database with test data

#### `test_gee_auth.py` — 4 tests
- GEE authentication (already authenticated)
- GEE authentication (needs auth)
- GEE ready check (success/failure)

**Mocking**: All Earth Engine API calls mocked

---

### Extraction Scripts

#### `test_s01_temperature.py` — 7 tests
- Kelvin to Celsius conversion
- Temperature extraction for year (success/no data)
- Seasonal window determination (normal/no growing season/partial overlap)

**Mocking**: GEE ImageCollection, reduceRegion

#### `test_s02_sentinel2.py` — 8 tests
- Loading seasonal windows from file
- Cloud masking (SCL band)
- Index computation (NDVI, NDRE, GNDVI, etc.)
- Feature extraction (success/no data)
- Processing fields for season

**Mocking**: GEE ImageCollection, geometry conversion, file I/O

#### `test_s03_landsat8.py` — 5 tests
- Cloud masking (QA_PIXEL)
- Index computation (NDVI, GNDVI, SAVI)
- Feature extraction (success/no data)

**Mocking**: GEE ImageCollection

#### `test_s04_s07.py` — 8 tests
- **Sentinel-1**: Feature extraction (success/no data)
- **Topography**: DEM/slope/aspect extraction
- **Soil maps**: Placeholder extraction
- **Hyperspectral**: PRISMA/EnMAP/LUCAS availability checks

**Mocking**: GEE API for S1 and topography

#### `test_s08_merge.py` — 9 tests
- Loading temperature/S2/L8/S1/topography data from CSV
- Loading soil samples from database
- Merging all features (with pivot)

**Fixtures**: Mock CSV files in temporary directory

---

## Running Tests

### All tests
```cmd
bat\run_tests.bat
```

Or directly:
```cmd
uv run pytest tests/ -v
```

### Specific test file
```cmd
uv run pytest tests/test_config.py -v
```

### Specific test function
```cmd
uv run pytest tests/test_config.py::test_years_configuration -v
```

### With coverage report
```cmd
uv run pytest tests/ --cov=src --cov-report=html
```

---

## Test Statistics

| Module | Tests | Coverage |
|--------|-------|----------|
| config.py | 15 | 100% |
| db_utils.py | 10 | ~90% |
| gee_auth.py | 4 | 100% |
| s01_temperature.py | 7 | ~80% |
| s02_sentinel2.py | 8 | ~75% |
| s03_landsat8.py | 5 | ~75% |
| s04-s07 | 8 | ~70% |
| s08_merge.py | 9 | ~80% |
| **Total** | **66** | **~80%** |

---

## Key Testing Strategies

### 1. **Mocking GEE API**
All Earth Engine calls are mocked to avoid:
- Network dependencies
- GEE authentication requirements
- Slow API calls

Example:
```python
@patch("src.s02_sentinel2.ee")
def test_extract_s2_features(mock_ee):
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    # ... test logic
```

### 2. **Fixture-Based Testing**
Reusable fixtures for common test data:
- `mock_db_connection` — temporary SQLite database
- `mock_data_files` — temporary CSV files
- `sample_bbox`, `sample_polygon`, `sample_dates` — shared test data

### 3. **Edge Case Coverage**
Tests cover:
- ✅ Success cases (data available)
- ✅ Failure cases (no data, missing files)
- ✅ Boundary conditions (empty DataFrames, None values)
- ✅ Data validation (correct types, ranges)

### 4. **Integration Points**
Tests verify:
- File I/O (CSV reading/writing)
- Database operations (SQLite queries)
- Data transformations (pivot, merge)
- GEE API interactions (mocked)

---

## Adding New Tests

### Template for new test file:
```python
"""Tests for new_module.py"""
import pytest
from unittest.mock import Mock, patch
from src import new_module

def test_function_name():
    """Test description."""
    result = new_module.function_to_test()
    assert result == expected_value
```

### Best practices:
1. **One test per function behavior**
2. **Descriptive test names** (`test_extract_features_when_no_data`)
3. **Mock external dependencies** (GEE, file I/O)
4. **Use fixtures** for shared setup
5. **Test edge cases** (empty inputs, None, errors)

---

## Continuous Integration

To run tests automatically on commit:
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install uv
      - run: uv sync
      - run: uv run pytest tests/
```
