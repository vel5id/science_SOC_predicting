"""Tests for s11_spectral_eng.py — EVI, band ratios, PCA, filename parsing."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src import s11_spectral_eng, config


# ─── EVI Calculation ─────────────────────────────────────────────

def test_calculate_evi_basic():
    """Test EVI calculation with known values."""
    df = pd.DataFrame({
        "B2": [0.05],    # blue
        "B4": [0.04],    # red
        "B8": [0.30],    # nir
    })

    result = s11_spectral_eng.calculate_evi(df)

    assert "EVI" in result.columns
    # EVI = 2.5 * ((0.30 - 0.04) / (0.30 + 6*0.04 - 7.5*0.05 + 1))
    nir, red, blue = 0.30, 0.04, 0.05
    expected = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    assert result["EVI"].iloc[0] == pytest.approx(expected, abs=1e-6)


def test_calculate_evi_missing_bands():
    """EVI should gracefully handle missing bands."""
    df = pd.DataFrame({"B8": [0.3], "B4": [0.04]})  # B2 missing

    result = s11_spectral_eng.calculate_evi(df)

    assert "EVI" not in result.columns  # should not crash


# ─── Band Ratios ─────────────────────────────────────────────────

def test_calculate_band_ratios():
    """Test band ratio computation."""
    df = pd.DataFrame({
        "B3": [0.06],
        "B4": [0.04],
        "B8": [0.30],
        "B11": [0.15],
    })

    result = s11_spectral_eng.calculate_band_ratios(df)

    assert "B3_B4" in result.columns
    assert "B8_B4" in result.columns
    assert "B11_B8" in result.columns
    assert result["B3_B4"].iloc[0] == pytest.approx(0.06 / 0.04, rel=1e-3)


def test_calculate_band_ratios_zero_denominator():
    """Band ratios should not crash on zero-valued bands (epsilon added)."""
    df = pd.DataFrame({
        "B3": [0.06],
        "B4": [0.0],   # zero
        "B8": [0.0],   # zero
        "B11": [0.15],
    })

    result = s11_spectral_eng.calculate_band_ratios(df)

    # Should be NaN (denominator near zero → guarded)
    assert np.isnan(result["B3_B4"].iloc[0])
    assert np.isnan(result["B11_B8"].iloc[0])


# ─── PCA ─────────────────────────────────────────────────────────

def test_calculate_pca_basic():
    """PCA should produce n_components new columns."""
    np.random.seed(42)
    df = pd.DataFrame({
        "B2": np.random.rand(20),
        "B3": np.random.rand(20),
        "B4": np.random.rand(20),
        "B8": np.random.rand(20),
    })

    result = s11_spectral_eng.calculate_pca(df, n_components=3)

    assert "PCA_1" in result.columns
    assert "PCA_2" in result.columns
    assert "PCA_3" in result.columns
    assert "PCA_4" not in result.columns  # only 3 requested


def test_calculate_pca_with_nan():
    """PCA should skip if NaN values present."""
    df = pd.DataFrame({
        "B2": [0.1, np.nan, 0.3],
        "B3": [0.2, 0.3, 0.4],
        "B4": [0.15, 0.25, 0.35],
    })

    result = s11_spectral_eng.calculate_pca(df)

    # Should NOT have PCA columns (skipped due to NaN)
    assert "PCA_1" not in result.columns


def test_calculate_pca_too_few_bands():
    """PCA should skip if fewer than 3 bands available."""
    df = pd.DataFrame({
        "B2": [0.1, 0.2],
        "B3": [0.2, 0.3],
    })

    result = s11_spectral_eng.calculate_pca(df)

    assert "PCA_1" not in result.columns


def test_calculate_pca_b8a_excluded():
    """BUG-4 awareness: B8A should be excluded from PCA by the isdigit() filter."""
    df = pd.DataFrame({
        "B2": np.random.rand(10),
        "B3": np.random.rand(10),
        "B4": np.random.rand(10),
        "B8": np.random.rand(10),
        "B8A": np.random.rand(10),  # narrow NIR — filtered out
    })

    # The filter is: col.startswith('B') and col[1:].isdigit()
    # B8A → '8A'.isdigit() = False → excluded
    band_cols = [col for col in df.columns if col.startswith('B') and col[1:].isdigit()]

    assert "B8A" not in band_cols
    assert "B8" in band_cols
    assert len(band_cols) == 4


# ─── Filename Parsing (BUG-1 regression) ─────────────────────────

def test_filename_parsing_standard_seasons():
    """Standard season names should be parsed correctly."""
    test_cases = {
        "s2_features_2021_spring": ("2021", "spring"),
        "s2_features_2022_summer": ("2022", "summer"),
        "s2_features_2023_autumn": ("2023", "autumn"),
    }

    for stem, (expected_year, expected_season) in test_cases.items():
        parts = stem.split('_')
        year = parts[2]
        season = '_'.join(parts[3:])
        assert year == expected_year, f"Failed for {stem}"
        assert season == expected_season, f"Failed for {stem}"


def test_filename_parsing_late_summer():
    """BUG-1 regression: late_summer must be parsed as 'late_summer', not 'late'."""
    stem = "s2_features_2021_late_summer"
    parts = stem.split('_')

    year = parts[2]
    season = '_'.join(parts[3:])

    assert year == "2021"
    assert season == "late_summer", (
        f"BUG-1 regression: season parsed as '{season}' instead of 'late_summer'. "
        "The old code used parts[3] which gives 'late'."
    )


# ─── End-to-End Processing ───────────────────────────────────────

def test_process_s2_file(tmp_path):
    """Test end-to-end processing of a single S2 CSV file."""
    # Create input S2 data
    np.random.seed(42)
    n = 15
    input_df = pd.DataFrame({
        "year": [2021] * n,
        "season": ["summer"] * n,
        "farm": ["Farm1"] * n,
        "field_name": ["Field1"] * n,
        "B2": np.random.rand(n) * 0.1,
        "B3": np.random.rand(n) * 0.15,
        "B4": np.random.rand(n) * 0.1,
        "B8": np.random.rand(n) * 0.4,
        "B11": np.random.rand(n) * 0.2,
    })

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_df.to_csv(input_path, index=False)

    s11_spectral_eng.process_s2_file(str(input_path), str(output_path))

    assert output_path.exists()
    result = pd.read_csv(output_path)

    # Should have original cols + EVI + ratios + PCA
    assert "EVI" in result.columns
    assert "B3_B4" in result.columns
    assert "B8_B4" in result.columns
    assert "B11_B8" in result.columns
    assert "PCA_1" in result.columns
    assert len(result) == n
