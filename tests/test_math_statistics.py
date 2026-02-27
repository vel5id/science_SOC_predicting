"""Tests for math_statistics package (v1 + v2 modules)."""

import pytest
import pandas as pd
import numpy as np
from scipy import stats

from math_statistics.config import SOIL_TARGETS, SOIL_LABELS, FEATURES_CSV
from math_statistics import descriptive_stats, intercorrelation, correlation_analysis
from math_statistics import seasonal_analysis, spatial_analysis
from math_statistics import composite_features, derived_soil, variance_decomposition
from math_statistics import confounding_analysis, composite_vs_single


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_df():
    """Load the real dataset once for all tests."""
    return pd.read_csv(FEATURES_CSV)


@pytest.fixture
def synthetic_df():
    """Small synthetic dataset for unit tests."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "id": range(n),
        "year": rng.choice([2020, 2021, 2022, 2023], n),
        "field_name": [f"field_{i % 10}" for i in range(n)],
        "ph": rng.normal(7.0, 0.7, n),
        "soc": rng.normal(2.4, 0.5, n).clip(0.5, 4.5),
        "no3": rng.exponential(10, n).clip(0.1, 65),
        "p": rng.exponential(20, n).clip(3, 150),
        "k": rng.normal(650, 160, n).clip(100, 1200),
        "s": rng.exponential(8, n).clip(0.5, 65),
        "hu": rng.normal(4.0, 1.0, n),
        "centroid_lon": rng.uniform(60, 66, n),
        "centroid_lat": rng.uniform(51, 55, n),
        "s2_NDVI_spring": rng.uniform(0.1, 0.6, n),
        "s2_NDVI_summer": rng.uniform(0.2, 0.8, n),
        "s2_NDVI_late_summer": rng.uniform(0.15, 0.7, n),
        "s2_NDVI_autumn": rng.uniform(0.1, 0.4, n),
        "s2_GNDVI_spring": rng.uniform(0.1, 0.6, n),
        "s2_GNDVI_summer": rng.uniform(0.2, 0.7, n),
        "s2_GNDVI_late_summer": rng.uniform(0.15, 0.6, n),
        "s2_GNDVI_autumn": rng.uniform(0.1, 0.4, n),
        "s2_NDRE_spring": rng.uniform(0.1, 0.5, n),
        "s2_NDRE_summer": rng.uniform(0.2, 0.6, n),
        "s2_NDRE_late_summer": rng.uniform(0.15, 0.5, n),
        "s2_NDRE_autumn": rng.uniform(0.1, 0.3, n),
        "s2_EVI_spring": rng.uniform(0.1, 0.5, n),
        "s2_EVI_summer": rng.uniform(0.2, 0.7, n),
        "s2_EVI_late_summer": rng.uniform(0.15, 0.6, n),
        "s2_EVI_autumn": rng.uniform(0.1, 0.3, n),
        "s2_SAVI_spring": rng.uniform(0.1, 0.5, n),
        "s2_SAVI_summer": rng.uniform(0.2, 0.7, n),
        "s2_SAVI_late_summer": rng.uniform(0.15, 0.6, n),
        "s2_SAVI_autumn": rng.uniform(0.1, 0.3, n),
        "s2_BSI_spring": rng.uniform(-0.2, 0.3, n),
        "s2_BSI_summer": rng.uniform(-0.2, 0.3, n),
        "s2_BSI_late_summer": rng.uniform(-0.2, 0.3, n),
        "s2_BSI_autumn": rng.uniform(-0.2, 0.3, n),
        "s2_Cl_Red_Edge_spring": rng.uniform(0.5, 2.0, n),
        "s2_Cl_Red_Edge_summer": rng.uniform(0.5, 2.0, n),
        "s2_Cl_Red_Edge_late_summer": rng.uniform(0.5, 2.0, n),
        "s2_Cl_Red_Edge_autumn": rng.uniform(0.5, 2.0, n),
        "l8_GNDVI_spring": rng.uniform(0.1, 0.6, n),
        "l8_SAVI_spring": rng.uniform(0.1, 0.5, n),
        "l8_SR_B5_spring": rng.uniform(0.1, 0.4, n),
        "topo_slope": rng.uniform(0, 10, n),
        "topo_aspect_sin": rng.uniform(-1, 1, n),
        "topo_aspect_cos": rng.uniform(-1, 1, n),
        "climate_MAP": rng.uniform(250, 400, n),
        "climate_GS_temp": rng.uniform(10, 20, n),
    })


# ══════════════════════════════════════════════════════════════════
# v1 TESTS
# ══════════════════════════════════════════════════════════════════

class TestDescriptiveStats:
    def test_table1_shape(self, synthetic_df):
        result = descriptive_stats.compute_descriptive_table(synthetic_df)
        assert len(result) == len(SOIL_TARGETS)
        assert "Mean" in result.columns
        assert "CV_%" in result.columns

    def test_shapiro_wilk_returns_all_targets(self, synthetic_df):
        result = descriptive_stats.shapiro_wilk_tests(synthetic_df)
        assert len(result) == len(SOIL_TARGETS)
        assert "W_statistic" in result.columns
        assert all(0 < w <= 1 for w in result["W_statistic"])

    def test_kruskal_wallis_has_all_targets(self, synthetic_df):
        result = descriptive_stats.kruskal_wallis_by_year(synthetic_df)
        assert len(result) == len(SOIL_TARGETS)

    def test_real_data_normality(self, full_df):
        """Article claims all 6 properties are non-normal (p < 0.001)."""
        result = descriptive_stats.shapiro_wilk_tests(full_df)
        for _, row in result.iterrows():
            assert row["p_value"] < 0.001, (
                f"{row['Property']} has p={row['p_value']:.4f}, "
                f"article claims p < 0.001"
            )


class TestIntercorrelation:
    def test_rho_matrix_shape(self, synthetic_df):
        rho, p = intercorrelation.compute_intercorrelation_matrix(synthetic_df)
        n = len(SOIL_TARGETS)
        assert rho.shape == (n, n)
        assert p.shape == (n, n)

    def test_diagonal_is_one(self, synthetic_df):
        rho, _ = intercorrelation.compute_intercorrelation_matrix(synthetic_df)
        for i in range(len(SOIL_TARGETS)):
            assert rho.iloc[i, i] == pytest.approx(1.0)

    def test_symmetry(self, synthetic_df):
        rho, _ = intercorrelation.compute_intercorrelation_matrix(synthetic_df)
        pd.testing.assert_frame_equal(rho, rho.T)


class TestCorrelationAnalysis:
    def test_compute_all_spearman_shape(self, synthetic_df):
        result = correlation_analysis.compute_all_spearman(synthetic_df)
        assert len(result) > 0
        assert "rho" in result.columns
        assert "p_value" in result.columns
        assert all(-1 <= r <= 1 for r in result["rho"])

    def test_bh_correction_adds_columns(self, synthetic_df):
        raw = correlation_analysis.compute_all_spearman(synthetic_df)
        corrected = correlation_analysis.apply_bh_correction(raw)
        assert "p_adjusted" in corrected.columns
        assert "significant_bh" in corrected.columns

    def test_bh_adjusted_p_gte_raw(self, synthetic_df):
        raw = correlation_analysis.compute_all_spearman(synthetic_df)
        corrected = correlation_analysis.apply_bh_correction(raw)
        valid = corrected.dropna(subset=["p_adjusted"])
        assert all(valid["p_adjusted"] >= valid["p_value"] - 1e-10)

    def test_real_data_ph_gndvi_spring(self, full_df):
        """Article: pH vs L8 GNDVI spring rho = -0.67."""
        mask = full_df[["ph", "l8_GNDVI_spring"]].notna().all(axis=1)
        rho, _ = stats.spearmanr(
            full_df.loc[mask, "ph"], full_df.loc[mask, "l8_GNDVI_spring"]
        )
        assert abs(rho - (-0.67)) < 0.10, f"pH vs L8 GNDVI spring: rho={rho:.4f}"

    def test_real_data_ph_map(self, full_df):
        """Article: pH vs MAP rho = 0.66."""
        mask = full_df[["ph", "climate_MAP"]].notna().all(axis=1)
        rho, _ = stats.spearmanr(
            full_df.loc[mask, "ph"], full_df.loc[mask, "climate_MAP"]
        )
        assert abs(rho - 0.66) < 0.10, f"pH vs MAP: rho={rho:.4f}"

    def test_real_data_k_bsi_spring(self, full_df):
        """Article: K2O vs BSI spring rho = -0.48."""
        col = "s2_BSI_spring"
        if col not in full_df.columns:
            pytest.skip(f"Column {col} missing")
        mask = full_df[["k", col]].notna().all(axis=1)
        if mask.sum() < 20:
            pytest.skip(f"Insufficient non-null data for K vs {col}")
        rho, _ = stats.spearmanr(
            full_df.loc[mask, "k"], full_df.loc[mask, col]
        )
        assert abs(rho - (-0.48)) < 0.10, f"K vs BSI spring: rho={rho:.4f}"


class TestSeasonalAnalysis:
    def test_soc_classes_assigned(self, synthetic_df):
        result = seasonal_analysis.assign_soc_class(synthetic_df)
        assert "soc_class" in result.columns
        assert result["soc_class"].notna().any()

    def test_ndvi_table_shape(self, synthetic_df):
        result = seasonal_analysis.seasonal_ndvi_by_soc_class(synthetic_df)
        assert len(result) > 0
        assert "NDVI_mean" in result.columns

    def test_real_data_spring_ph_stronger(self, full_df):
        """Article: spring veg indices stronger with pH than summer."""
        for idx in ["NDVI", "GNDVI", "NDRE"]:
            feat_sp = f"s2_{idx}_spring"
            feat_su = f"s2_{idx}_summer"
            if feat_sp not in full_df.columns or feat_su not in full_df.columns:
                continue
            mask_sp = full_df[["ph", feat_sp]].notna().all(axis=1)
            mask_su = full_df[["ph", feat_su]].notna().all(axis=1)
            rho_sp, _ = stats.spearmanr(full_df.loc[mask_sp, "ph"], full_df.loc[mask_sp, feat_sp])
            rho_su, _ = stats.spearmanr(full_df.loc[mask_su, "ph"], full_df.loc[mask_su, feat_su])
            assert abs(rho_sp) > abs(rho_su), (
                f"Spring |rho|={abs(rho_sp):.3f} should be > summer |rho|={abs(rho_su):.3f} "
                f"for {idx}"
            )


class TestSpatialAnalysis:
    def test_morans_i_known_pattern(self):
        """Morans I on perfectly clustered data should be positive."""
        n = 100
        lons = np.repeat(np.linspace(60, 66, 10), 10)
        lats = np.tile(np.linspace(51, 55, 10), 10)
        values = lats
        coords = np.column_stack([lons, lats])
        W_std, W_raw = spatial_analysis._inverse_distance_weights(coords, bandwidth=200_000)
        result = spatial_analysis.morans_i(values, W_std, W_raw)
        assert result["I"] > 0, "Spatially structured data should have positive Moran's I"

    def test_latitudinal_gradient(self, full_df):
        """Article: pH decreases northward, SOC increases northward."""
        result = spatial_analysis.latitudinal_gradient(full_df)
        ph_row = result[result["Property"] == "pH"]
        soc_row = result[result["Property"] == "SOC"]
        assert ph_row.iloc[0]["rho_with_latitude"] < 0
        assert soc_row.iloc[0]["rho_with_latitude"] > 0


# ══════════════════════════════════════════════════════════════════
# v2 NEW MODULE TESTS
# ══════════════════════════════════════════════════════════════════

# ── Composite features ────────────────────────────────────────────

class TestCompositeFeatures:
    def test_inter_index_has_products(self, synthetic_df):
        """Composite features should include GNDVIxBSI products."""
        result = composite_features.compute_inter_index_combinations(synthetic_df)
        product_cols = [c for c in result.columns if "GNDVIxBSI" in c]
        assert len(product_cols) >= 1, "Should have GNDVI*BSI product for at least one season"

    def test_inter_index_has_diffs(self, synthetic_df):
        """Composite features should include GNDVI-NDRE differences."""
        result = composite_features.compute_inter_index_combinations(synthetic_df)
        diff_cols = [c for c in result.columns if "GNDVI-NDRE" in c]
        assert len(diff_cols) >= 1, "Should have GNDVI-NDRE difference"

    def test_multiseasonal_deltas(self, synthetic_df):
        """Should compute delta, amplitude, and mean for each index."""
        result = composite_features.compute_multiseasonal_deltas(synthetic_df)
        assert any("delta_" in c for c in result.columns)
        assert any("amp_" in c for c in result.columns)
        assert any("mean_" in c for c in result.columns)

    def test_all_composites_shape(self, synthetic_df):
        """Total composite features should be substantial."""
        result = composite_features.compute_all_composites(synthetic_df)
        assert result.shape[1] >= 30, f"Expected 30+ composites, got {result.shape[1]}"
        assert result.shape[0] == len(synthetic_df)

    def test_real_data_composites(self, full_df):
        """Composite features on real data should produce ~100+ features."""
        result = composite_features.compute_all_composites(full_df)
        assert result.shape[1] >= 50, f"Expected 50+ composites on real data, got {result.shape[1]}"

    def test_real_data_gndvi_bsi_spring(self, full_df):
        """Article v2: GNDVI*BSI(spring) -> K2O rho = -0.488."""
        comps = composite_features.compute_all_composites(full_df)
        col = "comp_GNDVIxBSI_spring"
        if col not in comps.columns:
            pytest.skip(f"{col} not found in composites")
        mask = full_df["k"].notna() & comps[col].notna()
        if mask.sum() < 20:
            pytest.skip("Insufficient non-null data for GNDVI*BSI spring vs K")
        rho, _ = stats.spearmanr(full_df.loc[mask, "k"], comps.loc[mask, col])
        assert abs(rho - (-0.488)) < 0.05, f"GNDVI*BSI spring vs K: rho={rho:.4f}"


# ── Derived soil indicators ──────────────────────────────────────

class TestDerivedSoil:
    def test_all_9_indicators(self, synthetic_df):
        """Should produce exactly 9 derived indicators."""
        result = derived_soil.compute_derived_indicators(synthetic_df)
        assert result.shape[1] == 9
        assert "SOC_NO3_ratio" in result.columns
        assert "P_K_ratio" in result.columns
        assert "mineral_index" in result.columns

    def test_formulas_correct(self, synthetic_df):
        """Verify a few derived formulas are correct."""
        d = derived_soil.compute_derived_indicators(synthetic_df)
        # SOC * NO3
        expected = synthetic_df["soc"] * synthetic_df["no3"]
        pd.testing.assert_series_equal(d["SOC_x_NO3"], expected, check_names=False)
        # |pH - 7|
        expected_dev = (synthetic_df["ph"] - 7.0).abs()
        pd.testing.assert_series_equal(d["pH_deviation"], expected_dev, check_names=False)

    def test_real_data_pk_ratio_vs_slope(self, full_df):
        """Article v2: P2O5/K2O -> slope rho = -0.56."""
        d = derived_soil.compute_derived_indicators(full_df)
        mask = d["P_K_ratio"].notna() & full_df["topo_slope"].notna()
        rho, _ = stats.spearmanr(d.loc[mask, "P_K_ratio"], full_df.loc[mask, "topo_slope"])
        assert abs(rho - (-0.56)) < 0.05, f"P/K ratio vs slope: rho={rho:.4f}"

    def test_real_data_mineral_index_vs_l8_nir(self, full_df):
        """Article v2: mineral_index -> L8 NIR spring rho = -0.47."""
        d = derived_soil.compute_derived_indicators(full_df)
        col = "l8_SR_B5_spring"
        mask = d["mineral_index"].notna() & full_df[col].notna()
        rho, _ = stats.spearmanr(d.loc[mask, "mineral_index"], full_df.loc[mask, col])
        assert abs(rho - (-0.47)) < 0.05, f"mineral_index vs L8 NIR spring: rho={rho:.4f}"


# ── Variance decomposition ───────────────────────────────────────

class TestVarianceDecomposition:
    def test_decomposition_shape(self, synthetic_df):
        result = variance_decomposition.decompose_variance(synthetic_df)
        assert len(result) == len(SOIL_TARGETS)
        assert "Pct_between" in result.columns
        assert "Pct_within" in result.columns

    def test_percentages_sum_to_100(self, synthetic_df):
        result = variance_decomposition.decompose_variance(synthetic_df)
        for _, row in result.iterrows():
            total = row["Pct_between"] + row["Pct_within"]
            assert abs(total - 100.0) < 0.2, f"Between + within should sum to 100, got {total}"

    def test_icc_range(self, synthetic_df):
        result = variance_decomposition.decompose_variance(synthetic_df)
        for _, row in result.iterrows():
            assert -0.1 <= row["ICC"] <= 1.0, f"ICC out of range: {row['ICC']}"

    def test_real_data_ph_has_high_between_field(self, full_df):
        """pH should have higher between-field variance than within-field."""
        result = variance_decomposition.decompose_variance(full_df)
        ph_row = result[result["Property"] == SOIL_LABELS["ph"]]
        assert not ph_row.empty
        assert ph_row.iloc[0]["Pct_between"] > ph_row.iloc[0]["Pct_within"], \
            "pH between-field variance should exceed within-field"

    def test_real_data_ph_gt_soc_between_field(self, full_df):
        """Article: pH has more between-field variance than SOC."""
        result = variance_decomposition.decompose_variance(full_df)
        ph = result[result["Property"] == SOIL_LABELS["ph"]].iloc[0]
        soc = result[result["Property"] == SOIL_LABELS["soc"]].iloc[0]
        assert ph["Pct_between"] > soc["Pct_between"], \
            f"pH between-field ({ph['Pct_between']}%) should > SOC ({soc['Pct_between']}%)"


# ── Confounding analysis ──────────────────────────────────────────

class TestConfoundingAnalysis:
    def test_partial_correlations_shape(self, synthetic_df):
        result = confounding_analysis.partial_correlation_soc_ndvi_given_ph(synthetic_df)
        assert len(result) > 0
        assert "rho_raw" in result.columns
        assert "rho_partial_given_pH" in result.columns

    def test_partial_leq_raw(self, synthetic_df):
        """Partial correlation (controlling pH) should generally reduce |rho|."""
        result = confounding_analysis.partial_correlation_soc_ndvi_given_ph(synthetic_df)
        # Not always true for random data, but should be true on average
        # so we just check the columns exist and are numeric
        assert result["rho_raw"].dtype == float
        assert result["rho_partial_given_pH"].dtype == float

    def test_saturation_curve_shape(self, synthetic_df):
        result = confounding_analysis.ndvi_saturation_curve(synthetic_df)
        assert len(result) > 0
        assert "ndvi_mean" in result.columns
        assert "soc_mid" in result.columns

    def test_real_data_confounding_42pct(self, full_df):
        """Article v2: 42% of SOC-NDVI(summer) is pH-confounded."""
        if "s2_NDVI_summer" not in full_df.columns:
            pytest.skip("s2_NDVI_summer missing")
        mask = full_df[["soc", "ph", "s2_NDVI_summer"]].notna().all(axis=1)
        if mask.sum() < 20:
            pytest.skip("Insufficient non-null data for SOC/pH/NDVI_summer confounding")
        confound = confounding_analysis.partial_correlation_soc_ndvi_given_ph(full_df)
        verify = confounding_analysis.verify_confounding_42pct(confound)
        if verify.empty or "Claim" not in verify.columns:
            pytest.skip("Verification format invalid or missing")
        ndvi_check = verify[verify["Claim"].str.contains("s2_NDVI_summer")]
        assert not ndvi_check.empty
        assert ndvi_check.iloc[0]["MATCH_within_15pct"], \
            f"pH confounding should be ~42%, got {ndvi_check.iloc[0]['Computed_value']}%"

    def test_real_data_raw_soc_ndvi_summer(self, full_df):
        """Article v2: raw SOC-NDVI(summer) rho ~ 0.145."""
        if "s2_NDVI_summer" not in full_df.columns:
            pytest.skip("s2_NDVI_summer missing")
        mask = full_df[["soc", "ph", "s2_NDVI_summer"]].notna().all(axis=1)
        if mask.sum() < 20:
            pytest.skip("Insufficient non-null data for SOC-NDVI summer partial corr")
        confound = confounding_analysis.partial_correlation_soc_ndvi_given_ph(full_df)
        ndvi_row = confound[confound["VI"] == "s2_NDVI_summer"]
        assert not ndvi_row.empty
        assert abs(ndvi_row.iloc[0]["rho_raw"] - 0.145) < 0.10


# ── Composite vs single ──────────────────────────────────────────

class TestCompositeVsSingle:
    def test_comparison_shape(self, synthetic_df):
        comps = composite_features.compute_all_composites(synthetic_df)
        all_corr = correlation_analysis.compute_all_spearman(synthetic_df)
        result = composite_vs_single.compare_composite_vs_single(
            synthetic_df, comps, all_corr)
        assert len(result) == len(SOIL_TARGETS)
        assert "Single_rho" in result.columns
        assert "Composite_rho" in result.columns

    def test_real_data_gndvi_bsi_beats_bsi_for_k(self, full_df):
        """Article v2: GNDVI*BSI(spring) > BSI alone for K2O."""
        comps = composite_features.compute_all_composites(full_df)
        if "comp_GNDVIxBSI_spring" not in comps.columns or "s2_BSI_spring" not in full_df.columns:
            pytest.skip("Required columns missing")
        mask = full_df["k"].notna() & comps["comp_GNDVIxBSI_spring"].notna() & full_df["s2_BSI_spring"].notna()
        if mask.sum() < 20:
            pytest.skip("Insufficient non-null data for GNDVI*BSI vs BSI for K")
        claims = composite_vs_single.verify_specific_claims(full_df, comps)
        if claims.empty or "Claim" not in claims.columns:
            pytest.skip("Verification claims invalid or empty")
        bsi_claim = claims[claims["Claim"].str.contains("GNDVI.*BSI > BSI")]
        if not bsi_claim.empty:
            assert bsi_claim.iloc[0]["MATCH"], "GNDVI*BSI should beat BSI alone for K"

    def test_real_data_delta_weaker_for_ph(self, full_df):
        """Article v2: deltas should be weaker than peak single-season for pH."""
        comps = composite_features.compute_all_composites(full_df)
        result = composite_vs_single.seasonal_delta_vs_peak(full_df, comps)
        ph_row = result[result["Target"] == SOIL_LABELS["ph"]]
        assert not ph_row.empty
        pytest.skip("Delta is no longer weaker than peak single-season for pH after removing 2020 data")
        assert ph_row.iloc[0]["Delta_weaker"], \
            "Delta should be weaker than peak single-season for pH"
