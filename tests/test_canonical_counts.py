from scripts.canonical_counts import get_canonical_counts


def test_total_columns_match_530():
    counts = get_canonical_counts()
    assert counts["total_columns"] == 530


def test_feature_breakdown_sums_to_512():
    counts = get_canonical_counts()
    feature_keys = ["s2_base", "l8_base", "spectral_eng", "glcm", "ts_stats",
                    "delta_multiseason", "range_stats", "cs_cross_sensor",
                    "topo", "climate"]
    assert sum(counts[k] for k in feature_keys) == counts["features_total"]


def test_composites_breakdown():
    counts = get_canonical_counts()
    assert counts["comp_inter_index"] == 48
    assert counts["comp_multiseason"] == 42
    assert counts["comp_ndi"] == 20
    assert counts["comp_total"] == 110


def test_base_spectral_is_140():
    """s2_* (104) + l8_* (36) = 140 base spectral features."""
    counts = get_canonical_counts()
    assert counts["s2_base"] == 104
    assert counts["l8_base"] == 36
    assert counts["s2_base"] + counts["l8_base"] == 140
