╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    DATA FLOW DIAGRAM — science_article pipeline                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 0 — RAW DATA SOURCES                                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐
  │  SHP-files       │    │  Excel/PDF       │    │  Google Earth Engine (GEE)  │
  │  (field polygons │    │  protocols       │    │  ┌─────────┐ ┌────────────┐ │
  │   + soil data)   │    │  (agrochemical   │    │  │ ERA5    │ │ Sentinel-2 │ │
  └────────┬─────────┘    │   lab results)   │    │  │ climate │ │ (S2)       │ │
           │              └────────┬─────────┘    │  └─────────┘ └────────────┘ │
           │                       │              │  ┌─────────┐ ┌────────────┐ │
           ▼                       ▼              │  │Landsat 8│ │ Sentinel-1 │ │
  ┌─────────────────────────────────────┐         │  │ (L8)    │ │ (S1 SAR)   │ │
  │  build_soil_db.py                   │         │  └─────────┘ └────────────┘ │
  │  ┌──────────────────────────────┐   │         │  ┌─────────┐ ┌────────────┐ │
  │  │ extract_metadata()           │   │         │  │  SRTM   │ │ SoilGrids  │ │
  │  │ process_shapefile()          │   │         │  │ (DEM)   │ │            │ │
  │  │ merge_with_protocol_metadata │   │         │  └─────────┘ └────────────┘ │
  │  │  └─ normalize_field_name()   │   │         └─────────────────────────────┘
  │  └──────────────────────────────┘   │                      │
  │  raw_data/pdf_excel_data/           │                  gee_auth.py
  │  ├─ extract_protocol_metadata.py    │            authenticate_and_initialize()
  │  ├─ build_database.py               │                      │
  │  └─ extract_data.py                 │                      │
  └────────────────┬────────────────────┘                      │
                   │                                           │
                   ▼                                           │
  ┌────────────────────────────────┐                           │
  │  soil_analysis.db (SQLite)     │◄──────────────────────────┤
  │  table: soil_samples           │                           │
  │  (field_name, year, lat, lon,  │                           │
  │   pH, SOC/humus, P2O5, K2O...) │                           │
  └────────────────┬───────────────┘                           │
                   │                                           │
                   │                                           │
┌──────────────────▼───────────────────────────────────────────▼────────────────────┐
│  LAYER 1 — FEATURE EXTRACTION (GEE → CSV files)                                   │
│  Shared helpers: db_utils.py (get_field_polygons, get_sampling_dates,             │
│                               get_region_bbox, save_features_to_db)               │
│                  file_utils.py (should_skip_file)                                 │
└───────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s01_temperature.py                                                            │
  │  get_region_bbox() → ERA5 GEE → extract_temperature_for_year()                 │
  │  → determine_seasonal_windows()                                                │
  │  OUTPUT: data/temperature/temperature_{year}.csv                               │
  │          data/temperature/seasonal_windows_{year}.json   ◄── used by s02, s03  │
  └────────────────────────────────────────────────────────────────────────────────┘
           │ seasonal_windows_{year}.json
           ▼
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s02_sentinel2.py                                                              │
  │  get_field_polygons() + load_seasonal_windows()                                │
  │  → cloud_mask_s2() → compute_s2_indices()                                      │
  │  → extract_s2_features(polygon, start, end)                                    │
  │  → process_fields_for_season()                                                 │
  │  OUTPUT: data/sentinel2/s2_{year}_{season}.csv                                 │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s03_landsat8.py                                                               │
  │  get_field_polygons() + load_seasonal_windows()  (same pattern as s02)         │
  │  → cloud_mask_l8() → apply_l8_scale() → compute_l8_indices()                   │
  │  → extract_l8_features() → process_fields_for_season()                         │
  │  OUTPUT: data/landsat8/l8_{year}_{season}.csv                                  │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s04_sentinel1.py                                                              │
  │  get_field_polygons() + get_sampling_dates()                                   │
  │  → extract_s1_features(polygon, start, end)   [SAR VV/VH backscatter]          │
  │  OUTPUT: data/sentinel1/s1_{year}.csv                                          │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s05_topography.py                                                             │
  │  get_field_polygons() → extract_topo_features(polygon)   [SRTM DEM]            │
  │  OUTPUT: data/topography/topography.csv    (static, no year loop)              │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s06_soil_maps.py                                                              │
  │  get_field_polygons() → extract_soilgrids_features(polygon)                    │
  │  → _reduce_soilgrids_band()                                                    │
  │  OUTPUT: data/soilgrids/soilgrids.csv    (static)                              │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s07_hyperspectral.py                                                          │
  │  check_prisma_availability() / check_enmap_availability() / check_lucas()      │
  │  OUTPUT: availability report only (PRISMA/EnMAP not available → skipped)       │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s09_climate.py                                                                │
  │  get_field_polygons() + get_sampling_dates()                                   │
  │  → extract_climate_features(polygon, year, gs_start, gs_end)                   │
  │  → _reduce_with_fallback()   [ERA5: MAT, MAP, GS_temp, GS_precip]              │
  │  OUTPUT: data/climate/climate_{year}.csv                                       │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s11_spectral_eng.py  ← reads s02 output CSVs                                  │
  │  calculate_evi() → calculate_band_ratios() → calculate_pca()                   │
  │  process_s2_file()                                                             │
  │  OUTPUT: data/spectral_eng/spectral_{year}_{season}.csv                        │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s12_glcm.py  ← uses GEE S2 images                                             │
  │  extract_glcm_features(polygon, start, end)   [texture: contrast, entropy...]  │
  │  OUTPUT: data/glcm/glcm_{year}_{season}.csv                                    │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  s10_semivariogram.py  ← reads soil_samples from DB directly                   │
  │  load_soil_data() → calculate_semivariogram(coords, values)                    │
  │  OUTPUT: data/semivariogram/{property}_variogram.csv + .png                    │
  │  (spatial autocorrelation analysis, informational — not fed into merge)        │
  └────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — FEATURE MERGE                                                             │
└──────────────────────────────────────────────────────────────────────────────────────┘

  All CSVs from Layer 1 ──────────────────────────────────────────►┐
  soil_analysis.db (soil_samples) ──────────────────────────────►┐ │
                                                                   ▼ ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  s08_merge_features.py                                                       │
  │                                                                              │
  │  load_temperature_data()    ← data/temperature/*.csv                         │
  │  load_sentinel2_data()      ← data/sentinel2/*.csv                           │
  │  load_landsat8_data()       ← data/landsat8/*.csv                            │
  │  load_sentinel1_data()      ← data/sentinel1/*.csv                           │
  │  load_topography_data()     ← data/topography/topography.csv                 │
  │  load_climate_data()        ← data/climate/*.csv                             │
  │  load_soilgrids_data()      ← data/soilgrids/soilgrids.csv                   │
  │  load_spectral_eng_data()   ← data/spectral_eng/*.csv                        │
  │  load_glcm_data()           ← data/glcm/*.csv                                │
  │  load_soil_samples()        ← soil_analysis.db                               │
  │                                                                              │
  │  add_soc_column()           [humus → SOC via Van Bemmelen × 0.58]            │
  │  _aggregate_to_field_level()  [pixel-level → field-level mean]               │
  │  merge_static_features()    [S1 + topo + climate]                            │
  │  merge_all_features()       [everything joined on field_name + year]         │
  │  save_features_to_db()                                                       │
  └───────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │  soil_analysis.db              │
                  │  table: full_dataset           │
                  │  (~300+ feature columns)       │
                  │  field_name | year | season |  │
                  │  s2_* | l8_* | s1_* | topo_*   │
                  │  climate_* | soilgrids_* |     │
                  │  glcm_* | spectral_* |         │
                  │  pH | soc | P2O5 | K2O | ...   │
                  └───────────────┬────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────────────────┐
│  LAYER 3 — STATISTICAL ANALYSIS  (math_statistics/)                                  │
└──────────────────────────────────────────────────────────────────────────────────────┘

                         run_all.py → load_data()
                                │
                ┌───────────────┼─────────────────────────────────────────┐
                │               │                                         │
                ▼               ▼                                         ▼
  ┌─────────────────┐  ┌─────────────────┐                   ┌─────────────────────┐
  │descriptive_stats│  │intercorrelation │                   │ correlation_analysis│
  │ .compute_desc.. │  │ .compute_inter..│                   │ .compute_all_       │
  │ .shapiro_wilk() │  │ .verify_article │                   │   spearman()        │
  │ .kruskal_wallis │  │ .check_weak_..  │                   │ .apply_bh_          │
  └────────┬────────┘  └────────┬────────┘                   │   correction()      │
           │                    │                            │ .verify_article_    │
           │                    │                            │   claims()          │
           │                    │                            └─────────┬───────────┘
           │                    │                                       │
           ▼                    ▼                                       ▼
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  composite_features.py                                                         │
  │  compute_inter_index_combinations()  [pairwise: GNDVI×BSI, etc.]               │
  │  compute_multiseasonal_deltas()      [Δ summer−spring, amplitude]              │
  │  compute_normalised_band_differences()                                         │
  │  → 148 composite features DataFrame                                            │
  └───────────┬────────────────────────────────────────────────────────────────────┘
              │
    ┌─────────┴────────┐
    ▼                  ▼
  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  ┌───────────────────┐
  │ derived_soil │  │composite_vs_     │  │confounding_       │  │ variance_         │
  │ .compute_    │  │single.py         │  │analysis.py        │  │ decomposition.py  │
  │  derived_    │  │ compare_comp_vs_ │  │ partial_corr_     │  │ decompose_        │
  │  indicators()│  │  single()        │  │  soc_ndvi_ph()    │  │  variance()       │
  │ [9 new cols] │  │ verify_specific_ │  │ ndvi_saturation_  │  │ verify_claims()   │
  └──────┬───────┘  │  claims()        │  │  curve()          │  └─────────┬─────────┘
         │          └──────────────────┘  └───────────────────┘            │
         │                                                                 │
  ┌──────┴──────────┐  ┌──────────────────────┐  ┌───────────────────────┐ │
  │ seasonal_       │  │ spatial_analysis.py  │  │ s10_semivariogram.py  │ │
  │ analysis.py     │  │ morans_i()           │  │ calculate_semivario.. │ │
  │ assign_soc_     │  │ compute_morans_i_all │  │ (spatial autocorr.)   │ │
  │  class()        │  │ latitudinal_gradient │  └───────────────────────┘ │
  │ seasonal_ndvi_  │  └──────────────────────┘                            │
  │  by_soc_class() │                                                      │
  └─────────────────┘                                                      │
                                                                           │
┌──────────────────────────────────────────────────────────────────────────▼───────────┐
│  LAYER 4 — VISUALIZATION  (math_statistics/plots.py + approximated/)                 │
└──────────────────────────────────────────────────────────────────────────────────────┘

  plots.py (run_all_plots) receives results dicts from all analysis modules:
  ├── plot_histograms()               ← df
  ├── plot_intercorrelation_heatmap() ← rho_matrix, p_matrix
  ├── plot_s2_index_heatmap()         ← corr_df
  ├── plot_top_scatters()             ← df, corr_df
  ├── plot_seasonal_ndvi()            ← ndvi_table
  ├── plot_band_correlations()        ← corr_df
  ├── plot_spatial_maps()             ← df
  ├── plot_bootstrap_ci()             ← df
  ├── plot_claim_verification()       ← claims_df
  ├── plot_composite_vs_single()      ← comparison_df
  ├── plot_variance_decomposition()   ← decomp_df
  ├── plot_confounding()              ← confound_df
  ├── plot_ndvi_saturation()          ← sat_curve
  └── ... (20 plot functions total)
  OUTPUT: PNG figures in output dir

  approximated/ — pixel-level spatial prediction maps:
  ┌─────────────────────────────────────────────────────────────────┐
  │  full_dataset + pixel-level S2 CSVs                             │
  │        │                                                        │
  │  build_extra_features.py → derived band ratios                  │
  │        │                                                        │
  │  rf_train_cv.py → stage1_variance_filter()                      │
  │                   stage2_dedup()                                │
  │                   stage3_rf_importance()                        │
  │        │                                                        │
  │  rf_grid_train_all.py → hyperparameter grid search              │
  │        │                   make_heatmap()                       │
  │        │                                                        │
  │  rf_pixel_geo_maps.py  → lofo_cv_rho() / kfold5_cv_rho()        │
  │  rf_ensemble_maps.py   → ensemble predictions                   │
  │  rf_pixel_maps.py      → build_ridge_predictions()              │
  │        │                                                        │
  │  pixel_geo_approx.py   → _make_smooth_grid() → render_map()     │
  │  pixel_geo_cv.py       → build_X() → CV predictions             │
  │  pixel_heatmap.py      → save_single()                          │
  │  pixel_ndvi_real.py    → render_map() [dual-zone]               │
  │  pixel_p_enriched.py   → train_ridge() → predict_inside/outside │
  │  pixel_ranking_heatmap.py → train_and_predict()                 │
  └─────────────────────────────────────────────────────────────────┘
  OUTPUT: PNG maps (spatial heatmaps of soil properties)

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  KEY DATA TYPES FLOWING THROUGH THE PIPELINE                                         │
└──────────────────────────────────────────────────────────────────────────────────────┘

  ee.Geometry  ──►  GEE collections  ──►  dict[str, float]  ──►  pd.DataFrame  ──►  CSV
  SHP/Excel    ──►  pd.DataFrame     ──►  SQLite (soil_samples)
  CSVs ×N      ──►  merge           ──►  SQLite (full_dataset)  ──►  analysis  ──►  PNG

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  EXECUTION ORDER (main.py orchestrates)                                              │
└──────────────────────────────────────────────────────────────────────────────────────┘

  build_soil_db.py → s01 → s02 → s03 → s04 → s05 → s06 → s09 → s11 → s12
                                                                │
                                                       s08_merge_features ◄┘
                                                                │
                                                   math_statistics/run_all.py
                                                                │
                                                       approximated/*.py
Краткое резюме архитектуры:
СлойЧто происходитКлючевые хранилища0 — Сырые данныеSHP-файлы + Excel-протоколы → build_soil_db.pysoil_analysis.db::soil_samples1 — Извлечение признаковs01–s12 запрашивают GEE по полигонам полейCSV-файлы в data/2 — Слияниеs08_merge_features.py объединяет все CSV + таблицу с образцамиsoil_analysis.db::full_dataset3 — Статистикаmath_statistics/run_all.py → модули корреляций, дисперсии, пространственного анализаdict[str, pd.DataFrame] в памяти4 — Визуализацияplots.py + approximated/*.py → PNG-картыPNG-файлыКритические точки зависимости:

s01 → seasonal_windows_{year}.json → s02, s03 (без температурных окон нет сезонных признаков)
s02 → CSV → s11_spectral_eng.py (spectral engineering работает поверх S2)
s08 — единственная точка сборки: падение любого загрузчика (load_*_data) ведёт к пропуску группы признаков
gee_auth.py::authenticate_and_initialize() — общая зависимость для s01–s06, s09, s12

========================================================================================
  CODEBASE MAP  (AST-extracted signatures)
========================================================================================

────────────────────────────────────────────────────────────────────────────────────────
  src\db_utils.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def get_connection() -> sqlite3.Connection   # L13
    """Get SQLite connection to soil_analysis.db."""

  def get_field_polygons() -> gpd.GeoDataFrame   # L20
    """Extract unique field polygons from soil_samples table."""

  def get_sampling_dates() -> dict[int, list[str]]   # L52
    """Get sampling dates grouped by year."""

  def get_region_bbox() -> tuple[float, float, float, float]   # L78
    """Calculate bounding box of all sampling points."""

  def save_features_to_db(
                          table_name: str,
                          df: pd.DataFrame,
                          if_exists: str = 'replace'
  ) -> None   # L102
    """Save feature DataFrame to SQLite database."""

  def table_exists(table_name: str) -> bool   # L121
    """Check if a table exists in the database."""

────────────────────────────────────────────────────────────────────────────────────────
  src\file_utils.py  (line counts: 0 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def should_skip_file(file_path: Path, min_size_bytes: int = 1) -> bool   # L11
    """Check if file exists and has content."""

────────────────────────────────────────────────────────────────────────────────────────
  src\gee_auth.py  (line counts: 0 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def authenticate_and_initialize() -> None   # L5
    """Authenticate and initialize Google Earth Engine."""

  def check_gee_ready() -> bool   # L22
    """Check if GEE is ready to use."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s01_temperature.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def kelvin_to_celsius(kelvin: float) -> float   # L20
    """Convert Kelvin to Celsius."""

  def extract_temperature_for_year(
                                   year: int,
                                   bbox: tuple[float,
                                   float,
                                   float,
                                   float]
  ) -> pd.DataFrame   # L25
    """Extract ERA5 monthly temperature for a given year and region."""

  def determine_seasonal_windows(temp_df: pd.DataFrame) -> dict[str, tuple[str, str]]   # L90
    """Determine date ranges for each season based on growing season mask."""

  def main() -> None   # L133
    """Main execution: extract temperature for all years."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s02_sentinel2.py  (line counts: 9 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def load_seasonal_windows() -> dict[int, dict[str, tuple[str, str]]]   # L24
    """Load seasonal windows from temperature extraction output."""

  def cloud_mask_s2(image: ee.Image) -> ee.Image   # L62
    """Apply cloud mask to Sentinel-2 image using SCL band."""

  def compute_s2_indices(image: ee.Image) -> ee.Image   # L72
    """Compute spectral indices from Sentinel-2 bands."""

  def extract_s2_features(
                          polygon: ee.Geometry,
                          start_date: str,
                          end_date: str
  ) -> dict[str, float] | None   # L138
    """Extract S2 features for a single polygon and date range."""

  def process_fields_for_season(
                                fields_gdf: gpd.GeoDataFrame,
                                year: int,
                                season: str,
                                start_date: str,
                                end_date: str
  ) -> pd.DataFrame   # L194
    """Process all fields for a given year and season."""

  def main() -> None   # L248
    """Main execution: extract S2 features for all years and seasons."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s03_landsat8.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def cloud_mask_l8(image: ee.Image) -> ee.Image   # L24
    """Apply cloud mask to Landsat 8 using QA_PIXEL band."""

  def apply_l8_scale(image: ee.Image) -> ee.Image   # L37
    """Apply Collection 2 Level 2 scale factors to Landsat 8 SR bands."""

  def compute_l8_indices(image: ee.Image) -> ee.Image   # L55
    """Compute spectral indices from Landsat 8 bands."""

  def extract_l8_features(
                          polygon: ee.Geometry,
                          start_date: str,
                          end_date: str
  ) -> dict[str, float] | None   # L81
    """Extract L8 features for a single polygon and date range."""

  def process_fields_for_season(
                                fields_gdf: gpd.GeoDataFrame,
                                year: int,
                                season: str,
                                start_date: str,
                                end_date: str
  ) -> pd.DataFrame   # L125
    """Process all fields for a given year and season."""

  def main() -> None   # L163
    """Main execution: extract L8 features for all years and seasons."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s04_sentinel1.py  (line counts: 5 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def extract_s1_features(
                          polygon: ee.Geometry,
                          start_date: str,
                          end_date: str
  ) -> dict[str, float] | None   # L23
    """Extract S1 SAR features for a single polygon and date range."""

  def main() -> None   # L69
    """Main execution: extract S1 features for all years."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s05_topography.py  (line counts: 5 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def extract_topo_features(polygon: ee.Geometry) -> dict[str, float]   # L23
    """Extract topographic features for a single polygon."""

  def main() -> None   # L108
    """Main execution: extract topographic features for all fields."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s06_soil_maps.py  (line counts: 10 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _reduce_soilgrids_band(
                             image: ee.Image,
                             polygon: ee.Geometry,
                             band_name: str
  ) -> float | None   # L77
    """Extract a single SoilGrids band value for a polygon."""

  def extract_soilgrids_features(
                                 polygon: ee.Geometry,
                                 properties: list[str] | None = None
  ) -> dict[str, float | None]   # L111
    """Extract SoilGrids features for a single polygon."""

  def main() -> None   # L185
    """Main execution: extract SoilGrids features for all unique fields."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s07_hyperspectral.py  (line counts: 4 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def check_prisma_availability(bbox: tuple[float, float, float, float]) -> dict   # L17
    """Check PRISMA hyperspectral data availability."""

  def check_enmap_availability(bbox: tuple[float, float, float, float]) -> dict   # L32
    """Check EnMAP hyperspectral data availability."""

  def check_lucas_availability() -> dict   # L47
    """Check LUCAS soil spectral library availability."""

  def main() -> None   # L62
    """Main execution: check hyperspectral data availability."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s08_merge_features.py  (line counts: 20 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def load_temperature_data() -> pd.DataFrame   # L38
    """Load all temperature CSV files."""

  def load_sentinel2_data() -> pd.DataFrame   # L50
    """Load all Sentinel-2 CSV files."""

  def load_landsat8_data() -> pd.DataFrame   # L62
    """Load all Landsat 8 CSV files."""

  def load_sentinel1_data() -> pd.DataFrame   # L74
    """Load all Sentinel-1 CSV files."""

  def load_topography_data() -> pd.DataFrame   # L86
    """Load topography CSV."""

  def load_climate_data() -> pd.DataFrame   # L94
    """Load all climate CSV files."""

  def load_soilgrids_data() -> pd.DataFrame   # L106
    """Load SoilGrids texture/properties CSV."""

  def load_spectral_eng_data() -> pd.DataFrame   # L114
    """Load all spectral engineering CSV files."""

  def load_glcm_data() -> pd.DataFrame   # L126
    """Load all GLCM CSV files."""

  def load_soil_samples() -> pd.DataFrame   # L138
    """Load soil_samples table from database."""

  def add_soc_column(df: pd.DataFrame) -> pd.DataFrame   # L146
    """Convert humus (hu, %) to Soil Organic Carbon (soc, %)."""

  def _aggregate_to_field_level(
                                df: pd.DataFrame,
                                group_keys: list[str],
                                prefix: str
  ) -> pd.DataFrame   # L165
    """Aggregate pixel-level data to field-level by computing mean per group."""

  def merge_static_features() -> pd.DataFrame   # L205
    """Merge static field-level features (S1 + Topography + Climate)."""

  def merge_all_features() -> pd.DataFrame   # L279
    """Merge ALL features (seasonal + static) with soil samples into a single dataset."""

  def _print_coverage_report(df: pd.DataFrame) -> None   # L505
    """Print a coverage report for key feature groups."""

  def main() -> None   # L534
    """Main execution: merge all features and save to database."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s09_climate.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def kelvin_to_celsius(kelvin: float) -> float   # L23
    """Convert Kelvin to Celsius."""

  def _reduce_with_fallback(
                            image: ee.Image,
                            polygon: ee.Geometry,
                            band_name: str,
                            scale: int = 11132
  ) -> float | None   # L28
    """Extract a value from an image, falling back to centroid point"""

  def extract_climate_features(
                               polygon: ee.Geometry,
                               year: int,
                               growing_season_start: str,
                               growing_season_end: str
  ) -> dict[str, float]   # L65
    """Extract climate features for a single polygon and year."""

  def main() -> None   # L124
    """Main execution: extract climate features for all years."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s10_semivariogram.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def load_soil_data() -> pd.DataFrame   # L22
    """Load soil samples with coordinates and target properties."""

  def calculate_semivariogram(
                              coords: np.ndarray,
                              values: np.ndarray,
                              property_name: str,
                              output_dir: Path
  ) -> dict[str, float]   # L46
    """Calculate empirical semivariogram and fit model."""

  def main() -> None   # L133
    """Main execution: calculate semivariograms for all soil properties."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s11_spectral_eng.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def calculate_evi(df: pd.DataFrame) -> pd.DataFrame   # L21
    """Calculate EVI (Enhanced Vegetation Index) from S2 bands."""

  def calculate_band_ratios(df: pd.DataFrame) -> pd.DataFrame   # L55
    """Calculate band ratios for soil analysis."""

  def calculate_pca(
                    df: pd.DataFrame,
                    n_components: int = 5,
                    random_state: int = 42
  ) -> pd.DataFrame   # L91
    """Calculate PCA components from spectral bands."""

  def process_s2_file(s2_path: str, output_path: str) -> None   # L146
    """Process a single S2 CSV file and add spectral engineering features."""

  def main() -> None   # L170
    """Main execution: process all S2 files and add spectral engineering features."""

────────────────────────────────────────────────────────────────────────────────────────
  src\s12_glcm.py  (line counts: 5 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def extract_glcm_features(
                            polygon: ee.Geometry,
                            start_date: str,
                            end_date: str
  ) -> dict[str, float] | None   # L26
    """Extract GLCM texture features for a single polygon and date range."""

  def main() -> None   # L102
    """Main execution: extract GLCM features for all years and seasons."""

────────────────────────────────────────────────────────────────────────────────────────
  approximated\build_extra_features.py  (line counts: 25 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def safe_div(a, b, eps = EPS)   # L92
    """Safe division with NaN where |b| < eps."""

  def get_band(season: str, band: str) -> pd.Series   # L98
    """Return s2_{band}_{season} as Series (raw DN, divide by 10000 for reflectance)."""

  def add_col(name: str, values, display: str, group: str)   # L106

────────────────────────────────────────────────────────────────────────────────────────
  approximated\pixel_geo_approx.py  (line counts: 26 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lo, la)   # L154

  def norm_utm(ux, uy)   # L234
    """Normalize UTM coords to [0,1] using training range."""

  def _draw_validate_polygons(ax, fontsize = 5.5)   # L375

  def _draw_approximate_label(ax, fontsize = 9)   # L394

  def _make_smooth_grid(col_name, df_src)   # L414
    """Interpolate col_name from df_src onto the Mercator grid."""

  def render_map(col, title, cmap_, fname, vmin = None, vmax = None, footer_extra = '')   # L439

────────────────────────────────────────────────────────────────────────────────────────
  approximated\pixel_geo_cv.py  (line counts: 23 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def build_X(sub_df, csv_preds, x_min, x_max, y_min, y_max)   # L118
    """Normalize UTM and build [spec1, spec2, ..., nx, ny, nx*ny] matrix."""

────────────────────────────────────────────────────────────────────────────────────────
  approximated\pixel_heatmap.py  (line counts: 17 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lons, lats)   # L91

  def save_single(
                  col,
                  label,
                  cmap,
                  data_col,
                  vmin_override = None,
                  vmax_override = None,
                  fname_suffix = '',
                  title_suffix = ''
  )   # L259

────────────────────────────────────────────────────────────────────────────────────────
  approximated\pixel_ndvi_real.py  (line counts: 25 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lo, la)   # L105

  def _draw_validate_polygons(ax)   # L428
    """Draw black polygon outlines + 'Validate' labels for each field."""

  def _draw_approximate_label(ax)   # L459
    """Draw 'Approximate' label in the surrounding (non-field) area."""

  def render_map(
                 val_col,
                 title,
                 cmap_,
                 fname,
                 vmin = None,
                 vmax = None,
                 footer_extra = ''
  )   # L473
    """Render a dual-zone chemistry map:"""

────────────────────────────────────────────────────────────────────────────────────────
  approximated\pixel_p_enriched.py  (line counts: 35 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lo, la)   # L234

  def norm_utm(ux, uy)   # L243

  def train_ridge(px_col_names, fl_short_names, label)   # L284
    """Train Ridge model with RidgeCV alpha selection + constant-feature removal."""

  def _build_pred_X(df_src, px_col_names, fl_short_names, nx_arr, ny_arr)   # L380
    """Build raw feature matrix for prediction (same column order as training)."""

  def predict_inside(model_result, px_col_names, fl_short_names, col_out)   # L405

  def predict_outside(model_result, px_col_names, fl_short_names, col_out)   # L486

  def _draw_field_borders(ax, fontsize = 5.5)   # L510

  def _draw_approx_label(ax, text = 'Approximate\n(model extrapolation)', fontsize = 8)   # L527

  def _make_grid(col_name, df_src)   # L540

  def _render_panel(
                    ax,
                    col_in,
                    col_out,
                    title,
                    subtitle = '',
                    r_val = None,
                    basemap = True
  )   # L557
    """Render one map panel on ax."""

────────────────────────────────────────────────────────────────────────────────────────
  approximated\pixel_ranking_heatmap.py  (line counts: 26 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lo, la)   # L237

  def norm_utm(ux, uy)   # L246

  def _get_pixel_col(feat_name)   # L318
    """Map a csv feature name to its pixel-level column (or None if not available)."""

  def train_and_predict(tgt, feat_name, rho_val)   # L329
    """Train Ridge(CV) on farm data using best spectral feature."""

  def make_smooth_grid(mx_arr, my_arr, z_arr)   # L407

  def draw_borders(ax)   # L433

────────────────────────────────────────────────────────────────────────────────────────
  approximated\rf_ensemble_maps.py  (line counts: 24 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lo, la)   # L156

  def norm_utm(ux, uy)   # L176

  def _draw_polygons(ax, fontsize = 5.5)   # L211

  def _draw_approx_label(ax, fontsize = 9)   # L226

  def _smooth_grid(col, df_src)   # L237

  def render_single(ax, vals_in, vals_out, cmap_, v0, v1, title, footer = '')   # L443

────────────────────────────────────────────────────────────────────────────────────────
  approximated\rf_grid_train_all.py  (line counts: 25 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def make_labels(y, lo, hi)   # L121
    """0=low, 1=mid, 2=high based on thresholds lo/hi."""

  def make_heatmap(
                   fig_title,
                   metric_col,
                   label,
                   vmin_override = None,
                   fname = 'heatmap.png',
                   best_col = 'is_best_reg',
                   annotate_col2 = None,
                   annotate_label2 = ''
  )   # L366

────────────────────────────────────────────────────────────────────────────────────────
  approximated\rf_pixel_geo_maps.py  (line counts: 30 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_merc(lo, la)   # L186

  def lofo_cv_rho(X, y, field_ids, n_est, seed)   # L299
    """Leave-One-Field-Out CV, returns Spearman rho on OOF predictions."""

  def kfold5_cv_rho(X, y, n_est, seed)   # L323
    """5-fold CV Spearman rho (faster, less rigorous)."""

  def _draw_validate_polygons(ax, fontsize = 5.5)   # L482

  def _draw_approximate_label(ax, fontsize = 9)   # L499

  def _make_smooth_grid(col_name, df_src)   # L513

────────────────────────────────────────────────────────────────────────────────────────
  approximated\rf_pixel_maps.py  (line counts: 19 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def build_ridge_predictions(full_df, px_df, tgt, best_pred_pairs, alpha = 10.0)   # L198
    """Train Ridge on field-level full_dataset, predict on pixel CSV."""

  def get_scatter_coords(px_df, pred_df, key)   # L280
    """Return (lon, lat, values) arrays for scatter plotting."""

────────────────────────────────────────────────────────────────────────────────────────
  approximated\rf_train_cv.py  (line counts: 25 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def stage1_variance_filter(feat_cols, df_sub)   # L113
    """Remove near-constant features and high-NaN features."""

  def stage2_dedup(feat_cols, df_sub, target_col, threshold = CORR_THRESHOLD)   # L128
    """Hierarchical correlation clustering; keep best representative per cluster."""

  def stage3_rf_importance(feat_cols, X_tr, y_tr, top_n = TOP_FEATS)   # L184
    """Quick RF + permutation importance on held-out validation set."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\composite_features.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _col(prefix: str, name: str, season: str) -> str   # L26
    """Build column name, trying both 's2_' and 'spectral_' prefixes."""

  def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series   # L31
    """Division with 0-protection."""

  def _safe_norm_diff(a: pd.Series, b: pd.Series) -> pd.Series   # L36
    """Normalised difference: (a-b)/(a+b)."""

  def compute_inter_index_combinations(df: pd.DataFrame) -> pd.DataFrame   # L42
    """(a) Pairwise products, ratios, differences of S2 indices per season."""

  def compute_multiseasonal_deltas(df: pd.DataFrame) -> pd.DataFrame   # L77
    """(b) Multi-seasonal deltas: Δ(summer−spring), amplitude, seasonal mean."""

  def compute_normalised_band_differences(df: pd.DataFrame) -> pd.DataFrame   # L114
    """(c) Normalised band differences and ratios."""

  def compute_all_composites(df: pd.DataFrame) -> pd.DataFrame   # L139
    """Compute all 148 composite features and return concatenated."""

  def run(df: pd.DataFrame) -> dict   # L149
    """Compute composites and report summary."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\composite_vs_single.py  (line counts: 8 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def compare_composite_vs_single(
                                  df: pd.DataFrame,
                                  composites: pd.DataFrame,
                                  all_single_corr: pd.DataFrame
  ) -> pd.DataFrame   # L17
    """For each soil target, find best composite and best single, then compare."""

  def verify_specific_claims(df: pd.DataFrame, composites: pd.DataFrame) -> pd.DataFrame   # L60
    """Verify the specific composite feature claims in the article."""

  def seasonal_delta_vs_peak(df: pd.DataFrame, composites: pd.DataFrame) -> pd.DataFrame   # L145
    """Check: multi-seasonal deltas do NOT outperform peak single-season."""

  def run(
          df: pd.DataFrame,
          composites: pd.DataFrame,
          all_single_corr: pd.DataFrame
  ) -> dict   # L196
    """Run composite vs single comparison."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\confounding_analysis.py  (line counts: 7 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def partial_correlation_soc_ndvi_given_ph(df: pd.DataFrame) -> pd.DataFrame   # L23
    """Partial correlation: SOC ~ vegetation_index | pH."""

  def verify_confounding_42pct(confound_df: pd.DataFrame) -> pd.DataFrame   # L81
    """Check article claim: 42% of SOC-NDVI(summer) correlation is pH-confounded."""

  def ndvi_saturation_curve(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame   # L121
    """NDVI saturation curve: mean NDVI as function of SOC."""

  def verify_saturation_claim(sat_curve: pd.DataFrame) -> pd.DataFrame   # L142
    """Verify: NDVI plateau at SOC > 2.5%, only 10% of data in linear zone (SOC < 2.0)."""

  def cv_vs_correlation_strength(
                                 df: pd.DataFrame,
                                 all_corr: pd.DataFrame = None
  ) -> pd.DataFrame   # L162
    """CV of soil property vs maximum absolute correlation."""

  def run(df: pd.DataFrame, all_corr: pd.DataFrame = None) -> dict   # L186
    """Run confounding analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\correlation_analysis.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _get_rs_feature_columns(df: pd.DataFrame) -> list[str]   # L29
    """Return all remote sensing / covariate feature columns."""

  def compute_all_spearman(df: pd.DataFrame) -> pd.DataFrame   # L40
    """Spearman correlations: each soil target vs each RS feature."""

  def apply_bh_correction(corr_df: pd.DataFrame) -> pd.DataFrame   # L68
    """Apply Benjamini-Hochberg FDR correction per target."""

  def top_correlations(corr_df: pd.DataFrame, n_top: int = 20) -> pd.DataFrame   # L86
    """Top-N strongest correlations per soil target."""

  def verify_article_claims(corr_df: pd.DataFrame) -> pd.DataFrame   # L95
    """Verify specific rho values stated in the article."""

  def seasonal_comparison(corr_df: pd.DataFrame) -> pd.DataFrame   # L138
    """Compare spring vs summer correlations for vegetation indices."""

  def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]   # L169
    """Run the full correlation analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\derived_soil.py  (line counts: 9 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def compute_derived_indicators(df: pd.DataFrame) -> pd.DataFrame   # L65
    """Compute all 9 derived soil indicators."""

  def correlate_derived_with_rs(df: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame   # L101
    """Spearman correlations of derived indicators with all RS features."""

  def correlate_derived_with_composites(
                                        derived: pd.DataFrame,
                                        composites: pd.DataFrame
  ) -> pd.DataFrame   # L130
    """Spearman correlations of derived soil with composite spectral features."""

  def verify_article_claims(
                            corr_rs: pd.DataFrame,
                            corr_comp: pd.DataFrame
  ) -> pd.DataFrame   # L152
    """Verify specific article v2 claims about derived indicators."""

  def top_derived_correlations(corr_df: pd.DataFrame, n_top: int = 5) -> pd.DataFrame   # L206
    """Top N correlations for each derived indicator."""

  def run(df: pd.DataFrame, composites: pd.DataFrame = None) -> dict   # L217
    """Run derived soil analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\descriptive_stats.py  (line counts: 4 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def compute_descriptive_table(df: pd.DataFrame) -> pd.DataFrame   # L17
    """Reproduce Table 1: descriptive statistics for soil targets."""

  def shapiro_wilk_tests(df: pd.DataFrame) -> pd.DataFrame   # L37
    """Shapiro-Wilk normality test for each soil property."""

  def kruskal_wallis_by_year(df: pd.DataFrame) -> pd.DataFrame   # L63
    """Kruskal-Wallis test: are soil properties significantly different across years?"""

  def descriptive_stats_by_year(df: pd.DataFrame) -> pd.DataFrame   # L105
    """Per-year descriptive statistics to check for temporal bias."""

  def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]   # L125
    """Run all descriptive statistics analyses."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\intercorrelation.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def compute_intercorrelation_matrix(
                                      df: pd.DataFrame
  ) -> tuple[pd.DataFrame, pd.DataFrame]   # L18
    """Spearman rank correlation matrix among soil properties."""

  def verify_article_intercorrelations(df: pd.DataFrame) -> pd.DataFrame   # L45
    """Check specific inter-correlations stated in the article."""

  def check_weak_correlations(df: pd.DataFrame) -> pd.DataFrame   # L69
    """Verify article claim: P and K have |ρ| < 0.30 with other properties."""

  def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]   # L90
    """Run intercorrelation analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\plots.py  (line counts: 32 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _save(fig, name: str)   # L44
    """Save figure to output directory."""

  def plot_histograms(df: pd.DataFrame)   # L55
    """Distribution histograms for 6 soil properties."""

  def plot_intercorrelation_heatmap(rho_matrix: pd.DataFrame, p_matrix: pd.DataFrame)   # L87
    """Heatmap of soil property intercorrelations."""

  def plot_s2_index_heatmap(corr_df: pd.DataFrame)   # L117
    """Heatmap: soil properties vs S2 spectral indices by season."""

  def plot_top_scatters(df: pd.DataFrame, corr_df: pd.DataFrame, n_top: int = 9)   # L148
    """Scatter plots for strongest correlations."""

  def plot_seasonal_ndvi(ndvi_table: pd.DataFrame)   # L184
    """Line plot: seasonal NDVI trajectory per SOC class."""

  def plot_band_correlations(corr_df: pd.DataFrame)   # L212
    """Grouped bar chart: S2 summer band correlations with soil properties."""

  def plot_topo_climate_correlations(corr_df: pd.DataFrame)   # L243
    """Grouped bar chart: topo + climate feature correlations."""

  def plot_spatial_maps(df: pd.DataFrame)   # L272
    """Scatter-based spatial maps of soil properties."""

  def plot_qq(df: pd.DataFrame)   # L299
    """QQ-plots for normality assessment."""

  def plot_boxplots_by_year(df: pd.DataFrame)   # L318
    """Boxplots of soil properties split by year."""

  def plot_bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000)   # L348
    """Bootstrap 95% CI for the key article correlations."""

  def plot_claim_verification(claims_df: pd.DataFrame)   # L406
    """Visual comparison: article rho vs computed rho."""

  def plot_vif(df: pd.DataFrame, top_n: int = 30)   # L432
    """Variance Inflation Factor for top correlated features."""

  def plot_composite_vs_single(comparison_df: pd.DataFrame)   # L483
    """Side-by-side bars: best single vs best composite per soil target."""

  def plot_variance_decomposition(decomp_df: pd.DataFrame)   # L511
    """Stacked bar chart: between-field vs within-field variance."""

  def plot_confounding(confound_df: pd.DataFrame)   # L537
    """Grouped bar: raw SOC-VI correlation vs partial (controlling pH)."""

  def plot_ndvi_saturation(sat_curve: pd.DataFrame)   # L570
    """NDVI(summer) as function of SOC — saturation plateau."""

  def plot_cv_vs_rho(cv_rho_df: pd.DataFrame)   # L592
    """Scatter: CV of soil property vs max |rho| with RS features."""

  def plot_derived_soil_top(top_derived_df: pd.DataFrame)   # L614
    """Bar chart of top correlations for derived soil indicators."""

  def plot_delta_vs_peak(delta_df: pd.DataFrame)   # L641
    """Paired bars: seasonal delta vs peak single-season."""

  def run_all_plots(
                    df: pd.DataFrame,
                    rho_matrix: pd.DataFrame = None,
                    p_matrix: pd.DataFrame = None,
                    corr_df: pd.DataFrame = None,
                    ndvi_table: pd.DataFrame = None,
                    claims_df: pd.DataFrame = None,
                    comparison_df: pd.DataFrame = None,
                    decomp_df: pd.DataFrame = None,
                    confound_df: pd.DataFrame = None,
                    sat_curve: pd.DataFrame = None,
                    cv_rho_df: pd.DataFrame = None,
                    top_derived_df: pd.DataFrame = None,
                    delta_df: pd.DataFrame = None
  )   # L665
    """Generate all plots."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\run_all.py  (line counts: 11 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def load_data() -> pd.DataFrame   # L35
    """Load and validate the full dataset."""

  def generate_text_report(all_results: dict) -> str   # L49
    """Generate a human-readable verification report."""

  def main()   # L228

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\seasonal_analysis.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def assign_soc_class(df: pd.DataFrame) -> pd.DataFrame   # L24
    """Add SOC class column to dataframe."""

  def seasonal_ndvi_by_soc_class(df: pd.DataFrame) -> pd.DataFrame   # L31
    """Mean NDVI per season per SOC class (Sentinel-2)."""

  def kruskal_wallis_ndvi_per_season(df: pd.DataFrame) -> pd.DataFrame   # L57
    """Kruskal-Wallis test: does NDVI differ significantly across SOC classes?"""

  def verify_article_claims(df: pd.DataFrame) -> pd.DataFrame   # L98
    """Check specific numeric claims from section 3.4."""

  def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]   # L155
    """Run seasonal analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\spatial_analysis.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _inverse_distance_weights(
                                coords: np.ndarray,
                                bandwidth: float = 50000
  ) -> np.ndarray   # L17
    """Compute inverse-distance weight matrix (meters)."""

  def morans_i(values: np.ndarray, W: np.ndarray, W_raw: np.ndarray = None) -> dict   # L43
    """Compute Moran's I statistic with z-test."""

  def compute_morans_i_all(df: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame   # L91
    """Moran's I for all soil targets."""

  def latitudinal_gradient(df: pd.DataFrame) -> pd.DataFrame   # L132
    """Check article claim: pH increases from north to south, SOC decreases."""

  def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]   # L153
    """Run spatial analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  math_statistics\variance_decomposition.py  (line counts: 4 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def decompose_variance(
                         df: pd.DataFrame,
                         group_col: str = 'field_name'
  ) -> pd.DataFrame   # L18
    """One-way variance decomposition for each soil property."""

  def verify_article_claims(decomp: pd.DataFrame) -> pd.DataFrame   # L69
    """Verify specific article claims about variance decomposition."""

  def run(df: pd.DataFrame) -> dict   # L111
    """Run variance decomposition analysis."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\conftest.py  (line counts: 0 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @pytest.fixture
  def sample_bbox()   # L11
    """Sample bounding box for testing."""

  @pytest.fixture
  def sample_polygon()   # L17
    """Sample polygon geometry for testing."""

  @pytest.fixture
  def sample_dates()   # L24
    """Sample date range for testing."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_config.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def test_paths_exist()   # L7
    """Test that all path constants are defined."""

  def test_years_configuration()   # L14
    """Test years configuration."""

  def test_seasons_configuration()   # L21
    """Test seasonal composites configuration."""

  def test_s2_configuration()   # L36
    """Test Sentinel-2 configuration."""

  def test_s2_indices()   # L46
    """Test Sentinel-2 indices formulas."""

  def test_l8_configuration()   # L58
    """Test Landsat 8 configuration."""

  def test_l8_indices()   # L66
    """Test Landsat 8 indices formulas."""

  def test_s1_configuration()   # L75
    """Test Sentinel-1 configuration."""

  def test_dem_configuration()   # L81
    """Test DEM configuration."""

  def test_era5_configuration()   # L86
    """Test ERA5 configuration."""

  def test_gee_settings()   # L92
    """Test GEE settings."""

  def test_crs_configuration()   # L98
    """Test CRS configuration."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_db_utils.py  (line counts: 5 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @pytest.fixture
  def mock_db_connection(tmp_path)   # L14
    """Create a temporary test database."""

  def test_get_connection(mock_db_connection, monkeypatch)   # L51
    """Test database connection."""

  def test_get_connection_missing_db(tmp_path, monkeypatch)   # L67
    """Test connection with missing database."""

  def test_get_field_polygons(mock_db_connection, monkeypatch)   # L76
    """Test field polygon extraction."""

  def test_get_sampling_dates(mock_db_connection, monkeypatch)   # L98
    """Test sampling dates extraction."""

  def test_get_region_bbox(mock_db_connection, monkeypatch)   # L113
    """Test bounding box calculation."""

  def test_save_features_to_db(mock_db_connection, monkeypatch)   # L129
    """Test saving features to database."""

  def test_save_features_to_db_replace(mock_db_connection, monkeypatch)   # L155
    """Test replacing existing table."""

  def test_table_exists(mock_db_connection, monkeypatch)   # L176
    """Test table existence check."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_gee_auth.py  (line counts: 1 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @patch('src.gee_auth.ee')
  def test_authenticate_and_initialize_already_authenticated(mock_ee)   # L8
    """Test when GEE is already authenticated."""

  @patch('src.gee_auth.ee')
  def test_authenticate_and_initialize_needs_auth(mock_ee)   # L22
    """Test when GEE needs authentication."""

  @patch('src.gee_auth.ee')
  def test_check_gee_ready_success(mock_ee)   # L37
    """Test GEE ready check when successful."""

  @patch('src.gee_auth.ee')
  def test_check_gee_ready_failure(mock_ee)   # L52
    """Test GEE ready check when failed."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_math_statistics.py  (line counts: 20 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @pytest.fixture(scope='module')
  def full_df()   # L18
    """Load the real dataset once for all tests."""

  @pytest.fixture
  def synthetic_df()   # L24
    """Small synthetic dataset for unit tests."""

  class TestDescriptiveStats:   # L84
    def test_table1_shape(self, synthetic_df)   # L85
    def test_shapiro_wilk_returns_all_targets(self, synthetic_df)   # L91
    def test_kruskal_wallis_has_all_targets(self, synthetic_df)   # L97
    def test_real_data_normality(self, full_df)   # L101
        """Article claims all 6 properties are non-normal (p < 0.001)."""

  class TestIntercorrelation:   # L111
    def test_rho_matrix_shape(self, synthetic_df)   # L112
    def test_diagonal_is_one(self, synthetic_df)   # L118
    def test_symmetry(self, synthetic_df)   # L123

  class TestCorrelationAnalysis:   # L128
    def test_compute_all_spearman_shape(self, synthetic_df)   # L129
    def test_bh_correction_adds_columns(self, synthetic_df)   # L136
    def test_bh_adjusted_p_gte_raw(self, synthetic_df)   # L142
    def test_real_data_ph_gndvi_spring(self, full_df)   # L148
        """Article: pH vs L8 GNDVI spring rho = -0.67."""
    def test_real_data_ph_map(self, full_df)   # L156
        """Article: pH vs MAP rho = 0.66."""
    def test_real_data_k_bsi_spring(self, full_df)   # L164
        """Article: K2O vs BSI spring rho = -0.48."""

  class TestSeasonalAnalysis:   # L173
    def test_soc_classes_assigned(self, synthetic_df)   # L174
    def test_ndvi_table_shape(self, synthetic_df)   # L179
    def test_real_data_spring_ph_stronger(self, full_df)   # L184
        """Article: spring veg indices stronger with pH than summer."""

  class TestSpatialAnalysis:   # L201
    def test_morans_i_known_pattern(self)   # L202
        """Morans I on perfectly clustered data should be positive."""
    def test_latitudinal_gradient(self, full_df)   # L213
        """Article: pH decreases northward, SOC increases northward."""

  class TestCompositeFeatures:   # L228
    def test_inter_index_has_products(self, synthetic_df)   # L229
        """Composite features should include GNDVIxBSI products."""
    def test_inter_index_has_diffs(self, synthetic_df)   # L235
        """Composite features should include GNDVI-NDRE differences."""
    def test_multiseasonal_deltas(self, synthetic_df)   # L241
        """Should compute delta, amplitude, and mean for each index."""
    def test_all_composites_shape(self, synthetic_df)   # L248
        """Total composite features should be substantial."""
    def test_real_data_composites(self, full_df)   # L254
        """Composite features on real data should produce ~100+ features."""
    def test_real_data_gndvi_bsi_spring(self, full_df)   # L259
        """Article v2: GNDVI*BSI(spring) -> K2O rho = -0.488."""

  class TestDerivedSoil:   # L271
    def test_all_9_indicators(self, synthetic_df)   # L272
        """Should produce exactly 9 derived indicators."""
    def test_formulas_correct(self, synthetic_df)   # L280
        """Verify a few derived formulas are correct."""
    def test_real_data_pk_ratio_vs_slope(self, full_df)   # L290
        """Article v2: P2O5/K2O -> slope rho = -0.56."""
    def test_real_data_mineral_index_vs_l8_nir(self, full_df)   # L297
        """Article v2: mineral_index -> L8 NIR spring rho = -0.47."""

  class TestVarianceDecomposition:   # L308
    def test_decomposition_shape(self, synthetic_df)   # L309
    def test_percentages_sum_to_100(self, synthetic_df)   # L315
    def test_icc_range(self, synthetic_df)   # L321
    def test_real_data_ph_has_high_between_field(self, full_df)   # L326
        """pH should have higher between-field variance than within-field."""
    def test_real_data_ph_gt_soc_between_field(self, full_df)   # L334
        """Article: pH has more between-field variance than SOC."""

  class TestConfoundingAnalysis:   # L345
    def test_partial_correlations_shape(self, synthetic_df)   # L346
    def test_partial_leq_raw(self, synthetic_df)   # L352
        """Partial correlation (controlling pH) should generally reduce |rho|."""
    def test_saturation_curve_shape(self, synthetic_df)   # L360
    def test_real_data_confounding_42pct(self, full_df)   # L366
        """Article v2: 42% of SOC-NDVI(summer) is pH-confounded."""
    def test_real_data_raw_soc_ndvi_summer(self, full_df)   # L375
        """Article v2: raw SOC-NDVI(summer) rho ~ 0.145."""

  class TestCompositeVsSingle:   # L385
    def test_comparison_shape(self, synthetic_df)   # L386
    def test_real_data_gndvi_bsi_beats_bsi_for_k(self, full_df)   # L395
        """Article v2: GNDVI*BSI(spring) > BSI alone for K2O."""
    def test_real_data_delta_weaker_for_ph(self, full_df)   # L403
        """Article v2: deltas should be weaker than peak single-season for pH."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_pipeline_integrity.py  (line counts: 9 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def test_all_output_dirs_are_path_objects()   # L29
    """All *_DIR config constants should be pathlib.Path objects."""

  def test_years_are_sorted_and_continuous()   # L43
    """YEARS should be a sorted list of consecutive integers."""

  def test_seasons_month_ranges_valid()   # L53
    """Season month ranges should be valid (1-12) and non-overlapping."""

  def test_all_s2_indices_have_implementation()   # L71
    """Every index in config.S2_INDICES must have a branch in compute_s2_indices."""

  def test_all_l8_indices_have_implementation()   # L86
    """Every index in config.L8_INDICES must have a branch in compute_l8_indices."""

  def _extract_sql_columns(source_code: str) -> list[str]   # L101
    """Extract column names from SELECT ... FROM blocks in source code."""

  def test_s10_semivariogram_sql_columns()   # L123
    """s10_semivariogram.load_soil_data() SQL columns must match DB schema."""

  def test_db_utils_sql_columns()   # L137
    """db_utils.get_field_polygons() SQL columns must match DB schema."""

  def test_merge_glob_patterns_match_output_filenames()   # L153
    """s08_merge loader glob patterns must match actual script output filenames."""

  def test_topo_filename_matches_merge_loader()   # L178
    """Topography loader uses exact filename, not glob — must match."""

  def test_all_modules_import_successfully()   # L188
    """All pipeline modules should import without error."""

  def test_soilgrids_keys_complete()   # L219
    """SOILGRIDS config should have entries for all expected soil properties."""

  def test_soilgrids_depths_valid()   # L228
    """SOILGRIDS_DEPTHS should contain valid depth range strings."""

  def test_soilgrids_primary_depth_in_depths()   # L235
    """Primary depth must be listed in SOILGRIDS_DEPTHS."""

  def test_should_skip_file_nonexistent(tmp_path)   # L242
    """should_skip_file returns False for nonexistent files."""

  def test_should_skip_file_empty(tmp_path)   # L248
    """should_skip_file returns False for empty files."""

  def test_should_skip_file_with_content(tmp_path)   # L256
    """should_skip_file returns True for files with content."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s01_temperature.py  (line counts: 4 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def test_kelvin_to_celsius()   # L8
    """Test Kelvin to Celsius conversion."""

  @patch('src.s01_temperature.ee')
  def test_extract_temperature_for_year(mock_ee)   # L16
    """Test temperature extraction for a single year."""

  @patch('src.s01_temperature.ee')
  def test_extract_temperature_no_data(mock_ee)   # L46
    """Test temperature extraction when no data available."""

  def test_determine_seasonal_windows()   # L69
    """Test seasonal window determination."""

  def test_determine_seasonal_windows_no_growing_season()   # L93
    """Test seasonal windows when no growing season detected."""

  def test_determine_seasonal_windows_partial_overlap()   # L108
    """Test seasonal windows with partial season overlap."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s02_sentinel2.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @pytest.fixture
  def mock_seasonal_windows(tmp_path)   # L14
    """Create mock seasonal windows file."""

  def test_load_seasonal_windows(mock_seasonal_windows, monkeypatch)   # L32
    """Test loading seasonal windows from file."""

  def test_load_seasonal_windows_missing_file(tmp_path, monkeypatch)   # L46
    """Test loading seasonal windows when file doesn't exist."""

  @patch('src.s02_sentinel2.ee')
  def test_cloud_mask_s2(mock_ee)   # L55
    """Test Sentinel-2 cloud masking."""

  @patch('src.s02_sentinel2.ee')
  def test_compute_s2_indices(mock_ee)   # L74
    """Test Sentinel-2 index computation."""

  @patch('src.s02_sentinel2.ee')
  def test_extract_s2_features_success(mock_ee)   # L92
    """Test successful S2 feature extraction."""

  @patch('src.s02_sentinel2.ee')
  def test_extract_s2_features_no_data(mock_ee)   # L133
    """Test S2 feature extraction when no images available."""

  @patch('src.s02_sentinel2.ee')
  @patch('src.s02_sentinel2.json')
  @patch('src.s02_sentinel2.gpd')
  def test_process_fields_for_season(mock_gpd, mock_json, mock_ee)   # L155
    """Test processing fields for a season."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s03_landsat8.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @patch('src.s03_landsat8.ee')
  def test_cloud_mask_l8(mock_ee)   # L8
    """Test Landsat 8 cloud masking."""

  @patch('src.s03_landsat8.ee')
  def test_compute_l8_indices(mock_ee)   # L26
    """Test Landsat 8 index computation."""

  @patch('src.s03_landsat8.ee')
  def test_extract_l8_features_success(mock_ee)   # L44
    """Test successful Landsat 8 feature extraction."""

  @patch('src.s03_landsat8.ee')
  def test_extract_l8_features_no_data(mock_ee)   # L82
    """Test Landsat 8 extraction when no images available."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s04_s07.py  (line counts: 8 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @patch('src.s04_sentinel1.ee')
  def test_extract_s1_features_success(mock_ee)   # L10
    """Test successful Sentinel-1 feature extraction."""

  @patch('src.s04_sentinel1.ee')
  def test_extract_s1_features_no_data(mock_ee)   # L48
    """Test Sentinel-1 extraction when no images available."""

  @patch('src.s05_topography.ee')
  def test_extract_topo_features(mock_ee)   # L69
    """Test topographic feature extraction."""

  def test_unit_conversions()   # L123
    """Test that UNIT_CONVERSIONS dict is complete and reasonable."""

  def test_depth_weights_sum_to_one()   # L134
    """Depth weights for 0-30cm should sum to 1.0."""

  @patch('src.s06_soil_maps.ee')
  def test_extract_soilgrids_features(mock_ee)   # L142
    """Test SoilGrids feature extraction with mocked GEE."""

  @patch('src.s06_soil_maps.ee')
  def test_reduce_soilgrids_band_fallback(mock_ee)   # L170
    """Test centroid fallback when polygon reduction returns None."""

  def test_check_prisma_availability()   # L195
    """Test PRISMA availability check."""

  def test_check_enmap_availability()   # L207
    """Test EnMAP availability check."""

  def test_check_lucas_availability()   # L217
    """Test LUCAS availability check."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s08_merge.py  (line counts: 13 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @pytest.fixture
  def mock_data_files(tmp_path)   # L10
    """Create mock CSV files for testing."""

  def test_load_temperature_data(mock_data_files, monkeypatch)   # L100
    """Test loading temperature data."""

  def test_load_sentinel2_data(mock_data_files, monkeypatch)   # L112
    """Test loading Sentinel-2 data."""

  def test_load_landsat8_data(mock_data_files, monkeypatch)   # L124
    """Test loading Landsat 8 data."""

  def test_load_sentinel1_data(mock_data_files, monkeypatch)   # L135
    """Test loading Sentinel-1 data."""

  def test_load_topography_data(mock_data_files, monkeypatch)   # L146
    """Test loading topography data."""

  @patch('src.s08_merge_features.get_connection')
  def test_load_soil_samples(mock_get_connection)   # L158
    """Test loading soil samples from database."""

  class TestAddSocColumn:   # L178
    """Tests for SOC conversion."""
    def test_soc_conversion_basic(self)   # L181
        """SOC = hu * 0.58."""
    def test_soc_conversion_with_nulls(self)   # L191
        """SOC should be NaN where hu is NaN."""
    def test_soc_conversion_no_hu_column(self)   # L200
        """Should warn but not crash if hu is missing."""

  class TestAggregateToFieldLevel:   # L207
    """Tests for pixel-to-field aggregation."""
    def test_aggregation_mean(self)   # L210
        """Multiple pixels per field should be averaged."""
    def test_aggregation_preserves_groups(self)   # L229
        """Multiple fields should remain separate after aggregation."""
    def test_aggregation_drops_coords(self)   # L247
        """centroid_lon/lat should be dropped before aggregation."""

  @patch('src.s08_merge_features.load_soil_samples')
  @patch('src.s08_merge_features.load_sentinel2_data')
  @patch('src.s08_merge_features.load_landsat8_data')
  @patch('src.s08_merge_features.load_sentinel1_data')
  @patch('src.s08_merge_features.load_topography_data')
  @patch('src.s08_merge_features.load_climate_data')
  @patch('src.s08_merge_features.load_spectral_eng_data')
  @patch('src.s08_merge_features.load_glcm_data')
  def test_merge_all_features_with_static(
                                          mock_glcm,
                                          mock_spectral,
                                          mock_climate,
                                          mock_topo,
                                          mock_s1,
                                          mock_l8,
                                          mock_s2,
                                          mock_soil
  )   # L274
    """Test that merge_all_features includes S1, topo, and climate."""

  @patch('src.s08_merge_features.load_soil_samples')
  @patch('src.s08_merge_features.load_sentinel2_data')
  @patch('src.s08_merge_features.load_landsat8_data')
  @patch('src.s08_merge_features.load_sentinel1_data')
  @patch('src.s08_merge_features.load_topography_data')
  @patch('src.s08_merge_features.load_climate_data')
  @patch('src.s08_merge_features.load_spectral_eng_data')
  @patch('src.s08_merge_features.load_glcm_data')
  def test_merge_preserves_zeros(
                                 mock_glcm,
                                 mock_spectral,
                                 mock_climate,
                                 mock_topo,
                                 mock_s1,
                                 mock_l8,
                                 mock_s2,
                                 mock_soil
  )   # L359
    """Verify that legitimate 0.0 values are NOT replaced with NaN."""

  def test_van_bemmelen_factor()   # L388
    """Test that VAN_BEMMELEN_FACTOR is the standard 0.58."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s09_climate.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def test_kelvin_to_celsius()   # L10
    """Test Kelvin to Celsius conversion."""

  @patch('src.s09_climate.ee')
  def test_reduce_with_fallback_primary(mock_ee)   # L18
    """Test _reduce_with_fallback when primary reduction succeeds."""

  @patch('src.s09_climate.ee')
  def test_reduce_with_fallback_centroid(mock_ee)   # L38
    """Test _reduce_with_fallback centroid fallback when polygon returns None."""

  @patch('src.s09_climate._reduce_with_fallback')
  @patch('src.s09_climate.ee')
  def test_extract_climate_features(mock_ee, mock_reduce)   # L61
    """Test extract_climate_features returns MAT, MAP, GS_temp, GS_precip."""

  @patch('src.s09_climate._reduce_with_fallback')
  @patch('src.s09_climate.ee')
  def test_extract_climate_features_null_handling(mock_ee, mock_reduce)   # L91
    """Test that None values from GEE are handled gracefully."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s10_semivariogram.py  (line counts: 5 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def test_sql_columns_match_db_schema()   # L23
    """Regression test: all columns in load_soil_data() SQL must exist in real DB."""

  def test_properties_dict_uses_correct_columns()   # L45
    """Verify the properties dict in main() references valid DataFrame columns."""

  def test_calculate_semivariogram_insufficient_data(tmp_path)   # L59
    """Should return empty dict when fewer than 10 samples."""

  def test_calculate_semivariogram_with_nans(tmp_path)   # L71
    """NaN values should be filtered before fitting."""

  def test_calculate_semivariogram_all_nan(tmp_path)   # L94
    """All NaN values should return empty dict (< 10 valid)."""

  def test_calculate_semivariogram_output_files(tmp_path)   # L106
    """Successful variogram should create CSV and PNG files."""

  @patch('src.s10_semivariogram.get_connection')
  def test_load_soil_data(mock_get_conn)   # L132
    """Test load_soil_data returns DataFrame with correct columns."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s11_spectral_eng.py  (line counts: 6 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def test_calculate_evi_basic()   # L12
    """Test EVI calculation with known values."""

  def test_calculate_evi_missing_bands()   # L29
    """EVI should gracefully handle missing bands."""

  def test_calculate_band_ratios()   # L40
    """Test band ratio computation."""

  def test_calculate_band_ratios_zero_denominator()   # L57
    """Band ratios should not crash on zero-valued bands (epsilon added)."""

  def test_calculate_pca_basic()   # L75
    """PCA should produce n_components new columns."""

  def test_calculate_pca_with_nan()   # L93
    """PCA should skip if NaN values present."""

  def test_calculate_pca_too_few_bands()   # L107
    """PCA should skip if fewer than 3 bands available."""

  def test_calculate_pca_b8a_excluded()   # L119
    """BUG-4 awareness: B8A should be excluded from PCA by the isdigit() filter."""

  def test_filename_parsing_standard_seasons()   # L140
    """Standard season names should be parsed correctly."""

  def test_filename_parsing_late_summer()   # L156
    """BUG-1 regression: late_summer must be parsed as 'late_summer', not 'late'."""

  def test_process_s2_file(tmp_path)   # L173
    """Test end-to-end processing of a single S2 CSV file."""

────────────────────────────────────────────────────────────────────────────────────────
  tests\test_s12_glcm.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  @patch('src.s12_glcm.ee')
  @patch('src.s12_glcm.cloud_mask_s2')
  def test_extract_glcm_features_no_data(mock_cloud_mask, mock_ee)   # L10
    """Should return None when no S2 images available."""

  @patch('src.s12_glcm.ee')
  @patch('src.s12_glcm.cloud_mask_s2')
  def test_extract_glcm_features_success(mock_cloud_mask, mock_ee)   # L31
    """Should return renamed GLCM features on success."""

  def test_texture_feature_names()   # L89
    """Verify texture feature names match GEE GLCM band naming convention."""

────────────────────────────────────────────────────────────────────────────────────────
  raw_data\pdf_excel_data\analyze_structure.py  (line counts: 3 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def analyze_pdf(file_path)   # L14
    """Анализ структуры PDF файла"""

  def analyze_excel(file_path)   # L28
    """Анализ структуры Excel файла"""

────────────────────────────────────────────────────────────────────────────────────────
  raw_data\pdf_excel_data\build_database.py  (line counts: 9 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _generate_extraction_report(out: pd.DataFrame, stats: ExtractionStats) -> None   # L33
    """Create multi-sheet Excel report on extraction results."""

  def build_database() -> pd.DataFrame   # L166
    """Extract metadata from all Excel protocols, normalize, save CSV + report."""

────────────────────────────────────────────────────────────────────────────────────────
  raw_data\pdf_excel_data\extract_data.py  (line counts: 9 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def extract_date_from_text(text)   # L21
    """Извлекает дату из текста в различных форматах"""

  def extract_field_number(text)   # L43
    """Извлекает номер поля из текста"""

  def extract_parameter_name(text, filename)   # L57
    """Извлекает название параметра из текста или имени файла"""

  def extract_from_pdf(file_path)   # L90
    """Извлекает данные из PDF файла"""

  def extract_from_excel(file_path)   # L129
    """Извлекает данные из Excel файла"""

  def process_directory(base_path)   # L183
    """Обрабатывает все PDF и Excel файлы в директории"""

────────────────────────────────────────────────────────────────────────────────────────
  raw_data\pdf_excel_data\extract_protocol_metadata.py  (line counts: 11 KB)
────────────────────────────────────────────────────────────────────────────────────────

  class ExtractionStats:   # L30
    """Tracks what was extracted and what was rejected."""

  def _extract_header_metadata(df: pd.DataFrame) -> dict   # L41
    """Scan first 40 rows for protocol metadata."""

  def _find_data_start(df: pd.DataFrame) -> int   # L99
    """Find the column-number row (1, 2, 3, ...) that precedes data."""

  def _extract_field_names(df: pd.DataFrame, data_start: int) -> list[str]   # L114
    """Extract unique field names from col 2 (forward-filled)."""

  def extract_metadata_from_file(xlsx_path: Path, stats: ExtractionStats) -> list[dict]   # L157
    """Extract metadata rows from a single Excel protocol file."""

  def extract_all_metadata(base_path: Path) -> tuple[pd.DataFrame, ExtractionStats]   # L259
    """Scan all Excel protocol files and return metadata DataFrame + stats."""

────────────────────────────────────────────────────────────────────────────────────────
  raw_data\pdf_excel_data\farm_name_map.py  (line counts: 4 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def normalize_farm_name(raw: str) -> str   # L86
    """Normalize a farm name from Excel directory or applicant field."""

────────────────────────────────────────────────────────────────────────────────────────
  ast_map.py  (line counts: 15 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def _annotation_str(node: Optional[ast.expr]) -> str   # L43
    """Convert an annotation AST node to its source string."""

  def _arg_str(arg: ast.arg) -> str   # L53
    """Format a single function argument with its annotation."""

  def _default_str(node: ast.expr) -> str   # L62

  def _func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str   # L69
    """Reconstruct a human-readable function signature from AST."""

  def _docstring_first_line(
                            node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
  ) -> str   # L109
    """Extract only the first line of a docstring (or empty string)."""

  def parse_file(path: pathlib.Path) -> list[dict]   # L122
    """Parse a single Python file and return a list of entry dicts:"""

  def scan_directory(
                     directory: pathlib.Path,
                     recursive: bool = True
  ) -> dict[pathlib.Path, list[dict]]   # L185
    """Scan a directory and return {file_path: entries} mapping."""

  def _wrap_signature(sig: str, indent: str) -> str   # L208
    """Wrap long signatures gracefully."""

  def format_text(file_map: dict[pathlib.Path, list[dict]], root: pathlib.Path) -> str   # L227
    """Render the codebase map as readable text."""

  def format_json(file_map: dict[pathlib.Path, list[dict]], root: pathlib.Path) -> str   # L278
    """Render the codebase map as JSON."""

  def grep_map(
               file_map: dict[pathlib.Path,
               list[dict]],
               query: str,
               root: pathlib.Path
  ) -> str   # L289
    """Find all functions/methods/classes whose name contains `query`."""

  def summary_stats(file_map: dict[pathlib.Path, list[dict]]) -> str   # L325
    """Print a quick statistics summary at the end."""

  def build_file_map(target_dirs: list[str]) -> dict[pathlib.Path, list[dict]]   # L346
    """Scan target directories and merge into a single ordered dict."""

  def main() -> None   # L364

────────────────────────────────────────────────────────────────────────────────────────
  build_soil_db.py  (line counts: 15 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def extract_metadata(shp_path: Path) -> dict[str, str | int]   # L57
    """Parse year/farm/field from directory structure."""

  def process_shapefile(shp_path: Path) -> pd.DataFrame | None   # L71
    """Read a single shapefile and normalize to unified schema."""

  def _normalize_field_name(name: str) -> str   # L112
    """Normalize field_name for fuzzy matching between SHP and Excel."""

  def merge_with_protocol_metadata(shp_df: pd.DataFrame) -> pd.DataFrame   # L125
    """LEFT JOIN SHP data with protocol metadata on (year, farm, field_name)."""

  def generate_merge_report(
                            shp_df: pd.DataFrame,
                            result: pd.DataFrame,
                            shp_errors: list[str] | None = None
  ) -> None   # L168
    """Create a multi-sheet Excel report on merge coverage."""

  def build_database() -> pd.DataFrame   # L336
    """Main pipeline: scan → extract → normalize → merge → load."""

────────────────────────────────────────────────────────────────────────────────────────
  copernicus_s1_check.py  (line counts: 5 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def get_access_token() -> str   # L30
    """Get OAuth2 access token from Copernicus."""

  def search_s1_products(
                         bbox: tuple[float,
                         float,
                         float,
                         float],
                         start_date: str,
                         end_date: str,
                         token: str
  ) -> list[dict]   # L44
    """Search for S1 products in the catalog."""

  def check_s1_availability_2022(year: int = 2022) -> None   # L87
    """Check S1 product availability for 2022."""

────────────────────────────────────────────────────────────────────────────────────────
  debug_s1_2022.py  (line counts: 2 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def check_s1_by_year(polygon: ee.Geometry, year: int, start: str, end: str) -> int   # L19
    """Check S1 image count for a year."""

  def check_s2_by_year(polygon: ee.Geometry, year: int, start: str, end: str) -> int   # L25
    """Check S2 image count for a year."""

  def main() -> None   # L31
    """Main execution."""

────────────────────────────────────────────────────────────────────────────────────────
  heatmap_run.py  (line counts: 8 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def to_mercator(lons, lats)   # L54

  def _render_ax(ax, fig_single, col, label, valid)   # L105
    """Рисует одну тепловую карту на переданный ax."""

────────────────────────────────────────────────────────────────────────────────────────
  inspect_shp.py  (line counts: 2 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def read_dbf_fields(dbf_path: Path) -> list[str]   # L5
    """Read field names from a .dbf file (lightweight, no dependencies)."""

  def main()   # L30

────────────────────────────────────────────────────────────────────────────────────────
  main.py  (line counts: 0 KB)
────────────────────────────────────────────────────────────────────────────────────────

  def main()   # L1

========================================================================================
  Files: 66 | Functions: 353 | Classes: 13 | Methods: 47