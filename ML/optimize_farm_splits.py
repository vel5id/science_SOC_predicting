"""
optimize_farm_splits.py
=======================
Grid search for optimal train/val/test farm split.
Constraints: min 4 test farms, min 4 val farms, min 20 train farms.
Method: fast XGBoost + multiple random seeds + all 6 targets.
"""
import io, sys, warnings, argparse, itertools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_CSV    = ROOT / "data" / "features" / "master_dataset.csv"
FEAT_DIR    = ROOT / "data" / "features" / "selected"

TARGETS     = ["ph", "soc", "no3", "p", "k", "s"]
FARM_COL    = "field_name"
TOTAL_FARMS = 81
N_SEEDS     = 15       # random split iterations per combo
MIN_TRAIN   = 20       # minimum training farms

# Coarse grid search (keeps total unique combos manageable)
TEST_SIZES  = [4, 5, 6, 7, 8, 10, 12, 14]
VAL_SIZES   = [4, 5, 6, 8, 10, 12, 15]

# Fast XGBoost (early stop after 15 no-improve rounds)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0, tree_method="hist",
    early_stopping_rounds=15,
)

def get_fallback_features(df):
    num_cols = df.select_dtypes(include="number").columns
    spring   = sorted([c for c in num_cols if "_spring" in c])
    topo     = sorted([c for c in num_cols if c.startswith("topo_")])
    climate  = sorted([c for c in num_cols if c.startswith("climate_")])
    return spring + topo + climate

def load_features(df, target):
    feat_path = FEAT_DIR / f"{target}_best_features.txt"
    if feat_path.exists():
        with open(feat_path) as fh:
            feats = [l.strip() for l in fh if l.strip()]
        feats = [f for f in feats if f in df.columns]
        if feats:
            return feats
    return get_fallback_features(df)

def evaluate_split(X, y, y_log, farms, unique_farms, test_count, val_count, rng):
    shuffled = rng.permutation(unique_farms)
    test_f  = shuffled[:test_count]
    val_f   = shuffled[test_count:test_count + val_count]
    train_f = shuffled[test_count + val_count:]

    tr_mask = np.isin(farms, train_f)
    va_mask = np.isin(farms, val_f)
    te_mask = np.isin(farms, test_f)

    X_tr, y_tr = X[tr_mask], y_log[tr_mask]
    X_va, y_va = X[va_mask], y_log[va_mask]
    X_te, y_te = X[te_mask], y_log[te_mask]

    if len(X_tr) < 5 or len(X_va) < 2 or len(X_te) < 2:
        return None

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    preds_log = model.predict(X_te)
    preds     = np.expm1(preds_log)
    y_te_orig = np.expm1(y_te)

    r2   = r2_score(y_te_orig, preds)
    rmse = float(np.sqrt(mean_squared_error(y_te_orig, preds)))
    mae  = float(mean_absolute_error(y_te_orig, preds))
    rho, _ = spearmanr(y_te_orig, preds)
    if np.isnan(rho):
        rho = 0.0

    return dict(
        r2=r2, rmse=rmse, mae=mae, rho=rho,
        n_train=int(tr_mask.sum()), n_val=int(va_mask.sum()), n_test=int(te_mask.sum()),
    )

def main():
    print("=" * 65)
    print(" SEARCH: OPTIMAL TRAIN / VAL / TEST FARM SPLIT")
    print("=" * 65)
    print(f"  Dataset     : {DATA_CSV}")
    print(f"  Total farms : {TOTAL_FARMS}  |  Seeds: {N_SEEDS}  |  Targets: {len(TARGETS)}")
    print(f"  Test range  : {TEST_SIZES}")
    print(f"  Val  range  : {VAL_SIZES}")
    print()

    print("Loading data...")
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)

    target_data = {}
    for t in TARGETS:
        sub = df_raw.dropna(subset=[t]).reset_index(drop=True)
        feats = load_features(sub, t)
        sub_f = sub.copy()
        for c in feats:
            sub_f[c] = sub_f[c].fillna(sub_f[c].median())
        X = sub_f[feats].values.astype(np.float32)
        y = sub_f[t].values.astype(np.float32)
        y_log  = np.log1p(np.clip(y, 0, None))
        farms  = sub_f[FARM_COL].values
        u_farms = np.unique(farms)
        target_data[t] = (X, y, y_log, farms, u_farms)
        print(f"  {t:>4s}: {len(sub):5d} rows, {len(u_farms):3d} farms, {X.shape[1]:4d} features")

    print()

    valid_combos = [
        (tc, vc)
        for tc, vc in itertools.product(TEST_SIZES, VAL_SIZES)
        if tc + vc + MIN_TRAIN <= TOTAL_FARMS
    ]
    total_fits = len(valid_combos) * N_SEEDS * len(TARGETS)
    print(f"Grid: {len(valid_combos)} combos x {N_SEEDS} seeds x {len(TARGETS)} targets = {total_fits} XGB fits")
    print()

    results = []

    for combo_idx, (test_count, val_count) in enumerate(valid_combos):
        train_count = TOTAL_FARMS - test_count - val_count
        combo_metrics = {m: [] for m in ["rho", "r2", "rmse", "mae", "n_train", "n_test", "n_val"]}

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 37 + test_count * 100 + val_count)
            for t, (X, y, y_log, farms, u_farms) in target_data.items():
                res = evaluate_split(X, y, y_log, farms, u_farms, test_count, val_count, rng)
                if res is None:
                    continue
                for k in combo_metrics:
                    combo_metrics[k].append(res[k])

        if not combo_metrics["rho"]:
            continue

        r = {
            "test_farms":   test_count,
            "val_farms":    val_count,
            "train_farms":  train_count,
            "mean_rho":     float(np.mean(combo_metrics["rho"])),
            "std_rho":      float(np.std(combo_metrics["rho"])),
            "mean_r2":      float(np.mean(combo_metrics["r2"])),
            "std_r2":       float(np.std(combo_metrics["r2"])),
            "mean_rmse":    float(np.mean(combo_metrics["rmse"])),
            "mean_mae":     float(np.mean(combo_metrics["mae"])),
            "mean_n_train": float(np.mean(combo_metrics["n_train"])),
            "mean_n_val":   float(np.mean(combo_metrics["n_val"])),
            "mean_n_test":  float(np.mean(combo_metrics["n_test"])),
        }
        results.append(r)
        print(f"  [{combo_idx+1:2d}/{len(valid_combos)}] test={test_count:2d} val={val_count:2d} train={train_count:2d}"
              f"  samples(tr/va/te)={r['mean_n_train']:.0f}/{r['mean_n_val']:.0f}/{r['mean_n_test']:.0f}"
              f"  rho={r['mean_rho']:.4f}+/-{r['std_rho']:.4f}  R2={r['mean_r2']:.4f}")

    res_df = pd.DataFrame(results).sort_values("mean_rho", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(" TOP-20 SPLITS (sorted by mean Spearman rho, all targets + seeds)")
    print("=" * 90)
    top20 = res_df.head(20).copy()
    top20.index += 1
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(top20[["test_farms","val_farms","train_farms",
                 "mean_n_train","mean_n_val","mean_n_test",
                 "mean_rho","std_rho","mean_r2","std_r2","mean_rmse"]].to_string())

    best = res_df.iloc[0]
    print()
    print("=" * 65)
    print(" RECOMMENDED SPLIT")
    print("=" * 65)
    print(f"  Test  farms : {int(best['test_farms']):3d}  (~{best['mean_n_test']:.0f} samples)")
    print(f"  Val   farms : {int(best['val_farms']):3d}  (~{best['mean_n_val']:.0f} samples)")
    print(f"  Train farms : {int(best['train_farms']):3d}  (~{best['mean_n_train']:.0f} samples)")
    print(f"  mean rho    : {best['mean_rho']:.4f}  (std={best['std_rho']:.4f})")
    print(f"  mean R2     : {best['mean_r2']:.4f}  (std={best['std_r2']:.4f})")
    print(f"  mean RMSE   : {best['mean_rmse']:.4f}")
    total = int(best['test_farms']) + int(best['val_farms']) + int(best['train_farms'])
    print(f"  Ratio  --> Train {100*int(best['train_farms'])//total}% | Val {100*int(best['val_farms'])//total}% | Test {100*int(best['test_farms'])//total}%")

    print()
    print("=" * 65)
    print(" BEST SPLIT — PER-TARGET METRICS")
    print("=" * 65)
    tc_best = int(best["test_farms"])
    vc_best = int(best["val_farms"])
    for t, (X, y, y_log, farms, u_farms) in target_data.items():
        per_seed_rho, per_seed_r2 = [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 37 + tc_best * 100 + vc_best)
            res = evaluate_split(X, y, y_log, farms, u_farms, tc_best, vc_best, rng)
            if res:
                per_seed_rho.append(res["rho"])
                per_seed_r2.append(res["r2"])
        if per_seed_rho:
            print(f"  {t:>4s}: rho={np.mean(per_seed_rho):.4f} (std={np.std(per_seed_rho):.4f})"
                  f"  R2={np.mean(per_seed_r2):.4f} (std={np.std(per_seed_r2):.4f})")

    out_path = ROOT / "ML" / "farm_split_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nFull results saved: {out_path}")

if __name__ == "__main__":
    main()
