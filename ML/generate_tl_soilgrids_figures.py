"""
generate_tl_soilgrids_figures.py
---------------------------------
Generates additional figures for article2_prediction:

  fig14_tl_comparison.png        – ResNet-18 Scratch vs TL (ImageNet): grouped bars
  fig15_soilgrids_baseline.png   – SoilGrids v2.0 vs local models (bar + scatter)
  fig16_ml_vs_dl_farm.png        – One-farm prediction: Best ML vs DL (scratch) vs DL (TL)

Usage:
  python ML/generate_tl_soilgrids_figures.py
"""
import json, copy, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_CSV    = ROOT / "data" / "features" / "master_dataset.csv"
PATCHES_DIR = ROOT / "data" / "patches_18ch"
TL_JSON     = ROOT / "ML" / "results" / "transfer_learning" / "tl_summary.json"
SG_JSON     = ROOT / "ML" / "results" / "soilgrids" / "soilgrids_metrics.json"
SG_CSV      = ROOT / "ML" / "results" / "soilgrids" / "soilgrids_predictions.csv"
RF_OOF_PH   = ROOT / "ML" / "results" / "rf" / "ph_oof_predictions.csv"
RESNET_OOF  = ROOT / "ML" / "results" / "resnet" / "ph_resnet_oof.csv"
FIG_DIR     = ROOT / "articles" / "article2_prediction" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

TARGETS_NICE = {
    "ph": "pH", "soc": "SOC", "no3": "NO₃",
    "p": "P₂O₅", "k": "K₂O", "s": "S"
}


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 14: Transfer Learning comparison (grouped bars)
# ═══════════════════════════════════════════════════════════════════════════════
def fig14_tl_comparison():
    print("[Fig 14] Transfer Learning comparison...")
    with open(TL_JSON) as f:
        tl = json.load(f)

    targets = ["ph", "soc", "no3", "p", "k", "s"]
    strategies = ["field_lofo", "farm_lofo"]
    modes = ["scratch", "tl_imagenet"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax_i, strategy in enumerate(strategies):
        ax = axes[ax_i]
        x = np.arange(len(targets))
        width = 0.35

        scratch_vals = [tl[f"{t}_{strategy}_scratch"]["rho"] for t in targets]
        tl_vals      = [tl[f"{t}_{strategy}_tl_imagenet"]["rho"] for t in targets]

        bars1 = ax.bar(x - width/2, scratch_vals, width, label="From Scratch",
                       color="#4C72B0", edgecolor="white", linewidth=0.5, zorder=3)
        bars2 = ax.bar(x + width/2, tl_vals, width, label="TL (ImageNet)",
                       color="#DD8452", edgecolor="white", linewidth=0.5, zorder=3)

        # Annotate values
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                y_pos = max(h, 0) + 0.01
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([TARGETS_NICE[t] for t in targets])
        ax.set_ylabel("Spearman ρ")
        strategy_name = "Field-LOFO (81 folds)" if strategy == "field_lofo" else "Farm-LOFO (20 farms)"
        ax.set_title(f"ResNet-18: {strategy_name}")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(-0.15, 0.85)

    fig.suptitle("ResNet-18: From Scratch vs ImageNet Transfer Learning", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig14_tl_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 15: SoilGrids baseline (bar chart + scatter)
# ═══════════════════════════════════════════════════════════════════════════════
def fig15_soilgrids_baseline():
    print("[Fig 15] SoilGrids baseline...")
    with open(SG_JSON) as f:
        sg = json.load(f)

    sg_pred = pd.read_csv(SG_CSV)

    # Local best Farm-LOFO metrics (from article)
    local_best = {
        "ph":  {"rho": 0.750, "r2": 0.616, "rmse": 0.406, "model": "RF"},
        "soc": {"rho": 0.554, "r2": 0.257, "rmse": 0.472, "model": "CatBoost"},
    }

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Top row: bar charts ρ and R² ──────────────────────────────────────────
    props = ["ph", "soc"]
    prop_nice = {"ph": "pH", "soc": "SOC"}

    # Bar: Spearman ρ
    ax_rho = fig.add_subplot(gs[0, 0])
    x = np.arange(len(props))
    w = 0.3
    sg_rho  = [sg[p]["rho"] for p in props]
    loc_rho = [local_best[p]["rho"] for p in props]
    b1 = ax_rho.bar(x - w/2, sg_rho, w, label="SoilGrids v2.0", color="#C44E52", edgecolor="white", zorder=3)
    b2 = ax_rho.bar(x + w/2, loc_rho, w, label="Best local (Farm-LOFO)", color="#55A868", edgecolor="white", zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            ax_rho.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax_rho.set_xticks(x)
    ax_rho.set_xticklabels([prop_nice[p] for p in props])
    ax_rho.set_ylabel("Spearman ρ")
    ax_rho.set_title("(a) Spearman ρ")
    ax_rho.legend(fontsize=9)
    ax_rho.set_ylim(0, 0.95)
    ax_rho.grid(axis="y", alpha=0.3, zorder=0)

    # Bar: R²
    ax_r2 = fig.add_subplot(gs[0, 1])
    sg_r2  = [sg[p]["r2"] if sg[p]["r2"] > -5 else 0 for p in props]  # clip extreme
    loc_r2 = [local_best[p]["r2"] for p in props]
    b1 = ax_r2.bar(x - w/2, sg_r2, w, label="SoilGrids v2.0", color="#C44E52", edgecolor="white", zorder=3)
    b2 = ax_r2.bar(x + w/2, loc_r2, w, label="Best local (Farm-LOFO)", color="#55A868", edgecolor="white", zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            ax_r2.text(bar.get_x() + bar.get_width()/2., max(bar.get_height(), 0) + 0.015,
                       f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels([prop_nice[p] for p in props])
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("(b) R²")
    ax_r2.legend(fontsize=9)
    ax_r2.grid(axis="y", alpha=0.3, zorder=0)

    # Bar: RMSE
    ax_rmse = fig.add_subplot(gs[0, 2])
    sg_rmse  = [sg[p]["rmse"] for p in props]
    loc_rmse = [local_best[p]["rmse"] for p in props]
    # For SOC, SoilGrids RMSE is in g/kg (~69) vs local in % (~0.47), show only pH
    b1 = ax_rmse.bar(x - w/2, [sg_rmse[0], 0], w, label="SoilGrids v2.0", color="#C44E52", edgecolor="white", zorder=3)
    b2 = ax_rmse.bar(x + w/2, [loc_rmse[0], 0], w, label="Best local (Farm-LOFO)", color="#55A868", edgecolor="white", zorder=3)
    ax_rmse.text(b1[0].get_x() + b1[0].get_width()/2., b1[0].get_height() + 0.01,
                 f"{sg_rmse[0]:.3f}", ha="center", va="bottom", fontsize=9)
    ax_rmse.text(b2[0].get_x() + b2[0].get_width()/2., b2[0].get_height() + 0.01,
                 f"{loc_rmse[0]:.3f}", ha="center", va="bottom", fontsize=9)
    ax_rmse.text(x[1], 0.02, "N/A\n(unit mismatch)", ha="center", fontsize=8, color="grey")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([prop_nice[p] for p in props])
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("(c) RMSE (pH only; SOC units differ)")
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(axis="y", alpha=0.3, zorder=0)

    # ── Bottom row: scatter plots ─────────────────────────────────────────────
    for i, prop in enumerate(props):
        ax = fig.add_subplot(gs[1, i])
        gt_col = f"gt_{prop}"
        sg_col = f"sg_{prop}"
        sub = sg_pred[[gt_col, sg_col]].dropna()

        gt = sub[gt_col].values
        sg_v = sub[sg_col].values

        ax.scatter(gt, sg_v, s=8, alpha=0.4, c="#C44E52", edgecolors="none", zorder=3)

        mn, mx = min(gt.min(), sg_v.min()), max(gt.max(), sg_v.max())
        pad = (mx - mn) * 0.05
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=1, alpha=0.5, label="1:1 line")
        ax.set_xlabel(f"Observed {prop_nice[prop]}")
        ax.set_ylabel(f"SoilGrids predicted {prop_nice[prop]}")
        rho_val = sg[prop]["rho"]
        r2_val  = sg[prop]["r2"]
        title_r2 = f"R²={r2_val:.3f}" if abs(r2_val) < 100 else f"R²=−15613"
        ax.set_title(f"(d{i+1}) SoilGrids vs Observed: {prop_nice[prop]}\nρ={rho_val:.3f}, {title_r2}, N={len(sub)}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, zorder=0)

    # empty right subplot
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")
    info_text = (
        "SoilGrids v2.0 baseline:\n\n"
        "• pH:  ρ=0.208, R²=0.034\n"
        "  Local RF: ρ=0.750, R²=0.616\n"
        "  Improvement: 3.6× in ρ\n\n"
        "• SOC: ρ=0.042 (≈ random)\n"
        "  Local CatBoost: ρ=0.554\n"
        "  R² = −15613 (unit mismatch:\n"
        "  SoilGrids: g/kg, local: %)\n\n"
        "Conclusion: global 250m maps\n"
        "are inadequate for field-level\n"
        "agrochemical mapping."
    )
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                 verticalalignment="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                 family="monospace")

    out = FIG_DIR / "fig15_soilgrids_baseline.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 16: ML vs DL (Scratch) vs DL (TL) on one farm — pH
# ═══════════════════════════════════════════════════════════════════════════════

# ── ResNet building (same as transfer_learning_resnet.py) ─────────────────────
IN_CHANNELS = 18
BATCH_SIZE  = 32
LR          = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS      = 60
PATIENCE    = 10


def build_resnet(pretrained, in_channels=IN_CHANNELS):
    if pretrained:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        old_w = base.conv1.weight.data
        mean_w = old_w.mean(dim=1, keepdim=True)
        new_w = mean_w.repeat(1, in_channels, 1, 1)
        new_w = new_w / (in_channels / 3.0)
    else:
        base = models.resnet18(weights=None)
        new_w = None

    base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if new_w is not None:
        base.conv1.weight.data = new_w

    num_ftrs = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, 1)
    )
    return base


class PatchDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            if torch.rand(1) < 0.5: x = TF.hflip(x)
            if torch.rand(1) < 0.5: x = TF.vflip(x)
            k = np.random.randint(0, 4)
            if k: x = torch.rot90(x, k, dims=[1, 2])
        return x, y


def train_one_fold_resnet(X_all, y_all, train_idx, val_idx, test_idx, pretrained):
    """Return predictions for test_idx."""
    fold_mean = X_all[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    fold_std  = X_all[train_idx].std(dim=(0, 2, 3), keepdim=True)
    fold_std[fold_std == 0] = 1.0
    X_norm = (X_all - fold_mean) / fold_std
    y_t = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)

    tr_dl  = DataLoader(PatchDataset(X_norm[train_idx], y_t[train_idx], True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(PatchDataset(X_norm[val_idx], y_t[val_idx], False),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_dl  = DataLoader(PatchDataset(X_norm[test_idx], y_t[test_idx], False),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_resnet(pretrained=pretrained).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val, best_state, pat_cnt = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for bX, by in tr_dl:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward(); optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, by in val_dl:
                bX, by = bX.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= max(len(val_dl.dataset), 1)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; best_state = copy.deepcopy(model.state_dict()); pat_cnt = 0
        else:
            pat_cnt += 1
        if pat_cnt >= PATIENCE:
            break
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for bX, _ in te_dl:
            preds.extend(model(bX.to(DEVICE)).cpu().numpy().flatten())
    return np.array(preds)


def load_patches(df, target):
    patches, ys, farms, fields = [], [], [], []
    for df_idx, row in df.iterrows():
        if pd.isna(row[target]):
            continue
        pf = PATCHES_DIR / f"patch_idx_{df_idx}.npy"
        if not pf.exists():
            continue
        try:
            arr = np.load(pf).astype(np.float32)
            patches.append(torch.from_numpy(arr))
            ys.append(float(row[target]))
            farms.append(str(row["farm"]))
            fields.append(str(row["field_name"]))
        except Exception:
            pass
    X = torch.stack(patches)
    return X, np.array(ys, dtype=np.float32), np.array(farms), np.array(fields)


def fig16_ml_vs_dl_farm():
    """
    For one test farm (Агро Парасат - largest, 151 points):
    Train on remaining 19 farms, predict the test farm.
    Compare RF OOF, ResNet scratch, ResNet TL.
    """
    print("[Fig 16] ML vs DL vs TL on one farm (pH)...")
    target = "ph"
    TEST_FARM = "Агро Парасат"

    # ── 1) RF OOF predictions (already computed) ──────────────────────────────
    rf_df = pd.read_csv(RF_OOF_PH, low_memory=False)
    rf_test = rf_df[rf_df["farm"] == TEST_FARM].copy()
    rf_y_true = rf_test[target].values
    rf_y_pred = rf_test["oof_pred"].values
    rf_lon    = rf_test["centroid_lon"].values
    rf_lat    = rf_test["centroid_lat"].values
    print(f"  RF: {len(rf_test)} test points on '{TEST_FARM}'")

    # ── 2) ResNet: load patches, train scratch & TL ───────────────────────────
    print("  Loading patches...")
    df_master = pd.read_csv(DATA_CSV, low_memory=False)
    X, y, farms, fields = load_patches(df_master, target)

    test_idx  = np.where(farms == TEST_FARM)[0]
    other_farms = np.unique(farms[farms != TEST_FARM])
    # val_farm: second-largest
    val_farm  = "КХ Алмаз"
    val_idx   = np.where(farms == val_farm)[0]
    train_idx = np.where((farms != TEST_FARM) & (farms != val_farm))[0]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Scratch
    print("  Training ResNet-18 (scratch)...")
    preds_scratch = train_one_fold_resnet(X, y, train_idx, val_idx, test_idx, pretrained=False)

    # TL
    print("  Training ResNet-18 (TL ImageNet)...")
    preds_tl = train_one_fold_resnet(X, y, train_idx, val_idx, test_idx, pretrained=True)

    y_test = y[test_idx]

    # ── Metrics ───────────────────────────────────────────────────────────────
    def calc_metrics(yt, yp):
        rho, _ = spearmanr(yt, yp)
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        return rho, r2, rmse

    m_rf      = calc_metrics(rf_y_true, rf_y_pred)
    m_scratch = calc_metrics(y_test, preds_scratch)
    m_tl      = calc_metrics(y_test, preds_tl)

    print(f"  RF:      ρ={m_rf[0]:.3f}, R²={m_rf[1]:.3f}, RMSE={m_rf[2]:.3f}")
    print(f"  Scratch: ρ={m_scratch[0]:.3f}, R²={m_scratch[1]:.3f}, RMSE={m_scratch[2]:.3f}")
    print(f"  TL:      ρ={m_tl[0]:.3f}, R²={m_tl[1]:.3f}, RMSE={m_tl[2]:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    configs = [
        ("RF (tabular, 15 features)", rf_y_true, rf_y_pred, m_rf, "#55A868"),
        ("ResNet-18 (scratch, 18ch)", y_test, preds_scratch, m_scratch, "#4C72B0"),
        ("ResNet-18 (TL ImageNet)",   y_test, preds_tl,      m_tl,      "#DD8452"),
    ]

    for ax, (title, yt, yp, (rho, r2, rmse), color) in zip(axes, configs):
        ax.scatter(yt, yp, s=20, alpha=0.5, c=color, edgecolors="none", zorder=3)

        mn = min(yt.min(), yp.min()) - 0.2
        mx = max(yt.max(), yp.max()) + 0.2
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.5)

        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed pH")
        ax.set_ylabel("Predicted pH")
        ax.set_title(title, fontsize=11)

        metrics_text = f"ρ = {rho:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}\nN = {len(yt)}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
        ax.grid(alpha=0.3, zorder=0)

    fig.suptitle(f"pH Prediction on Test Farm \"{TEST_FARM}\" (Farm-LOFO)\n"
                 "Best ML vs DL (Scratch) vs DL (Transfer Learning)",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    out = FIG_DIR / "fig16_ml_vs_dl_farm.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig14_tl_comparison()
    fig15_soilgrids_baseline()
    fig16_ml_vs_dl_farm()
    print("\n[DONE] All figures generated.")
