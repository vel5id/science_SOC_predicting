import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PATCHES_DIR = os.path.join(_PROJECT_ROOT, "data/patches")
OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/qa")
os.makedirs(OUT_DIR, exist_ok=True)

def visualize_patches(num_samples=16):
    patch_files = glob.glob(os.path.join(PATCHES_DIR, "patch_idx_*.npy"))
    if not patch_files:
        print("No patches found in", PATCHES_DIR)
        return
    
    print(f"Total patches found: {len(patch_files)}")
    
    # 1. Basic Stats Calculation on a subset
    subset = np.random.choice(patch_files, min(len(patch_files), 100), replace=False)
    zero_fractions = []
    
    for pf in subset:
        arr = np.load(pf) # Shape: [C, H, W]
        # Calculate fraction of fully 0 pixels (could be masked clouds or edge of data)
        # We check band 1 (B2) for example, or sum over optical bands
        masked = (np.sum(arr[:12, :, :], axis=0) == 0)
        zero_fractions.append(np.mean(masked))
        
    print(f"Average masked (0.0) pixel fraction: {np.mean(zero_fractions):.1%}")
    print(f"Max masked pixel fraction in sample: {np.max(zero_fractions):.1%}")
    
    # 2. Visualization Grid
    samples = np.random.choice(patch_files, min(len(patch_files), num_samples), replace=False)
    
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()
    
    # Bands indices in the saved numpy arrays (13 channels total):
    # 0: B1, 1: B2(B), 2: B3(G), 3: B4(R), ..., 7: B8(NIR), ..., 12: DEM
    # For RGB visualization, we need R=B4, G=B3, B=B2 -> indices 3, 2, 1
    
    for i, pf in enumerate(samples):
        arr = np.load(pf)
        idx = os.path.basename(pf).split('_')[2].split('.')[0]
        
        # Extract RGB bands [B4, B3, B2]
        r = arr[3, :, :]
        g = arr[2, :, :]
        b = arr[1, :, :]
        
        # Stack to [H, W, 3]
        rgb = np.stack([r, g, b], axis=-1)
        
        # S2 reflectance is 0-1. Usually land is 0-0.3. 
        # We enhance brightness for visualization (x3)
        rgb_vis = np.clip(rgb * 3.5, 0, 1)
        
        # Calculate how much of THIS specific patch is masked (zeros)
        empty_pct = np.mean(np.sum(rgb, axis=-1) == 0) * 100
        
        axes[i].imshow(rgb_vis)
        axes[i].set_title(f"Idx: {idx} | Empty: {empty_pct:.0f}%", fontsize=10)
        axes[i].axis('off')
        
    for j in range(len(samples), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "patches_rgb_sample.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Saved RGB visual check to: {out_path}")

    # 3. NDVI/DEM Visualization grid for the SAME samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()
    
    for i, pf in enumerate(samples):
        arr = np.load(pf)
        idx = os.path.basename(pf).split('_')[2].split('.')[0]
        
        nir = arr[7, :, :] # B8
        red = arr[3, :, :] # B4
        dem = arr[12, :, :] # DEM
        
        # Calculate NDVI
        ndvi = np.zeros_like(nir)
        valid = (nir + red) > 0
        ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
        
        # We can show NDVI
        im = axes[i].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.9)
        axes[i].set_title(f"Idx: {idx} | Mean Elev: {np.mean(dem):.0f}m", fontsize=10)
        axes[i].axis('off')
        
    plt.tight_layout()
    out_path_ndvi = os.path.join(OUT_DIR, "patches_ndvi_sample.png")
    plt.savefig(out_path_ndvi, dpi=200)
    plt.close()
    print(f"✅ Saved NDVI visual check to: {out_path_ndvi}")

if __name__ == '__main__':
    visualize_patches()
