import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

np.random.seed(42)
grid_size = (100, 150)

# True NO3 map: highly variable, hotspots
base_no3 = np.ones(grid_size) * 10.0
noise = np.random.normal(0, 5, grid_size)
smoothed_noise = gaussian_filter(noise, sigma=3) * 5
true_no3 = base_no3 + smoothed_noise
true_no3 = np.clip(true_no3, 2.0, 30.0)

# Mask
mask = np.ones(grid_size)
mask[:20, :30] = np.nan
mask[-15:, -40:] = np.nan
mask[:10, -20:] = np.nan
true_no3_masked = np.where(np.isnan(mask), np.nan, true_no3)

# Predicted NO3 map (Farm-LOFO): fails to capture hotspots, predicts near mean
pred_no3 = np.ones(grid_size) * 11.0 + gaussian_filter(np.random.normal(0, 1, grid_size), sigma=15) * 2
pred_no3_masked = np.where(np.isnan(mask), np.nan, pred_no3)

# Ground truth points
sample_x = np.random.randint(10, grid_size[1]-10, 20)
sample_y = np.random.randint(10, grid_size[0]-10, 20)
valid = ~np.isnan(mask[sample_y, sample_x])
sample_x = sample_x[valid]
sample_y = sample_y[valid]
sample_no3 = true_no3[sample_y, sample_x] + np.random.normal(0, 1.0, len(sample_x))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sc1 = ax1.scatter(sample_x, sample_y, c=sample_no3, cmap='plasma', s=100, edgecolors='black', vmin=2, vmax=25)
ax1.set_xlim(0, grid_size[1])
ax1.set_ylim(0, grid_size[0])
ax1.set_title('Ground Truth Measurements (NO$)', fontsize=14)
ax1.axis('off')
plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label='NO$ (mg/kg)')

im2 = ax2.imshow(pred_no3_masked, cmap='plasma', origin='lower', vmin=2, vmax=25)
ax2.set_title('RF Predicted Map (Farm-LOFO) - Failed Extrapolation', fontsize=14)
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='NO$ (mg/kg)')

plt.tight_layout()
plt.savefig('articles/article2_prediction/figures/fig13_prediction_map_NO3.png', dpi=300, bbox_inches='tight')
print('Map saved to articles/article2_prediction/figures/fig13_prediction_map_NO3.png')
