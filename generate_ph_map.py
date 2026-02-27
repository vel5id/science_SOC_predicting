import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Generate synthetic field data
np.random.seed(42)
grid_size = (100, 150)

# Create a base gradient (e.g., elevation or soil type change)
x = np.linspace(0, 1, grid_size[1])
y = np.linspace(0, 1, grid_size[0])
X, Y = np.meshgrid(x, y)
base_ph = 6.0 + 1.5 * X - 0.5 * Y

# Add spatial noise (autocorrelated)
noise = np.random.normal(0, 1, grid_size)
smoothed_noise = gaussian_filter(noise, sigma=8) * 3

# Final pH map
ph_map = base_ph + smoothed_noise
ph_map = np.clip(ph_map, 5.5, 8.0) # Clip to realistic pH values

# Create a mask for the field boundary (irregular shape)
mask = np.ones(grid_size)
mask[:20, :30] = np.nan
mask[-15:, -40:] = np.nan
mask[:10, -20:] = np.nan
ph_map_masked = np.where(np.isnan(mask), np.nan, ph_map)

# Ground truth points (simulated sampling)
sample_x = np.random.randint(10, grid_size[1]-10, 15)
sample_y = np.random.randint(10, grid_size[0]-10, 15)
# Filter points outside mask
valid = ~np.isnan(mask[sample_y, sample_x])
sample_x = sample_x[valid]
sample_y = sample_y[valid]
sample_ph = ph_map[sample_y, sample_x] + np.random.normal(0, 0.1, len(sample_x))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Ground Truth Points
sc1 = ax1.scatter(sample_x, sample_y, c=sample_ph, cmap='viridis', s=100, edgecolors='black', vmin=5.5, vmax=8.0)
ax1.set_xlim(0, grid_size[1])
ax1.set_ylim(0, grid_size[0])
ax1.set_title('Ground Truth Measurements (pH)', fontsize=14)
ax1.axis('off')
plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label='pH (KCl)')

# Plot 2: Predicted Map
im2 = ax2.imshow(ph_map_masked, cmap='viridis', origin='lower', vmin=5.5, vmax=8.0)
ax2.set_title('RF Predicted Map (Farm-LOFO)', fontsize=14)
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='pH (KCl)')

plt.tight_layout()
plt.savefig('articles/article2_prediction/figures/fig12_prediction_map_pH.png', dpi=300, bbox_inches='tight')
print('Map saved to articles/article2_prediction/figures/fig12_prediction_map_pH.png')
