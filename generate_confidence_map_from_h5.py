#!/usr/bin/env python3
"""
Generate and visualize ultrasound confidence map from TUS challenge dataset
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import UltrasoundConfidenceMapTransform

# H5 file path
h5_path = Path("data/datasets/000/LH_Par_C_DtP.h5")

# Load h5 file
with h5py.File(h5_path, 'r') as f:
    frames = f['frames'][:]
    tforms = f['tforms'][:]
    print(f"Loaded {h5_path}")
    print(f"Frames shape: {frames.shape}")
    print(f"Transforms shape: {tforms.shape}")

# Extract first frame
frame_idx = 20
img = frames[frame_idx]

print(f"\nFrame {frame_idx}:")
print(f"  Shape: {img.shape}")
print(f"  Dtype: {img.dtype}")
print(f"  Min-max: {img.min()}-{img.max()}")

# Convert to float32 for MONAI transforms
img_float = img.astype(np.float32) / 255.0

# Add channel dimension: [1, H, W] as required by MONAI
img_float = np.expand_dims(img_float, axis=0)
print(f"  Input shape for transform: {img_float.shape}")

# Create UltrasoundConfidenceMapTransform
confidence_transform = UltrasoundConfidenceMapTransform()

# Apply transform
confidence_map = confidence_transform(img_float)
print(f"  Confidence map shape: {confidence_map.shape}")
print(f"  Confidence map dtype: {confidence_map.dtype}")
print(f"  Confidence map min-max: {confidence_map.min():.4f}-{confidence_map.max():.4f}")

# Remove channel dimension if present
if len(confidence_map.shape) == 3:
    confidence_map = confidence_map[0]  # [H, W]

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original Image (Frame {frame_idx})', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Confidence map
im = axes[1].imshow(confidence_map, cmap='jet')
axes[1].set_title('Confidence Map', fontsize=14, fontweight='bold')
axes[1].axis('off')
cbar = plt.colorbar(im, ax=axes[1], label='Confidence')

# Overlay: Confidence map on original
img_normalized = img / 255.0
axes[2].imshow(img_normalized, cmap='gray', alpha=0.7)
axes[2].imshow(confidence_map, cmap='jet', alpha=0.3)
axes[2].set_title('Confidence Map Overlay', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
output_path = Path("confidence_map_from_h5.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Also show statistics
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(confidence_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Confidence Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Confidence Map Distribution (Frame {frame_idx})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
hist_path = Path("confidence_map_histogram_h5.png")
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
print(f"Histogram saved to: {hist_path}")

print("\nDone!")
