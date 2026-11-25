#!/usr/bin/env python3
"""
Generate and visualize ultrasound confidence map using MONAI transforms
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import UltrasoundConfidenceMapTransform

# Image path
image_path = Path("/home/syahid/github/DualTrack/data/data_custom/screen_captures/screen_000093.png")

# Load image
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"Failed to load image: {image_path}")

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image min-max: {img.min()}-{img.max()}")

# Convert to float32 for MONAI transforms (MONAI expects float32)
img_float = img.astype(np.float32) / 255.0

# Add channel dimension: [1, H, W] as required by MONAI
img_float = np.expand_dims(img_float, axis=0)
print(f"Input shape for transform: {img_float.shape}")

# Create UltrasoundConfidenceMapTransform
confidence_transform = UltrasoundConfidenceMapTransform()

# Apply transform
confidence_map = confidence_transform(img_float)
print(f"Confidence map shape after transform: {confidence_map.shape}")
print(f"Confidence map dtype: {confidence_map.dtype}")
print(f"Confidence map min-max: {confidence_map.min():.4f}-{confidence_map.max():.4f}")

# Remove channel dimension for visualization if present
if len(confidence_map.shape) == 3:
    confidence_map = confidence_map[0]  # [H, W]
    print(f"Confidence map shape after removing channel: {confidence_map.shape}")
elif len(confidence_map.shape) != 2:
    print(f"Warning: unexpected shape {confidence_map.shape}")
    confidence_map = confidence_map.squeeze()

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image (normalized)
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Confidence map
im = axes[1].imshow(confidence_map, cmap='jet')
axes[1].set_title('Confidence Map', fontsize=14, fontweight='bold')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], label='Confidence')

# Overlay: Confidence map on original
img_normalized = img / 255.0
axes[2].imshow(img_normalized, cmap='gray', alpha=0.7)
axes[2].imshow(confidence_map, cmap='jet', alpha=0.3)
axes[2].set_title('Confidence Map Overlay', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
output_path = Path("confidence_map_visualization.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

plt.show()
print("Done!")
