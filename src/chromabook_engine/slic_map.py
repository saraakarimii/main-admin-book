from __future__ import annotations
import cv2
import numpy as np
from skimage.segmentation import slic, find_boundaries


def make_slic(processed_rgb, n_segments=320, compactness=10.0):
    """Low-level superpixel over-segmentation (NOT semantic object detection)."""
    labels = slic(processed_rgb.astype(np.float32) / 255.0, n_segments=int(n_segments),
                  compactness=float(compactness), sigma=0, start_label=0, convert2lab=True,
                  enforce_connectivity=True, channel_axis=-1).astype(np.int32)
    b = find_boundaries(labels, mode="thick").astype(np.uint8)
    b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 0.8)
    if b.max() > 0:
        b /= b.max()
    return labels, b
