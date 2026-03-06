"""
Feature Extraction Module

Extracts a numerical representation of the detected hand region
using multiple classical computer vision techniques.

Features included:
- Centroid (x, y)
- Contour area
- Perimeter
- Bounding box aspect ratio
- Extent (area / bounding box area)
- Solidity (area / convex hull area)
- Edge density (Canny edges inside mask)
"""

import cv2
import numpy as np


def extract_features(mask):
    """
    Extract shape descriptors from binary mask.

    Steps:
    1. Find contours
    2. Select largest contour (assumed hand)
    3. Compute geometric and edge-based features
    4. Return feature vector (numpy array)
    """

    if mask is None:
        return None

    # --------------------------------------------------
    # 1. Find contours
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # --------------------------------------------------
    # 2. Select largest contour (hand assumption)
    # --------------------------------------------------
    largest = max(contours, key=cv2.contourArea)

    # --------------------------------------------------
    # 3. Basic geometric properties
    # --------------------------------------------------
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    # --------------------------------------------------
    # 4. Centroid calculation using image moments
    # --------------------------------------------------
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # --------------------------------------------------
    # 5. Bounding box features
    # --------------------------------------------------
    x, y, w, h = cv2.boundingRect(largest)

    aspect_ratio = w / float(h) if h != 0 else 0
    bounding_area = w * h
    extent = area / float(bounding_area) if bounding_area != 0 else 0

    # --------------------------------------------------
    # 6. Convex hull + solidity
    # --------------------------------------------------
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)

    solidity = area / float(hull_area) if hull_area != 0 else 0

    # --------------------------------------------------
    # 7. Edge-based feature (Canny edge density)
    # --------------------------------------------------
    edges = cv2.Canny(mask, 100, 200)
    edge_pixels = np.sum(edges > 0)
    mask_pixels = np.sum(mask > 0)

    edge_density = edge_pixels / float(mask_pixels) if mask_pixels != 0 else 0

    # --------------------------------------------------
    # 8. Create final feature vector
    # --------------------------------------------------
    features = np.array([
        cx,
        cy,
        area,
        perimeter,
        aspect_ratio,
        extent,
        solidity,
        edge_density
    ], dtype=np.float32)

    return features
