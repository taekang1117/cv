
"""
Recognition Module

Compares extracted feature vectors to stored templates
using Euclidean distance.
"""

import numpy as np


def match_gesture(features, templates, threshold):
    """
    Compare extracted features to each template.

    Returns:
        best matching label or "unknown"
    """

    if features is None:
        return "unknown"

    best_label = "unknown"
    best_score = float("inf")

    for template in templates:
        template_features = template["features"]

        if template_features is None:
            continue

        # Euclidean distance
        score = np.linalg.norm(features - template_features)

        if score < best_score:
            best_score = score
            best_label = template["label"]

    if best_score < threshold:
        return best_label
    else:
        return "unknown"
