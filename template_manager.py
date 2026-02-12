"""
Loads and stores gesture templates.
Supports:
- Custom recorded gestures
- Dataset-based gestures (MSR Gesture3D)
"""
import os
import numpy as np
import feature_extraction

TEMPLATE_FOLDER = "templates"


def save_template(mask, label):

    features = feature_extraction.extract_features(mask)

    if features is None:
        print("Could not extract features.")
        return

    if not os.path.exists(TEMPLATE_FOLDER):
        os.makedirs(TEMPLATE_FOLDER)

    filename = os.path.join(TEMPLATE_FOLDER, f"{label}.npy")

    np.save(filename, features)

    print(f"Saved template for {label}")

