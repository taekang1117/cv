"""
Extracts numerical representation of hand shape.
Designed for template matching (Hu moments or contour features).
"""

def extract_features(mask):
    """
    Extract shape descriptors from binary mask.

    Steps:
    1. Find contours
    2. Select largest contour
    3. Compute shape descriptor (Hu moments recommended)
    4. Return feature vector
    """

    if mask is None:
        return None

    # contours = find_contours(mask)
    # largest = select_largest(contours)
    # features = compute_hu_moments(largest)

    features = None  # placeholder

    return features
