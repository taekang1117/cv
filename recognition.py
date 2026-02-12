"""
Compares live gesture features against stored templates.
"""

def match_gesture(features, templates, threshold):
    """
    Compare extracted features to each template.

    Steps:
    1. Compute distance between feature vectors
    2. Select best match
    3. If distance < threshold → return label
    4. Otherwise return "unknown"
    """

    if features is None:
        return "unknown"

    best_label = "unknown"
    best_score = None

    for template in templates:
        template_features = template["features"]

        # score = compute_distance(features, template_features)

        # update best match

    # compare best_score to threshold

    return best_label
