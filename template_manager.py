"""
Loads and stores gesture templates.
Supports:
- Custom recorded gestures
- Dataset-based gestures (MSR Gesture3D)
"""

def load_templates(use_custom=True, use_dataset=False):
    """
    Load templates from selected sources.

    Returns:
        list of template dictionaries:
        [
            {"label": "swipe_left", "features": [...]},
            ...
        ]
    """

    templates = []

    if use_custom:
        # load custom templates from folder
        pass

    if use_dataset:
        # load dataset-based templates
        pass

    return templates


def save_template(mask, label):
    """
    Save new gesture template.

    Steps:
    1. Extract features
    2. Store mask and features
    3. Associate with label
    """
    pass
