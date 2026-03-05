"""
Depth thresholding and noise removal.
Converts raw depth frame into clean binary hand mask.
"""

import config

def isolate_hand(depth_frame):
    """
    Apply depth threshold and filtering to isolate hand.

    Steps:
    1. Threshold depth between DEPTH_MIN and DEPTH_MAX
    2. Convert to binary mask
    3. Apply noise filtering (median blur / morphology)
    4. Return cleaned mask
    """

    if depth_frame is None:
        return None

    # mask = threshold(depth_frame)

    # mask = apply_noise_removal(mask)

    mask = None  # placeholder

    return mask
