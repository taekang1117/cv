"""
Configuration file for gesture recognition system.
All adjustable parameters live here.
"""

# Depth threshold range (in millimeters)
DEPTH_MIN = 500
DEPTH_MAX = 1200

# Matching threshold (lower = stricter match)
MATCH_THRESHOLD = 0.5

# Toggle data sources
USE_CUSTOM_TEMPLATES = True
USE_DATASET_TEMPLATES = False

# Available gesture labels
GESTURES = [
    "swipe_left",
    "swipe_right",
    "stop"
]

# Development flags
ENABLE_KINECT = False      # Set True when hardware is connected
ENABLE_HARDWARE_OUTPUT = False  # For GPIO / media control later

SHOW_DEBUG_WINDOWS = True
