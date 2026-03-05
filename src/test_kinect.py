"""
Simple Kinect Depth Test Script

Purpose:
- Initialize Kinect
- Continuously grab depth frames
- Normalize depth values for display
- Show live depth window
- Press 'q' to exit

This script ONLY tests depth capture.
No gesture processing is done here.
"""

import cv2
import time
from kinect_camera import initialize_kinect, get_depth_frame


# --------------------------------------------------
# Initialize Kinect device
# --------------------------------------------------
initialize_kinect()

print("Starting depth stream...")
print("Press 'q' to quit.")


while True:

    # --------------------------------------------------
    # Acquire one depth frame from Kinect
    # --------------------------------------------------
    depth = get_depth_frame()

    # If Kinect is not connected or frame fails
    if depth is None:
        print("No depth frame received.")
        time.sleep(0.5)  # Prevent CPU from maxing out
        continue

    # --------------------------------------------------
    # Normalize depth values for visualization
    #
    # Raw Kinect depth is typically 11–16 bit values.
    # We scale them to 0–255 so OpenCV can display them.
    # --------------------------------------------------
    depth_display = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_display.astype("uint8")

    # --------------------------------------------------
    # Show depth image
    # --------------------------------------------------
    cv2.imshow("Kinect Depth View", depth_display)

    # --------------------------------------------------
    # Exit condition (press 'q')
    # --------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cv2.destroyAllWindows()
print("Depth test ended.")
