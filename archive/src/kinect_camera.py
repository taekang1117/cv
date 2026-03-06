"""
Handles communication with Kinect depth sensor.
Currently pseudocode to allow hardware flexibility.
"""

def initialize_kinect():
    """
    Connect to Kinect using libfreenect.
    Start depth stream.
    """
    # connect to device
    # start depth stream
    print("Kinect initialization placeholder.")


def get_depth_frame():
    """
    Retrieve one depth frame from Kinect.

    Returns:
        depth_frame (2D array) OR None
    """

    # depth_frame = read depth data from device

    depth_frame = None  # placeholder for development

    return depth_frame
