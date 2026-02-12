"""
Main program loop for real-time gesture recognition.
"""

import config
import kinect_camera
import preprocessing
import feature_extraction
import template_manager
import recognition
import actions
import hardware_interface


def main():

    # Initialize systems
    if config.ENABLE_KINECT:
        kinect_camera.initialize_kinect()

    hardware_interface.initialize_hardware()

    templates = template_manager.load_templates(
        use_custom=config.USE_CUSTOM_TEMPLATES,
        use_dataset=config.USE_DATASET_TEMPLATES
    )

    print("System initialized.")

    while True:

        # Acquire depth frame
        if config.ENABLE_KINECT:
            depth_frame = kinect_camera.get_depth_frame()
        else:
            depth_frame = None  # development mode

        # Preprocess
        mask = preprocessing.isolate_hand(depth_frame)

        # Feature extraction
        features = feature_extraction.extract_features(mask)

        # Recognition
        gesture = recognition.match_gesture(
            features,
            templates,
            config.MATCH_THRESHOLD
        )

        # Execute action
        if config.ENABLE_HARDWARE_OUTPUT:
            actions.execute_action(gesture)


if __name__ == "__main__":
    main()
