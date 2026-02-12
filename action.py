"""
High-level gesture-to-action mapping.
Separate from hardware implementation.
"""

import hardware_interface

def execute_action(gesture):
    """
    Interprets gesture label and triggers output layer.
    """

    if gesture == "unknown":
        return

    print("Gesture detected:", gesture)

    hardware_interface.execute_output(gesture)
