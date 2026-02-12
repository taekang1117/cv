"""
Abstract hardware output layer.
Can later support:
- GPIO LEDs
- Media control
- Menu navigation
- Serial communication
"""

def initialize_hardware():
    """
    Initialize hardware output system.
    Leave empty during development.
    """
    pass


def execute_output(gesture):
    """
    Map gesture to physical/system action.

    Examples:
    - Turn LED on/off
    - Play/pause media
    - Navigate menu
    """

    # if gesture == "swipe_left":
    #     perform action

    pass


def cleanup():
    """
    Cleanup hardware resources.
    """
    pass
