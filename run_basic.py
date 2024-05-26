# Main function for the o2c project

from basic_utilities import *
from constants import *

import time
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    map = create_map()
    # Force stop command when the user presses 'q', and run the function stop_car
    keyboard.on_press(stop_car)