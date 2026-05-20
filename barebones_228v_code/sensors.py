import numpy as np
import config as cfg

class OnlineSensor:
    # the class we'll be using to design our front facing sensors
    
    def __init__(self, view_depth = 4, view_angle = np.pi/2, false_negative_rate = 0.05, false_positive_rate = 0.01, rng=None):
        # initializing our sensor
        # the view depth and view angle are fairly self-explanatory, showing how far and wide the drone's visual field is
        