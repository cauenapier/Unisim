"""
Wind Models
-----------

"""
import numpy as np


class NoWind(object):

    def __init__(self):
        # Wind velocity: FROM North to South, FROM East to West,
        # Wind velocity in the UPSIDE direction
        self.horizon = np.zeros([3], dtype=float)
        self.body = np.zeros([3], dtype=float)

    def update(self, state):
        pass
