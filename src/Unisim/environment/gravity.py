"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Gravity Models
--------------

"""
from abc import abstractmethod

import numpy as np

from Unisim.utils.constants import GRAVITY, STD_GRAVITATIONAL_PARAMETER
from Unisim.utils.constants import EARTH_MEAN_RADIUS
from Unisim.utils.coordinates import hor2body


class Gravity(object):
    """Generic gravity model"""

    def __init__(self):
        self._magnitude = None
        self._versor = np.zeros([3])  # Body axis
        self._vector = np.zeros([3])  # Body axis

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def versor(self):
        return self._versor

    @property
    def vector(self):
        return self._vector

    @abstractmethod
    def update(self, system):
        pass

class NoGravity(Gravity):
    """No Gravity.
    """

    def __init__(self):
        self._magnitude = 0
        self._versor = np.zeros(3)
        self._vector = np.zeros(3)

    def update(self, height):
        pass


class VerticalConstant(Gravity):
    """Vertical constant gravity model.
    """

    def __init__(self):
        self._magnitude = GRAVITY
        self._versor = np.array([0, 0, 1], dtype=float)
        self._vector = self.magnitude * self.versor

    def update(self):
        pass


class VerticalNewton(Gravity):
    """Vertical gravity model with magnitude varying according to Newton's
    universal law of gravitation.
    """

    def __init__(self):
        self._versor = np.array([0, 0, -1], dtype=float)
        self._vector = np.zeros([3])

    def update(self, _height):
        try:
            height = float(_height)
        except:
            raise Exception("Height Value must be an integer")

        r_squared = (height+EARTH_MEAN_RADIUS)**2
        self._magnitude = STD_GRAVITATIONAL_PARAMETER / r_squared
        self._vector = self.magnitude * self._versor

class SimpleNewton(Gravity):
    """SImple gravity model with magnitude varying according to Newton's
    universal law of gravitation.
    """

    def __init__(self):
        self._versor = np.zeros([3])
        self._vector = np.zeros([3])

    def update(self, _pos):
        r = -np.linalg.norm(_pos)
        self._versor = _pos/r

        r_squared = r**2
        self._magnitude = STD_GRAVITATIONAL_PARAMETER / r_squared
        self._vector = self.magnitude * self._versor

class LatitudeModel(Gravity):
    # TODO: https://en.wikipedia.org/wiki/Gravity_of_Earth#Latitude_model

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def update(self, system):
        raise NotImplementedError
