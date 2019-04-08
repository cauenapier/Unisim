from abc import abstractmethod


class Planet(object):
    """ Base class for planetary bodies
        It will reference to an environment objects
        Do the coordinate system conversions

    """
    def __ini__(self, mass, radius, grav_param, name):
        self._mass = mass
        self._radius = radius
        self._grav_param = grav_param
        self._name = name

    def cartesian(self, lat, long, alt):
        """
        """
        raise NotImplementedError

    def polar(self, x, y, z):
        """
        """
        raise NotImplementedError


class Earth_Round(Planet):
    """
    """
    def __init__(self):
        Planet.__init__(self, 5.9722e24, 6371000, 3.986004418e14, "Earth")
