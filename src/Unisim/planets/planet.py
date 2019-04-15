from abc import abstractmethod
import numpy as np

from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import SimpleNewton
from Unisim.environment.environment import Environment


class Planet(object):
    """ Base class for planetary bodies
        It will reference to an environment objects
        Do the coordinate system conversions
    """
    def __init__(self, mass, radius, grav_param, name):
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

    def __init__(self, gravity="SimpleNewton", atmos="ISA1976", wind = "NoWind"):
        self.earth_mass = 5.9722e24
        self.earth_radius = 6371000
        self.earth_grav_param = 3.986004418e14

        Planet.__init__(self, self.earth_mass,self.earth_radius, self.earth_grav_param, "Earth")

        if gravity == "SimpleNewton":
            gravity = SimpleNewton()
        if atmos == "ISA1976":
            atmo = ISA1976()
        if wind == "NoWind":
            wind = NoWind()

        self._environment = Environment(atmo, gravity, wind)

    def polar(self, x, y, z):
        """
        Take Position in Earth Centered Frame.
        Returns Latitude, Longitude and Altitude. (Degrees and meters)
        """
        lon = np.rad2deg(np.arctan2(y, x))
        xy = np.sqrt(x**2+y**2)
        lat = np.rad2deg(np.arctan2(z, xy))
        alt = np.sqrt(x**2+y**2+z**2) - self.earth_radius

        return lat, lon, alt

    def cartesian(self, lat, lon, alt):
        """
        Take Latitude, Longitude and Altitude. (Degrees and meters)
        Returns Position in Earth Centered Frame.
        """
        xy = np.cos(np.deg2rad(lat))*(self.earth_radius+alt)
        z = np.sin(np.deg2rad(lat))*(self.earth_radius+alt)

        x = np.cos(np.deg2rad(lon))*xy
        y = np.sin(np.deg2rad(lon))*xy

        return x, y, z


    def _get_height(self, pos):
        """
        Position in Earth Centered Frame
        """
        r = np.linalg.norm(pos)
        height = r - self.earth_radius
        return height
