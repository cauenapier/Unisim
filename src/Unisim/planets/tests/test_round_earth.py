import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal)
import pytest

from Unisim.planets.planet import *

def test_polar_transformation():
    Earth = Earth_Round()

    assert_almost_equal(Earth.polar(6371000, 0, 0), (0, 0, 0))
    assert_almost_equal(Earth.polar(-6371000, 0, 0), (0, 180, 0))
    assert_almost_equal(Earth.polar(0, 6371000, 0), (0, 90, 0))
    assert_almost_equal(Earth.polar(0, -6371000, 0), (0, -90, 0))
    assert_almost_equal(Earth.polar(6371100, 0, 0), (0, 0, 100))
    assert_almost_equal(Earth.polar(0, 0, 6371000), (90, 0, 0))
    assert_almost_equal(Earth.polar(6371100*np.cos(np.deg2rad(45)), 6371100*np.sin(np.deg2rad(45)), 0), (0, 45, 100))
    assert_almost_equal(Earth.polar(6371100*np.cos(np.deg2rad(30)), 6371100*np.sin(np.deg2rad(30)), 0), (0, 30, 100))


    assert_almost_equal(Earth.cartesian(0, 0, 0), (6371000, 0, 0))
    assert_almost_equal(Earth.cartesian(0, 0, 100), (6371100, 0, 0))
