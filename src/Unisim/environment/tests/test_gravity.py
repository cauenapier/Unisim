"""

"""

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal)
import pytest


from Unisim.environment.gravity import VerticalConstant

def test_constant_gravity_magnitude():
    expected_grav_magnitude = 9.81
    gravity = VerticalConstant()

    assert_almost_equal(gravity._magnitude, expected_grav_magnitude, decimal=2)


from Unisim.environment.gravity import VerticalNewton

def test_VerticalNewton():
    expected_grav_vector = np.array([0, 0, -9.81], dtype=float)
    height = 0

    gravity = VerticalNewton()
    gravity.update(height)
    assert_almost_equal(gravity._vector, expected_grav_vector, decimal=2)
