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


from Unisim.models.state.objectstate import ObjectState
from Unisim.models.state import objectstate, attitude, velocity
from Unisim.models.state.position import EarthPosition

def test_constant_gravity_vector():
    expected_grav_vector = np.array([0, 0, 9.81], dtype=float)

    gravity = VerticalConstant()

    assert_almost_equal(gravity._vector, expected_grav_vector, decimal=2)


from Unisim.environment.gravity import VerticalNewton

def test_VerticalNewton():
    expected_grav_vector = np.array([0, 0, 9.81], dtype=float)
    pos = EarthPosition(x=0, y=0, height=0)
    att = attitude.EulerAttitude(theta=0, phi=0, psi=0)
    vel = velocity.BodyVelocity(u=0, v=0, w=0, attitude=att)
    dumb_state = ObjectState(pos, att, vel)

    gravity = VerticalNewton()
    gravity.update(dumb_state)
    assert_almost_equal(gravity._vector, expected_grav_vector, decimal=2)
