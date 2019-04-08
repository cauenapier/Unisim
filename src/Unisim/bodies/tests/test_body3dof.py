from numpy.testing import (assert_equal, assert_almost_equal)
import pytest
import numpy as np

from Unisim.bodies.body_3dof import *

t0 = 0
#% Position, velocity, acceleration
x0 = np.zeros(6)

Ball = Body_RoundEarth(t0,x0)

def test_change_name():
    assert Ball._name is None
    Ball._set_name("Ball")
    assert Ball._name ==  "Ball"

def test_change_mass():
    assert Ball._mass == None
    Ball._set_mass(10)
    assert Ball._mass == 10
