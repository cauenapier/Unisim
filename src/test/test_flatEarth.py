import numpy as np
import matplotlib.pyplot as plt
from Unisim.bodies.body_3dof import *
from Unisim.environment import *

from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import VerticalNewton
from Unisim.environment.environment import Environment

from numpy.testing import (assert_equal, assert_almost_equal)
import pytest


t0 = 0
x0 = np.zeros(6)
x0[2] = 10 # Initial Height
mass = 10

Ball = Body_FlatEarth(t0,x0)
Ball._set_mass(mass)

atmo = ISA1976()
gravity = VerticalNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)
Ball.set_environment(Env)

def test_gravity_force():
    Ball.step(0.01)
    expected_Gravity_Force = np.array([0, 0, -98.1])

    assert_almost_equal(Ball.calc_gravity(mass), expected_Gravity_Force, decimal=1)
