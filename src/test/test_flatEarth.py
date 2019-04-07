import numpy as np
import matplotlib.pyplot as plt
from Unisim.body import Body_FlatEarth
from Unisim.environment import *

from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import VerticalNewton
from Unisim.environment.environment import Environment

from numpy.testing import (assert_equal, assert_almost_equal)
import pytest


t0 = 0
x0 = np.zeros(6)
x0[2] = 10

Ball1 = Body_FlatEarth(t0,x0)

atmo = ISA1976()
gravity = VerticalNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)
