from numpy.testing import (assert_equal, assert_almost_equal)
import pytest
import numpy as np

from Unisim.bodies.body_6dof import *


t0 = 0
x0 = np.zeros(12)

Ball = Body_FlatEarth(t0,x0)
