"""

"""

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal)
import pytest


from Unisim.environment.gravity import VerticalConstant


def test_constant_gravity():
    expected_grav_magnitude = 9.81
    gravity = VerticalConstant()

    assert_almost_equal(gravity._magnitude, expected_grav_magnitude, decimal=2)


if __name__ == '__main__':
    pytest.main()
