import pytest


def test_vector_creation():
    position_ECI = vector(6371000, 0, 0, "ECI")
    r = position_ECI.magnitude()
    assert_equal(r, 637100)
