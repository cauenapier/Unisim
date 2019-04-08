"""A constraint is something that describes how two bodies interact with
each other. (how they constrain each other).
"""

class Constraint(object):
    """ Base class for constraints
    """


    def _set_bodies(self,a,b):
        self._a = a
        self._b = b


class DampedSpring(Constraint):
    """
    """
    def __init__(self, a, b, anchor_a, anchor_b, rest_length, sitffness, damping):
        """

        :param Body a: Body a
        :param Body b: Body b
        :param anchor_a: Anchor point a, relative to body a
        :type anchor_a: `(float,float)`
        :param anchor_b: Anchor point b, relative to body b
        :type anchor_b: `(float,float)`
        :param float rest_length: The distance the spring wants to be.
        :param float stiffness: The spring constant (Young's modulus).
        :param float damping: How soft to make the damping of the spring.
        """
