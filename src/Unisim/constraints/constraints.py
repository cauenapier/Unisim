"""A constraint is something that describes how two bodies interact with
each other. (how they constrain each other).
"""
import numpy as np

class Constraint(object):
    """ Base class for constraints
    """


    def _set_bodies(self,a,b):
        self._a = a
        self._b = b

    a = property(lambda self: self._a,
        doc="""The first of the two bodies constrained""")
    b = property(lambda self: self._b,
        doc="""The second of the two bodies constrained""")

class Tension_Only_Spring_3DOF(Constraint):
    """
    """
    def __init__(self, a, b, rest_length, stiffness, damping):
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
        super()._set_bodies(a,b)
        self._rest_length = rest_length
        self._stiffness = stiffness
        self._damping = damping
        self._force_vector_a2b = None

        a.set_constraint(self)
        b.set_constraint(self)

    def _calcForce_a2b(self):
        pos_a = self._a._state_vector[0:3]
        pos_b = self._b._state_vector[0:3]

        dist_vector = (pos_b - pos_a)
        print(self._a._time, dist_vector)
        dist_norm = np.linalg.norm(pos_b - pos_a)
        dist_versor = dist_vector/dist_norm

        if dist_norm > self._rest_length:
            F_k = (dist_norm-self._rest_length)*self._stiffness*dist_versor
        else:
            F_k = 0

        self._force_vector_a2b = F_k

        return F_k

    def _get_force_vector_a2b(self):
        return self._force_vector_a2b
