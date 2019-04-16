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

class DampedSpring(Constraint):
    def __init__(self, a, b, rest_length, stiffness, damping):
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
        vel_a = self._a._state_vector[3:6]
        vel_b = self._b._state_vector[3:6]

        # Relative Position
        rel_position_norm = np.linalg.norm(pos_b - pos_a)
        if rel_position_norm == 0:
            rel_position_versor = np.zeros(3)
        else:
            rel_position_versor = (pos_b - pos_a)/rel_position_norm

        # Relative Velocity
        rel_velocity_norm = np.linalg.norm(vel_b - vel_a)
        if rel_velocity_norm == 0:
            rel_velocity_versor = np.zeros(3)
        else:
            rel_velocity_versor = (vel_b - vel_a)/rel_velocity_norm

        F_k = (rel_position_norm-self._rest_length)*self._stiffness*rel_position_versor
        F_c = (rel_velocity_norm)*self._damping*rel_velocity_versor
        Force = (F_k+F_c)
        self._force_vector_a2b = Force

        return Force

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
        vel_a = self._a._state_vector[3:6]
        pos_b = self._b._state_vector[0:3]
        vel_b = self._b._state_vector[3:6]

        # Relative Position
        rel_position_norm = np.linalg.norm(pos_b - pos_a)
        if rel_position_norm == 0:
            rel_position_versor = np.zeros(3)
        else:
            rel_position_versor = (pos_b - pos_a)/rel_position_norm

        # Relative Velocity
        rel_velocity_norm = np.linalg.norm(vel_b - vel_a)
        if rel_velocity_norm == 0:
            rel_velocity_versor = np.zeros(3)
        else:
            rel_velocity_versor = (vel_b - vel_a)/rel_velocity_norm

        # Tension Only
        if rel_position_norm >= self._rest_length:
            F_k = (rel_position_norm-self._rest_length)*self._stiffness*rel_position_versor
            F_c = (rel_velocity_norm)*self._damping*rel_velocity_versor
        else:
            F_k = 0
            F_c = 0

        Force = (F_k + F_c)
        self._force_vector_a2b = Force

        return Force

    def _get_force_vector_a2b(self):
        return self._force_vector_a2b
