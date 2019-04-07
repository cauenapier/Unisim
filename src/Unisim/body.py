import numpy as np
from scipy.integrate import solve_ivp
from abc import abstractmethod

from Unisim.environment.gravity import *


class Body(object):
    """A rigid body

    TODO: create a type for static objects
    """

#    _default_save_vars = {
#    'u': 'self._state_vector[3]',
#    'v': 'self._state_vector[4]',
#    'w': 'self._state_vector[5]',
#    }

    def __init__(self, t0, x0, method='RK45', options=None):
        """
        """
        self._time = t0
        self._state_vector = x0
        self._state_vector_dot = np.zeros_like(x0)

        # Mass % Inertia
        self._mass = 10 # kg
        # Forces & Moments
        self.total_forces = np.zeros(3)

        # Environment
        self._environment = None

        if options is None:
            options = {}
        self._method = method
        self._options = options

        self._results = np.empty((0,6))

    @property
    def results(self):
        return self._results

    @property
    def state_vector(self):
        return self._state_vector

    @property
    def state_vector_dot(self):
        return self._state_vector_dot

    @property
    def time(self):
        return self._time

    @property
    def environment(self):
        return self._environment

    def set_environment(self, environment):
        self._environment = environment

    @abstractmethod
    def fun(self, t, x):
        """
        """
        raise NotImplementedError


    def fun_wrapped(self, t, x):
        # First way that comes to my mind in order to store the derivatives
        # that are useful for full_state calculation
        state_dot = self.fun(t, x)
        self._state_vector_dot = state_dot
        return state_dot

    def step(self, dt):
        """Integrate the system from current time to t_end.

        Parameters
        dt : float
        ----------
            Time step.

        Returns
        -------
        y : ndarray, shape (n)
            Solution values at t_end.
        """

        x0 = self.state_vector
        t_ini = self.time

        t_span = (t_ini, t_ini + dt)
        method = self._method

        # TODO: prepare to use jacobian in case it is defined
        sol = solve_ivp(self.fun_wrapped, t_span, x0, method=method,
                        **self._options)

        if sol.status == -1:
            raise RuntimeError(f"Integration did not converge at t={t_ini}")

        self._time = sol.t[-1]
        self._state_vector = sol.y[:, -1]

        self._results = np.append(self._results, [x0], axis=0)

        return self._state_vector

class Body_FlatEarth(Body):
    """
    """
    def fun(self, t, x):
        """
        """
        mass = self._mass
        height = self._state_vector[2]


        self._environment.gravity.update(height)
        Fg = self._environment.gravity._vector*mass


        forces = Fg

        self.total_forces = forces

        rv = self._system_equations_3DOF(t,x,mass,forces)

        return rv

    def _system_equations_3DOF(self, time, state_vector, mass, forces):
        """
        """
        Fx, Fy, Fz = forces
        u, v, w = state_vector[3:6]

        dx_dt = u
        dy_dt = v
        dz_dt = w

        du_dt = Fx/mass
        dv_dt = Fy/mass
        dw_dt = Fz/mass

        return np.array([dx_dt, dy_dt, dz_dt, du_dt, dv_dt, dw_dt])

class Body_RoundEarth(Body):
    """
    """
    def fun(self, t, x):
        """
        """
        mass = self._mass

        pos = self._state_vector[0:3]
        self._environment.gravity.update(pos)
        Fg = self._environment.gravity._vector*mass

        forces = Fg

        self.total_forces = forces

        rv = self._system_equations_3DOF(t,x,mass,forces)

        return rv

    def _system_equations_3DOF(self, time, state_vector, mass, forces):
        """
        """
        Fx, Fy, Fz = forces
        u, v, w = state_vector[3:6]

        dx_dt = u
        dy_dt = v
        dz_dt = w

        du_dt = Fx/mass
        dv_dt = Fy/mass
        dw_dt = Fz/mass

        return np.array([dx_dt, dy_dt, dz_dt, du_dt, dv_dt, dw_dt])
