import operator
import numpy as np
from scipy.integrate import solve_ivp
from abc import abstractmethod
import pandas as pd

"""
TODO: A Body class should reference to an orbiting body, instead of fixing it to earth.
Then update its parameters
TODO: Should I separate system equations from the Body Class?
"""

class Body(object):
    """A rigid body

    TODO: create a type for static objects
    """

    _default_save_vars = {
    'Time': '_time',
    'Pos x': 'pos_x',
    'Pos y': 'pos_y',
    'Pos z' : 'pos_z',
    'Vel x': 'vel_x',
    'Vel y': 'vel_y',
    'Vel z' : 'vel_z',
    'Acc x': 'acc_x',
    'Acc y': 'acc_y',
    'Acc z' : 'acc_z',
    'Mass' : '_mass',
    'Total Forces' : 'total_forces'
    }

    def __init__(self, t0, x0, method='RK45', name=None, options=None, save_vars=None):
        """

        x0 = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        """
        self._name = name
        self._time = t0
        self._state_vector = x0
        self._state_vector_dot = np.zeros_like(x0)

        # Mass
        self._mass = None # kg
        # Forces
        self.total_forces = np.zeros(3)

        # Environment
        self._environment = None
        # Aerodynamics
        self._aerodynamics = None
        # Constraints
        self._constraints = []

        if options is None:
            options = {}
        self._method = method
        self._options = options

        if not save_vars:
            self._save_vars = self._default_save_vars
        # Initialize results structure
        self.results = {name: [] for name in self._save_vars}

        # Other variables
        self._isSleeping = 0

    @property
    def pos_x(self):
        return self._state_vector[0]
    @property
    def pos_y(self):
        return self._state_vector[1]
    @property
    def pos_z(self):
        return self._state_vector[2]
    @property
    def vel_x(self):
        return self._state_vector[3]
    @property
    def vel_y(self):
        return self._state_vector[4]
    @property
    def vel_z(self):
        return self._state_vector[5]
    @property
    def acc_x(self):
        return self._state_vector_dot[3]
    @property
    def acc_y(self):
        return self._state_vector_dot[4]
    @property
    def acc_z(self):
        return self._state_vector_dot[5]

    def _set_mass(self, mass):
        if mass <= 0:
            raise Exception("Object Mass is equal or lower than zero")
        else:
            self._mass = mass
    def _get_mass(self):
        return self._mass
    mass = property(_get_mass, _set_mass, doc="""Mass of the body.""")

    def _set_name(self, name):
        self._name = name
    def _get_name(self):
        return self._name
    name = property(_get_name, _set_name, doc="""Name of the body.""")

    @property
    def time(self):
        return self._time

    @property
    def environment(self):
        return self._environment

    def set_environment(self, environment):
        self._environment = environment

    @property
    def aerodynamics(self):
        return self._aerodynamics

    def set_aerodynamics(self, aerodynamics):
        self._aerodynamics = aerodynamics

    @property
    def constraints(self):
        return self._constraints

    def add_constraint(self, constraint):

        self._constraints.append(constraint)

    def sleep(self):
        """ Forces a body to sleep and stop propagation.
            Time will still pass but object will not be propagated
        """
        self._isSleeping = 1
    def awake(self):
        """ Awaken the body, returning its propagation
        """
        self._isSleeping = 0

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

    def calculate_forces(self):
        """
        """
        raise NotImplementedError

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
        t_ini = self.time
        t_span = (t_ini, t_ini + dt)

        if self._isSleeping:
            print(self._name, "is sleeping")
            self._time = t_ini+dt
        else:
            x0 = self._state_vector

            method = self._method
            # TODO: prepare to use jacobian in case it is defined
            self.calculate_forces()
            sol = solve_ivp(self.fun_wrapped, t_span, x0, method=method,
                            **self._options)

            if sol.status == -1:
                raise RuntimeError(f"Integration did not converge at t={t_ini}")

            self._time = sol.t[-1]
            self._state_vector = sol.y[:, -1]

        self._save_time_step()
        return self._state_vector

    def _save_time_step(self):
        """ Saves the selected variables for the current body
        """
        for var_name, value_pointer in self._save_vars.items():
            self.results[var_name].append(
                operator.attrgetter(value_pointer)(self)
            )

class Body_FlatEarth(Body):
    """
    """

    def fun(self, t, x):
        """
        """
        rv = self._system_equations_3DOF(t,x,self._mass,self.total_forces)

        return rv

    def calculate_forces(self):
        """
        """
        mass = self._mass
        height = self._state_vector[2]

        # Zeroing the total forces variable
        self.total_forces = np.zeros(3)

        # === GRAVITY ===
        self._environment.gravity.update(height)
        self.calc_gravity(mass)

        # === ATMOSPHERE ===
        #print(self._environment.atmosphere.rho)

        # === AERO ===
        #self.calc_drag_TEMP(self._state_vector[3:6])
        if self._aerodynamics is not None:
            self._aerodynamics.update(self._environment,self._state_vector[3:6])
            Fa = self._aerodynamics.forces_wind()
            self.total_forces = self.total_forces + Fa

        # === CONSTRAINTS ===
        if self._constraints is not []:
            for constraint in self._constraints:
                self.calc_constraints_forces(constraint)

    def calc_constraints_forces(self, constraint):
        """
        Calculates Constraint Force and add up to total Forces
        Forces are from A to B. If applied to body B, it inverts the sign.
        """
        Fk = constraint._force_vector_a2b
        if constraint.b is self:
            Fk = -Fk
        self.total_forces = self.total_forces + Fk
        return Fk

    def calc_aero_forces(self, state_vector):
        raise NotImplementedError

    def calc_gravity(self, mass):
        Fg = self._environment.gravity._vector*mass
        self.total_forces = self.total_forces + Fg
        return Fg

    def calc_drag_TEMP(self, vel):
        """
        TEMPORARY METHOD
        """
        #height

        #self._environment.atmosphere.update(height)
        #rho = self._environment.atmosphere._rho
        v = np.linalg.norm(vel)

        if v == 0:
            Fd = 0
        else:
            versor = vel/v
            #Fd = -(0.5*rho*v**2)*cD*versor
            Fd = -(v**2)*versor

        self.total_forces = self.total_forces + Fd
        return Fd

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
        rv = self._system_equations_3DOF(t,x,self._mass,self.total_forces)

        return rv

    def calculate_forces(self):
        """
        """
        mass = self._mass
        pos = self._state_vector[0:3]
        vel = self._state_vector[3:6]

        # Zeroing the total forces variable
        self.total_forces = np.zeros(3)

        # === GRAVITY ===
        self._environment.gravity.update(pos)
        self.calc_gravity(mass)

        # === AERO ===
        if self._aerodynamics is not None:
            self.calc_aero_forces()

        # === CONSTRAINTS ===
        if self._constraints is not []:
            for constraint in self._constraints:
                self.calc_constraints_forces(constraint)

    def calc_constraints_forces(self):
        """
        Calculates Constraint Force and add up to total Forces
        Forces are from A to B. If applied to body B, it inverts the sign.
        """
        Fk = self._constraints._force_vector_a2b
        if self._constraints.b is self:
            Fk = -Fk

        self.total_forces = self.total_forces + Fk
        return Fk

    def calc_gravity(self, mass):
        Fg = self._environment.gravity._vector*mass
        self.total_forces = self.total_forces + Fg
        return Fg

    def calc_aero_forces(self):
        """
        """
        raise NotImplementedError

    def calc_drag_TEMP(self, vel):
        """
        TEMPORARY METHOD
        """
        #height

        #self._environment.atmosphere.update(height)
        #rho = self._environment.atmosphere._rho
        v = np.linalg.norm(vel)

        if v == 0:
            Fd = 0
        else:
            versor = vel/v
            #Fd = -(0.5*rho*v**2)*cD*versor
            Fd = -(0.1*v**2)*versor

        self.total_forces = self.total_forces + Fd

        return Fd

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
