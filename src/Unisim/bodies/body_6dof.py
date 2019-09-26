import operator
import numpy as np
from scipy.integrate import solve_ivp
from abc import abstractmethod
import pandas as pd

from Unisim.utils.quaternions import *
from Unisim.utils.quaternions import quaternion_inverse

class Body(object):

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
    'Psi' : 'psi',
    'Theta' : 'theta',
    'Phi' : 'phi',
    'p' : 'p',
    'q' : 'q',
    'r' : 'r',
    'p_dot' : 'p_dot',
    'q_dot' : 'q_dot',
    'r_dot' : 'r_dot',
    'Total Forces' : 'total_forces'
    }

    def __init__(self, t0, x0, method='RK45', name=None, options=None, save_vars=None, k_quat=1):
        """

        x0 = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, q0, q1, q2, q3, p, q, r,]
        """
        self._name = name
        self._time = t0
        self._state_vector = x0
        self._state_vector_dot = np.zeros_like(x0)
        self._DCM = np.zeros((3,3))
        self._euler = np.zeros(3)

        self._k_quat = k_quat # orthonormality error factor

        # Mass & Inertia
        self._mass = None # kg
        self._MOI = np.zeros((3,3))

        # Forces and Torques
        self.total_forces = np.zeros(3)
        self.total_torques = np.zeros(3)

        # Environment
        self._environment = None
        # Aerodynamics
        self._aerodynamics = None
        # Constraiunts
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
    def quat(self):
        return self._state_vector[6:10]
    @property
    def p(self):
        return self._state_vector[10]
    @property
    def q(self):
        return self._state_vector[11]
    @property
    def r(self):
        return self._state_vector[12]
    @property
    def acc_x(self):
        return self._state_vector_dot[3]
    @property
    def acc_y(self):
        return self._state_vector_dot[4]
    @property
    def acc_z(self):
        return self._state_vector_dot[5]
    @property
    def quat_dot(self):
        return self._state_vector_dot[6:10]
    @property
    def p_dot(self):
        return self._state_vector_dot[10]
    @property
    def q_dot(self):
        return self._state_vector_dot[11]
    @property
    def r_dot(self):
        return self._state_vector_dot[12]
    @property
    def psi(self):
        return self._euler[0]
    @property
    def theta(self):
        return self._euler[1]
    @property
    def phi(self):
        return self._euler[2]

    def _set_mass(self, mass):
        if mass <= 0:
            raise Exception("Object Mass is equal or lower than zero")
        else:
            self._mass = mass
    def _get_mass(self):
        return self._mass
    mass = property(_get_mass, _set_mass, doc="""Mass of the body.""")

    def _set_inertia(self, Ixx, Iyy, Izz):
        self._MOI[0,0] = Ixx
        self._MOI[1,1] = Iyy
        self._MOI[2,2] = Izz
    def _get_inertia(self):
        return self._MOI
    MOI = property(_get_mass, _set_mass, doc="""Moments of Inertia of the body.""")

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
            Time will still pass but object will not be propagated.
            It will keep its position
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

    def initialize_statevector(self):
        NotImplementedError

    def calculate_forces_and_torques(self):
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

            Quat = self._state_vector[6:10]
            # Direction Cossine Matrix
            self._DCM = quat2DCM(Quat)
            # Euler angles
            self._euler = quat2euler(Quat)

            self.calculate_forces_and_torques()
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
        Roll - Phi
        Pitch - Theta
        Yaw - Psi
        """
        rv = self._system_equations_6DOF(t,x,self._mass, self._MOI, self.total_forces, self.total_torques)

        return rv

    def initialize_statevector(self, roll_0, pitch_0, yaw_0):
        quat_0 = euler2quat(roll_0, pitch_0, yaw_0)
        self._DCM = quat2DCM(quat_0)
        self._state_vector[6:10] = quat_0

    def calculate_forces_and_torques(self):
        """
        """
        mass = self._mass
        MOI = self._MOI

        height = self._state_vector[2]

        # Zeroing the total forces variable
        self.total_forces = np.zeros(3)
        self.total_torques = np.zeros(3)

        if self._time <= 1:
            self.total_torques = np.array([1,1,1])
        #if self._time >= 3 and self._time < 4:
        #    self.total_torques = np.array([-1,-1,-1])
        else:
            self.total_torques = np.zeros(3)

        # === GRAVITY ===
        self._environment.gravity.update(height)
        self.calc_gravity(mass, self._DCM)
        #self.calc_gravity_quat(mass, self._state_vector[6:10])

        self._environment.atmosphere.update(height)
        # === AERO ===
        if self._aerodynamics is not None:
            self._aerodynamics.update(self._environment,self._state_vector[3:6])
            Fa = self._aerodynamics.forces_wind()
            self.total_forces = self.total_forces + Fa

        # === CONSTRAINTS ===
        if self._constraints is not []:
            for constraint in self._constraints:
                self.calc_constraints(constraint)

    def calc_constraints(self, constraint):
        NotImplementedError

    def calc_gravity(self, mass, DCM):
        Fg = np.matmul(DCM, self._environment.gravity._vector*mass)
        self.total_forces = self.total_forces + Fg
        return Fg

    def calc_gravity_quat(self, mass, quat):
        Fg = quaternion_rotation(quat, self._environment.gravity._vector*mass)
        self.total_forces = self.total_forces + Fg
        return Fg

    def calc_aero(self):
        Fa = self._aerodynamics.forces_body()
        Ma = self._aerodynamics.torques()
        self.total_forces = self.total_forces + Fa
        self.total_torques = self.total_torques + Ma
        return Fa

    def _system_equations_6DOF(self, time, state_vector, mass, MOI, forces, torques):
        """
        All forces and torques should be passed in body coordinates
        Missing the term if Inertia varies with time
        """
        Pos = state_vector[0:3]
        Vel = state_vector[3:6]
        Quat = state_vector[6:10]
        Ang_Vel = state_vector[10:13]

        # Tangencial Acceleration
        Acc_t = np.cross(Vel,Ang_Vel)
        # Acceleration in Body Coordinates
        Acc = forces/mass + Acc_t
        # Transform Velocity Into Local coordinates
        Vel_earth = np.matmul(self._DCM.transpose(), Vel)
        #Quat_inv = quaternion_inverse(Quat)
        #Vel_earth = quaternion_rotation(Quat_inv, Vel)
        #print(Vel_earth)

        Ang_Acc = np.matmul(np.linalg.inv(MOI), (torques - np.cross(Ang_Vel, np.matmul(MOI, Ang_Vel))))
        # Missing term when Inertia is changing (I_dot different from zero)

        p = Ang_Vel[0]
        q = Ang_Vel[1]
        r = Ang_Vel[2]

        lamb = self._k_quat * (1.0 - np.dot(Quat, Quat))
        quat_dot0 = - 0.5 * (p*Quat[1] + q*Quat[2] + r*Quat[3] ) + lamb * Quat[0]
        quat_dot1 = + 0.5 * (p*Quat[0] + r*Quat[2] - q*Quat[3] ) + lamb * Quat[1]
        quat_dot2 = + 0.5 * (q*Quat[0] + p*Quat[3] - r*Quat[1] ) + lamb * Quat[2]
        quat_dot3 = + 0.5 * (r*Quat[0] + q*Quat[1] - p*Quat[2] ) + lamb * Quat[3]
        Quat_dot = np.array([quat_dot0, quat_dot1, quat_dot2, quat_dot3])

        return np.concatenate((Vel_earth, Acc, Quat_dot, Ang_Acc))
