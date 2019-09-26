import numpy as np

class Aerodynamics(object):
    """
    """
    def __init__(self):
        ""
        self._Drag = 0
        self._Lift = 0

    def _set_cD(self, mass):
        ""
    def _get_cD(self):
        return self._mass
    cD = property(_get_cD, _set_cD, doc="""Drag Coefficient""")


    def forces_body(self):
        """
        Returns the forces on the body axis
        """
        raise NotImplementedError
    def forces_wind(self):
        """
        Returns the forces on the wind axis
        """
        raise NotImplementedError
    def torques(self):
        """

        """
        raise NotImplementedError


class OnlyDrag(Aerodynamics):
    """
    """
    def __init__(self, cD, ref_Area):
        ""
        self._cD = cD
        self._ref_Area = ref_Area

    def update(self, environment, vel):

        v = np.linalg.norm(vel)

        if v == 0:
            self._Drag = 0
        else:
            versor = vel/v
            rho = environment.atmosphere.rho
            q = 0.5*rho*v**2
            self._Drag = -q*self._cD*self._ref_Area*versor


    def forces_wind(self):
        return self._Drag
