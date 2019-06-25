class Aerodynamics(object):
    """
    """
    def __init__(self):
        ""



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
