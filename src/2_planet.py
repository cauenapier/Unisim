from Unisim.planets.planet import *
from Unisim.bodies.body_3dof import Body_RoundEarth


Earth = Earth_Round()

t0 = 0
x0 = np.zeros(6)
x0[0] = 6371000 + 100
Ball = Body_RoundEarth(t0,x0, name="Ball")
Ball._set_mass(10)
