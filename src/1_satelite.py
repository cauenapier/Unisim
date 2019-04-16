
import numpy as np
import matplotlib.pyplot as plt
from Unisim.bodies.body_3dof import *
from Unisim.environment import *
from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import SimpleNewton
from Unisim.environment.environment import Environment


t0 = 0
#% Position, velocity, acceleration
x0 = np.zeros(6)

altitude = 242000 # Meters
x0[0] = 6371000 + altitude
# Velocity
x0[4] = 7800

Sat = Body_RoundEarth(t0,x0)
Sat._set_mass(500)

atmo = ISA1976()
gravity = SimpleNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Sat.set_environment(Env)

step_size = 1
t0 = 0
tf = 10000

for ii in np.arange(t0,tf, step_size):
    Sat.step(step_size)
    #print(Sat.altitude())
results = pd.DataFrame(Sat.results)

plt.plot(results['Pos x'], results['Pos y'])
plt.show()
