
import numpy as np
import matplotlib.pyplot as plt
from Unisim.body import *
from Unisim.environment import *


from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import VerticalNewton
from Unisim.environment.environment import Environment


t0 = 0
#% Position, velocity, acceleration
x0 = np.zeros(6)

altitude = 242000 # Meters
x0[0] = 6371000 + altitude
# Velocity
x0[4] = 7800

Sat = Body_RoundEarth(t0,x0)

atmo = ISA1976()
gravity = SimpleNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Sat.set_environment(Env)


for ii in range(0,10000):
    Sat.step(1)
    #print(Sat.altitude())

plt.plot(Sat.results[:,0], Sat.results[:,1])
plt.show()
