import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pymap3d
import datetime
from Unisim.bodies.body_3dof import *
from Unisim.environment import *
from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import SimpleNewton
from Unisim.environment.environment import Environment


t0 = 0
#% Position, velocity, acceleration
x0 = np.zeros(6)

lat = 0
long = 0
alt = 242000

time_now = datetime.datetime.now()

pos_ecef = pymap3d.geodetic2ecef(lat, long, alt)
print(pos_ecef)
pos_eci = pymap3d.ecef2eci(pos_ecef[0], pos_ecef[1], pos_ecef[2], time_now)
print(pos_eci)

x0[0:3] = pos_ecef
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


bar = tqdm.tqdm(total=tf, desc='time', initial=t0)

for ii in np.arange(t0,tf, step_size):
    Sat.step(step_size)
    #print(Sat.altitude())
    bar.update(step_size)
bar.close()
results = pd.DataFrame(Sat.results)


plt.plot(results['Pos x'], results['Pos y'])
plt.show()
