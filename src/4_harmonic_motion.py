
import numpy as np
import matplotlib.pyplot as plt
import copy

from Unisim.bodies.body_3dof import *
from Unisim.environment import *
from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import *
from Unisim.environment.environment import Environment
from Unisim.constraints.constraints import *


t0 = 0
#% Position, velocity, acceleration
x0_1 = np.zeros(6)
x0_2 = np.zeros(6)

x0_1[2] = 4
x0_2[2] = 10

Ball1 = Body_FlatEarth(t0,x0_1)
Ball1._set_mass(2)
Ball1._set_name("Ball 1")
FixedPoint = Body_FlatEarth(t0,x0_2)
FixedPoint._set_mass(10000)
FixedPoint._set_name("Fixed Point")

atmo = ISA1976()
gravity = VerticalNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball1.set_environment(Env)
FixedPoint.set_environment(Env)


rest_length = 5
stiffness = 10
damping = 1
spring = Tension_Only_Spring_3DOF(FixedPoint, Ball1, rest_length, stiffness, damping)

step_size = 0.01
t0 = 0
tf = 200

"""
Iteraton loop. For every time step, the spring (constraint) forces are calculated
and the Ball object is propagated. The FixedPoint object is not in the iteration loop.
"""
for ii in np.arange(t0,tf,step_size):
    spring._calcForce_a2b()
    Ball1.step(step_size)
#        print(Ball._get_name(), Ball.total_forces)
#    print("")


results1 = pd.DataFrame(Ball1.results)
results2 = pd.DataFrame(FixedPoint.results)

#plt.plot(results1['Pos z'])
#plt.plot(results2['Pos z'])
#plt.plot(results1['Acc z'])
#plt.plot(results2['Acc z'])
#plt.show()

#fig, ax1 = plt.subplots()
#ax1.plot(results1['Time'], results1['Pos z'])
#ax2 = ax1.twinx()
#ax2.plot(results1['Time'], results1['Acc z'])
#plt.grid(which='major', axis='both', linestyle='--')
#plt.show()
