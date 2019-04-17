
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

x0_1[2] = -5
x0_1[0] = 0

Ball = Body_FlatEarth(t0,x0_1)
Ball._set_mass(10)
Ball._set_name("Ball")
FixedPoint = Body_FlatEarth(t0,x0_2)
FixedPoint._set_mass(10000)
FixedPoint._set_name("Fixed Point")

atmo = ISA1976()
gravity = VerticalNewton()
#gravity = NoGravity()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball.set_environment(Env)
#FixedPoint.set_environment(Env) # If the fixed point is not being integrate, there is no need to assign an environment to it

rest_length = 5
stiffness = 10
damping = 1
spring = Rope_3DOF(FixedPoint, Ball, rest_length, stiffness, damping)
#spring = DampedSpring(FixedPoint, Ball, rest_length, stiffness, damping)

step_size = 0.01
t0 = 0
tf = 100

"""
Iteraton loop. For every time step, the spring (constraint) forces are calculated
and the Ball object is propagated. The FixedPoint object is not in the iteration loop.
"""
for ii in np.arange(t0, tf, step_size):
    spring._calcForce_a2b()
    Ball.step(step_size)
#        print(Ball._get_name(), Ball.total_forces)
#    print("")


results1 = pd.DataFrame(Ball.results)
results2 = pd.DataFrame(FixedPoint.results)

#plt.plot(results1['Pos z'])
#plt.plot(results2['Pos z'])
#plt.plot(results1['Acc z'])
#plt.plot(results2['Acc z'])
#plt.show()

#fig, ax1 = plt.subplots()
#ax1.plot(results1['Pos x'], results1['Pos z'])
#ax1.plot(results1['Time'], results1['Pos z'])
#ax2 = ax1.twinx()
#ax2.plot(results1['Time'], results1['Acc z'])
#plt.grid(which='major', axis='both', linestyle='--')
#plt.show()

plt.plot(results1['Time'], results1['Pos z'])
plt.show()
plt.plot(results1['Time'], results1['Pos x'])
plt.show()
