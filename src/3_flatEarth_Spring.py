
import numpy as np
import matplotlib.pyplot as plt
import copy

from Unisim.bodies.body_3dof import *
from Unisim.environment import *
from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import VerticalNewton
from Unisim.environment.environment import Environment
from Unisim.constraints.constraints import Tension_Only_Spring_3DOF


t0 = 0
#% Position, velocity, acceleration
x0_1 = np.zeros(6)
x0_2 = np.zeros(6)

x0_1[2] = 1010
x0_2[2] = 1000

Ball1 = Body_FlatEarth(t0,x0_1)
Ball1._set_mass(500)
Ball1._set_name("Ball 1")
Ball2 = Body_FlatEarth(t0,x0_2)
Ball2._set_mass(500)
Ball2._set_name("Ball 2")

atmo = ISA1976()
gravity = VerticalNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball1.set_environment(Env)
Ball2.set_environment(Env)

Balls = [Ball1, Ball2]

rest_length = 7
stiffness = 1000
damping = 0
spring = Tension_Only_Spring_3DOF(Ball1, Ball2, rest_length, stiffness, damping)

step_size = 0.01
t0 = 0
tf = 10

spring_forces = np.zeros(3)

for ii in np.arange(t0,tf, step_size):
    spring._calcForce_a2b()
    for Ball in Balls:
        Ball.step(step_size)
#        print(Ball._get_name(), Ball.total_forces)
#    print("")


results1 = pd.DataFrame(Ball1.results)
results2 = pd.DataFrame(Ball2.results)

plt.plot(results1['Pos z'])
plt.plot(results2['Pos z'])
#plt.plot(results1['Acc z'])
#plt.plot(results2['Acc z'])
plt.show()
#plt.plot(results1['Pos z']-results2['Pos z'])
#plt.show()
