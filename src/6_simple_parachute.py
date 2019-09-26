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
from Unisim.aerodynamics.aerodynamics import OnlyDrag


t0 = 0
#% Position, velocity, acceleration
x0_1 = np.zeros(6)
x0_2 = np.zeros(6)

x0_1[2] = 1000
x0_1[4] = 0

x0_2[2] = 1000
x0_2[4] = 0

Ball = Body_FlatEarth(t0,x0_1)
Ball._set_mass(300)
Ball._set_name("Ball")

Para = Body_FlatEarth(t0,x0_2)
Para._set_mass(10)
Para._set_name("Para")


atmo = ISA1976()
gravity = VerticalNewton()
drag = OnlyDrag(0.5, 1)
#gravity = NoGravity()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball.set_environment(Env)
Para.set_environment(Env)

Para.set_aerodynamics(drag)

rest_length = 1
stiffness = 7500
damping = 100
rope = Ellastic_Rope_3DOF(Para, Ball, rest_length, stiffness, damping)

step_size = 0.01
t0 = 0
tf = 50


import tqdm
bar = tqdm.tqdm(total=tf, desc='time', initial=t0)

for ii in np.arange(t0, tf, step_size):
    rope._calcForce_a2b()
    Para.step(step_size)
    #print(Para._name, Para.total_forces)
    Ball.step(step_size)
    #print(Ball._name, Ball.total_forces)
    bar.update(step_size)


bar.close()

results_ball = pd.DataFrame(Ball.results)
results_para = pd.DataFrame(Para.results)


plt.plot(results_ball['Time'], results_ball['Pos z'])
plt.show()
#plt.plot(results_para['Time'], results_para['Pos z'])
plt.plot(results_ball['Time'], results_ball['Pos z']-results_para['Pos z'])
plt.show()
plt.plot(results_ball['Time'], results_ball['Acc z']-results_para['Acc z'])
plt.show()
