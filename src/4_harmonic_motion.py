
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

x0_1[2] = 0
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
stiffness = 1000
damping = 600
spring = Ellastic_Rope_3DOF(FixedPoint, Ball, rest_length, stiffness, damping)
#spring = DampedSpring(FixedPoint, Ball, rest_length, stiffness, damping)

step_size = 0.001
t0 = 0
tf = 30

"""
Iteraton loop. For every time step, the spring (constraint) forces are calculated
and the Ball object is propagated. The FixedPoint object is not in the iteration loop.
"""
import tqdm
bar = tqdm.tqdm(total=tf, desc='time', initial=t0)

for ii in np.arange(t0, tf, step_size):
    spring._calcForce_a2b()
    Ball.step(step_size)
    bar.update(step_size)

bar.close()

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


#import time
#plt.show()
#axes = plt.gca()
#axes.set_xlim(-5, 5)
#axes.set_ylim(-7, 0)
#
#xdata1 = []
#ydata1 = []
#xdata2 = []
#ydata2 = []
#line1, = axes.plot(xdata1, ydata1, 'r-')
#line2, = axes.plot(xdata2, ydata2, 'b-')
#anim_step = 100
#
#for i in range(0, len(results1),anim_step):
#    xdata1.append(results1['Pos x'][i])
#    ydata1.append(results1['Pos z'][i])
#    #xdata2.append(results2['Pos x'][i])
#    #ydata2.append(results2['Pos z'][i])
#    line1.set_xdata(xdata1)
#    line1.set_ydata(ydata1)
#    #line2.set_xdata(xdata2)
#    #line2.set_ydata(ydata2)
#    plt.draw()
#    plt.pause(1e-17)
#    time.sleep(1e-5)
## add this if you don't want the window to disappear at the end
#plt.show()
