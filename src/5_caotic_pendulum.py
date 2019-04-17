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
x0_fix = np.zeros(6)

x0_1[0] = 1
x0_1[2] = 0

x0_2 = copy.deepcopy(x0_1)
x0_2[0] = 2
x0_2[2] = 0

FixedPoint = Body_FlatEarth(t0,x0_fix)
FixedPoint._set_mass(10000)
FixedPoint._set_name("Fixed Point")

Ball = Body_FlatEarth(t0,x0_1)
Ball._set_mass(10)
Ball._set_name("Ball 1")
Ball2 = Body_FlatEarth(t0, x0_2)
Ball2._set_mass(10)
Ball2._set_name("Ball 2")

atmo = ISA1976()
gravity = VerticalNewton()
#gravity = NoGravity()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball.set_environment(Env)
Ball2.set_environment(Env)
#FixedPoint.set_environment(Env) # If the fixed point is not being integrate, there is no need to assign an environment to it

rest_length = 1
stiffness = 10000
damping = 10
spring = Ellastic_Rope_3DOF(FixedPoint, Ball, rest_length, stiffness, damping)
spring2 = Ellastic_Rope_3DOF(Ball, Ball2, rest_length, stiffness, damping)
#spring = DampedSpring(FixedPoint, Ball, rest_length, stiffness, damping)

step_size = 0.0005
t0 = 0
tf = 10

"""
Iteraton loop. For every time step, the spring (constraint) forces are calculated
and the Ball object is propagated. The FixedPoint object is not in the iteration loop.
"""
import tqdm
bar = tqdm.tqdm(total=tf, desc='time', initial=t0)

for ii in np.arange(t0, tf, step_size):
    spring._calcForce_a2b()
    spring2._calcForce_a2b()
    Ball.step(step_size)
    Ball2.step(step_size)
    bar.update(step_size)

bar.close()

results1 = pd.DataFrame(Ball.results)
results2 = pd.DataFrame(Ball2.results)
#results2 = pd.DataFrame(FixedPoint.results)

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

#plt.plot(results1['Pos x'], results1['Pos z'])
#plt.show()
#plt.plot(results1['Time'], results1['Pos x'])
#plt.show()

#######
#from matplotlib import animation
## First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
#ax = plt.axes(xlim=(-2, 2), ylim=(-30, 0))
#line, = ax.plot([], [], lw=2)
#
#def init():
#    line.set_data([], [])
#    return line,
#
## animation function of dataframes' list
#def animate(i):
#    line.set_data(results1[i]['Pos x'], results1[i]['Pos z'])
#    return line,
#
## call the animator, animate every 300 ms
## set number of frames to the length of your list of dataframes
#anim = animation.FuncAnimation(fig, animate, frames=len(results1), init_func=init, interval=300, blit=True)
#
#plt.show()
#############

import time
plt.show()
axes = plt.gca()
axes.set_xlim(-3, 3)
axes.set_ylim(-3, 3)

xdata1 = []
ydata1 = []
xdata2 = []
ydata2 = []
line1, = axes.plot(xdata1, ydata1, 'r-')
line2, = axes.plot(xdata2, ydata2, 'b-')
anim_step = 10

for i in range(0, len(results1),anim_step):
    xdata1.append(results1['Pos x'][i])
    ydata1.append(results1['Pos z'][i])
    xdata2.append(results2['Pos x'][i])
    ydata2.append(results2['Pos z'][i])
    line1.set_xdata(xdata1)
    line1.set_ydata(ydata1)
    line2.set_xdata(xdata2)
    line2.set_ydata(ydata2)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(1e-8)
# add this if you don't want the window to disappear at the end
plt.show()
