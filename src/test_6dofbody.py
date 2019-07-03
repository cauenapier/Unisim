from numpy.testing import (assert_equal, assert_almost_equal)
import pytest
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from Unisim.bodies.body_6dof import *
from Unisim.environment import *
from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import *
from Unisim.environment.environment import Environment


t0 = 0
x0 = np.zeros(13)
x0[2] = 1000

Ball = Body_FlatEarth(t0,x0, k_quat=1)
Ball._set_mass(1)
Ball._set_inertia(10,10,10)

atmo = ISA1976()
gravity = VerticalConstant()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball.set_environment(Env)


Roll=0*np.pi/180
Pitch=0*np.pi/180
Yaw=0*np.pi/180
Ball.initialize_statevector(Roll, Pitch, Yaw)


step_size = 0.01
t0 = 0
tf = 100
bar = tqdm.tqdm(total=tf, desc='time', initial=t0)
for ii in np.arange(t0,tf,step_size):
    #print(Ball._state_vector[6:10])
    Ball.step(step_size)
    bar.update(step_size)

bar.close()
results = pd.DataFrame(Ball.results)

fig, ax = plt.subplots()
#ax.plot(results['Time'], results['Psi'],  label='Psi')
#ax.plot(results['Time'], results['Theta'],  label='Theta')
#ax.plot(results['Time'], results['Phi'],  label='Phi')

ax.plot(results['Time'], results['Pos x'],  label='Pos_x')
ax.plot(results['Time'], results['Pos y'],  label='Pos_y')
ax.plot(results['Time'], results['Pos z'],  label='Pos_z')
legend = ax.legend(loc = 'upper right')
plt.show()

#fig, ax = plt.subplots()
#ax.plot(results['Time'], results['Psi'],  label='Psi')
#ax.plot(results['Time'], results['Theta'],  label='Theta')
#ax.plot(results['Time'], results['Phi'],  label='Phi')
#legend = ax.legend(loc = 'upper right')
#plt.show()
