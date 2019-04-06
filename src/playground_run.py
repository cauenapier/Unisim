
import numpy as np
import matplotlib.pyplot as plt
from Unisim.body import Body_FlatEarth
from Unisim.environment import *


from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import VerticalNewton
from Unisim.environment.environment import Environment


t0 = 0
#% Position, velocity, acceleration
x0 = np.zeros(6)
x0[2] = 10
x0_2 = np.zeros(6)
x0_2[2] = 20

Ball1 = Body_FlatEarth(t0,x0)
Ball2 = Body_FlatEarth(t0,x0_2)


atmo = ISA1976()
gravity = VerticalNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

#print(Ball.state_vector)
#print(Ball.state_vector_dot)
#
#print(Ball.fun(0, Ball.state_vector))
#
#B#all.step(0.1)
###print(Ball.state_vector)
###print(Ball.state_vector_dot)

#print(Ball.state_vector)

print("Has Attribute Test")
#Ball1.environment.gravity

Ball1.set_environment(Env)

print("Has Attribute Test")
print(Ball1.environment.gravity)
#print(hasattr(Ball1, 'environment'))

Ball1.step(0.01)
print(Ball1.total_forces)

#for ii in range(0,100):
#    Ball1.step(0.01)
#    print(Ball1.state_vector_dot)
#    Ball2.step(0.01)
#    print(Ball.results)
#    print("/n")

#print(Ball1.results[:,2])

#plt.plot(Ball1.results[:,2])
#plt.plot(Ball2.results[:,2])
#plt.show()

#    print(Ball.state_vector)


#print(Ball.results)
