
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
x0[0] = 6371000 + 242000
x0[4] = 6000

Ball1 = Body_RoundEarth(t0,x0)


atmo = ISA1976()
gravity = SimpleNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball1.set_environment(Env)

Ball1.step(0.01)
print(Ball1.total_forces)

for ii in range(0,10000):
    Ball1.step(1)
#    print(Ball1.state_vector_dot)
#    Ball2.step(0.01)
#    print(Ball.results)
#    print("/n")

#print(Ball1.results[:,2])

plt.plot(Ball1.results[:,0], Ball1.results[:,1])
plt.show()

#    print(Ball.state_vector)


#print(Ball.results)
