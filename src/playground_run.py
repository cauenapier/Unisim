
import numpy as np
import matplotlib.pyplot as plt
from Unisim.bodies.body_3dof import *
from Unisim.environment import *


from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import SimpleNewton
from Unisim.environment.environment import Environment


t0 = 0
#% Position, velocity, acceleration
x0 = np.zeros(6)
x0[0] = 6371000 + 1000
x0[4] = 7600

Ball1 = Body_RoundEarth(t0,x0, name="Ball")
Ball1._set_mass(10)


atmo = ISA1976()
gravity = SimpleNewton()
wind = NoWind()
Env = Environment(atmo, gravity, wind)

Ball1.set_environment(Env)

Ball1.step(0.01)

step_size = 1
for ii in np.arange(0,100, step_size):
    if ii == 25:
        Ball1.sleep()
    if ii == 75:
        Ball1.awake()
    Ball1.step(step_size)


    #print(Ball1.total_forces)
    #print("====")
#    print(Ball1.state_vector_dot)
#    Ball2.step(0.01)
#    print(Ball.results)
#    print("/n")

results = pd.DataFrame(Ball1.results)
results.set_index('Time', inplace=True)
#results.to_csv("out.csv")

#print(Ball1.results[:,2])

plt.plot(results['Pos x'])
plt.show()

#    print(Ball.state_vector)


#print(Ball.results)
