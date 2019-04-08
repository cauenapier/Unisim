
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Unisim.bodies.body_3dof import *
from Unisim.environment import *
import copy


from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import SimpleNewton
from Unisim.environment.environment import Environment


def main():

    t0 = 0
    #% Position, velocity, acceleration
    x0 = np.zeros(6)
    x0[0] = 6371000 + 200000
    #x0[3] = 20
    x0[4] = 7600


    Ball1 = Body_RoundEarth(t0,x0, name="Ball1")
    Ball1._set_mass(10)

    #x0_2 = copy.deepcopy(x0)
    #x0_2[0] = x0_2[0] + 10
    #Ball2 = Body_RoundEarth(t0,x0_2, name="Ball2")
    #Ball2._set_mass(10)

    #x0_3 = copy.deepcopy(x0_2)
    #x0_3[0] = x0_3[0] + 10
    #Ball3 = Body_RoundEarth(t0,x0_3, name="Ball3")
    #Ball3._set_mass(10)



    atmo = ISA1976()
    gravity = SimpleNewton()
    wind = NoWind()
    Env = Environment(atmo, gravity, wind)

    Ball1.set_environment(Env)
    #Ball2.set_environment(Env)
    #Ball3.set_environment(Env)


    #Balls = [Ball1, Ball2, Ball3]
    Balls = [Ball1]


    step_size = 1
    t0 = 0
    tf = 3


    for Ball in Balls:
        for ii in np.arange(t0,tf, step_size):
            Ball.step(step_size)


    results1 = pd.DataFrame(Ball1.results)
    #results2 = pd.DataFrame(Ball2.results)
    #results3 = pd.DataFrame(Ball3.results)


    #results.set_index('Time', inplace=True)
    #results.to_csv("out.csv")
    #print(Ball1.results)


    plt.plot(results1['Pos x'])
    #plt.plot(results2['Pos x'])
    #plt.plot(results3['Pos x'])
    plt.show()

    #    print(Ball.state_vector)

    #print(Ball.results)

    ### ANIMATION ###
    #x = results1['Pos x']
    #y = results1['Pos y']
    #fig, ax = plt.subplots()
    #line, = ax.plot(x,y)

    #def update(num, x, y, line):
    #    line.set_data(x[:num], y[:num])
    #    #line.axes
    #    return line,
    #ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],interval=1, blit=True)
    #plt.show()





if __name__ == '__main__':
	main()
