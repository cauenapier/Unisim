
import numpy as np
import matplotlib.pyplot as plt
from Unisim.bodies.body_3dof import *
from Unisim.environment import *
import copy

import multiprocessing as mp

from Unisim.environment.atmosphere import ISA1976
from Unisim.environment.wind import NoWind
from Unisim.environment.gravity import SimpleNewton
from Unisim.environment.environment import Environment


def run_Body(body, t0, tf, step_size):
    for tt in np.arange(t0,tf,step_size):
        body.step(step_size)
    return body

def main():

    t0 = 0
    #% Position, velocity, acceleration
    x0 = np.zeros(6)
    x0[0] = 6371000 + 10
    #x0[4] = 7600


    Ball1 = Body_RoundEarth(t0,x0, name="Ball1")
    Ball1._set_mass(10)

    x0_2 = copy.deepcopy(x0)
    x0_2[0] = x0_2[0] + 10
    Ball2 = Body_RoundEarth(t0,x0_2, name="Ball2")
    Ball2._set_mass(10)

    x0_3 = copy.deepcopy(x0_2)
    x0_3[0] = x0_3[0] + 10
    Ball3 = Body_RoundEarth(t0,x0_3, name="Ball3")
    Ball3._set_mass(10)

    x0_4 = copy.deepcopy(x0_3)
    x0_4[0] = x0_4[0] + 10
    Ball4 = Body_RoundEarth(t0,x0_4, name="Ball4")
    Ball4._set_mass(10)

    x0_5 = copy.deepcopy(x0_4)
    x0_5[0] = x0_5[0] + 10
    Ball5 = Body_RoundEarth(t0,x0_5, name="Ball5")
    Ball5._set_mass(10)

    x0_6 = copy.deepcopy(x0_5)
    x0_6[0] = x0_6[0] + 10
    Ball6 = Body_RoundEarth(t0,x0_6, name="Ball6")
    Ball6._set_mass(10)


    atmo = ISA1976()
    gravity = SimpleNewton()
    wind = NoWind()
    Env = Environment(atmo, gravity, wind)

    Ball1.set_environment(Env)
    Ball2.set_environment(Env)
    Ball3.set_environment(Env)
    Ball4.set_environment(Env)
    Ball5.set_environment(Env)
    Ball6.set_environment(Env)

    Balls = [Ball1, Ball2, Ball3, Ball4, Ball5, Ball6]


    step_size = 0.01
    t0 = 0
    tf = 100

    pool = mp.Pool(processes=6)

    results = [pool.apply(run_Body, args=(body, t0, tf, step_size)) for body in Balls]
    #for Ball in Balls:
        #for ii in np.arange(0,10, step_size):
            #Ball.step(step_size)



    results1 = pd.DataFrame(results[0].results)
    results2 = pd.DataFrame(results[1].results)
    results3 = pd.DataFrame(results[2].results)

    #print(results[0]._name)
    #print(results[1]._name)
    #print(results[2]._name)


    #results.set_index('Time', inplace=True)
    #results.to_csv("out.csv")
    #print(Ball1.results)


    #plt.plot(results1['Pos x'])
    #plt.plot(results2['Pos x'])
    #plt.plot(results3['Pos x'])
    #plt.show()

    #    print(Ball.state_vector)


    #print(Ball.results)

if __name__ == '__main__':
	main()
