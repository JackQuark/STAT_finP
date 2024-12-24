# _summary_
# ==================================================
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from time import perf_counter

from Functions import Get_Functions_details
from SFO import SFO
from HHO import HHO
# ==================================================

# ==================================================

def main():
    # Parameters
    Nhawks = 30  # Number of search agents
    NSF = 5  # Number of sailfish
    NSA = 100  # Number of sardines
    T = 500  # Maximum number of iterations
    Function_name = "F12"
    fobj, lb, ub, ndim = Get_Functions_details(Function_name)


    hho = HHO(fobj, Nhawks, lb, ub, ndim, T)
    
    sfo = SFO(fobj, NSF, NSA, lb, ub, ndim, T)

    # hho.Visualization()
    # return
    Rabbit_Energy, Rabbit_Location, CNVG = hho.run()
    elite_SFpos, elite_SFfit, elite_SFfitrec = sfo.run()
        
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax[0].semilogy(CNVG, label='HHO')
    ax[1].semilogy(elite_SFfitrec, label='SFO')
    
    ax[0].set_title("HHO")
    ax[0].set_ylabel("fitness")
    ax[1].set_title("SFO")
    ax[1].set_ylabel("fitness")
    ax[1].set_xlabel('Iteration')
    ax[1].set_xlim(0, T)    
    
    # fig2 = plt.figure(figsize=(6, 6))
    # ax3d = fig2.add_subplot(111, projection='3d')

    # x = np.linspace(-5, 5, 100)
    # y = np.linspace(-5, 5, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = fobj(np.array([X, Y]))
    # ax3d.plot_surface(X, Y, Z, cmap='coolwarm', vmin=0, vmax=5)
    # ax3d.set_xlim(-5, 5)
    # ax3d.set_ylim(-5, 5)
    # ax3d.set_zlim(-1, 5)
    plt.savefig(f"./results/{Function_name}.png")
    plt.show()
    
    print(f"The best location of HHO is: {Rabbit_Location}")
    print(f"The best fitness of HHO is: {Rabbit_Energy}")
    print(f"The best location of SFO is: {elite_SFpos}")
    print(f"The best fitness of SFO is: {elite_SFfit}")
    
    

# ==================================================

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))