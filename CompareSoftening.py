import warnings
from math import sqrt

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

import DistributionFunctions as DF
import tools

import h5py


import DIY

import units as u


TEST_SNAP_DIR = "test_snapshots"
TEST_FIG_DIR = "test_figures"

    
def main():
    print("> Running simulation...")
    N0 = 4*12000 #DKD needs about 500 steps per orbit in order to be good enough for our purposes
    
    #N0 = 4*4*12000 works for 1000 Msun, 3e-8, 1000 particles
    #N0 = 4*12000 works for 1000 Msun, 1e-9, 1000 particles
    
    N_DM = 1000
    
    a_i = 1e-9*u.PC
    
    M_1 = 1000*u.MSUN
    
    ID  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = int(N0),
                             method="DKD", dynamic_BH=True, soft_factor=0.3333)
    ID2  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = int(N0),
                             method="DKD", dynamic_BH=True, soft_factor=1, init_from_file=ID)
    ID3  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = int(N0),
                             method="DKD", dynamic_BH=True, soft_factor=3, init_from_file=ID)
                             
    ID4  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = int(N0),
                             method="DKD", dynamic_BH=True, soft_factor=10, init_from_file=ID)
                             

    print("> Generating plots...")

    plt.figure()
    
    plt.axhline(0, linestyle='--', color='grey')
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a, label=r"$\epsilon = 3\bar{x}$")
    
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID2)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a, label=r"$\epsilon = \bar{x}$")
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID3)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a, label=r"$\epsilon = \bar{x}$/3")
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID4)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a, label=r"$\epsilon = \bar{x}$/10")
    
    
    plt.legend()
    
    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta a/a$")
    
    plt.savefig(f"{TEST_FIG_DIR}/SofteningComparison.pdf", bbox_inches='tight')
    
    plt.show()
    
main()

    
    
