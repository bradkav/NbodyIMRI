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
    N0 = 10000 #DKD needs about 500 steps per orbit in order to be good enough for our purposes
    
    #N0 = 4*4*12000 works for 1000 Msun, 3e-8, 1000 particles
    #N0 = 4*12000 works for 1000 Msun, 1e-9, 1000 particles
    
    #Nstep = 100000, N_DM = 10000, soft_factor=100 works well!
    
    #RERUN THIS - IT KIND OF WORKS!!!
    
    N_DM = 10000
    
    a_i = 3e-8*u.PC
    
    M_1 = 1000*u.MSUN
    
    #ID_in = "EF777"
    

    #ID1 = "f2144"
    #ID2 = "270b2"
    #ID3 = "ceDf8"
    
    
    ID1  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = N0,
                             method="PEFRL", dynamic_BH=True, soft_factor=3,
                             add_to_list = False)
    
    """
    ID2  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = N0,
                             method="DKD", dynamic_BH=True, soft_factor=3, 
                             add_to_list = False, init_from_file = ID1)
                             
    ID3  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
                             a_i = a_i, e_i = 0.0, 
                             N_DM = N_DM, N_step = N0,
                             method="DKD", dynamic_BH=True, soft_factor=10, 
                             add_to_list = False, init_from_file = ID1)
    
    """
    #ID3  = DIY.run_simulation(M_1 = M_1, M_2 = 1*u.MSUN,
    #                         a_i = a_i, e_i = 0.0, 
    #                         N_DM = N_DM*100, N_step = N0,
    #                         method="DKD", dynamic_BH=True, soft_factor=10, 
    #                         add_to_list = False)
                    

            
    #ID1 = "FB4cB"
    #ID2 = "7D752"
    #ID3 = "A3962"              


    print("> Generating plots...")

    plt.figure()
    
    plt.axhline(0, linestyle='--', color='grey')
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID1)
    delta_a = (a_list - a_list[0])/a_list[0]
    #plt.plot(N_orb, delta_a, label=r"$N_\mathrm{step} = 10000$")
    plt.plot(N_orb, delta_a, label=r"$N_\mathrm{step} = 10000$")
    
    """
    N_orb, a_list, e_list = DIY.load_trajectory(ID2)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a, label=r"$N_\mathrm{step} = 30000$")
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID3)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a, label=r"$N_\mathrm{step} = 100000$")
    """
    
    """
    N_orb, a_list, e_list = DIY.load_trajectory(ID3)
    delta_a = (a_list - a_list[0])/a_list[0]
    plt.plot(N_orb, delta_a,  label=r"$N_\mathrm{step} = 10^6$")
    """
    
    
    plt.legend()
    
    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta a/a$")
    
#plt.savefig(f"{TEST_FIG_DIR}/SofteningComparison.pdf", bbox_inches='tight')
    
    plt.show()
    
main()

    
    
