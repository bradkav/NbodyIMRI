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
    
    
#def main(N_DM, noDM = False):
def main():
    print("> Running simulation...")
    N0 = 4*12000 #DKD needs about 500 steps per orbit in order to be good enough for our purposes
    N_DM = 1000
    ID  = DIY.run_simulation(M_1 = 1000*u.MSUN, M_2 = 1*u.MSUN, a_i = 1e-9*u.PC, e_i = 0.0, N_DM = N_DM, N_step = int(N0), method="DKD", dynamic_BH=True)
    ID2 = DIY.run_simulation(M_1 = 1000*u.MSUN, M_2 = 1*u.MSUN, a_i = 1e-9*u.PC, e_i = 0.0, N_DM = N_DM, N_step = int(N0/3), method="FR", dynamic_BH=True, init_from_file=ID)
    ID3 = DIY.run_simulation(M_1 = 1000*u.MSUN, M_2 = 1*u.MSUN, a_i = 1e-9*u.PC, e_i = 0.0, N_DM = N_DM, N_step = int(N0/4), method="PEFRL", dynamic_BH=True, init_from_file=ID)

    print("> Generating plots...")
        
    plt.figure()
    
    plt.axhline(0, linestyle='--', color='grey')
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID)
    delta_a = (a_list - a_list[0])/a_list[0]
    
    plt.plot(N_orb, delta_a, label=r"$N_\mathrm{force} = 1$ (DKD)")
    
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID2)
    delta_a = (a_list - a_list[0])/a_list[0]
    
    plt.plot(N_orb, delta_a, label=r"$N_\mathrm{force} = 3$ (FR)")
    
    N_orb, a_list, e_list = DIY.load_trajectory(ID3)
    delta_a = (a_list - a_list[0])/a_list[0]
    
    plt.plot(N_orb, delta_a, label=r"$N_\mathrm{force} = 4$ (PEFRL)")
    
    plt.legend()
    
    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta a/a$")
    
    plt.savefig(f"{TEST_FIG_DIR}/IntegratorComparison.pdf", bbox_inches='tight')
    
    plt.show()

main()
    
    
