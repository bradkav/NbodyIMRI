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

    
def main(M_1_sel, M_2_sel, a_i_sel, e_i_sel):
    hashes, M_1, M_2, a_i, e_i, N_DM, M_DM, N_step, N_orb, r_soft, method = DIY.load_simulation_list()
    
    criteria = (M_1 == M_1_sel) & (M_2 == M_2_sel) & (a_i == a_i_sel) & (e_i == e_i_sel) & (method == "DKD") & (N_DM == 32768)
    
    hashes_list = hashes[criteria]
    
    M_tot = M_1_sel + M_2_sel
    T_orb = 2*np.pi*np.sqrt(a_i_sel**3/(u.G*M_tot))
    
    print(hashes_list)
    
    energy_loss = np.zeros(len(hashes_list))
    
    plt.figure()
    
    for i, ID in enumerate(hashes_list):
        t, a, e = DIY.load_trajectory(ID)
        #print(r_soft[hashes == ID]/u.PC)
        
        delta_a = (a - a[0])/a
        plt.plot(t, delta_a)
            
        N_o = N_orb[hashes == ID]
        energy_loss[i] = delta_a[-1]/(T_orb*N_o)
        
    #print(energy_loss)
    print("Energy loss rate [s^-1]:", np.mean(energy_loss), "+-", np.std(energy_loss)/np.sqrt(len(hashes_list)))
    
    t = np.linspace(0, N_orb[hashes == hashes_list[0]])
    plt.plot(t, np.mean(energy_loss)*t*T_orb, linestyle='--', color='k')
    
    plt.ylabel(r'$\Delta a/a$')
    plt.xlabel(r'$N_\mathrm{orbit}$')
    
    plt.show()
        


main(M_1_sel = 1000*u.MSUN, M_2_sel = 1*u.MSUN, a_i_sel = 1e-9*u.PC, e_i_sel = 0)
    
