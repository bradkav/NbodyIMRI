import warnings
from math import sqrt

from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

from NbodyIMRI import distributionfunctions as DF
from NbodyIMRI import tools, particles
from NbodyIMRI import units as u
import NbodyIMRI

import h5py
import copy




def load_trajectory(IDhash):
    """
    Load a simulation output file and return the binary trajectory (semi-major axis and eccentricity as a function of time)
    
    Parameters:
        IDhash (string):    ID hash of the file you'd like to load.
    
    Returns:
        t (array_like): Output timesteps from the simulation.
        a (array_like): Semi-major axis at each timestep
        e (array_like): Eccentricity at each timestep
    
    """
    
    f = h5py.File(f"{NbodyIMRI.snapshot_dir}/{IDhash}.hdf5", 'r')
    ts       = np.array(f['data']['t'])
    xBH1_list = np.array(f['data']['xBH1'])
    vBH1_list = np.array(f['data']['vBH1'])
    
    xBH2_list = np.array(f['data']['xBH2'])
    vBH2_list = np.array(f['data']['vBH2'])
    
    xBH_list = xBH2_list - xBH1_list
    vBH_list = vBH2_list - vBH1_list
    
    N_step = len(ts-1)
    
    M_1 = f['data'].attrs["M_1"]*u.Msun
    M_2 = f['data'].attrs["M_2"]*u.Msun
    a_i = f['data'].attrs["a_i"]*u.pc
    dynamic_BH = f['data'].attrs["dynamic"]

    
    f.close()
    
    #if (dynamic_BH == 1):
    M_tot = M_1 + M_2
    #else:
    #    M_tot = 1.0*M_1
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (u.G_N*M_tot))
    
    a_list, e_list = tools.calc_orbital_elements(xBH_list, vBH_list, M_tot)
    
    #return ts/T_orb, a_list, e_list
    return ts, a_list, e_list
    
    
    
def load_DMparticles(IDhash, which="initial"):
    """
    Load the configuration of the DM particles. 
    
    Parameters:
        IDhash (string):    ID hash of the file you'd like to load.
        which (string): Determine whether to load "initial" or "final" configuration of DM particles form the file. 
    
    Returns:
        xDM (2d array): Positions of DM particles (N_DM, 3)
        vDM (2d array): Velocities of DM particles (N_DM, 3)
    """
    
    f = h5py.File(f"{NbodyIMRI.snapshot_dir}/{IDhash}.hdf5", 'r')
    
    if (which == "final"):
        tag = "f"
    else:
        tag = "i"
        
    xDM_list =  np.array(f['data']['xDM_' + tag])
    vDM_list =  np.array(f['data']['vDM_' + tag])
    
    return xDM_list, vDM_list
    
def show_simulation_summary(IDhash):
    """
    Print details about the simulation. 
    
    Parameters:
        IDhash (string):    ID hash of the file you'd like to load.
    
    """
    
    f = h5py.File(f"{NbodyIMRI.snapshot_dir}/{IDhash}.hdf5", 'r')
    M_1 = f['data'].attrs["M_1"]*u.Msun
    M_2 = f['data'].attrs["M_2"]*u.Msun
    a_i = f['data'].attrs["a_i"]*u.pc
    e_i = f['data'].attrs['e_i']
    dynamic_BH = f['data'].attrs["dynamic"]

    N_DM = f['data'].attrs['N_DM']
    M_DM = f['data'].attrs['M_DM']*u.Msun
    r_soft = f['data'].attrs['r_soft']*u.pc
    
    f.close()
    
    print(f"> File: {IDhash}")
    print(f">    (M_1, M_2) = ({M_1/u.Msun}, {M_2/u.Msun}) Msun")
    print(f">    (a_i, e_i) = ({a_i/u.pc} pc, {e_i[0]})")
    
    print(f">    N_DM = {N_DM}")
    print(f">    r_soft = {r_soft/u.pc} pc")
    

def load_entry(i, dtype='float'):
    """
    Load a specific column from the "SimulationList.txt" file.
    
    Parameters:
        i (int): index of the column to be loaded
        dtype (float): convert column to a given dtype
    """
    
    listfile = f'{NbodyIMRI.snapshot_dir}/SimulationList.txt'
    return np.loadtxt(listfile, unpack=True, usecols=(i,), dtype=dtype)
    
def load_simulation_list():
    """
    Load the file "SimulationList.txt" into a dictionary.
    
    Returns:
        sim_dict (dictionary): Dictionary of simulations, with the following fields:
            'ID', 'M_1', 'M_2', 'a_i', 'e_i', 'N_DM', 'M_DM', 'Nstep_per_orb', 'N_orb', 'r_soft', 'method',  'rho_6', 'gamma_sp', 'alpha', 'r_t'
    
    """

    hashes = load_entry(0, dtype=str)
    M_1     = load_entry(1)
    M_2     = load_entry(2)
    a_i     = load_entry(3)
    e_i     = load_entry(4)
    N_DM    = load_entry(5, dtype=int)
    M_DM    = load_entry(6)
    Nstep_per_orb      = load_entry(7, dtype=int)
    N_orb   = load_entry(8, dtype=int)
    r_soft  = load_entry(9)
    method  = load_entry(10, dtype=str)
    rho_6   = load_entry(11)
    gamma_sp = load_entry(12)
    alpha   = load_entry(13)
    r_t     = load_entry(14)
    
    sim_dict = {'ID': hashes,
                'M_1'  : M_1, 
                'M_2'  : M_2, 
                'a_i'  : a_i,
                'e_i'  : e_i,
                'N_DM' : N_DM, 
                'M_DM' : M_DM, 
                'Nstep_per_orb': Nstep_per_orb,
                'N_orb': N_orb,
                'r_soft': r_soft, 
                'method': method,
                'rho_6': rho_6,
                'gamma_sp': gamma_sp, 
                'alpha'  : alpha, 
                'r_t'   : r_t}
    
    return sim_dict
    

def plot_trajectory(IDhash):
    t, a, e = load_trajectory(IDhash)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 5))
    
    axes = ax[:]
    
    axes[0].plot(t, (a - a[0])/a[0])
    axes[0].axhline(0, linestyle='--', color='grey')
    
    axes[0].xlabel(r"$t$ [s]")
    axes[0].ylabel(r"$\Delta a/a_i$")
    
    axes[0].plot(t, e - e[0])
    axes[0].axhline(e[0], linestyle='--', color='grey')
    
    axes[0].xlabel(r"$t$ [s]")
    axes[0].ylabel(r"$\Delta e$")
    
    fig.title(IDhash)
    
    return fig
    
    
    
