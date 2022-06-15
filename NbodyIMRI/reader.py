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
    
def load_DMparticles(IDhash, final=True):
    f = h5py.File(f"{NbodyIMRI.snapshot_dir}/{IDhash}.hdf5", 'r')
    
    if (final == True):
        tag = "f"
    else:
        tag = "i"
        
    xDM_list =  np.array(f['data']['xDM_' + tag])
    vDM_list =  np.array(f['data']['vDM_' + tag])
    
    return xDM_list, vDM_list
    
def show_simulation_summary(IDhash):
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
    listfile = f'{NbodyIMRI.snapshot_dir}/SimulationList.txt'
    return np.loadtxt(listfile, unpack=True, usecols=(i,), dtype=dtype)
    
def load_simulation_list():

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
    
def make_plots(IDhash):
    
    f = h5py.File(f"{NbodyIMRI.snapshot_dir}/{IDhash}.hdf5", 'r')
    ts       = np.array(f['data']['t'])
    xBH1_list = np.array(f['data']['xBH1'])
    vBH1_list = np.array(f['data']['vBH1'])
    
    xBH2_list = np.array(f['data']['xBH2'])
    vBH2_list = np.array(f['data']['vBH2'])
    
    xBH_list = xBH2_list - xBH1_list
    vBH_list = vBH2_list - vBH1_list
    
    N_step = len(ts-1)
    
    M_1 = f['data'].attrs["M_1"]*u.MSUN
    M_2 = f['data'].attrs["M_2"]*u.pc
    a_i = f['data'].attrs["a_i"]*u.pc
    dynamic_BH = f['data'].attrs["dynamic"]

    
    f.close()
    
    #if (dynamic_BH == 1):
    M_tot = M_1 + M_2
    #else:
    #    M_tot = 1.0*M_1
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (u.G*M_tot))
    
    a_list, e_list = calc_orbital_elements(xBH_list, vBH_list, M_tot)
    
    #print(xBH_list.shape)
    #print(list(f['data'].keys()))
    
    #----------------------          
    """
    plt.figure()

    plt.loglog(r_vals/u.pc, T_orb/_YR)

    plt.axhline(dt/_YR, linestyle='--', color='k')
    plt.axhline(t_end/_YR, linestyle='--', color='k')
    """
        
    #----------------------
        
    """
    plt.figure()


    rvals_i = np.sqrt(np.sum(xs_i**2, axis=-1))
    rvals = np.sqrt(np.sum(xs**2, axis=-1))

    bins = np.linspace(0, 1.5*r_max/u.pc, 50)

    #r_c = 0.5*(bins[1:] + bins[:-1])

    #P_r = 4*np.pi*r_c**2*SpikeDF.rho_ini(r_c)
    #P_r /= np.trapz(P_r, r_c)
    #axes[1,0].plot(r_c, P_r, linestyle='--', color='k')

    plt.hist(rvals_i/u.pc, bins, alpha=0.75)
    plt.hist(rvals/u.pc, bins, alpha=0.75)   
    """
    #----------------------
    """
    plt.figure()

    plt.plot(xBH_list[:,0], xBH_list[:,1])
    """

    if (a_list[3] < a_list[0]):
        peak_func = np.argmax
    else:
        peak_func = np.argmin

    t_end = ts[-1]
    dt = ts[-1]/N_step
    
    
    N_orb = int(t_end/T_orb)
    
    """
    plt.figure()
    
    i_vals = np.zeros(N_orb + 1, dtype=int)
    i_vals[0] = 0
    for j in range(N_orb):
        inds = ((0.5+j)*T_orb < ts)  & (ts < (1.5+j)*T_orb)
        t_sub = ts[inds]
        t_min = t_sub[peak_func(a_list[inds])]
        #print("Check")
        #print( np.where(ts == t_min))
        i_vals[j+1] = np.where(ts == t_min)[0]
        #print(i_vals[j+1])
    #for j in range(N_orbs)
        #plt.axvline(i_vals[j+1], linestyle='--', color='grey')
    

    #print(ts)
    

    plt.plot(a_list/u.pc)
    plt.axhline(a_i/u.pc, linestyle='--', color='grey')

    plt.xlabel(r"Timestep")
    plt.ylabel(r"$a$")



    plt.tight_layout()
    """
    
    delta_a = a_list[-1] - a_list[0]
    dlogadt = (delta_a/a_list[0])/ts[-1]
    
    
    
    print("> a-dot/a [s^-1]:", dlogadt)
    

    #----------------------

    #i_peaks, _ = signal.find_peaks(a_list)
    #i_peaks = np.arange(len(a_list))
    i_vals = np.arange(len(a_list))
    a_peaks = a_list[i_vals]

    
    plt.figure()

    plt.plot(ts[i_vals]/T_orb, (a_peaks-a_peaks[0])/a_peaks[0])

    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta a/a$")

    plt.title(IDhash)
    plt.tight_layout()

    #----------------------

    #i_peaks, _ = signal.find_peaks(e_list)
    #i_peaks = np.arange(len(a_list))
    e_peaks = e_list[i_vals]

    """
    plt.figure()

    plt.axhline(e_peaks[0], linestyle='--', color='grey')
    plt.plot(ts[i_vals]/T_orb, e_peaks)


    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$e$")

    #---------------------
    
    plt.figure()
    
    plt.plot(norm(xBH1_list)/u.pc)
    plt.xlabel(r"$r_1$")

    #----------------------
    plt.figure()

    r_list = norm(xBH_list)

    plt.axhline(r_list[0]/u.pc, linestyle='--', color='grey')
    plt.plot(ts/_S, r_list/u.pc)


    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$r$ [pc]")

    #----------------------
    
    plt.figure()

    rBH_list = np.sqrt(np.sum(xBH_list**2, axis=-1))
    plt.plot(ts/T0, 1e4*(rBH_list - r_i)/r_i)

    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta r_2/r_2$ [$10^{-4}$]")

    plt.savefig("../test_figures/test_binary_1e2_e.pdf", bbox_inches='tight')

    plt.show()
    """
    
    #plt.show()
    
