import warnings
from math import sqrt

from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

import DistributionFunctions as DF
import tools

import h5py

import units as u


theta = 1/(2 - 2**(1/3))

xi  = +0.1786178958448091e+00
lam = -0.2123418310626054e+00
chi = -0.6626458266981849e-01


TEST_SNAP_DIR = "test_snapshots"
TEST_FIG_DIR = "test_figures"

class CentralPotential():
    def __init__(self, M = 1e3*u.MSUN):
        self.M = M
        self.mu = u.G*self.M
        
    def get_gravity_at_point(self, pos, eps=0): 
        rsq = np.atleast_2d(np.sum(pos**2, axis=-1) + eps**2).T
        return -self.mu*pos/rsq**1.5
      
def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))

def calc_orbital_elements(x, v, M_tot):
    x_mag = norm(x)
    v_mag = norm(v)
    mu = u.G*M_tot

    a = (2/x_mag - v_mag**2/mu)**-1
        #https://astronomy.stackexchange.com/questions/29005/calculation-of-eccentricity-of-orbit-from-velocity-and-radius
    h = np.cross(x,v)
    e_vec = np.cross(v,h)/mu - x/np.atleast_2d(x_mag).T
    e = norm(e_vec)
        
    return a, e

    
class particles():
    def __init__(self, M_1, M_2, N_DM=2, M_DM = 0, dynamic_BH=True, r_soft=0):
    
        self.M_1 = M_1
        self.M_2 = M_2
        
        self.M_DM = M_DM
        self.N_DM = N_DM
        
        #self.r_isco1 = 6*u.G*M_1/u.C**2
        #self.r_isco2 = 6*u.G*M_2/u.C**2
        
        #self.r_isco2 = 10*1.8849555921538754e-10*u.PC
        self.r_isco2 = r_soft
        
        self.r_isco1  = 0
        #self.r_isco2  = 0
    
        self.xBH1 = np.zeros((1,3), dtype=np.float64)
        self.vBH1 = np.zeros((1,3), dtype=np.float64)
        
        self.xBH2 = np.zeros((1,3), dtype=np.float64)
        self.vBH2 = np.zeros((1,3), dtype=np.float64)
        
        self.xDM = np.zeros((N_DM, 3))
        self.vDM = np.zeros((N_DM, 3))
        
        self.BH = CentralPotential(M_1)
    
        self.dvdtBH1 = None
        self.dvdtBH2 = None
        self.dxdtDM = None
        
        self.dynamic_BH = dynamic_BH
        
    def update_acc(self):
        #Initial accelerations
        dx  = (self.xDM - self.xBH2)
        r = norm(dx)
        #if (np.sum(r < self.r_isco2)):
        #    print("Close encounter!")
        #rsq = np.clip(np.atleast_2d(r**2).T, self.r_isco2**2, 1e30)
        rsq = np.atleast_2d(r**2).T
        acc_DM2 = -u.G*self.M_2*dx/rsq**1.5
        
        #print((rsq < self.r_isco2**2).shape)
        inds = rsq < self.r_isco2**2
        
        #if (np.sum(inds) > 0):
        #    print(np.sum(inds))
        acc_DM2[inds.flatten(),:] *= 0.0
    
        dx = self.xDM - self.xBH1
        acc_DM1 = self.BH.get_gravity_at_point(dx, eps=self.r_isco1)
        
        dx = self.xBH2 - self.xBH1
        acc_BH = self.BH.get_gravity_at_point(dx, eps=self.r_isco1)
    
        if (self.dynamic_BH):
            self.dvdtBH1 = -(self.M_2/self.M_1)*acc_BH
        else:
            self.dvdtBH1 = 0.0
            
        self.dvdtBH2 = acc_BH - (1/self.M_2)*np.sum(self.M_DM*acc_DM2, axis=0)
        self.dvdtDM  = acc_DM1 + acc_DM2

        
    def xstep(self, h):
        if (self.dynamic_BH):
            self.xBH1 += self.vBH1*h
        self.xBH2 += self.vBH2*h
        self.xDM += self.vDM*h


    def vstep(self, h):
        self.update_acc()
        #self.dvdtBH1 = -u.G*self.M_1*self.xBH/norm(self.xBH)**3
        if (self.dynamic_BH):
            self.vBH1 += self.dvdtBH1*h
        self.vBH2 += self.dvdtBH2*h
        self.vDM += self.dvdtDM*h

        
    
    
def run_simulation(M_1, M_2, a_i, e_i, N_DM = 0, gamma = 7/3, r_max = 1e-6*u.PC, N_step = 10000, method = "DKD", dynamic_BH = True, init_from_file=None, soft_factor = 10, add_to_list = True):

 
    #Initialise BH properties
    r_i = a_i * ( 1 + e_i)
    
    if (dynamic_BH):
        M_tot = M_1 + M_2
    else:
        M_tot = M_1
    mu = u.G*M_tot
    v_i = np.sqrt( mu * (2.0/r_i - 1.0/a_i) )
    
    r_p = a_i * (1 - e_i)
    v_p = np.sqrt( mu * (2.0/r_p - 1.0/a_i) )
    
    # Simulation parameters
    N_orb = 100
    
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (u.G*M_tot))
    t_end = N_orb*T_orb
    dt = t_end/N_step
    
    print(f"> (M_1, M_2) = ({M_1/u.MSUN}, {M_2/u.MSUN}) Msun")
    
    #N = 0
    #r_soft = (N**2*dt**2*u.G*M_2)**(1/3) #DO NOT CHANGE _ THIS SEEMS TO WORK!!!
    #r_soft = 2*np.pi*N*a_i*dt/T_orb

    rho_sp = 226*u.MSUN/u.PC**3
    r_6   = 1e-6*u.PC
    k = (3-gamma)*(0.2)**(3-gamma)/(2*np.pi)
    rho_6 = (rho_sp)**(1-gamma/3)*(k*M_1)**(gamma/3)*r_6**-gamma
    print(f"> rho_6 [Msun/pc^3]: {rho_6/(u.MSUN/u.PC**3):e}")
    # Initialise central potential and binary orbit
    SpikeDF = DF.SpikeDistribution(M_1/u.MSUN, rho_6=rho_6/(u.MSUN/u.PC**3), gamma_sp=gamma)
       
    #l_avg = (SpikeDF.rho_ini(r_i/u.PC)*u.MSUN/u.PC**3/M_DM)**(-1/3)
    #print("> Mean DM particle separation [pc]:", l_avg/u.PC)
    
    #b_90 = u.G*M_2/v_i**2    
    #r_min = 0.5*(np.sqrt(4*b_90**2 + 4*l_avg**2) - 2*b_90)
    #print("> Typical r_min [pc]:", r_min/u.PC)
    
    #r_soft = l_avg/10
    #print("> Drift per timestep [pc]: ", v_i*dt/u.PC)
    #r_soft = 10000*6*u.G*M_2/u.C**2
    #r_soft = 10*v_p*dt
    
        
    #dt_est = np.sqrt(0.05*r_soft**3/(u.G*M_2))
    #print("> XXX/T_orb: ", dt_est/T_orb)
        
    
    #-----------------------------
    
    if (dynamic_BH):
        factor = M_2/M_tot
    else:
        factor = 0
        
    print(f"> N_DM = {N_DM}")
        
    print("> Initialising...")
    if (init_from_file is None):
    
        if (N_DM > 0):
            M_spike = SpikeDF.M_DM_ini(r_max/u.PC)*u.MSUN
            M_DM    = (M_spike/N_DM)
            
            l_avg = (SpikeDF.rho_ini(r_i/u.PC)*u.MSUN/u.PC**3/M_DM)**(-1/3)
            print("> Mean DM particle separation [pc]:", l_avg/u.PC)
            #r_soft = l_avg/10
            r_soft = l_avg/soft_factor
            
            
            
        else:
            N_DM = 2 #Keep N_DM = 2 so that all the arrays work as expected...
            M_DM = 0.0
            r_soft = 0.0

        
        p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM, dynamic_BH=dynamic_BH, r_soft=r_soft)
    
        p.xBH1[:] = np.atleast_2d([-r_i*factor,   0, 0])
        p.xBH2[:] = np.atleast_2d([r_i*(1-factor),   0, 0])
    
        p.vBH1[:] = np.atleast_2d([0.0, v_i*factor, 0])
        p.vBH2[:] = np.atleast_2d([0.0, -v_i*(1-factor), 0])
    
        #Initialise DM properties

        r, v = SpikeDF.draw_particle(r_max/u.PC, N = N_DM)

        for i in range(N_DM):
            rhat = tools.get_random_direction()
            vhat = tools.get_random_direction()
            p.xDM[i,:] = r[i]*rhat * u.PC
            p.vDM[i,:] = v[i]*vhat * u.PC/u.MYR
        
        p.xDM += p.xBH1
        p.vDM += p.vBH1
    
    else:
        print(f"> Re-starting from {init_from_file}...")
        f = h5py.File(f"{TEST_SNAP_DIR}/{init_from_file}.hdf5", 'r')
        N_DM = f['data'].attrs["N_DM"]
        M_DM = f['data'].attrs["M_DM"]*u.MSUN
        #r_soft = f['data'].attrs["r_soft"]*u.PC
        
        if (N_DM <= 0):
            N_DM = 2 #Keep N_DM = 2 so that all the arrays work as expected...
            M_DM = 0.0
            r_soft = 0.0
        else:
            l_avg = (SpikeDF.rho_ini(r_i/u.PC)*u.MSUN/u.PC**3/M_DM)**(-1/3)
            r_soft = l_avg/soft_factor
        
        p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM, dynamic_BH=dynamic_BH, r_soft=r_soft)
        
        #print(f['data']['xBH1'])
        p.xBH1 = np.array(f['data']['xBH1'])[:1,:]
        p.vBH1 = np.array(f['data']['vBH1'])[:1,:]
        
        #print(p.xBH1/u.PC)
        
        p.xBH2 = np.array(f['data']['xBH2'])[:1,:]
        p.vBH2 = np.array(f['data']['vBH2'])[:1,:]
        
        p.xDM  = np.array(f['data']['xDM_i'])
        p.vDM  = np.array(f['data']['vDM_i'])
        
        
    print("> Softening length [pc]: ", p.r_isco2/u.PC)
    dt_est = np.sqrt(0.05*r_soft**3/(u.G*M_2))
    print("> dt_est [s]:", dt_est/u.S)
    
    """
    if (dt_est/dt < 100):
        dt = dt_est/100
        N_step_per_orb = int(T_orb/dt)
        dt =  T_orb/N_step_per_orb
        N_step = N_step_per_orb*N_orb
    """
    

    elements = calc_orbital_elements(p.xBH1 - p.xBH2 , p.vBH1 - p.vBH2, M_tot)
    print("> Orbital elements (a_i, e_i):", float(elements[0]/u.PC), ",", float(elements[1]))
    print("> DM pseudoparticle mass [Msun]:", M_DM/u.MSUN)
    
    ts = np.linspace(0, t_end, N_step)
    xBH_list = np.zeros((N_step, 3))
    
    
    print("> r_isco(1) [pc]: ", 6*u.G*M_1/u.C**2/u.PC)
    print("> r_isco(2) [pc]: ", 6*u.G*M_2/u.C**2/u.PC)
    
    print("> dt [s]:", dt/u.S)
    print("> t_end [s]:", t_end/u.S)
    print("> T_orb [s]:", T_orb/u.S)
    print("> Steps per orbit:", T_orb/dt)
    
    
    #-----------------------------
    IDhash = tools.generate_hash()
    print("> Hash: " + IDhash)
    f = h5py.File(f"{TEST_SNAP_DIR}/{IDhash}.hdf5", "w")
    grp = f.create_group("data")
    grp.attrs['M_1'] = M_1/u.MSUN
    grp.attrs['M_2'] = M_2/u.MSUN
    grp.attrs['a_i'] = a_i/u.PC
    grp.attrs['e_i'] = e_i
    grp.attrs['N_DM'] = N_DM
    grp.attrs['M_DM'] = M_DM/u.MSUN
    grp.attrs['r_soft'] = r_soft/u.PC
    if (dynamic_BH):
        grp.attrs['dynamic'] = 1
    else:
        grp.attrs['dynamic'] = 0
        

    #Save more stuff here...
    
    datatype = np.float64
    t_data   = grp.create_dataset("t", (N_step,), dtype=datatype, compression="gzip")
    xBH1_data = grp.create_dataset("xBH1", (N_step,3), dtype=datatype, compression="gzip")
    vBH1_data = grp.create_dataset("vBH1", (N_step,3), dtype=datatype, compression="gzip")
    
    xBH2_data = grp.create_dataset("xBH2", (N_step,3), dtype=datatype, compression="gzip")
    vBH2_data = grp.create_dataset("vBH2", (N_step,3), dtype=datatype, compression="gzip")
    
    xDM_i_data = grp.create_dataset("xDM_i", (N_DM,3), dtype=datatype, compression="gzip")
    xDM_f_data = grp.create_dataset("xDM_f", (N_DM,3), dtype=datatype, compression="gzip")
    
    vDM_i_data = grp.create_dataset("vDM_i", (N_DM,3), dtype=datatype, compression="gzip")
    vDM_f_data = grp.create_dataset("vDM_f", (N_DM,3), dtype=datatype, compression="gzip")
    
    xBH1_list = np.zeros((N_step, 3))
    vBH1_list = np.zeros((N_step, 3))
    
    xBH2_list = np.zeros((N_step, 3))
    vBH2_list = np.zeros((N_step, 3))
    t_list   = np.zeros(N_step)
    N_update = 10000
    
    rmin_list = np.zeros(N_step)
    rmax_list = np.zeros(N_step)
    vmin_list = np.zeros(N_step)
    vmax_list = np.zeros(N_step)
    
    xDM_i_data[:,:] = 1.0*p.xDM
    vDM_i_data[:,:] = 1.0*p.vDM
    
    #--------------------------
    print("> Simulating...")

    
    for it in tqdm(range(N_step)):
           
        t_list[it]     = ts[it]     
        xBH1_list[it,:] = p.xBH1
        vBH1_list[it,:] = p.vBH1
        
        xBH2_list[it,:] = p.xBH2
        vBH2_list[it,:] = p.vBH2
        
        r_BH = norm(p.xBH2 - p.xDM)
        v_BH = norm(p.vBH2 - p.vDM)
        rmin_list[it] = np.min(r_BH)
        rmax_list[it] = np.max(r_BH)
        vmin_list[it] = np.min(v_BH)
        vmax_list[it] = np.max(v_BH)
        
        if (it%N_update == 0):
            t_data[:]     = 1.0*t_list
            
            xBH1_data[:,:] = 1.0*xBH1_list
            vBH1_data[:,:] = 1.0*vBH1_list
            
            xBH2_data[:,:] = 1.0*xBH2_list
            vBH2_data[:,:] = 1.0*vBH2_list     
            
    
        #https://arxiv.org/abs/2007.05308
        if (method == "DKD"):
            
            p.xstep(0.5*dt)
            p.vstep(1.0*dt)
            p.xstep(0.5*dt)

            
        elif (method == "FR"):
            
            p.xstep(theta*dt/2)
            
            p.vstep(theta*dt)
            
            p.xstep((1-theta)*dt/2)
            
            p.vstep((1-2*theta)*dt)
            
            p.xstep((1-theta)*dt/2)
            
            p.vstep(theta*dt)
            
            p.xstep(theta*dt/2)
            
        elif (method == "PEFRL"):
            
            p.xstep(xi*dt)
            
            p.vstep((1-2*lam)*dt/2)
            
            p.xstep(chi*dt)
            
            p.vstep(lam*dt)
            
            p.xstep((1-2*(chi + xi))*dt)
            
            p.vstep(lam*dt)
            
            p.xstep(chi*dt)
            
            p.vstep((1-2*lam)*dt/2)
            
            p.xstep(xi*dt)
        
            
    #One final update of the output data
    t_data[:]     = 1.0*t_list
    
    xBH1_data[:,:] = 1.0*xBH1_list
    vBH1_data[:,:] = 1.0*vBH1_list
    
    xBH2_data[:,:] = 1.0*xBH2_list
    vBH2_data[:,:] = 1.0*vBH2_list
    
    xDM_f_data[:,:] = 1.0*p.xDM
    vDM_f_data[:,:] = 1.0*p.vDM
    
    print("> Simulation completed.")
    
    listfile = f'{TEST_SNAP_DIR}/SimulationList.txt'
    hdrtxt = "Columns: IDhash, M_1/MSUN, M_2/MSUN, a_i/PC, e_i, N_DM, M_DM/MSUN, N_step, N_orb, r_soft/PC, method"
    
    meta_data = np.array([IDhash, M_1/u.MSUN, M_2/u.MSUN, a_i/u.PC, e_i, N_DM, M_DM/u.MSUN, N_step, N_orb, r_soft/u.PC, method])
    
    meta_data = np.reshape(meta_data, (1,  len(meta_data)))
    
    
    if (add_to_list):
        if (os.path.isfile(listfile)):
                with open(listfile,'a') as g:
                    np.savetxt(g, meta_data, fmt='%s')
                g.close()
        else:
                np.savetxt(listfile, meta_data, header=hdrtxt, fmt='%s')
    
    
    f.close()
    
    """
    plt.figure()
    
    plt.semilogy(t_list/T_orb, rmin_list/u.PC)
    plt.semilogy(t_list/T_orb, rmax_list/u.PC)
    plt.axhline(r_soft/u.PC, linestyle='--', color='r')
    
    plt.ylabel(r'$r_\mathrm{min}$ [pc]')
    
    plt.figure()
    
    plt.semilogy(t_list/T_orb, vmin_list/(u.PC/u.S))
    plt.semilogy(t_list/T_orb, vmax_list/(u.PC/u.S))
    plt.axhline(v_i/(u.PC/u.S), linestyle='--', color='r')
    
    plt.ylabel(r'$v_\mathrm{max}$ [pc]')
    
    #plt.show()
    """
    
    return IDhash
    

def load_trajectory(IDhash):
    f = h5py.File(f"{TEST_SNAP_DIR}/{IDhash}.hdf5", 'r')
    ts       = np.array(f['data']['t'])
    xBH1_list = np.array(f['data']['xBH1'])
    vBH1_list = np.array(f['data']['vBH1'])
    
    xBH2_list = np.array(f['data']['xBH2'])
    vBH2_list = np.array(f['data']['vBH2'])
    
    xBH_list = xBH2_list - xBH1_list
    vBH_list = vBH2_list - vBH1_list
    
    N_step = len(ts-1)
    
    M_1 = f['data'].attrs["M_1"]*u.MSUN
    M_2 = f['data'].attrs["M_2"]*u.MSUN
    a_i = f['data'].attrs["a_i"]*u.PC
    dynamic_BH = f['data'].attrs["dynamic"]

    
    f.close()
    
    if (dynamic_BH == 1):
        M_tot = M_1 + M_2
    else:
        M_tot = 1.0*M_1
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (u.G*M_tot))
    
    a_list, e_list = calc_orbital_elements(xBH_list, vBH_list, M_tot)
    
    return ts/T_orb, a_list, e_list
    

def load_entry(i, dtype='float'):
    listfile = f'{TEST_SNAP_DIR}/SimulationList.txt'
    return np.loadtxt(listfile, unpack=True, usecols=(i,), dtype=dtype)
    
def load_simulation_list():

    hashes = load_entry(0, dtype=str)
    M_1     = load_entry(1)*u.MSUN
    M_2     = load_entry(2)*u.MSUN
    a_i     = load_entry(3)*u.PC
    e_i     = load_entry(4)
    N_DM    = load_entry(5, dtype=int)
    M_DM    = load_entry(6)*u.MSUN
    N_step  = load_entry(7, dtype=int)
    N_orb   = load_entry(8, dtype=int)
    r_soft  = load_entry(9)*u.PC
    method  = load_entry(10, dtype=str)
    
    return hashes, M_1, M_2, a_i, e_i, N_DM, M_DM, N_step, N_orb, r_soft, method

    
def make_plots(IDhash):
    
    f = h5py.File(f"{TEST_SNAP_DIR}/{IDhash}.hdf5", 'r')
    ts       = np.array(f['data']['t'])
    xBH1_list = np.array(f['data']['xBH1'])
    vBH1_list = np.array(f['data']['vBH1'])
    
    xBH2_list = np.array(f['data']['xBH2'])
    vBH2_list = np.array(f['data']['vBH2'])
    
    xBH_list = xBH2_list - xBH1_list
    vBH_list = vBH2_list - vBH1_list
    
    N_step = len(ts-1)
    
    M_1 = f['data'].attrs["M_1"]*u.MSUN
    M_2 = f['data'].attrs["M_2"]*u.PC
    a_i = f['data'].attrs["a_i"]*u.PC
    dynamic_BH = f['data'].attrs["dynamic"]

    
    f.close()
    
    if (dynamic_BH == 1):
        M_tot = M_1 + M_2
    else:
        M_tot = 1.0*M_1
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (u.G*M_tot))
    
    a_list, e_list = calc_orbital_elements(xBH_list, vBH_list, M_tot)
    
    #print(xBH_list.shape)
    #print(list(f['data'].keys()))
    
    #----------------------          
    """
    plt.figure()

    plt.loglog(r_vals/u.PC, T_orb/_YR)

    plt.axhline(dt/_YR, linestyle='--', color='k')
    plt.axhline(t_end/_YR, linestyle='--', color='k')
    """
        
    #----------------------
        
    """
    plt.figure()


    rvals_i = np.sqrt(np.sum(xs_i**2, axis=-1))
    rvals = np.sqrt(np.sum(xs**2, axis=-1))

    bins = np.linspace(0, 1.5*r_max/u.PC, 50)

    #r_c = 0.5*(bins[1:] + bins[:-1])

    #P_r = 4*np.pi*r_c**2*SpikeDF.rho_ini(r_c)
    #P_r /= np.trapz(P_r, r_c)
    #axes[1,0].plot(r_c, P_r, linestyle='--', color='k')

    plt.hist(rvals_i/u.PC, bins, alpha=0.75)
    plt.hist(rvals/u.PC, bins, alpha=0.75)   
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
    

    plt.plot(a_list/u.PC)
    plt.axhline(a_i/u.PC, linestyle='--', color='grey')

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
    
    plt.plot(norm(xBH1_list)/u.PC)
    plt.xlabel(r"$r_1$")

    #----------------------
    plt.figure()

    r_list = norm(xBH_list)

    plt.axhline(r_list[0]/u.PC, linestyle='--', color='grey')
    plt.plot(ts/_S, r_list/u.PC)


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
    
