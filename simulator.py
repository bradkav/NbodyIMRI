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

import copy

import particles


# Leapfrog stepsize parameters
#-----------------------------
theta = 1/(2 - 2**(1/3))

xi  = +0.1786178958448091e+00
lam = -0.2123418310626054e+00
chi = -0.6626458266981849e-01
#-----------------------------

OUT_SNAP_DIR = "test_snapshots"
OUT_FIG_DIR = "test_figures"

    

    
class simulator():
    def __init__(self, particle_set, r_soft_sq = 0.0, soft_method="plummer"):
            
        self.p = copy.deepcopy(particle_set)
        self.r_soft_sq = r_soft_sq
        self.IDhash = tools.generate_hash()    
        self.soft_method = soft_method    
                    
            
    def full_step(self, dt, method="DKD"):
        """
        Perform a full leapfrog step, with timestep dt
        """
        
        #https://arxiv.org/abs/2007.05308
        if (method == "DKD"):
            self.p.xstep(0.5*dt)
            self.update_acceleration()
            self.p.vstep(1.0*dt)
            self.p.xstep(0.5*dt)

        
        elif (method == "FR"):
            self.p.xstep(theta*dt/2)
            self.update_acceleration()
            self.p.vstep(theta*dt)
            self.p.xstep((1-theta)*dt/2)
            self.update_acceleration()
            self.p.vstep((1-2*theta)*dt)
            self.p.xstep((1-theta)*dt/2)
            self.update_acceleration()
            self.p.vstep(theta*dt)
            self.p.xstep(theta*dt/2)
        
        elif (method == "PEFRL"):
            self.p.xstep(xi*dt)
            self.update_acceleration()
            self.p.vstep((1-2*lam)*dt/2)
            self.p.xstep(chi*dt)
            self.update_acceleration()
            self.p.vstep(lam*dt)
            self.p.xstep((1-2*(chi + xi))*dt)
            self.update_acceleration()
            self.p.vstep(lam*dt)
            self.p.xstep(chi*dt)
            self.update_acceleration()
            self.p.vstep((1-2*lam)*dt/2)
            self.p.xstep(xi*dt)
        
                
    def update_acceleration(self):
        
        dx1     = (self.p.xDM - self.p.xBH1)
        r1      = np.linalg.norm(dx1, axis=-1, keepdims=True)
        dx1    /= r1
        r1_sq   = r1**2
        
        
        if (self.soft_method == "plummer"):
            acc_DM1 = -u.G_N*self.p.M_1*dx1*(r1_sq + self.r_soft_sq)**-1
            
        elif (self.soft_method == "plummer2"):
            acc_DM1 = -u.G_N*self.p.M_1*r1*(dx1/2)*(2*r1_sq + 5*self.r_soft_sq)*(r1_sq + self.r_soft_sq)**(-5/2)
            
        elif (self.soft_method == "uniform"):
            x = np.sqrt(r1_sq/self.r_soft_sq)
            acc_DM1 = -u.G_N*self.p.M_1*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) > 1):
                inds = inds.flatten()
                acc_DM1[inds] = -u.G_N*self.p.M_1*dx1[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq)
                
        elif (self.soft_method == "truncate"):
            r1_sq = np.clip(r1_sq, self.r_soft_sq, 1e50)
            acc_DM1 = -u.G_N*self.p.M_1*dx1/r1_sq
            
        else:
            print("WHAT?!")
        
        dx2     = (self.p.xDM - self.p.xBH2)
        r2      = np.linalg.norm(dx2, axis=-1, keepdims=True)
        dx2     /= r2
        r2_sq   = r2**2 
        
        #Consider also Eq. (2.227) in Binney and Tremaine
        #https://arxiv.org/pdf/2104.05643.pdf
        
        if (self.soft_method == "plummer"):
            acc_DM2 = -u.G_N*self.p.M_2*dx2*(r2_sq + self.r_soft_sq)**-1
            
        elif (self.soft_method == "plummer2"):
            acc_DM2 = -u.G_N*self.p.M_2*r2*(dx2/2)*(2*r2_sq + 5*self.r_soft_sq)*(r2_sq + self.r_soft_sq)**(-5/2)
            
        elif (self.soft_method == "uniform"):
            x = np.sqrt(r2_sq/self.r_soft_sq)
            acc_DM2 = -u.G_N*self.p.M_2*dx2*(r2_sq)**-1
            inds = x < 1
            if (np.sum(inds) > 1):
                inds = inds.flatten()
                acc_DM2[inds] = -u.G_N*self.p.M_2*dx2[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq)
                
        elif (self.soft_method == "truncate"):
            r2_sq = np.clip(r2_sq, self.r_soft_sq, 1e50)
            acc_DM2 = -u.G_N*self.p.M_2*dx2/r2_sq
            
        else:
            print("WHAT?!")
        
        dx12    = (self.p.xBH1 - self.p.xBH2)
        r12_sq  = np.linalg.norm(dx12, axis=-1, keepdims=True)**2
        acc_BH = -u.G_N*self.p.M_2*dx12*(r12_sq)**-1.5

        
        if (self.p.dynamic_BH):
            self.p.dvdtBH1 = acc_BH
        else:
            self.p.dvdtBH1 = 0.0
            
        self.p.dvdtBH2 = -(self.p.M_1/self.p.M_2)*acc_BH - (self.p.M_DM/self.p.M_2)*np.sum(acc_DM2, axis=0)
        self.p.dvdtDM  = acc_DM1 + acc_DM2
        
    
            
    def run_simulation(self, dt, t_end, method="DKD", save_to_file = True, add_to_list = False, show_progress=False):
        #--------------------------
        print("> Simulating...")
        
        self.t_end   = t_end
        self.dt      = dt
        N_step = int(np.ceil(t_end/dt)) 

        a_i, e_i = self.p.orbital_elements()
        self.a_i = float(a_i)
        self.e_i = float(e_i)
        T_orb    = self.p.T_orb()        
        
        self.method = method
        
        if (save_to_file):
            fname = f"{OUT_SNAP_DIR}/{self.IDhash}.hdf5"
            f = self.open_outputfile(fname, N_step)
            
        self.xBH1_list = np.zeros((N_step, 3))
        self.vBH1_list = np.zeros((N_step, 3))
    
        self.xBH2_list = np.zeros((N_step, 3))
        self.vBH2_list = np.zeros((N_step, 3))
        self.ts        = np.linspace(0, t_end, N_step)
        
        self.rmin_list = np.zeros(N_step)
        self.vrel_list = np.zeros(N_step)
        
        if (save_to_file):
            self.t_data[:] = 1.0*self.ts
        
            self.xDM_i_data[:,:] = 1.0*self.p.xDM
            self.vDM_i_data[:,:] = 1.0*self.p.vDM
        
        N_update  = 100000
    

        stepper = lambda x: x
        if (show_progress):
            stepper = tqdm
        for it in stepper(range(N_step)):
               
            self.xBH1_list[it,:] = self.p.xBH1
            self.vBH1_list[it,:] = self.p.vBH1
        
            self.xBH2_list[it,:] = self.p.xBH2
            self.vBH2_list[it,:] = self.p.vBH2
            
            #BJK: Remove this for production
            #rDM = tools.norm(self.p.xBH2 - self.p.xDM)
            #ic  = np.argmin(rDM)
            #self.rmin_list[it] = rDM[ic]
            #self.vrel_list[it] = tools.norm(self.p.vBH2 - self.p.vDM[ic, :])
        
            if ((it%N_update == 0) and (save_to_file)):
        
                self.xBH1_data[:,:] = 1.0*self.xBH1_list
                self.vBH1_data[:,:] = 1.0*self.vBH1_list
            
                self.xBH2_data[:,:] = 1.0*self.xBH2_list
                self.vBH2_data[:,:] = 1.0*self.vBH2_list     
            
    
            #Step forward by dt
            self.full_step(dt, method)
        
        if (save_to_file):
            #One final update of the output data    
            self.xBH1_data[:,:] = 1.0*self.xBH1_list
            self.vBH1_data[:,:] = 1.0*self.vBH1_list
    
            self.xBH2_data[:,:] = 1.0*self.xBH2_list
            self.vBH2_data[:,:] = 1.0*self.vBH2_list
    
            self.xDM_f_data[:,:] = 1.0*self.p.xDM
            self.vDM_f_data[:,:] = 1.0*self.p.vDM
    
        print("> Simulation completed.")
    
        if (add_to_list):
            self.output_metadata()
    
    
        if (save_to_file):
            f.close()
        
        

    def open_outputfile(self, fname, N_step):
        f = h5py.File(fname, "w")
        grp = f.create_group("data")
        grp.attrs['M_1'] = self.p.M_1/u.Msun
        grp.attrs['M_2'] = self.p.M_2/u.Msun
        
        a_i, e_i = self.p.orbital_elements()
        
        #Still need to add other stuff here!
        
        grp.attrs['a_i'] = a_i/u.pc
        grp.attrs['e_i'] = e_i
        grp.attrs['N_DM'] = self.p.N_DM
        grp.attrs['M_DM'] = self.p.M_DM/u.Msun
        grp.attrs['r_soft'] = np.sqrt(self.r_soft_sq)/u.pc
        if (self.p.dynamic_BH):
            grp.attrs['dynamic'] = 1
        else:
            grp.attrs['dynamic'] = 0
        
    
        datatype = np.float64
        self.t_data   = grp.create_dataset("t", (N_step,), dtype=datatype, compression="gzip")
        self.xBH1_data = grp.create_dataset("xBH1", (N_step,3), dtype=datatype, compression="gzip")
        self.vBH1_data = grp.create_dataset("vBH1", (N_step,3), dtype=datatype, compression="gzip")
    
        self.xBH2_data = grp.create_dataset("xBH2", (N_step,3), dtype=datatype, compression="gzip")
        self.vBH2_data = grp.create_dataset("vBH2", (N_step,3), dtype=datatype, compression="gzip")
    
        self.xDM_i_data = grp.create_dataset("xDM_i", (self.p.N_DM,3), dtype=datatype, compression="gzip")
        self.xDM_f_data = grp.create_dataset("xDM_f", (self.p.N_DM,3), dtype=datatype, compression="gzip")
    
        self.vDM_i_data = grp.create_dataset("vDM_i", (self.p.N_DM,3), dtype=datatype, compression="gzip")
        self.vDM_f_data = grp.create_dataset("vDM_f", (self.p.N_DM,3), dtype=datatype, compression="gzip")
    
        return f
        
    def output_metadata(self):
        
        listfile = f'{OUT_SNAP_DIR}/SimulationList.txt'
        hdrtxt = "Columns: IDhash, M_1/MSUN, M_2/MSUN, a_i/r_isco(M1), e_i, N_DM, M_DM/MSUN, Nstep_per_orb, N_orb, r_soft/PC, method, rho_6/(MSUN/PC**3), gamma, alpha, r_t/PC"
    
        T_orb = 2*np.pi*np.sqrt(self.a_i**3/(u.G_N*self.p.M_tot))
    
        meta_data = np.array([self.IDhash, self.p.M_1/u.Msun, self.p.M_2/u.Msun, 
                            self.a_i/tools.calc_risco(self.p.M_1), self.e_i, self.p.N_DM, self.p.M_DM/u.Msun, 
                            int(np.round(T_orb/self.dt)), int(np.round(self.t_end/T_orb)), np.sqrt(self.r_soft_sq)/u.pc, self.method,
                            self.p.rho_6/(u.Msun/u.pc**3), self.p.gamma_sp, self.p.alpha, self.p.r_t/u.pc])
                            
        meta_data = np.reshape(meta_data, (1,  len(meta_data)))
    
    
        if (os.path.isfile(listfile)):
            with open(listfile,'a') as g:
                np.savetxt(g, meta_data, fmt='%s')
            g.close()
        else:
            np.savetxt(listfile, meta_data, header=hdrtxt, fmt='%s')
        
        

def load_trajectory(IDhash):
    f = h5py.File(f"{OUT_SNAP_DIR}/{IDhash}.hdf5", 'r')
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
    
    if (dynamic_BH == 1):
        M_tot = M_1 + M_2
    else:
        M_tot = 1.0*M_1
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (u.G_N*M_tot))
    
    a_list, e_list = tools.calc_orbital_elements(xBH_list, vBH_list, M_tot)
    
    #return ts/T_orb, a_list, e_list
    return ts, a_list, e_list
    
def load_DMparticles(IDhash, final=True):
    f = h5py.File(f"{OUT_SNAP_DIR}/{IDhash}.hdf5", 'r')
    
    if (final == True):
        tag = "f"
    else:
        tag = "i"
        
    xDM_list =  np.array(f['data']['xDM_' + tag])
    vDM_list =  np.array(f['data']['vDM_' + tag])
    
    return xDM_list, vDM_list
    
def show_simulation_summary(IDhash):
    f = h5py.File(f"{OUT_SNAP_DIR}/{IDhash}.hdf5", 'r')
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
    listfile = f'{OUT_SNAP_DIR}/SimulationList.txt'
    return np.loadtxt(listfile, unpack=True, usecols=(i,), dtype=dtype)
    
print("This needs to be a dictionary! Also - fix e_i...")
def load_simulation_list():

    hashes = load_entry(0, dtype=str)
    M_1     = load_entry(1)*u.Msun
    M_2     = load_entry(2)*u.Msun
    a_i     = load_entry(3)*tools.calc_risco(M_1)
    e_i     = load_entry(4)
    N_DM    = load_entry(5, dtype=int)
    M_DM    = load_entry(6)*u.Msun
    Nstep_per_orb      = load_entry(7, dtype=int)
    N_orb   = load_entry(8, dtype=int)
    r_soft  = load_entry(9)*u.pc
    method  = load_entry(10, dtype=str)
    rho_6   = load_entry(11)*u.Msun/u.pc**3
    gamma_sp = load_entry(12)
    alpha   = load_entry(13)
    r_t     = load_entry(14)*u.pc
    
    return hashes, M_1, M_2, a_i, e_i, N_DM, M_DM, Nstep_per_orb, N_orb, r_soft, method, rho_6, gamma_sp, alpha, r_t

    
def make_plots(IDhash):
    
    f = h5py.File(f"{OUT_SNAP_DIR}/{IDhash}.hdf5", 'r')
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
    
