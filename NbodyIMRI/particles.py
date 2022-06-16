import warnings
from math import sqrt

from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

from NbodyIMRI import distributionfunctions as DF
from NbodyIMRI import tools
from NbodyIMRI import units as u


import h5py





import copy

def load_particles_from_file(snap_shot_dir, IDhash, which="initial"):
    fname = join(snap_shot_dir, IDhash) + ".hdf5"
    f = h5py.File(fname, 'r')
    M_1 = f['data'].attrs["M_1"]*u.Msun
    M_2 = f['data'].attrs["M_2"]*u.Msun
    N_DM = f['data'].attrs["N_DM"]
    M_DM = f['data'].attrs["M_DM"]*u.Msun
    dynamic = f['data'].attrs["dynamic"]
    if (dynamic == 1):
        dynamic_BH = True
    else:
        dynamic_BH = False
    
    if (N_DM <= 0):
        N_DM = 2 #Keep N_DM = 2 so that all the arrays work as expected...
        M_DM = 0.0
    
    p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM, dynamic_BH=dynamic_BH)
    
    if (which == "initial"):
        p.xBH1 = np.array(f['data']['xBH1'])[1,:]
        p.vBH1 = np.array(f['data']['vBH1'])[1,:]
    
        p.xBH2 = np.array(f['data']['xBH2'])[1,:]
        p.vBH2 = np.array(f['data']['vBH2'])[1,:]
    
        p.xDM  = np.array(f['data']['xDM_i'])
        p.vDM  = np.array(f['data']['vDM_i'])
    elif (which == "final"):
        p.xBH1 = np.array(f['data']['xBH1'])[-1,:]
        p.vBH1 = np.array(f['data']['vBH1'])[-1,:]
    
        p.xBH2 = np.array(f['data']['xBH2'])[-1,:]
        p.vBH2 = np.array(f['data']['vBH2'])[-1,:]
    
        p.xDM  = np.array(f['data']['xDM_f'])
        p.vDM  = np.array(f['data']['vDM_f'])
    
    return p
    
    
def single_BH(M_1, N_DM=0, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=-1, r_t = -1, alpha = 2):
    if (r_max < 0):
        r_max = 1e5*tools.calc_risco(M_1)
    
    
    if (N_DM > 0):
        if (r_t < 0):
            SpikeDF = DF.PowerLawSpike(M_1/u.Msun, rho_6/(u.Msun/u.pc**3), gamma_sp)
        else:
            SpikeDF = DF.GeneralizedNFWSpike(M_1/u.Msun, rho_6/(u.Msun/u.pc**3), gamma_sp, r_t/u.pc, alpha)
        M_spike = SpikeDF.M_DM_ini(r_max/u.pc)*u.Msun
        M_DM    = (M_spike/N_DM)
    else:
        M_DM = 0.0
    
    p = particles(M_1, M_2=0.0, N_DM=N_DM, M_DM=M_DM, dynamic_BH=False)
    
    if (N_DM > 0):
        p.initialize_spike(rho_6, gamma_sp, r_max, r_t, alpha)
    
    return p
    
def particles_in_binary(M_1, M_2, a_i, e_i=0.0, N_DM=0, M_DM=0.0, dynamic_BH=True, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=-1, r_t = -1, alpha = 2, include_DM_mass=False):
    
    if (r_max < 0):
        r_max = 1e5*tools.calc_risco(M_1)
    
    if (N_DM > 0):
        if (r_t < 0):
            SpikeDF = DF.PowerLawSpike(M_1/u.Msun, rho_6/(u.Msun/u.pc**3), gamma_sp)
        else:
            SpikeDF = DF.GeneralizedNFWSpike(M_1/u.Msun, rho_6/(u.Msun/u.pc**3), gamma_sp, r_t/u.pc, alpha)
            
        M_spike = SpikeDF.M_DM_ini(r_max/u.pc)*u.Msun
        M_DM    = (M_spike/N_DM)
    else:
        M_DM = 0.0
    
    p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM, dynamic_BH=dynamic_BH)
    
    #Initialise BH properties
    r_i = a_i * ( 1 + e_i)
    
    if (include_DM_mass):
        mu = u.G_N*(p.M_tot + SpikeDF.M_DM_ini(a_i/u.pc)*u.Msun)
    else:
        mu = u.G_N*p.M_tot
    v_i = np.sqrt( mu * (2.0/r_i - 1.0/a_i) )
    
    if (dynamic_BH):
        factor = M_2/p.M_tot
    else:
        factor = 0
            
    p.xBH1[:] = np.atleast_2d([-r_i*factor,   0, 0])
    p.xBH2[:] = np.atleast_2d([r_i*(1-factor),   0, 0])

    p.vBH1[:] = np.atleast_2d([0.0, v_i*factor, 0])
    p.vBH2[:] = np.atleast_2d([0.0, -v_i*(1-factor), 0])
    
    if (N_DM > 0):
        p.initialize_spike(rho_6, gamma_sp, r_max, r_t, alpha)
        
    return p
    
    
class particles():
    def __init__(self, M_1, M_2, N_DM=2, M_DM = 0, dynamic_BH=True):

        self.M_1 = M_1
        self.M_2 = M_2
        
        self.M_tot = M_1 + M_2
        
        self.M_DM = M_DM
        self.N_DM = N_DM
            
        self.xBH1 = np.zeros((3), dtype=np.float64)
        self.vBH1 = np.zeros((3), dtype=np.float64)
        
        self.xBH2 = np.zeros((3), dtype=np.float64)
        self.vBH2 = np.zeros((3), dtype=np.float64)
        
        self.xDM = np.zeros((N_DM, 3))
        self.vDM = np.zeros((N_DM, 3))
    
        self.dvdtBH1 = None
        self.dvdtBH2 = None
        self.dxdtDM = None
        
        self.dynamic_BH = dynamic_BH
        
        #Null values for the spike parameters
        self.rho_6    = 0.0
        self.gamma_sp = 0.0
        self.alpha    = 0.0
        self.r_t      = -1.0

    def xstep(self, h):
        if (self.dynamic_BH):
            self.xBH1 += self.vBH1*h
        self.xBH2 += self.vBH2*h
        self.xDM += self.vDM*h


    def vstep(self, h):
        if (self.dynamic_BH):
            self.vBH1 += self.dvdtBH1*h
        self.vBH2 += self.dvdtBH2*h
        self.vDM += self.dvdtDM*h
        
    def orbital_elements(self):
        return tools.calc_orbital_elements(self.xBH1 - self.xBH2, self.vBH1 - self.vBH2, self.M_tot)
    
    def T_orb(self):
        a_i, e_i = self.orbital_elements()
        return tools.calc_Torb(a_i, self.M_tot)
    
    def initialize_spike(self, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=1e-6*u.pc, r_t = -1, alpha  = 2):
        
        self.rho_6    = rho_6
        self.gamma_sp = gamma_sp
        self.r_t      = r_t
        self.alpha    = alpha
        
        
        if (self.N_DM > 2):
            if (r_t < 0):
                SpikeDF = DF.PowerLawSpike(self.M_1/u.Msun, rho_6/(u.Msun/u.pc**3), gamma_sp)
            else:
                SpikeDF = DF.GeneralizedNFWSpike(self.M_1/u.Msun, rho_6/(u.Msun/u.pc**3), gamma_sp, r_t/u.pc, alpha)
            r, v = SpikeDF.draw_particle(r_max/u.pc, N = self.N_DM)

            for i in range(self.N_DM):
                rhat = tools.get_random_direction()
                vhat = tools.get_random_direction()
                self.xDM[i,:] = r[i]*rhat * u.pc
                self.vDM[i,:] = v[i]*vhat * u.pc/u.Myr
    
            self.xDM += self.xBH1
            self.vDM += self.vBH1
    
    def summary(self):
        print("> Particle set:")
        print(f">     M_1 [M_sun] = {self.M_1/u.Msun}")
        if (self.M_2 > 0): 
            print(f">     M_2 [M_sun] = {self.M_2/u.Msun}")
            a, e = self.orbital_elements()
            print(f">     (a [pc], e) = ({a/u.pc}, {e})")
        if (self.M_DM > 0):
            print(" ")
            print(f">     N_DM = {self.N_DM}")
            print(f">     M_DM [M_sun] = {self.M_DM/u.Msun}")
            
    
    
    def plot(self):
        if (self.M_DM > 0):    
            ncols = 3
        else:
            ncols = 2
            
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(16, 5))
        
        axes = ax[:]
        #----------------------------------------
        
        if (self.M_DM > 0):
            axes[0].scatter(self.xDM[:,0]/u.pc, self.xDM[:,1]/u.pc, color='C0', marker='o', alpha=0.75)
        
        axes[0].scatter(self.xBH1[0]/u.pc, self.xBH1[1]/u.pc, color='k', marker='o', s=250)
        
        if (self.M_2 > 0):
            a_pc = self.orbital_elements()[0]/u.pc
            axes[0].scatter(self.xBH2[0]/u.pc, self.xBH2[1]/u.pc, color='k', marker='o', s=40)
            axes[0].set_xlim(-1.5*a_pc, 1.5*a_pc)
            axes[0].set_ylim(-1.5*a_pc, 1.5*a_pc)
        
        axes[0].set_xlabel(r"$x$ [pc]")
        axes[0].set_ylabel(r"$y$ [pc]")
        axes[0].set_aspect('equal')
        
        #----------------------------------------
     
        if (self.M_DM > 0):
            axes[1].scatter(self.xDM[:,0]/u.pc, self.xDM[:,2]/u.pc, color='C0', marker='o', alpha=0.75)
     
        axes[1].scatter(self.xBH1[0]/u.pc, self.xBH1[2]/u.pc, color='k', marker='o', s=250)
        
        if (self.M_2 > 0):
            a_pc = self.orbital_elements()[0]/u.pc
            axes[1].scatter(self.xBH2[0]/u.pc, self.xBH2[2]/u.pc, color='k', marker='o', s=40)
            axes[1].set_xlim(-1.5*a_pc, 1.5*a_pc)
            axes[1].set_ylim(-1.5*a_pc, 1.5*a_pc)
        
            
        axes[1].set_xlabel(r"$x$ [pc]")
        axes[1].set_ylabel(r"$z$ [pc]")
        axes[1].set_aspect('equal')
            
        #---------------------------------------
        if (self.M_DM > 0):
 
            r_vals = tools.norm(self.xDM - self.xBH1)
            axes[2].hist(np.log10(r_vals/u.pc), 50, density=True)
            
            axes[2].set_xlabel(r"$\log_{10}(r/\mathrm{pc})$")
            axes[2].set_ylabel(r"$P(\log_{10}(r/\mathrm{pc}))$")
    
            if (self.M_2 <= 0):
                r_max = np.max(r_vals)/u.pc/1e3
                axes[0].set_xlim(-r_max, r_max)
                axes[0].set_ylim(-r_max, r_max)
                axes[1].set_xlim(-r_max, r_max)
                axes[1].set_ylim(-r_max, r_max)
        
        
        plt.tight_layout()