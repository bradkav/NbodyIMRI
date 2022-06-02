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


# Leapfrog stepsize parameters
#-----------------------------
theta = 1/(2 - 2**(1/3))

xi  = +0.1786178958448091e+00
lam = -0.2123418310626054e+00
chi = -0.6626458266981849e-01
#-----------------------------

OUT_SNAP_DIR = "test_snapshots"
OUT_FIG_DIR = "test_figures"

    
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
    
    
def single_BH(M_1, N_DM=0, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=1e-6*u.pc, r_t = -1, alpha = 2):
    
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
    
def particles_in_binary(M_1, M_2, a_i, e_i=0.0, N_DM=0, M_DM=0.0, dynamic_BH=True, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=1e-6*u.pc, r_t = -1, alpha = 2):
    
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
        
        if (dynamic_BH  == True):
            self.M_tot = M_1 + M_2
        else:
            self.M_tot = M_1
        
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
        return 2*np.pi*np.sqrt(a_i**3/(u.G_N*self.M_tot))
    
    def initialize_spike(self, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=1e-6*u.pc, r_t = -1, alpha  = 2):
        
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
        
        plt.tight_layout()
    
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
        r1_sq   = np.linalg.norm(dx1, axis=-1, keepdims=True)**2
        dx1    /= np.sqrt(r1_sq)
        
        if (self.soft_method == "plummer"):
            acc_DM1 = -u.G_N*self.p.M_1*dx1*(r1_sq + self.r_soft_sq)**-1
            
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
        r2_sq   = np.linalg.norm(dx2, axis=-1, keepdims=True)**2
        dx2     /= np.sqrt(r2_sq)
        
        if (self.soft_method == "plummer"):
            acc_DM2 = -u.G_N*self.p.M_2*dx2*(r2_sq + self.r_soft_sq)**-1
            
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
        
    
            
    def run_simulation(self, dt, t_end, method="DKD", save_to_file = True, add_to_list = False):
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
    

    
        for it in range(N_step):
               
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
        hdrtxt = "Columns: IDhash, M_1/MSUN, M_2/MSUN, a_i/PC, e_i, N_DM, M_DM/MSUN, dt, t_end, r_soft/PC, method"
    
        
    
        meta_data = np.array([self.IDhash, self.p.M_1/u.Msun, self.p.M_2/u.Msun, 
                            self.a_i/u.pc, self.e_i, self.p.N_DM, self.p.M_DM/u.Msun, 
                            self.dt, self.t_end, np.sqrt(self.r_soft_sq)/u.pc, self.method])
                            
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
    
    return ts/T_orb, a_list, e_list
    
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
    
def load_simulation_list():

    hashes = load_entry(0, dtype=str)
    M_1     = load_entry(1)*u.MSUN
    M_2     = load_entry(2)*u.MSUN
    a_i     = load_entry(3)*u.pc
    e_i     = load_entry(4)
    N_DM    = load_entry(5, dtype=int)
    M_DM    = load_entry(6)*u.MSUN
    N_step  = load_entry(7, dtype=int)
    N_orb   = load_entry(8, dtype=int)
    r_soft  = load_entry(9)*u.pc
    method  = load_entry(10, dtype=str)
    
    return hashes, M_1, M_2, a_i, e_i, N_DM, M_DM, N_step, N_orb, r_soft, method

    
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
    
