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



# Leapfrog stepsize parameters
#-----------------------------
theta = 1/(2 - 2**(1/3))

xi  = +0.1786178958448091e+00
lam = -0.2123418310626054e+00
chi = -0.6626458266981849e-01
#-----------------------------




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
        
        if (self.p.dynamic_BH):
            M1_eff  = self.p.M_1
            M2_eff  = self.p.M_2
        else:
            M1_eff  = self.p.M_1 + self.p.M_2
            M2_eff  = (self.p.M_1*self.p.M_2)/(self.p.M_1 + self.p.M_2)
        
        if (self.soft_method == "plummer"):
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq + self.r_soft_sq)**-1
            
        elif (self.soft_method == "plummer2"):
            acc_DM1 = -u.G_N*M1_eff*r1*(dx1/2)*(2*r1_sq + 5*self.r_soft_sq)*(r1_sq + self.r_soft_sq)**(-5/2)
            
        elif (self.soft_method == "uniform"):
            x = np.sqrt(r1_sq/self.r_soft_sq)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) > 1):
                inds = inds.flatten()
                acc_DM1[inds] = -u.G_N*M1_eff*dx1[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq)
                
        elif (self.soft_method == "truncate"):
            r1_sq = np.clip(r1_sq, self.r_soft_sq, 1e50)
            acc_DM1 = -u.G_N*M1_eff*dx1/r1_sq
            
        else:
            print("WHAT?!")
        
        dx2     = (self.p.xDM - self.p.xBH2)
        r2      = np.linalg.norm(dx2, axis=-1, keepdims=True)
        dx2     /= r2
        r2_sq   = r2**2 
        
        #Consider also Eq. (2.227) in Binney and Tremaine
        #https://arxiv.org/pdf/2104.05643.pdf
        
        if (self.soft_method == "plummer"):
            acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq + self.r_soft_sq)**-1
            
        elif (self.soft_method == "plummer2"):
            acc_DM2 = -u.G_N*M2_eff*r2*(dx2/2)*(2*r2_sq + 5*self.r_soft_sq)*(r2_sq + self.r_soft_sq)**(-5/2)
            
        elif (self.soft_method == "uniform"):
            x = np.sqrt(r2_sq/self.r_soft_sq)
            acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
            inds = x < 1
            if (np.sum(inds) > 1):
                inds = inds.flatten()
                acc_DM2[inds] = -u.G_N*M2_eff*dx2[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq)
                
        elif (self.soft_method == "truncate"):
            r2_sq = np.clip(r2_sq, self.r_soft_sq, 1e50)
            acc_DM2 = -u.G_N*M2_eff*dx2/r2_sq
            
        else:
            print("WHAT?!")
        
        dx12    = (self.p.xBH1 - self.p.xBH2)
        r12_sq  = np.linalg.norm(dx12, axis=-1, keepdims=True)**2
        acc_BH = -u.G_N*M2_eff*dx12*(r12_sq)**-1.5

        
        if (self.p.dynamic_BH):
            self.p.dvdtBH1 = acc_BH
        else:
            self.p.dvdtBH1 = 0.0
            
        self.p.dvdtBH2 = -(M1_eff/M2_eff)*acc_BH - (self.p.M_DM/M2_eff)*np.sum(acc_DM2, axis=0)
        self.p.dvdtDM  = acc_DM1 + acc_DM2
        
    
            
    def run_simulation(self, dt, t_end, method="PEFRL", save_to_file = False, add_to_list = False, show_progress=False):
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
        self.finished = False
        
        if (save_to_file):
            fname = f"{NbodyIMRI.snapshot_dir}/{self.IDhash}.hdf5"
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
            
        self.finished = True
        
        

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
        
        listfile = f'{NbodyIMRI.snapshot_dir}/SimulationList.txt'
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
        
    def plot_orbital_elements(self):
        if (self.finished == False):
            print("Simulation has not been finished. Please run using `rum_simulation()`.")
            return 0
            
        else:
            xBH_list = self.xBH1_list - self.xBH2_list
            vBH_list = self.vBH1_list - self.vBH2_list
            a_list, e_list = tools.calc_orbital_elements(xBH_list, vBH_list, self.p.M_tot)
            
            delta_a = (a_list - a_list[0])/a_list
            delta_e = (e_list - e_list[0])
            
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
        
            axes = ax[:]
        

            axes[0].plot(self.ts, delta_a)
            axes[0].set_xlabel(r"$t$ [s]")
            axes[0].set_ylabel(r"$\Delta a/a$")
            
            axes[1].plot(self.ts, delta_e)
            axes[1].set_xlabel(r"$t$ [s]")
            axes[1].set_ylabel(r"$\Delta e$")   

            #plt.title(IDhash)
            plt.tight_layout()
            
            plt.show()
            return fig
            
    def plot_trajectory(self):
        if (self.finished == False):
            print("Simulation has not been finished. Please run using `rum_simulation()`.")
            return 0
            
        else:
            fig = plt.figure()
            
            plt.plot(self.xBH1_list[:,0]/u.pc, self.xBH1_list[:,1]/u.pc, lw=2)
            #plt.plot(self.xBH2_list[:,0]/u.pc, self.xBH2_list[:,1]/u.pc)
            
            plt.xlabel(r"$x$ [pc]")
            plt.ylabel(r"$y$ [pc]")
            plt.gca().set_aspect('equal')
            
            plt.show()
            return fig
        