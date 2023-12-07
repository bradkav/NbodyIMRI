import warnings
from math import sqrt

from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

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

#NB: For PN corrections, see https://arxiv.org/abs/1312.1289
#NB: Deal with saving of M1 and M2 values for wheel and spoke


class simulator():
    """
    Class for evolving the N-body system.
    
    Attributes:
        p (particles)       : A particles object, which specifies the initial conditions of the objects to simulate (a deep copy is made and used internally by the simulator)
        r_soft_sq (float)   : The square of the softening length to be used for the DM particles (the BH-BH forces are not softened)
        soft_method (string): Softening method to be used. Options are: "plummer", "plummer2", "uniform", "truncate". 
                            Default is "uniform" which computes the softening assuming that each DM particle is a finite sphere of uniform density.
        IDhash (string)     : A hash made up of 5 hexadecimal digits which identifies the simulation (and the output files). This is generated and saved when he simulation is run.
        check_state (function): A function which will be called in between each timestep to check the state of the simulation and perform any required operations. 
                                (For example, removing certain particles). Must have the signature `check_state(simulator)`. Default is None. 
    
    """
    
    def __init__(self, particle_set, r_soft_sq = 0.0, r_soft_sq1 = -1, soft_method="uniform", check_state = None):
            
        self.p = copy.deepcopy(particle_set)
        self.r_soft_sq = r_soft_sq
        self.r_soft_sq1 = r_soft_sq1
        if (self.r_soft_sq1 < 0):
            self.r_soft_sq1 = 1.0*r_soft_sq
        self.soft_method = soft_method    
        self.soft_method1 = "uniform"
        self.check_state = check_state
        self.background_field = None
        
                    
            
    def full_step(self, dt, method="PEFRL"):
        """
        Perform a full leapfrog step, See e.g. https://arxiv.org/abs/2007.05308, http://physics.ucsc.edu/~peter/242/leapfrog.pdf
        
        Parameters:
            dt (float)      : size of the timestep (the leapfrog is made up of many sub-steps, with dt being the size of one full leapfrog step)
            method (string) : leapfrog method to use. Options are: "DKD", "FR", "PEFRL" [default]. These corresponds to 2nd, 4th and 4th order methods.
        
        Returns:
            None
        
        """
        
        
        #2nd order 'standard' leapfrog
        if (method == "DKD"):
            self.p.xstep(0.5*dt)
            self.update_acceleration()
            self.p.vstep(1.0*dt)
            self.p.xstep(0.5*dt)

        #4th order Ruth-Forest (FR) leapfrog
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
        
        #Improved 4th order "Position Extended Forest-Ruth Like" (PEFRL) leapfrog
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
        """
        Update the acceleration of all particles in p, based on current positions.
        
        Returns:
            None
        """
        
        #Calculate separations between DM particles and central BH
        dx1     = (self.p.xDM - self.p.xBH1)
        r1      = np.linalg.norm(dx1, axis=-1, keepdims=True)
        dx1    /= r1
        r1_sq   = r1**2
        
        if (self.p.dynamic_BH):
            M1_eff  = self.p.M_1
            M2_eff  = self.p.M_2
        else:
            #M1_eff  = self.p.M_1
            #M2_eff  = self.p.M_2
            M1_eff  = self.p.M_1 + self.p.M_2
            M2_eff  = (self.p.M_1*self.p.M_2)/(self.p.M_1 + self.p.M_2)

        #Calculate forces (including softening)
        if (self.soft_method1 == "plummer"):
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq + self.r_soft_sq1)**-1
            
        elif (self.soft_method1 == "plummer2"):
            acc_DM1 = -u.G_N*M1_eff*r1*(dx1/2)*(2*r1_sq + 5*self.r_soft_sq1)*(r1_sq + self.r_soft_sq1)**(-5/2)
            
        elif (self.soft_method1 == "uniform_old"):
            x = np.sqrt(r1_sq/self.r_soft_sq1)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) > 1):
                inds = inds.flatten()
                acc_DM1[inds] = -u.G_N*M1_eff*dx1[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq1)

        elif (self.soft_method1 == "uniform"):
            x = np.sqrt(r1_sq/self.r_soft_sq1)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) > 1):
                inds = inds.flatten()
                acc_DM1[inds] = -u.G_N*M1_eff*dx1[inds,:]*x[inds]/(self.r_soft_sq1)
                
        elif (self.soft_method1 == "truncate"):
            r1_sq = np.clip(r1_sq, self.r_soft_sq1, 1e50)
            acc_DM1 = -u.G_N*M1_eff*dx1/r1_sq

        elif (self.soft_method1 == "empty_shell"):
            x = np.sqrt(r1_sq/self.r_soft_sq1)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) >= 1):
                inds = inds.flatten()      
                acc_DM1[inds] *= 0.0

        else:
            raise ValueError("Invalid softening method:" + self.soft_method1)
        
        #Calculate forces on second BH (if it exists)
        if (self.p.M_2 > 0):
            dx2     = (self.p.xDM - self.p.xBH2)
            r2      = np.linalg.norm(dx2, axis=-1, keepdims=True)
            dx2     /= r2
            r2_sq   = r2**2 


            if (self.soft_method == "plummer"):
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq + self.r_soft_sq)**-1

            elif (self.soft_method == "plummer2"):
                acc_DM2 = -u.G_N*M2_eff*r2*(dx2/2)*(2*r2_sq + 5*self.r_soft_sq)*(r2_sq + self.r_soft_sq)**(-5/2)

            elif (self.soft_method == "uniform_old"):
                x = np.sqrt(r2_sq/self.r_soft_sq)
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
                inds = x < 1
                if (np.sum(inds) > 1):
                    inds = inds.flatten()
                    acc_DM2[inds] = -u.G_N*M2_eff*dx2[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq)

            elif (self.soft_method == "uniform"):
                x = np.sqrt(r2_sq/self.r_soft_sq)
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
                inds = x < 1
                if (np.sum(inds) > 1):
                    inds = inds.flatten()
                    acc_DM2[inds] = -u.G_N*M2_eff*dx2[inds,:]*x[inds]/(self.r_soft_sq)

            elif (self.soft_method == "truncate"):
                r2_sq = np.clip(r2_sq, self.r_soft_sq, 1e50)
                acc_DM2 = -u.G_N*M2_eff*dx2/r2_sq

            elif (self.soft_method == "empty_shell"):
                x = np.sqrt(r2_sq/self.r_soft_sq)
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
                inds = x < 1
                if (np.sum(inds) >= 1):
                    inds = inds.flatten()      
                    acc_DM2[inds] *= 0.0

            else:
                raise ValueError("Invalid softening method:" + self.soft_method)

            #Calculate forces between the 2 BHs  
            dx12    = (self.p.xBH1 - self.p.xBH2)
            r12_sq  = np.linalg.norm(dx12, axis=-1, keepdims=True)**2
            acc_BH = -u.G_N*M2_eff*dx12*(r12_sq)**-1.5
        else:
            acc_BH = 0.0
            acc_DM2 = 0.0
        
        #Save the values of the acceleration
        if (self.p.dynamic_BH):
            self.p.dvdtBH1 = acc_BH - 0.0*(1/M1_eff)*np.sum(np.atleast_2d(self.p.M_DM).T*acc_DM1, axis=0)
        else:
            self.p.dvdtBH1 = 0.0
        
        if (self.p.M_2 > 0):
            self.p.dvdtBH2 = -(M1_eff/M2_eff)*acc_BH - 0.0*(1/M2_eff)*np.sum(np.atleast_2d(self.p.M_DM).T*acc_DM2, axis=0)
        else:
            self.p.dvdtBH2 = 0.0
        
        self.p.dvdtDM  = acc_DM1 + acc_DM2
        #if (not self.p.dynamic_BH):
        #    self.p.dvdtDM += -acc_BH
        #else:
            
        
        #Now, if a background force field has been set, calculate the acceleration
        if self.background_field is not None:
            self.p.dvdtBH1 += self.background_field(self.p.xBH1)
            self.p.dvdtBH2 += self.background_field(self.p.xBH2)
            self.p.dvdtDM  += self.background_field(self.p.xDM)
        
    
            
    def run_simulation(self, dt, t_end, method="PEFRL", save_to_file = False, add_to_list = False, show_progress=False, save_DM_states=False, N_save=1, label=None):
        """
        Run the simulator, starting from the current state of particles in p, running for a time t_end.
        Times and timesteps are in physical times (as opposed to being in terms of number of orbits etc.)
        BEWARE: run_simulation will erase a previous version of the simulation with the same IDhash before starting.
        
        Parameters:
            dt (float)      : Size of the individual timesteps
            t_end (float)   : End time of the simulation
            method (string) : Leapfrog method. See `simulator.full_step` for more details.
            save_to_file (bool):    Set to True in order to output the simulation data to file. Default = False.
            add_to_list (bool):     Set to True in order to save metadata about the simulation to `SimulationList.txt`. Default = False. 
            show_progress (bool):   Set to True in order to show a progress bar during the simulation. Default = False
            save_DM_states (bool):  Set to True in order to save the initial and final configuration of the DM particles in the output. Default = False
            N_save (int):    Number of time steps in between saving the data to file. Default = 1
            label (str):    String to be used in the name of the output file (along with the IDhash). 
        
        Returns:
            None
        
        """
        
        print("> Simulating...")
        
        #Determine total number of steps
        self.t_end   = t_end
        self.dt      = dt
        self.current_step = 0
        self.runID   = label
        self.IDhash = tools.generate_hash() 
        
        if (self.runID is None):
            self.fileID = self.IDhash
        else:
            self.fileID = f"{self.runID}_{self.IDhash}"
            
        N_step = int(np.ceil(t_end/dt)) 
        
        #Initialise lists to save the BH positions
        self.ts        = np.linspace(0, t_end, N_step)
        self.xBH1_list = np.zeros((N_step, 3))
        self.vBH1_list = np.zeros((N_step, 3))
    
        self.xBH2_list = np.zeros((N_step, 3))
        self.vBH2_list = np.zeros((N_step, 3))
        
        self.M1_list   = np.zeros(N_step)
        self.M2_list   = np.zeros(N_step)
        
        
        #N_save = 100 #Save only every 100 timesteps
        #N_save = 1
        #N_out = int(N_step/N_save)
        N_out = len(self.ts[::N_save])
        N_update = 100_000 #Update the output file only every 100_000 steps
        #N_update = 1

        #Determine initial orbital parameters of the system
        if (self.p.M_2 > 0):
            
            a_i, e_i = self.p.orbital_elements()
            self.a_i = float(a_i)
            self.e_i = float(e_i)
            T_orb    = self.p.T_orb()
        else:
            self.a_i = 0
            self.e_i = 0    

        self.M_2_ini = self.p.M_2
        
        self.method = method
        self.finished = False
        
        #Open output file
        if (save_to_file):
            fname = f"{NbodyIMRI.snapshot_dir}/{self.fileID}.hdf5"
    
            try:
                os.remove(fname)
                print("Old file removed successfully:", fname)
            except: 
                print("No old snapshot file found...")
            f = self.open_outputfile(fname, N_out, save_DM_states)

        
        
        #Save the time steps and the initial DM configuration
        if (save_to_file):
            print(N_step, len(self.t_data[:]), len(1.0*self.ts[::N_save]))
            self.t_data[:] = 1.0*self.ts[::N_save]
            self.M1_list[0] = self.p.M_1
            self.M2_list[0] = self.p.M_2
        
            if (save_DM_states):
                self.xDM_i_data[:,:] = 1.0*self.p.xDM
                self.vDM_i_data[:,:] = 1.0*self.p.vDM
        
    
        #Define a dummy in case we're not using a progress bar
        stepper = lambda x: x
        if (show_progress):
            stepper = tqdm
        
        #print("N_steps:", N_step)
        #Simulate for N_step time-steps
        for it in stepper(range(N_step)):
            
            #Do any checks of the state of the system in between timesteps
            if (self.check_state is not None):
                self.check_state(self)
              
            #Save current binary configuration to array
            self.M1_list[it]     = self.p.M_1
            self.M2_list[it]     = self.p.M_2
            
            self.xBH1_list[it,:] = self.p.xBH1
            self.vBH1_list[it,:] = self.p.vBH1
        
            self.xBH2_list[it,:] = self.p.xBH2
            self.vBH2_list[it,:] = self.p.vBH2
        
            #Update data saved in file
            if ((it%N_update == 0) and (save_to_file)):
                #print(N_step, N_save, N_out, N_update)
                self.M_1_data[:]    = 1.0*self.M1_list[::N_save]
                self.M_2_data[:]    = 1.0*self.M2_list[::N_save]
                
                self.xBH1_data[:,:] = 1.0*self.xBH1_list[::N_save,:]
                self.vBH1_data[:,:] = 1.0*self.vBH1_list[::N_save,:]
            
                self.xBH2_data[:,:] = 1.0*self.xBH2_list[::N_save,:]
                self.vBH2_data[:,:] = 1.0*self.vBH2_list[::N_save,:]
            
            #Step forward by dt
            self.full_step(dt, method)
            
            #Increment the current step number (this is primarily so that the 
            #check_state function has some idea about how far in the simulation we are...)
            self.current_step += 1
        

        #One final update of the output data   
        if (save_to_file):
            self.M_1_data[:]    = 1.0*self.M1_list[::N_save]
            self.M_2_data[:]    = 1.0*self.M2_list[::N_save]
            
            self.xBH1_data[:,:] = 1.0*self.xBH1_list[::N_save,:]
            self.vBH1_data[:,:] = 1.0*self.vBH1_list[::N_save,:]
    
            self.xBH2_data[:,:] = 1.0*self.xBH2_list[::N_save,:]
            self.vBH2_data[:,:] = 1.0*self.vBH2_list[::N_save,:]
    
            if (save_DM_states):
                self.xDM_f_data[:,:] = 1.0*self.p.xDM
                self.vDM_f_data[:,:] = 1.0*self.p.vDM
                
                self.M_DM_data[:] =  1.0*self.p.M_DM
    
        print("> Simulation completed.")
    
        #Add information to SimulationList.txt if required
        if (add_to_list):
            self.output_metadata()
    
        if (save_to_file):
            f.close()
            
        self.finished = True
        
        

    def open_outputfile(self, fname, N_step, save_DM_states):
        """
        ...
        
        """
        f = h5py.File(fname, "w")
        grp = f.create_group("data")
        grp.attrs['M_1'] = self.p.M_1/u.Msun
        grp.attrs['M_2'] = self.M_2_ini/u.Msun
        
        a_i, e_i = self.p.orbital_elements()
        
        #Still need to add other stuff here!
        
        grp.attrs['a_i'] = a_i/u.pc
        grp.attrs['e_i'] = e_i
        grp.attrs['N_DM'] = self.p.N_DM
        grp.attrs['M_DM'] = self.p.M_DM[0]/u.Msun
        grp.attrs['r_soft'] = np.sqrt(self.r_soft_sq)/u.pc
        if (self.p.dynamic_BH):
            grp.attrs['dynamic'] = 1
        else:
            grp.attrs['dynamic'] = 0
        
    
        datatype = np.float64
        self.t_data   = grp.create_dataset("t", (N_step,), dtype=datatype, compression="gzip")
        self.M_1_data = grp.create_dataset("M_1", (N_step,), dtype=datatype, compression="gzip")
        self.M_2_data = grp.create_dataset("M_2", (N_step,), dtype=datatype, compression="gzip")
        
        self.xBH1_data = grp.create_dataset("xBH1", (N_step,3), dtype=datatype, compression="gzip")
        self.vBH1_data = grp.create_dataset("vBH1", (N_step,3), dtype=datatype, compression="gzip")
    
        self.xBH2_data = grp.create_dataset("xBH2", (N_step,3), dtype=datatype, compression="gzip")
        self.vBH2_data = grp.create_dataset("vBH2", (N_step,3), dtype=datatype, compression="gzip")
    
        
    
        if (save_DM_states):
            self.xDM_i_data = grp.create_dataset("xDM_i", (self.p.N_DM,3), dtype=datatype, compression="gzip")
            self.xDM_f_data = grp.create_dataset("xDM_f", (self.p.N_DM,3), dtype=datatype, compression="gzip")
    
            self.vDM_i_data = grp.create_dataset("vDM_i", (self.p.N_DM,3), dtype=datatype, compression="gzip")
            self.vDM_f_data = grp.create_dataset("vDM_f", (self.p.N_DM,3), dtype=datatype, compression="gzip")
            
            self.M_DM_data = grp.create_dataset("M_DM", (self.p.N_DM,), dtype=datatype, compression="gzip")
    
        return f
        
    def output_metadata(self):
        """
        ...
        """
        
        listfile = f'{NbodyIMRI.snapshot_dir}/SimulationList.txt'
        hdrtxt = "Columns: FileID, M_1/MSUN, M_2/MSUN, a_i/r_isco(M1), e_i, N_DM, M_DM/MSUN, Nstep_per_orb, N_orb, r_soft/PC, method, rho_6/(MSUN/PC**3), gamma, alpha, r_t/PC"
    
        T_orb = 2*np.pi*np.sqrt(self.a_i**3/(u.G_N*self.p.M_tot()))
    
        meta_data = np.array([self.fileID, self.p.M_1/u.Msun, self.M_2_ini/u.Msun, 
                            self.a_i/tools.calc_risco(self.p.M_1), self.e_i, self.p.N_DM, self.p.M_DM[0]/u.Msun, 
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
        """
        ...
        """
        
        if (self.finished == False):
            print("Simulation has not been finished. Please run using `rum_simulation()`.")
            return 0
            
        else:
            xBH_list = self.xBH1_list - self.xBH2_list
            vBH_list = self.vBH1_list - self.vBH2_list
            a_list, e_list = tools.calc_orbital_elements(xBH_list, vBH_list, self.M1_list + self.M2_list)
            
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

            #plt.title(self.fileID)
            plt.tight_layout()
            
            plt.show()
            return fig
            
    def plot_trajectory(self):
        """
        ...
        """
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
        
    def plot(self):
        self.p.plot()
    
