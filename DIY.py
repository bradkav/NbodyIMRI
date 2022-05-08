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


_S      = 1

_MSUN   = 1.98855e30 #kg
_PC     = 3.08567758149137e16 #m
_CM     = 1e-2
_YR     = 365.25 * 24 * 3600*_S #s
_MYR    = 1e6*_YR
_GEV    = 1.79e-27
_KM     = 1e3


_G      = 4.302e-3*(_PC/_MSUN)*(_KM/_S)**2
_C      = 3e8


theta = 1/(2 - 2**(1/3))

TEST_SNAP_DIR = "/code/test_snapshots"
TEST_FIG_DIR = "/code/test_figures"

class CentralPotential():
    def __init__(self, M = 1e3*_MSUN):
        self.M = M
        self.mu = _G*self.M
        
    def get_gravity_at_point(self, pos, eps=0): 
        rsq = np.atleast_2d(np.sum(pos**2, axis=-1) + eps**2).T
        return -self.mu*pos/rsq**1.5
      
def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))

def calc_orbital_elements(x, v, M_tot):
    x_mag = norm(x)
    v_mag = norm(v)
    mu = _G*M_tot

    a = (2/x_mag - v_mag**2/mu)**-1
        #https://astronomy.stackexchange.com/questions/29005/calculation-of-eccentricity-of-orbit-from-velocity-and-radius
    h = np.cross(x,v)
    e_vec = np.cross(v,h)/mu - x/np.atleast_2d(x_mag).T
    e = norm(e_vec)
        
    return a, e

    
class particles():
    def __init__(self, M_1, M_2, N_DM=2, M_DM = 0):
    
        self.M_1 = M_1
        self.M_2 = M_2
        
        self.M_DM = M_DM
        self.N_DM = N_DM
        
        #self.r_isco1 = 6*_G*M_1/_C**2
        #self.r_isco2 = 6*_G*M_2/_C**2
        
        self.r_isco1  = 0
        self.r_isco2  = 0
    
        self.xBH = np.zeros(3)
        self.vBH = np.zeros(3)
        
        self.xDM = np.zeros((N_DM, 3))
        self.vDM = np.zeros((N_DM, 3))
        
        self.BH = CentralPotential(M_1)
    
        self.dvdtBH = None
        self.dxdtDM = None
        
    def update_acc(self):
        #Initial accelerations
        dx  = (self.xBH - self.xDM)
        rsq = np.atleast_2d(np.sum(dx**2, axis=-1) + self.r_isco2**2).T
        acc = -_G*dx/rsq**1.5
    
        self.dvdtBH = 1.0*self.BH.get_gravity_at_point(self.xBH, eps=self.r_isco1) + np.sum(self.M_DM*acc, axis=0)
        self.dvdtDM = 1.0*self.BH.get_gravity_at_point(self.xDM, eps=self.r_isco1) - self.M_2*acc
   
        
    def xstep(self, h):
        self.xBH += self.vBH*h
        #self.xDM += self.vDM*h

        
    def vstep(self, h):
        #self.update_acc()
        self.dvdtBH = -_G*self.M_1*self.xBH/norm(self.xBH)**3
        self.vBH += self.dvdtBH*h
        #self.vDM += self.dvdtDM*h

        
    
    
def run_simulation(M_1, M_2, a_i, e_i, N_DM = 0, gamma = 7/3, r_max = 1e-6*_PC, method = "DKD"):
    
    # Initialise central potential and binary orbit
    SpikeDF = DF.SpikeDistribution(M_1/_MSUN, rho_6=1e15, gamma_sp=gamma)
 
    
    if (N_DM > 0):
        M_spike = SpikeDF.M_DM_ini(r_max/_PC)*_MSUN
        M_DM    = (M_spike/N_DM)
    else:
        N_DM = 2 #Keep N_DM = 2 so that all the arrays work as expected...
        M_DM = 0.0
        
    p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM)
        
    #Initialise BH properties
    r_i = a_i * ( 1 + e_i)
    
    M_tot = M_1 + M_2
    mu = _G*M_tot
    v_i = np.sqrt( mu * (2/r_i - 1/a_i) )
    
    
    #-----------------------------
    
    p.xBH = np.atleast_2d([r_i,   0, 0])
    p.vBH = np.atleast_2d([0.0, v_i, 0])
    
    elements = calc_orbital_elements(p.xBH, p.vBH, M_tot)
    print("(a_i, e_i):", float(elements[0]/_PC), ",", float(elements[1]))
    print("DM pseudoparticle mass [Msun]:", M_DM/_MSUN)
    
    #Initialise DM properties
    print("> Initialising...")
    r, v = SpikeDF.draw_particle(r_max/_PC, N = N_DM)
    

    for i in range(N_DM):
        rhat = tools.get_random_direction()
        vhat = tools.get_random_direction()
        p.xDM[i,:] = r[i]*rhat * _PC
        p.vDM[i,:] = v[i]*vhat * _PC/_MYR
    
    # Simulation parameters
    N_step = 10000
    N_orb = 100
    
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (_G*M_tot))
    t_end = N_orb*T_orb
    dt = t_end/N_step
    print("> Timestep [s]:", dt)
    
    ts = np.linspace(0, t_end, N_step)
    xBH_list = np.zeros((N_step, 3))
    
    
    print("> r_isco(1) [pc]: ", p.r_isco1/_PC)
    print("> r_isco(2) [pc]: ", p.r_isco2/_PC)
    
    print("> dt [s]:", dt/_S)
    print("> t_end [s]:", t_end/_S)
    print("> T_orb [s]:", T_orb/_S)
    print("> Steps per orbit:", T_orb/dt)
    
    
    #-----------------------------
    IDhash = tools.generate_hash()
    print("> Hash: " + IDhash)
    f = h5py.File(f"../test_snapshots/{IDhash}.hdf5", "w")
    grp = f.create_group("data")
    grp.attrs['M_1'] = M_1/_MSUN
    grp.attrs['M_2'] = M_2/_MSUN
    grp.attrs['a_i'] = a_i/_PC
    grp.attrs['e_i'] = e_i
    grp.attrs['N_DM'] = N_DM
    #Save more stuff here...
    
    t_data   = grp.create_dataset("t", (N_step,), dtype='f', compression="gzip")
    xBH_data = grp.create_dataset("xBH", (N_step,3), dtype='f', compression="gzip")
    vBH_data = grp.create_dataset("vBH", (N_step,3), dtype='f', compression="gzip")
    #print(list(grp.attrs.keys()))
    #dset = f.create_dataset("mydataset", (100,), dtype='i')
    #dset.attrs['temperature'] = 99.5
    
    xBH_list = np.zeros((N_step, 3))
    vBH_list = np.zeros((N_step, 3))
    t_list   = np.zeros(N_step)
    N_update = 2500
    
    
    #--------------------------
    print("> Simulating...")
    
    
    for it in tqdm(range(N_step)):
           
        t_list[it]     = ts[it]     
        xBH_list[it,:] = 1.0*p.xBH
        vBH_list[it,:] = 1.0*p.vBH
        
        if (it%N_update == 0):
            t_data[:]     = 1.0*t_list
            xBH_data[:,:] = 1.0*xBH_list
            vBH_data[:,:] = 1.0*vBH_list
            
    
        if (method == "DKD"):
            
            p.xstep(0.5*dt)
            p.vstep(1.0*dt)
            p.xstep(0.5*dt)
    
        elif (method == "KDK"):

            p.vstep(0.5*dt)
            p.xstep(1.0*dt)
            p.vstep(0.5*dt)
            
        elif (method == "FR"):
            
            p.xstep(theta*dt/2)
            
            p.vstep(theta*dt)
            
            p.xstep((1-theta)*dt/2)
            
            p.vstep((1-2*theta)*dt)
            
            p.xstep((1-theta)*dt/2)
            
            p.vstep(theta*dt)
            
            p.xstep(theta*dt/2)
        
            
    #One final update of the output data
    t_data[:]     = 1.0*t_list
    xBH_data[:,:] = 1.0*xBH_list
    vBH_data[:,:] = 1.0*vBH_list
    
    print("> Simulation completed.")
    f.close()
    
    return IDhash
    

    
        
    
def make_plots(IDhash):
    
    f = h5py.File(f"../test_snapshots/{IDhash}.hdf5", 'r')
    ts       = np.array(f['data']['t'])
    xBH_list = np.array(f['data']['xBH'])
    vBH_list = np.array(f['data']['vBH'])
    
    N_step = len(ts-1)
    
    M_1 = f['data'].attrs["M_1"]*_MSUN
    M_2 = f['data'].attrs["M_2"]*_MSUN
    a_i = f['data'].attrs["a_i"]*_PC
    
    f.close()
    
    M_tot = M_1 + M_2
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (_G*M_tot))
    
    a_list, e_list = calc_orbital_elements(xBH_list, vBH_list, M_tot)
    
    #print(xBH_list.shape)
    #print(list(f['data'].keys()))
    
    #----------------------          
    """
    plt.figure()

    plt.loglog(r_vals/_PC, T_orb/_YR)

    plt.axhline(dt/_YR, linestyle='--', color='k')
    plt.axhline(t_end/_YR, linestyle='--', color='k')
    """
        
    #----------------------
        
    """
    plt.figure()


    rvals_i = np.sqrt(np.sum(xs_i**2, axis=-1))
    rvals = np.sqrt(np.sum(xs**2, axis=-1))

    bins = np.linspace(0, 1.5*r_max/_PC, 50)

    #r_c = 0.5*(bins[1:] + bins[:-1])

    #P_r = 4*np.pi*r_c**2*SpikeDF.rho_ini(r_c)
    #P_r /= np.trapz(P_r, r_c)
    #axes[1,0].plot(r_c, P_r, linestyle='--', color='k')

    plt.hist(rvals_i/_PC, bins, alpha=0.75)
    plt.hist(rvals/_PC, bins, alpha=0.75)   
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
        plt.axvline(i_vals[j+1], linestyle='--', color='grey')
    

    #print(ts)
    

    plt.plot(a_list/_PC)
    plt.axhline(a_i/_PC, linestyle='--', color='grey')

    plt.xlabel(r"Timestep")
    plt.ylabel(r"$a$")


    plt.tight_layout()
    

    #----------------------

    #i_peaks, _ = signal.find_peaks(a_list)
    #i_peaks = np.arange(len(a_list))
    i_vals = np.arange(len(a_list))
    a_peaks = a_list[i_vals]


    plt.figure()

    plt.plot(ts[i_vals]/T_orb, (a_peaks-a_peaks[0])/a_peaks[0])

    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta a/a$")


    plt.tight_layout()

    #----------------------

    #i_peaks, _ = signal.find_peaks(e_list)
    #i_peaks = np.arange(len(a_list))
    e_peaks = e_list[i_vals]

    plt.figure()

    plt.axhline(e_peaks[0], linestyle='--', color='grey')
    plt.plot(ts[i_vals]/T_orb, e_peaks)


    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$e$")

    #----------------------
    plt.figure()

    r_list = norm(xBH_list)

    plt.axhline(r_list[0]/_PC, linestyle='--', color='grey')
    plt.plot(ts/_S, r_list/_PC)


    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$r$ [pc]")

    #----------------------
    """
    plt.figure()

    rBH_list = np.sqrt(np.sum(xBH_list**2, axis=-1))
    plt.plot(ts/T0, 1e4*(rBH_list - r_i)/r_i)

    plt.xlabel(r"$N_\mathrm{orbits}$")
    plt.ylabel(r"$\Delta r_2/r_2$ [$10^{-4}$]")

    plt.savefig("../test_figures/test_binary_1e2_e.pdf", bbox_inches='tight')

    plt.show()
    """
    
    plt.show()
    
#def main(N_DM, noDM = False):
def main():
    print("> Running simulation...")
    ID = run_simulation(M_1 = 1000*_MSUN, M_2 = 1*_MSUN, a_i = 1e-9*_PC, e_i = 0.0, N_DM = 0, method="DKD")
    print("> Generating plots...")
    make_plots(ID)
    #make_plots()
#2**15
#main(N_DM = 2**15, noDM=False)    
    
main()

#cE6dA.hdf5 original leapfrog
    
    
