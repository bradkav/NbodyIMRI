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

#_MSUN   = 1.98855e30 #kg
_MSUN   = 1.0
_KG     = 1/1.98855e30
_PC     = 3.08567758149137e16 #m
_CM     = 1e-2
_YR     = 365.25 * 24 * 3600*_S #s
_MYR    = 1e6*_YR
_GEV    = 1.79e-27*_KG
_KM     = 1e3


_G      = 4.302e-3*(_PC/_MSUN)*(_KM/_S)**2
_C      = 3e8


theta = 1/(2 - 2**(1/3))

TEST_SNAP_DIR = "test_snapshots"
TEST_FIG_DIR = "test_figures"

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
    def __init__(self, M_1, M_2, N_DM=2, M_DM = 0, dynamic_BH=True, r_soft=0):
    
        self.M_1 = M_1
        self.M_2 = M_2
        
        self.M_DM = M_DM
        self.N_DM = N_DM
        
        #self.r_isco1 = 6*_G*M_1/_C**2
        #self.r_isco2 = 6*_G*M_2/_C**2
        
        #self.r_isco2 = 10*1.8849555921538754e-10*_PC
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
        rsq = np.atleast_2d(r**2 + 0.0*self.r_isco2**2).T
        acc_DM2 = -_G*self.M_2*dx/rsq**1.5
        
        #print((rsq < self.r_isco2**2).shape)
        inds = rsq < self.r_isco2**2
        
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
        #self.dvdtBH1 = -_G*self.M_1*self.xBH/norm(self.xBH)**3
        if (self.dynamic_BH):
            self.vBH1 += self.dvdtBH1*h
        self.vBH2 += self.dvdtBH2*h
        self.vDM += self.dvdtDM*h

        
    
    
def run_simulation(M_1, M_2, a_i, e_i, N_DM = 0, gamma = 7/3, r_max = 1e-6*_PC, method = "DKD", dynamic_BH = True):

 
    #Initialise BH properties
    r_i = a_i * ( 1 + e_i)
    
    if (dynamic_BH):
        M_tot = M_1 + M_2
    else:
        M_tot = M_1
    mu = _G*M_tot
    v_i = np.sqrt( mu * (2.0/r_i - 1.0/a_i) )
 
    # Simulation parameters
    N_step = 10000
    N_orb = 100
    
    T_orb = 2 * np.pi * np.sqrt(a_i ** 3 / (_G*M_tot))
    t_end = N_orb*T_orb
    dt = t_end/N_step
    print("> Timestep [s]:", dt)
    
    #N = 0
    #r_soft = (N**2*dt**2*_G*M_2)**(1/3) #DO NOT CHANGE _ THIS SEEMS TO WORK!!!
    #r_soft = 2*np.pi*N*a_i*dt/T_orb
    print("> Drift per timestep [pc]: ", v_i*dt/_PC)
    #r_soft = 10000*6*_G*M_2/_C**2
    r_soft = v_i*dt
    print("> Softening length [pc]: ",r_soft/_PC)
    
    # Initialise central potential and binary orbit
    SpikeDF = DF.SpikeDistribution(M_1/_MSUN, rho_6=1e15, gamma_sp=gamma)
    
    if (N_DM > 0):
        M_spike = SpikeDF.M_DM_ini(r_max/_PC)*_MSUN
        M_DM    = (M_spike/N_DM)
    else:
        N_DM = 2 #Keep N_DM = 2 so that all the arrays work as expected...
        M_DM = 0.0
        
    p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM, dynamic_BH=dynamic_BH, r_soft=r_soft)
        
    

    
    #-----------------------------
    
    if (dynamic_BH):
        factor = M_2/M_tot
    else:
        factor = 0
        
    
    p.xBH1[:] = np.atleast_2d([-r_i*factor,   0, 0])
    p.xBH2[:] = np.atleast_2d([r_i*(1-factor),   0, 0])
    
    p.vBH1[:] = np.atleast_2d([0.0, v_i*factor, 0])
    p.vBH2[:] = np.atleast_2d([0.0, -v_i*(1-factor), 0])
    

    
    elements = calc_orbital_elements(p.xBH1 - p.xBH2 , p.vBH1 - p.vBH2, M_tot)
    print("(a_i, e_i):", float(elements[0]/_PC), ",", float(elements[1]))
    print("DM pseudoparticle mass [Msun]:", M_DM/_MSUN)
    
    b_90 = _G*M_2/v_i**2
    

    
    #Initialise DM properties
    print("> Initialising...")
    r, v = SpikeDF.draw_particle(r_max/_PC, N = N_DM)
    

    for i in range(N_DM):
        rhat = tools.get_random_direction()
        vhat = tools.get_random_direction()
        p.xDM[i,:] = r[i]*rhat * _PC
        p.vDM[i,:] = v[i]*vhat * _PC/_MYR
        
    p.xDM += p.xBH1
    p.vDM += p.vBH1
    
    print(b_90/_PC)
    print(p.r_isco2/_PC)
    print(v_i*dt/_PC)
    
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
    f = h5py.File(f"{TEST_SNAP_DIR}/{IDhash}.hdf5", "w")
    grp = f.create_group("data")
    grp.attrs['M_1'] = M_1/_MSUN
    grp.attrs['M_2'] = M_2/_MSUN
    grp.attrs['a_i'] = a_i/_PC
    grp.attrs['e_i'] = e_i
    grp.attrs['N_DM'] = N_DM
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
    
    xBH1_list = np.zeros((N_step, 3))
    vBH1_list = np.zeros((N_step, 3))
    
    xBH2_list = np.zeros((N_step, 3))
    vBH2_list = np.zeros((N_step, 3))
    t_list   = np.zeros(N_step)
    N_update = 2500
    
    
    #--------------------------
    print("> Simulating...")

    
    for it in tqdm(range(N_step)):
           
        t_list[it]     = ts[it]     
        xBH1_list[it,:] = p.xBH1
        vBH1_list[it,:] = p.vBH1
        
        xBH2_list[it,:] = p.xBH2
        vBH2_list[it,:] = p.vBH2
        
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
        
            
    #One final update of the output data
    t_data[:]     = 1.0*t_list
    
    xBH1_data[:,:] = 1.0*xBH1_list
    vBH1_data[:,:] = 1.0*vBH1_list
    
    xBH2_data[:,:] = 1.0*xBH2_list
    vBH2_data[:,:] = 1.0*vBH2_list
    
    print("> Simulation completed.")
    f.close()
    
    
    return IDhash
    

    
        
    
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
    
    M_1 = f['data'].attrs["M_1"]*_MSUN
    M_2 = f['data'].attrs["M_2"]*_MSUN
    a_i = f['data'].attrs["a_i"]*_PC
    dynamic_BH = f['data'].attrs["dynamic"]

    
    f.close()
    
    if (dynamic_BH == 1):
        M_tot = M_1 + M_2
    else:
        M_tot = 1.0*M_1
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
        #plt.axvline(i_vals[j+1], linestyle='--', color='grey')
    

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

    #---------------------
    
    plt.figure()
    
    plt.plot(norm(xBH1_list)/_PC)
    plt.xlabel(r"$r_1$")

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
    ID = run_simulation(M_1 = 100*_MSUN, M_2 = 1*_MSUN, a_i = 1e-9*_PC, e_i = 0.0, N_DM = 10000, method="FR", dynamic_BH=True)
    print("> Generating plots...")
    make_plots(ID)
    #make_plots()
#2**15
#main(N_DM = 2**15, noDM=False)    
    
main()

#cE6dA.hdf5 original leapfrog
    
    
