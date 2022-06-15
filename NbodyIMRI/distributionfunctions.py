import numpy as np
from scipy.special import gamma as Gamma_func
from scipy.integrate import cumtrapz
import sys

from NbodyIMRI import tools
import constants as const
from scipy.special import hyp2f1

from abc import ABC


"""
Units:
    - Masses in MSun
    - Times in Myr
    - Distances in pc

Some definitions:
    - E is always defined as the relative energy E = Psi(r) - 0.5*v^2,
         which is everywhere positive (for bound particles)

"""

r_6 = 1e-6

class SpikeDistribution(ABC):
    
    
    def f_ini(self, E):
        pass
    
    def rho_ini(self, r):
        pass
        
    def M_DM_ini(self, r):
        pass
    
    def Psi(self, r):
        return const.G_N * self.M_BH/r
    
    def v_max(self, r):
        return np.sqrt(2*self.Psi(r)) 
         
    def L_max(self, E):
        return const.G_N * self.M_BH/(np.sqrt(2)*E)

    def f_v_ini(self, v, r):
        f_v = 0.0*v
        mask = v < self.v_max(r)
        E = self.Psi(r) - 0.5*v**2
        f_v[mask] = 4*np.pi*v[mask]**2*self.f_ini(E[mask])/self.rho_ini(r)
        return f_v

    def density_of_states_E(self, E):
        mu = const.G_N * self.M_BH
        return np.sqrt(2) * np.pi**3 * mu**3 * E**(-5/2)
        
    def density_of_states_E_L(self, E, L):
        mu = const.G_N * self.M_BH
        return 4 * np.sqrt(2) * np.pi**3 * L * mu * E**-(3/2)
        
    def draw_radius(self, r_max, r_min = -1, N = 1):
        if (r_min < 0):
            r_min = self.r_min
        r_grid = np.geomspace(r_min, r_max, 1000)
        M_grid = self.M_DM_ini(r_grid)
        P_grid = (M_grid - M_grid[0])/(M_grid[-1] - M_grid[0])  
        u = np.random.rand(N)
        r = np.interp(u, P_grid, r_grid)
        return r
        
    def draw_velocity(self, r, N = 1):
        integ = lambda v: self.f_v_ini(v, r)
        v = tools.inverse_transform_sample(integ, 0, self.v_max(r), N,  N_grid = 1000, log=False)
        return v
        
    def draw_particle(self, r_max, r_min = -1, N = 1):
        """
        
        """
    
        if (N > 1):
            r = self.draw_radius(r_max, r_min, N)
            v = 0.0*r
            for i, ri in enumerate(r):
                v[i] = self.draw_velocity(ri, N=1)
        elif (N == 1):
            r = self.draw_radius(r_max, r_min, 1)
            v = self.draw_velocity(r, 1)
        else:
            sys.exit("Error: N must be positive!")
            
            
        return r, v
        
    def draw_E(self, r_max, N = 1):
        E_min = self.Psi(r_max)
        E_max = self.Psi(self.r_min)
        integ = lambda E: self.density_of_states_E(E)*self.f_ini(E)
        E = tools.inverse_transform_sample(integ, E_min, E_max, N, N_grid=1000, log=True)
        return E
    
    def draw_L(self, E, N = 1):
        u = np.random.rand(N)
        L = self.L_max(E)*np.sqrt(u)
        return L
    
    def draw_E_L(self, r_max, N = 1):
        if (N > 1):
            E = self.draw_E(r_max, N)
            L = 0.0*E
            for i, Ei in enumerate(E):
                L[i] = self.draw_L(Ei, N = 1)
        elif (N == 1):
            E = self.draw_E(r_max, 1)
            L = self.draw_L(E, 1)
        else:
            sys.exit("Error: N must be positive!")
        
        return E, L
    
    def reconstruct_rho(self, r):
        v = np.linspace(0, 0.999*self.v_max(r))
        f_v = 0.0*v
        E = self.Psi(r) - 0.5*v**2
        f_v = 4*np.pi*v**2*self.f_ini(E)
        return np.trapz(f_v, v)
        
    

class PowerLawSpike(SpikeDistribution):
    """
    
    """

    def __init__(self, M_BH, rho_6, gamma_sp = 7.0/3.0, rho_core = -1):
        
        self.M_BH   = M_BH
        self.rho_6  = rho_6
        #self.gamma_i = gamma_i
        #self.gamma_sp = (9-2*gamma_i)/(4-gamma_i)
        self.gamma_sp = gamma_sp
        if (rho_core > 0):
            self.cored = True
            self.rho_core = rho_core
        else:
            self.cored = False
            self.rho_core = 0.0
        
        self.r_isco = 6.0 * const.G_N * M_BH / const.C ** 2
        
        #Strictly speaking, r_min should be 2*r_isco (https://arxiv.org/abs/1305.2619)
        self.r_min  = self.r_isco
    
    def f_ini(self, E):
        A1 = r_6 / (const.G_N * self.M_BH)
        return (
            self.gamma_sp * (self.gamma_sp - 1)*self.rho_6
            * (2*np.pi) ** -1.5 * A1 ** self.gamma_sp  
            * (Gamma_func(self.gamma_sp - 1) / Gamma_func(self.gamma_sp - 1/2))
            * E ** (self.gamma_sp - 3/2)
        )
        
    def rho_ini(self, r):
        return self.rho_6*(r/r_6)**(-self.gamma_sp)
        
    def M_DM_ini(self, r):
        x = r/r_6
        return (4*np.pi*r_6**3*self.rho_6/(3 - self.gamma_sp))*x**(3 - self.gamma_sp)
        
        
        
class GeneralizedNFWSpike(SpikeDistribution):
    """
    
    """

    def __init__(self, M_BH, rho_6, gamma_sp, r_t, alpha):
        
        self.M_BH   = M_BH
        self.rho_6  = rho_6
        self.r_t    = r_t
        self.gamma_sp = gamma_sp
        self.alpha  = alpha
        
        self.r_isco = 6.0 * const.G_N * M_BH / const.C ** 2
        
        #Strictly speaking, r_min should be 2*r_isco (https://arxiv.org/abs/1305.2619)
        self.r_min  = self.r_isco
    
    def f_ini(self, E):
        A1 = r_6 / (const.G_N * self.M_BH)
        psi_t =  (const.G_N * self.M_BH)/self.r_t
        alpha = self.alpha
        gamma = self.gamma_sp
        
        AB_pre = 4*(gamma - 1)/(4*alpha**2 + 8*alpha*gamma + 4*gamma**2 - 1)
        A = psi_t*(2*alpha + 2*gamma + 1)*hyp2f1(alpha+2, alpha + gamma, alpha+gamma + 1/2, -E/psi_t)
        B = gamma*E*hyp2f1(alpha + 2, alpha + gamma + 1, alpha + gamma + 3/2, -E/psi_t)
        C = (psi_t**2/E)*hyp2f1(alpha + 2, alpha + gamma - 1, alpha + gamma - 1/2, -E/psi_t) 
        return (
            self.rho_6 * (2*np.pi) ** -1.5 * (psi_t**(-alpha-2)) *(A1 ** gamma)
            * (Gamma_func(alpha + gamma + 1) / Gamma_func(alpha + gamma - 1/2))
            * E ** (alpha + gamma - 1/2)
            * (AB_pre*(A + B) + C)
        )
        
    def rho_ini(self, r):
        return self.rho_6*(r/r_6)**(-self.gamma_sp)*(1 + r/self.r_t)**-self.alpha
        
    def M_DM_ini(self, r):
        x = r/r_6
        H = hyp2f1(self.alpha, 3 - self.gamma_sp, 4 - self.gamma_sp, -r/self.r_t)
        return (4*np.pi*r_6**3*self.rho_6/(3 - self.gamma_sp))*x**(3 - self.gamma_sp)*H

        

        
