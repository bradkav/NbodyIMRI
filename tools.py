import numpy as np

from scipy.integrate import cumtrapz
import string
import random

import units as u

def inverse_transform_sample(integrand, x_min, x_max, N_samps=1, N_grid=1000, log=True):
    if (log == True):
        x_grid = np.geomspace(x_min, x_max, N_grid)
    else:
        x_grid = np.linspace(x_min, x_max, N_grid)
        
    P_grid = cumtrapz(integrand(x_grid), x_grid, initial = 0.0)
    P_grid /= P_grid[-1]
    
    u = np.random.rand(N_samps)
    x_samps = np.interp(u, P_grid, x_grid)
    return x_samps
    
    
def get_random_direction():
    costheta = 2*np.random.rand() - 1
    theta    = np.arccos(costheta)
    phi      = 2*np.pi*np.random.rand()
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    
def generate_hash(length=5):
    h = np.zeros(length, dtype=str)
    for i in range(length):
        h[i] = random.choice(string.hexdigits)
    return ''.join(h)
    
    
def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))

def calc_orbital_elements(x, v, M_tot):
    x_mag = norm(x)
    v_mag = norm(v)
    mu = u.G_N*M_tot

    a = (2/x_mag - v_mag**2/mu)**-1
        #https://astronomy.stackexchange.com/questions/29005/calculation-of-eccentricity-of-orbit-from-velocity-and-radius
    h = np.cross(x,v)
    e_vec = np.cross(v,h)/mu - x/np.atleast_2d(x_mag).T
    e = norm(e_vec)
        
    return a, e
    
def calc_rho_6(rho_sp, M_1, gamma):
    r_6   = 1e-6*u.pc
    k = (3-gamma)*(0.2)**(3-gamma)/(2*np.pi)
    rho_6 = (rho_sp)**(1-gamma/3)*(k*M_1)**(gamma/3)*r_6**-gamma
    return rho_6
    
def calc_risco(M_1):
    return 6*u.G_N*M_1/u.c**2