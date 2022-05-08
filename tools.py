import numpy as np

from scipy.integrate import cumtrapz
import string
import random

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