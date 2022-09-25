import units as u
import tools
import particles
import simulator

import DistributionFunctions as DF

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m_1', '--m_1', type=float, default=1e3)
parser.add_argument('-a_i', '--a_i', type=float, default=100)
parser.add_argument('-N_orb', '--N_orb', type=float, default=100)
parser.add_argument('-e_i', '--e_i', type=float, default=0)

args = parser.parse_args()

M_1 = args.m_1*u.Msun
M_2 = 1*u.Msun

a_i = args.a_i*tools.calc_risco(M_1)
e_i = args.e_i

#rho_sp = 226*u.Msun/u.pc**3
#rho_6 = tools.calc_rho_6(rho_sp, M_1, gamma=7/3)

rho_6 = 1e16*u.Msun/u.pc**3

N_DM = 2**20

r_t = 20*a_i
alpha = 2

print("> Initialising system...")

p_dressed_binary = particles.particles_in_binary(M_1, M_2, a_i, e_i, N_DM = N_DM, r_max = 1.0*u.pc, rho_6=rho_6, r_t=r_t, alpha=alpha,dynamic_BH=True)

sim = simulator.simulator(p_dressed_binary, soft_method="uniform")

b_max = a_i*(M_2/(3*M_1))**(1/3)
#eps = 3e-11*u.pc
#eps = 0.015*b_max
eps = 0.001*b_max 
sim.r_soft_sq = eps**2


T_orb = sim.p.T_orb()
N_step_per_orb = 1000

N_orb = args.N_orb
t_end = T_orb*N_orb
dt    = T_orb/N_step_per_orb

print("> Running simulation...")

sim.run_simulation(dt, t_end, method='PEFRL', save_to_file = True, add_to_list=True)

print("> Simulation Complete:", sim.IDhash)
