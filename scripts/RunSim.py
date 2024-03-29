from NbodyIMRI import units as u
from NbodyIMRI import tools
from NbodyIMRI import particles
from NbodyIMRI import simulator

from NbodyIMRI import distributionfunctions as DF

import NbodyIMRI
NbodyIMRI.snapshot_dir = "/gpfs/users/kavanaghb/IMRI_Nbody/test_snapshots"

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m_1', '--m_1', type=float, default=1e3)
parser.add_argument('-a_i', '--a_i', type=float, default=100)
parser.add_argument('-N_orb', '--N_orb', type=float, default=100)
parser.add_argument('-e_i', '--e_i', type=float, default=0)
parser.add_argument('-N_DM', '--N_DM', type=int, default=2**14)
parser.add_argument('-x_eps', '--x_eps', type=float, default=0.025)
parser.add_argument('-label', '--label', type=str, default=None)

args = parser.parse_args()

M_1 = args.m_1*u.Msun
M_2 = 1*u.Msun

a_i = args.a_i*tools.calc_risco(M_1)
e_i = args.e_i


label = args.label

#print(label)
#rho_sp = 226*u.Msun/u.pc**3
#rho_6 = tools.calc_rho_6(rho_sp, M_1, gamma=7/3)

#Change this normalisation such that the errors are smaller? 
#Fix the mass ratio between the DM particles and the secondary BH?
#rho_6 = ((M_1/(1000*u.Msun))**(7/3))*1e16*u.Msun/u.pc**3
rho_6 = 1e16*u.Msun/u.pc**3

#N_DM = 2**14
N_DM = args.N_DM
x_eps = args.x_eps


r_t = 20*a_i
alpha = 2

print("> Initialising system...")

p_dressed_binary = particles.particles_in_binary(M_1, M_2, a_i, e_i, N_DM = N_DM, r_max = 1.0*u.pc, rho_6=rho_6, r_t=r_t, alpha=alpha,dynamic_BH=True, circular = 0, include_DM_mass=True)

sim = simulator.simulator(p_dressed_binary, soft_method="empty_shell")

b_max = a_i
#b_max = a_i*(M_2/(3*M_1))**(1/3)
#eps = 3e-11*u.pc
#eps = 0.015*b_max
eps = x_eps*b_max 
eps1 = 0.1*a_i
print(" Softening length [pc]:", eps/u.pc)

sim.r_soft_sq  = eps**2
sim.r_soft_sq1 = eps1**2

T_orb = sim.p.T_orb()
N_step_per_orb = 10000

N_orb = args.N_orb
t_end = T_orb*N_orb
dt    = T_orb/N_step_per_orb

print(" Semi-major axis [pc]:", a_i/u.pc)

#print(label)

#assert (1 == 0)

print("> Running simulation...")

sim.run_simulation(dt, t_end, method='PEFRL', save_to_file = True, add_to_list=True, save_DM_states=False, N_save=100, label=label)

print("> Simulation Complete:", sim.IDhash)
