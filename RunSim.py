import units as u
import tools
import DIY

import DistributionFunctions as DF

M_1 = 1000.0*u.Msun
M_2 = 1*u.Msun

a_i = 3e-8*u.pc
e_i = 0.0

rho_sp = 226*u.Msun/u.pc**3
rho_6 = tools.calc_rho_6(rho_sp, M_1, gamma=7/3)

N_DM = 2**15

r_t = 1.5e-7*u.pc
alpha = 2

p_dressed_binary = DIY.particles_in_binary(M_1, M_2, a_i, e_i, N_DM = 2**15, r_max = 1e-3*u.pc, rho_6=rho_6, r_t=r_t, alpha=alpha)

sim = DIY.simulator(p_dressed_binary, soft_method="uniform")

eps = 2.4e-11*u.pc
sim.r_soft_sq = eps**2


T_orb = sim.p.T_orb()
N_step_per_orb = 100000

N_orb = 1
t_end = T_orb*N_orb
dt    = T_orb/N_step_per_orb

sim.run_simulation(dt, t_end, method='PEFRL', save_to_file = True, add_to_list=True)

print("> Simulation Complete:", sim.IDhash)