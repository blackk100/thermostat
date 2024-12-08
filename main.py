"""
Solves the Newton-Lorentz equation for charged particle motion in an n-body system with no external E- & B-fields.
Implements the Andersen thermostat and the Berendsen thermostat.
References:
[1] H. C. Andersen, “Molecular dynamics simulations at constant pressure and/or temperature,” The Journal of Chemical Physics, vol. 72, no. 4, pp. 2384–2393, Feb. 1980, doi: 10.1063/1.439486.
[2] H. Tanaka, K. Nakanishi, and N. Watanabe, “Constant temperature molecular dynamics calculation on Lennard-Jones fluid and its application to watera),” The Journal of Chemical Physics, vol. 78, no. 5, pp. 2626–2634, Mar. 1983, doi: 10.1063/1.445020.
[3] H. J. C. Berendsen, J. P. M. Postma, W. F. Van Gunsteren, A. DiNola, and J. R. Haak, “Molecular dynamics with coupling to an external bath,” The Journal of Chemical Physics, vol. 81, no. 8, pp. 3684–3690, Oct. 1984, doi: 10.1063/1.448118.
[4] “Andersen thermostat,” SklogWiki. [Online]. Available: http://www.sklogwiki.org/SklogWiki/index.php/Andersen_thermostat
[5] “Berendsen thermostat,” SklogWiki. [Online]. Available: http://www.sklogwiki.org/SklogWiki/index.php/Berendsen_thermostat
[6] E. A. Koopman and C. P. Lowe, “Advantages of a Lowe-Andersen thermostat in molecular dynamics simulations,” The Journal of Chemical Physics, vol. 124, no. 20, p. 204103, May 2006, doi: 10.1063/1.2198824.
[7] D. Frenkel and B. Smit, “Molecular Dynamics in Various Ensembles,” in Understanding Molecular Simulation, Elsevier, 2002, pp. 139–163. doi: 10.1016/B978-012267351-1/50008-0.
"""

import constants as const
import input as inp
import utility as util
import solvers
import numpy as np

Nt     = inp.Nt
Np     = inp.Np
ne_exp = inp.ne_exp
L      = inp.L
TB     = const.TB
QE     = const.QE

if __name__ == '__main__':
	Y0    = util.init()
	tspan = util.time()

	# No thermostat
	Y       = solvers.dynamics(Y0, tspan)
	Rx      = Y[:, 0 * Nt:1 * Nt]
	Ry      = Y[:, 1 * Nt:2 * Nt]
	Rz      = Y[:, 2 * Nt:3 * Nt]
	Vx      = Y[:, 3 * Nt:4 * Nt]
	Vy      = Y[:, 4 * Nt:5 * Nt]
	Vz      = Y[:, 5 * Nt:6 * Nt]
	t90     = util.t90(Y, tspan)
	U       = util.potential(Y, tspan)
	K       = util.kinetic(Y, tspan)
	gamma   = util.gamma(U, K)
	util.plots(Rx, Ry, tspan, U, K, "dynamics", True)

	# Andersen thermostat
	Y_a     = solvers.andersen(Y0, tspan)
	Rx_a    = Y_a[:, 0 * Nt:1 * Nt]
	Ry_a    = Y_a[:, 1 * Nt:2 * Nt]
	Rz_a    = Y_a[:, 2 * Nt:3 * Nt]
	Vx_a    = Y_a[:, 3 * Nt:4 * Nt]
	Vy_a    = Y_a[:, 4 * Nt:5 * Nt]
	Vz_a    = Y_a[:, 5 * Nt:6 * Nt]
	print(f'Times thermalized = {solvers.thermalized}')
	print(f'Average thermalization time = {np.mean(solvers.times[1:])} [s]')
	t90_a   = util.t90(Y_a, tspan)
	U_a     = util.potential(Y_a, tspan)
	K_a     = util.kinetic(Y_a, tspan)
	gamma_a = util.gamma(U_a, K_a)
	util.plots(Rx_a, Ry_a, tspan, U_a, K_a, "andersen", True)

	# Berendsen thermostat
	Y_b     = solvers.berendsen(Y0, tspan)
	Rx_b    = Y_b[:, 0 * Nt:1 * Nt]
	Ry_b    = Y_b[:, 1 * Nt:2 * Nt]
	Rz_b    = Y_b[:, 2 * Nt:3 * Nt]
	Vx_b    = Y_b[:, 3 * Nt:4 * Nt]
	Vy_b    = Y_b[:, 4 * Nt:5 * Nt]
	Vz_b    = Y_b[:, 5 * Nt:6 * Nt]
	print(f'Average scaling factor = {solvers.scales / (tspan.size - 1)}')
	t90_b   = util.t90(Y_b, tspan)
	U_b     = util.potential(Y_b, tspan)
	K_b     = util.kinetic(Y_b, tspan)
	gamma_b = util.gamma(U_b, K_b)
	util.plots(Rx_b, Ry_b, tspan, U_b, K_b, "berendsen", True)

# L_20= 5.848035476425736e-07  [m]
# T_20= 7.599149230311296e-09  [s]
# Solved!
# t90_20= 1.4734440467425525e-09  [s]
# gamma_20= 0.0008562454572336629
# Solved!
# Times thermalized = 1089
# Average thermalization time = 1.9497999213789863e-09 [s]
# t90_20= 2.3706487746890304e-10  [s]
# gamma_20= 0.03339979317200078
# Solved!
# Average scaling factor = 0.9983951800705542
# t90_20= 9.032128761105478e-10  [s]
# gamma_20= 0.025100065409883375

# L_24= 2.714417616594909e-08  [m]
# T_24= 1.519829846062259e-11  [s]
# Solved!
# t90_24= 1.1358440297836768e-12  [s]
# gamma_24= 0.012041352382266439
# Solved!
# Times thermalized = 991
# Average thermalization time = 3.794229603571491e-12 [s]
# t90_24= 4.2411735052147457e-13  [s]
# gamma_24= 0.5424995765898822
# Solved!
# Average scaling factor = 0.9982776799988087
# t90_24= 1.9964424863256985e-12  [s]
# gamma_24= 0.5632987951445297

# L_28= 1.2599210498948747e-09  [m]
# T_28= 7.599149230311296e-14  [s]
# Solved!
# t90_28= 6.017558516615495e-15  [s]
# gamma_28= 0.05662976653266693
# Solved!
# Times thermalized = 997
# Average thermalization time = 1.9276837462647754e-14 [s]
# t90_28= 1.7572747569993715e-15  [s]
# gamma_28= 3.681294057326516
# Solved!
# Average scaling factor = 0.9958954287950974
# t90_28= 3.8565796353870295e-15  [s]
# gamma_28= 2.2797307087859657
