import warnings
import numpy as np
from matplotlib import pyplot as plt
import constants as const
import input as inp

Nt     = inp.Nt
Np     = inp.Np
ne_exp = inp.ne_exp
L      = inp.L
q      = inp.q
m      = inp.m
a_p    = inp.a_p
a_e    = inp.a_e

def init() -> np.ndarray:
	"""Generate the initial state vector

	Returns:
		np.ndarray: Initial state vector of size 6
	"""

	rng = np.random.default_rng()

	Rx = rng.random(size=Nt) * L
	Ry = rng.random(size=Nt) * L
	Rz = rng.random(size=Nt) * L
	Vx = np.concatenate((rng.normal(loc=0.0, scale=a_p, size=Np), rng.normal(loc=0.0, scale=a_e, size=Np)))
	Vy = np.concatenate((rng.normal(loc=0.0, scale=a_p, size=Np), rng.normal(loc=0.0, scale=a_e, size=Np)))
	Vz = np.concatenate((rng.normal(loc=0.0, scale=a_p, size=Np), rng.normal(loc=0.0, scale=a_e, size=Np)))

	return np.concatenate((Rx, Ry, Rz, Vx, Vy, Vz))


def time() -> np.ndarray:
	"""Generate the time vector

	Returns:
		np.ndarray: Time vector of size 5000
	"""

	# Characteristic time [s]
	if   ne_exp == int(np.log10(const.NE_HIGH)):
		t =  5e2 * const.TB
	elif ne_exp == int(np.log10(const.NE_MED)):
		t =  1e5 * const.TB
	elif ne_exp == int(np.log10(const.NE_LOW)):
		t =  5e7 * const.TB
	else:
		t =  1e8 * const.TB

	print(f'T_{ne_exp:.0f}=', t, ' [s]')

	# Time interval
	tsize = 5000
	tspan = np.linspace(0.0, t, tsize)

	return tspan


def t90(
		Y    : np.ndarray,
		tspan: np.ndarray,
	) -> np.float64:
	"""Calculate the average collision time of the second species (Change in velocity direction > 90deg)

	Args:
		Y (np.ndarray): State vector of the particles at each timestep. Has a shape of ``(tsize, 6 * Number of Particles)``
		tspan (np.ndarray): Time vector of size ``tsize``

	Returns:
		np.float64: The average collision time
	"""

	tsize = np.size(tspan)

	Vx = Y[:, 3 * Nt:4 * Nt]
	Vy = Y[:, 4 * Nt:5 * Nt]
	Vz = Y[:, 5 * Nt:6 * Nt]

	# Collision Time
	# machineeps = np.finfo(np.float64).eps
	t90f = np.full((Np, tsize), np.nan) # Collision times for each particle for each timestep
	t90p = np.full( Np        , np.nan) # Average collision times for each particle

	for n in range(Np, Nt): # For each particle
		i = 0
		while i < tsize:     # For each "start" timestep
			Vi = np.array((Vx[i, n], Vy[i, n], Vz[i, n]))
			Vimag = np.linalg.norm(Vi)
			Vi /= Vimag

			j = i + 1
			while j < tsize:  # Find collision timestep
				Vf = np.array((Vx[j, n], Vy[j, n], Vz[j, n]))
				Vfmag = np.linalg.norm(Vf)
				Vf /= Vfmag

				angle = np.arccos(np.clip(np.dot(Vi, Vf), -1.0, 1.0)) # rad in [0, np.pi]

				if angle >= np.pi / 2.0:
					t90f[n - Np, i] = tspan[j] - tspan[i]

					# Next start timestep is current collision timestep
					i = j - 1
					break

				j += 1

			if np.isnan(t90f[n - Np, i]): # Particle did not undergo collisions
				break

			i += 1

	# If only nans exist in t90f, the calculation for t90p will raise a RuntimeWarning and return a nan
	# Similarly with the calculation for t90
	# This block suppresses the RuntimeWarning
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		t90p = np.nanmean(t90f, axis=1) # Find average collision time for each particle
		t90  = np.nanmean(t90p)         # Find average collision time of the system

	print(f't90_{ne_exp:.0f}=', t90, ' [s]')
	return t90


def potential(
		Y    : np.ndarray,
		tspan: np.ndarray,
	) -> np.ndarray:
	"""Calculate the potential energy of the system (= sum(kc * qi * qj / r_ij) for every particle i, j for i != j)

	Args:
		Y (np.ndarray): State vector of the particles at each timestep. Has a shape of ``(tsize, 6 * Number of Particles)``
		tspan (np.ndarray): Time vector of size ``tsize``

	Returns:
		np.ndarray: Vector of the total kinetic energy of size ``tsize ``.
	"""

	tsize = np.size(tspan)

	Rx = Y[:, 0 * Nt:1 * Nt]
	Ry = Y[:, 1 * Nt:2 * Nt]
	Rz = Y[:, 2 * Nt:3 * Nt]

	# Potential Energy
	U = np.zeros(tsize)
	for i in range(Nt - 1):
		Ri = np.array((Rx[:, i], Ry[:, i], Rz[:, i]))

		for j in range(i + 1, Nt):
			Rj = np.array((Rx[:, j], Ry[:, j], Rz[:, j]))

			U += q[i] * q[j] / np.linalg.norm(Ri - Rj, axis=0)
	U = U * const.KC

	return U


def kinetic(
		Y    : np.ndarray,
		tspan: np.ndarray
	) -> np.ndarray:
	"""Calculate the kinetic energy of the system (= sum(m v^2 / 2) for every particle)

	Args:
		Y (np.ndarray): State vector of the particles at each timestep. Has a shape of ``(tsize, 6 * Number of Particles)``
		tspan (np.ndarray): Time vector of size ``tsize``

	Returns:
		np.ndarray: Vector of the total kinetic energy of size ``tsize ``.
	"""

	tsize = np.size(tspan)

	Vx = Y[:, 3 * Nt:4 * Nt]
	Vy = Y[:, 4 * Nt:5 * Nt]
	Vz = Y[:, 5 * Nt:6 * Nt]

	K = np.zeros(tsize)
	for i in range(Nt):
		Vi = np.linalg.norm(np.array((Vx[:, i], Vy[:, i], Vz[:, i])), axis=0)

		K += m[i] * Vi**2
	K /= (2.0 * Nt)

	# T = 2 * K / (3 * Nt * const.KB)

	return K


def gamma(
		U: np.ndarray,
		K: np.ndarray
	) -> np.float64:
	"""Calculate the plasma coupling parameter (= < |U| > / < K >)

	Args:
		U (np.ndarray): Vector of the total potential energy of size ``tsize``
		K (np.ndarray): Vector of the total kinetic energy of size ``tsize``

	Returns:
		np.float64: Plasma coupling parameter
	"""

	gamma = np.mean(np.abs(U)) / np.mean(K)

	print(f'gamma_{ne_exp:.0f}=', gamma)
	return gamma


def plots(
		Rx    : np.ndarray,
		Ry    : np.ndarray,
		tspan : np.ndarray,
		U     : np.ndarray,
		K     : np.ndarray,
		append: str        = "",
		pdf   : bool       = True
	) -> None:
	"""Generate and save plots. Filenames will be 'type_{append}_{ne_exp}.{png|pdf}'.

	Args:
		Rx (np.ndarray): Vector of particle X-Coordinates of size ``Nt``
		Ry (np.ndarray): Vector of particle Y-Coordinates of size ``Nt``
		tspan (np.ndarray): Time vector of size ``tsize``
		U (np.ndarray): Vector of system potential energy of size ``tsize``
		K (np.ndarray): Vector of system kinetic energy of size ``tsize``
		append (str): String to append to filename. Defaults to an empty string.
		pdf (bool): Flag to indicate if a vector graphics .pdf is generated. Defaults to True.
	"""

	file_append = f'{append}_{ne_exp:.0f}.{'pdf' if pdf else 'png'}'
	tmax = tspan[-1] / const.TB

	fig = plt.figure(1)
	fig, axes = plt.subplots()
	axes.set_xlim(0, 1)
	axes.set_ylim(0, 1)
	axes.grid(True)
	axes.set_xlabel('x / L')
	axes.set_ylabel('y / L')
	axes.plot(Rx[:, Np:Nt] / L, Ry[:, Np:Nt] / L, 'b.', markersize=1, label='Electrons')
	axes.plot(Rx[:,  0:Np] / L, Ry[:,  0:Np] / L, 'r.', markersize=1, label='Protons')
	handles, labels = axes.get_legend_handles_labels()
	labels, ids = np.unique(labels, return_index=True)
	handles = [handles[i] for i in ids]
	axes.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.0))
	fig.savefig(f'trajectories_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()

	fig = plt.figure(2)
	fig, axes = plt.subplots()
	axes.set_xlim(0, tmax)
	axes.grid(True)
	axes.set_xlabel(r'Time / $\tau_b$')
	axes.set_ylabel('Potential Energy [eV]')
	axes.plot(tspan / const.TB, U / const.QE, 'k-', markersize=1)
	fig.savefig(f'U_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()

	fig = plt.figure(3)
	fig, axes = plt.subplots()
	axes.set_xlim(0, tmax)
	axes.grid(True)
	axes.set_xlabel(r'Time / $\tau_b$')
	axes.set_ylabel('Average Kinetic Energy [eV]')
	axes.plot(tspan / const.TB, K / const.QE, 'k-', markersize=1)
	axes.axhline(y=(inp.T0 * 3.0 * const.KB / (2.0 * const.QE)), color='r', linestyle='--', markersize=1)
	fig.savefig(f'K_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()
