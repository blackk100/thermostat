import numpy as np
import constants as const
import input as inp

Nt = inp.Nt
Np = inp.Np
ne = inp.ne
q  = inp.q
m  = inp.m
L  = inp.L

# Andersen thermostat statistics
thermalized = 0
times       = [0.0]

# Berendsen thermostat statistics
scales      = 0.0


# Newton-Lorentz Equation
def nl(
		time: float,
		X   : np.ndarray
	) -> np.ndarray:
	"""Calculate particle dynamics using the Newton-Lorentz equations

	Args:
		time (float): Current time coordinate
		X (np.ndarray): State vector of size 6 * ``Nt``

	Returns:
		np.ndarray: State vector at the next time-step
	"""

	# Setup
	r = np.reshape(X[0 * Nt:3 * Nt], (Nt, 3), order='F')
	v = np.reshape(X[3 * Nt:6 * Nt], (Nt, 3), order='F') # = r_dot
	a = np.zeros((Nt, 3)) # = v_dot

	# Solve for derivatives
	for i in range(Nt):
		for j in range(Nt):
			if (j != i):
				r_ij  = r[i, :] - r[j, :]
				r_ij -= np.floor(r_ij / L) * L # Periodic cell

				a[i, :] += const.KC * q[i] * q[j] * r_ij / ((np.linalg.norm(r_ij)**3) * m[i])

	return np.concatenate((v.flatten('F'), a.flatten('F')))


# RK4
def dynamics(
		X0  : np.ndarray,
		time: np.ndarray
	) -> np.ndarray:
	"""Performs Runge-Kutta (4th order) integration without a thermostat.

	Args:
		X0 (np.ndarray): Initial state vector of size 6 * ``Nt``
		time (np.ndarray): Independent variable to integrate along

	Returns:
		np.ndarray: Array of state vectors for each value of ``x``
	"""

	# Setup RK4
	tspan    = np.size(time)
	num_vars = np.size(X0)
	h        = time[1] - time[0]

	X = np.zeros((tspan, num_vars))
	X[0, :] = X0

	# Run RK4
	for n in range(0, tspan - 1):
		k1 = h * nl(time[n]          , X[n, :]           )
		k2 = h * nl(time[n] + h / 2.0, X[n, :] + k1 / 2.0)
		k3 = h * nl(time[n] + h / 2.0, X[n, :] + k2 / 2.0)
		k4 = h * nl(time[n] + h      , X[n, :] + k3      )

		X[n + 1, :] = X[n, :] + k1 / 3.0 + k2 / 6.0 + k3 / 6.0 + k4 / 3.0
		X[n + 1, 0 * Nt:3 * Nt] -= np.floor(X[n + 1, 0 * Nt:3 * Nt] / L) * L # Periodic cell

	print("Solved!")
	return X


# RK4 + Andersen thermostat
def andersen(
		X0  : np.ndarray,
		time: np.ndarray
	) -> np.ndarray:
	"""Performs Runge-Kutta (4th order) integration with an Andersen thermostat [1].

	Args:
		X0 (np.ndarray): Initial state vector of size 6 * ``Nt``
		time (np.ndarray): Independent variable to integrate along

	Returns:
		np.ndarray: Array of state vectors for each value of ``x``
	"""

	# Setup RK4
	tspan    = np.size(time)
	num_vars = np.size(X0)
	h        = time[1] - time[0]

	X = np.zeros((tspan, num_vars))
	X[0, :] = X0

	# Setup thermostat
	global thermalized, times

	A     = 1 # Non-dimensional model constant.                  []
	kappa = 1 # Thermal conductivity of the system. Assumed == 1 [W / m / K]

	# Set A such that `nu * h ~=> 1%`
	if   ne == const.NE_HIGH:
		A	 = 2.1627e2
	elif ne == const.NE_MED:
		A    = 5.0190e-2
	elif ne == const.NE_LOW:
		A    = 4.6593e-6

	nu_c = 2 * A * kappa / (3 * const.KB * ne**(1.0 / 3.0)) # Collision frequency for 1 particle
	nu   = nu_c * Nt**(-2.0 / 3.0)                          # Collision frequency for ensemble
	rng  = np.random.default_rng()

	# Run RK4
	for n in range(0, tspan - 1):
		k1 = h * nl(time[n]          , X[n, :]           )
		k2 = h * nl(time[n] + h / 2.0, X[n, :] + k1 / 2.0)
		k3 = h * nl(time[n] + h / 2.0, X[n, :] + k2 / 2.0)
		k4 = h * nl(time[n] + h      , X[n, :] + k3      )

		X[n + 1, :] = X[n, :] + k1 / 3.0 + k2 / 6.0 + k3 / 6.0 + k4 / 3.0
		X[n + 1, 0 * Nt:3 * Nt] -= np.floor(X[n + 1, 0 * Nt:3 * Nt] / L) * L # Periodic cell

		# Apply thermostat
		for i in range(Nt):
			if (rng.random() < nu * h): # Check if the particle collides with the bath
				vi = rng.normal(loc=0.0, scale=(inp.a_p if i < Np else inp.a_e), size=3)

				X[n + 1, 3 * Nt + i] = vi[0]
				X[n + 1, 4 * Nt + i] = vi[1]
				X[n + 1, 5 * Nt + i] = vi[2]

				thermalized += 1
				times.append(time[n] - times[-1])

	print("Solved!")
	return X


# RK4 + Berendsen thermostat
def berendsen(
		X0  : np.ndarray,
		time: np.ndarray
	) -> np.ndarray:
	"""Performs Runge-Kutta (4th order) integration with a Berendsen thermostat [3].

	Args:
		X0 (np.ndarray): Initial state vector of size 6 * ``Nt``
		time (np.ndarray): Independent variable to integrate along

	Returns:
		np.ndarray: Array of state vectors for each value of ``x``
	"""

	global scales

	# Setup RK4
	tspan    = np.size(time)
	num_vars = np.size(X0)
	h        = time[1] - time[0]

	X = np.zeros((tspan, num_vars))
	X[0, :] = X0

	# Setup thermostat
	tau = h * 100 # The system is coupled to the thermostat bath ~every tau [s] / 100 timesteps

	# Run RK4
	for n in range(0, tspan - 1):
		k1 = h * nl(time[n]          , X[n, :]           )
		k2 = h * nl(time[n] + h / 2.0, X[n, :] + k1 / 2.0)
		k3 = h * nl(time[n] + h / 2.0, X[n, :] + k2 / 2.0)
		k4 = h * nl(time[n] + h      , X[n, :] + k3      )

		X[n + 1, :] = X[n, :] + k1 / 3.0 + k2 / 6.0 + k3 / 6.0 + k4 / 3.0
		X[n + 1, 0 * Nt:3 * Nt] -= np.floor(X[n + 1, 0 * Nt:3 * Nt] / L) * L # Periodic cell

		# Apply thermostat
		v = np.reshape(X[n + 1, 3 * Nt:6 * Nt], (Nt, 3), order='F')

		Vmag = np.linalg.norm(v, axis=1)
		K    = np.mean(Vmag**2 * m / 2.0) # Average kinetic energy
		T    = 2 * K / (3 * const.KB)     # Instantaneous temperature

		scale = np.sqrt(1 + (inp.T0 / T - 1) * h / tau) # Scaling factor (lambda)
		v    *= scale

		X[n + 1, 3 * Nt:6 * Nt] = v.flatten('F')

		scales += scale

	print("Solved!")
	return X
