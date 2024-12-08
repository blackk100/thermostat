import numpy as np
import constants as const

# Number of particles
Nt     = 20            # Total number of particles
Np     = int(Nt / 2.0) # Number of particles per species
ne     = const.NE_HIGH
ne_exp = int(np.log10(ne))

# Characteristic size of the domain [m]
# L = 100.0 * a0
L = (Nt / ne)**(1.0 / 3.0)
print(f'L_{ne_exp:.0f}=', L, ' [m]')

# Charge and Mass
q = np.concatenate((np.full(Np, const.QE), np.full(Np, -const.QE)))
m = np.concatenate((np.full(Np, const.MP), np.full(Np,  const.ME)))

# Particle Temperature
T0  = const.QE / const.KB # = 1.0 eV -> K
a_p = np.sqrt(const.KB * T0 / const.MP) # Standard deviation of the velocity distribution of the protons
a_e = np.sqrt(const.KB * T0 / const.ME) # Standard deviation of the velocity distribution of the electrons
