import numpy as np
import scipy.constants as const

# Physical Constants (SI units, CODATA2018)
QE    = const.elementary_charge
ME    = const.electron_mass
MP    = const.proton_mass
C     = const.speed_of_light
H     = const.Planck
KB    = const.Boltzmann
EPS0  = const.epsilon_0 # = 1.0 / (LUX * LUX * MU0)
ALPHA = const.fine_structure
KC    = 1.0 / (4.0 * const.pi * EPS0)
HBAR  = H / (2.0 * const.pi)
EPS   = const.physical_constants["neutron Compton wavelength"][0] # Small number (nucleus size) [m]
# MU0   = const.mu_0
# MUREF = MU0 / (4.0 * const.pi)

# Bohr model (SI units)
A0    = HBAR / (ME * C * ALPHA)                 # Bohr radius
MK    = KC * QE**2 / ME
# VB    = (MK / A0)**2                          # Bohr speed
TB    = 2.0 * np.pi * (A0**3 / MK)**(1.0 / 2.0) # Bohr period

# Particle number densities
NE_LOW  = 1e20
NE_MED  = 1e24
NE_HIGH = 1e28

print(TB)
