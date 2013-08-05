#!/usr/bin/env python

import numpy as np
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components import *

# TODO: astropy units for spectrum

# ----------------
# Read in spectrum
# ----------------
datafile = "../Data/LBQS_qsotemplate.dat"
wavelengths, flux = np.loadtxt(datafile, unpack=True)

spectrum = Spectrum()
spectrum.flux = flux
spectrum.wavelengths = wavelengths

# -----------------
# Create components
# -----------------
nuclear_comp = NuclearContinuumComponent()

# ------------
# Create model
# ------------
model = Model()

model.spectrum = spectrum
model.components.append(nuclear_comp)

model.run_mcmc(n_walkers=200)

# try:
# 	cf.run_mcmc_analysis(plot=False)
# except MCMCDidNotConverge:
# 	...







