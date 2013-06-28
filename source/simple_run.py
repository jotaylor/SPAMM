#!/usr/bin/python

import numpy as np
import module

# ----------------
# Read in spectrum
# ----------------
datafile = "../Data/LBQS_qsotemplate.dat"
wavelengths, flux = np.loadtxt(datafile, unpack=True)

spectrum = module.Spectrum()
spectrum.flux = flux
spectrum.wavelengths = wavelengths

# -----------------
# Create components
# -----------------
nuclear_comp = module.NuclearComponent()

# ------------
# Create model
# ------------
model = module.Model()

model.spectrum = spectrum
model.components.append(nuclear_comp)

model.run_mcmc(n_walkers=200)

# try:
# 	cf.run_mcmc_analysis(plot=False)
# except MCMCDidNotConverge:
# 	...







