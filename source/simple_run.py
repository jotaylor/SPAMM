#!/usr/bin/env python

import sys
import triangle
import numpy as np
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent

# TODO: astropy units for spectrum

# -----------------------------------------------------------------
# This block of code tells Python to drop into the debugger
# if there is an uncaught exception when run from the command line.
def info(type, value, tb):
   if hasattr(sys, 'ps1') or not sys.stderr.isatty():
      # we are in interactive mode or we don't have a tty-like
      # device, so we call the default hook
      sys.__excepthook__(type, value, tb)
   else:
      import traceback, pdb
      # we are NOT in interactive mode, print the exception...
      traceback.print_exception(type, value, tb)
      print
      # ...then start the debugger in post-mortem mode.
      pdb.pm()
sys.excepthook = info
# -----------------------------------------------------------------

# ----------------
# Read in spectrum
# ----------------
datafile = "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat"
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)

spectrum = Spectrum()
spectrum.flux = flux
spectrum.flux_error = flux_err
spectrum.wavelengths = wavelengths

# -----------------
# Create components
# -----------------
nuclear_comp = NuclearContinuumComponent()

# ------------
# Create model
# ------------
model = Model()
model.components.append(nuclear_comp)

model.data_spectrum = spectrum # add data

model.run_mcmc(n_walkers=10, n_iterations=50)

samples = model.sampler.chain[:, 5:, :].reshape((-1, model.total_parameter_count))

fig = triangle.corner(samples)
fig.savefig("triangle.png")

# try:
# 	cf.run_mcmc_analysis(plot=False)
# except MCMCDidNotConverge:
# 	...







