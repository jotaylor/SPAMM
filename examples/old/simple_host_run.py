#!/usr/bin/env python

import os
import sys
import triangle
import numpy as np
import matplotlib.pyplot as pl

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.HostGalaxyComponent import HostGalaxyComponent

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
      print()
      # ...then start the debugger in post-mortem mode.
      pdb.pm()
sys.excepthook = info
# -----------------------------------------------------------------

# ----------------
# Read in spectrum
# ----------------

#Fake spectrum is CWW_E only. Hence, best-fit parameters should only
#be non-zero for that SED.
datafile = "../Data/FakeData/fake_host_spectrum.dat"
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)

spectrum = Spectrum()
spectrum.flux = flux
spectrum.flux_error = flux_err
spectrum.wavelengths = wavelengths

# -----------------
# Initialize components
# -----------------
host_comp    = HostGalaxyComponent()

# ------------
# Initialize model
# ------------
model = Model()
model.components.append(host_comp)
model.data_spectrum = spectrum # add data

# ------------
# Run MCMC
# ------------
model.run_mcmc(n_walkers=100, n_iterations=500)
print(("Mean acceptance fraction: {0:.3f}".format(np.mean(model.sampler.acceptance_fraction))))

# ------------
# Analyze & Plot results
# ------------
#discard the first 100 steps (burn in)
#Flattens the chain to have a flat list of samples
samples = model.sampler.chain[:, 100:, :].reshape((-1, model.total_parameter_count))

# Save the samples into a text file to be read by other codes
np.savetxt("host_samples.text",samples)

# Plot the chains to check for convergence
labels=["$\\rm E$","$\\rm Im$","$\\rm sigma$"]
if np.size(labels) != np.size(samples[0,:]):
	print("Caution: The number of label names is not correct!")
for i in range(0,np.size(labels)):
	pl.clf()
	pl.plot(samples[:,0],'-b')
	pl.xlabel("$\\rm Chain$")
	pl.ylabel(labels[i])
	pl.show()

## add plot distributions with histograms instead of triangle
fig = triangle.corner(samples[0:],labels=["$E$","$Im$"])
fig.savefig("host_fit.png")

#add analysis of the samples -- producing numbers to quote!
#global minimum -- percentiles -- et al.

# try:
# 	cf.run_mcmc_analysis(plot=False)
# except MCMCDidNotConverge:
# 	...







