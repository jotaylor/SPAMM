#!/usr/bin/env python

import os
import sys
import gzip
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as pl

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
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
      print
      # ...then start the debugger in post-mortem mode.
      pdb.pm()
sys.excepthook = info
# -----------------------------------------------------------------

n_walkers = 30
n_iterations = 800
show_plots = False

# ----------------
# Read in spectrum
# ----------------

# ------------
# Initialize model
# ------------
model = Model()
model.print_parameters = False

# -----------------
# Initialize components
# -----------------
if True:
	nuclear_comp = NuclearContinuumComponent()

	datafile = "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat"
	#datafile = "../Data/FakeData/PLcompOnly/fakepowlaw2_werr.dat"
	wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
	spectrum = Spectrum()
	spectrum.wavelengths = wavelengths
	spectrum.flux = flux
	spectrum.flux_error = flux_err
	
	model.components.append(nuclear_comp)

if False:
	host_galaxy_comp = HostGalaxyComponent()
	
	datafile = "../Data/FakeData/for_gisella/fake_host_spectrum.dat"
	wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
	spectrum = Spectrum()
	spectrum.wavelengths = wavelengths
	spectrum.flux = flux
	spectrum.flux_error = flux_err
	
	model.components.append(host_galaxy_comp)

model.data_spectrum = spectrum # add data

# ------------
# Run MCMC
# ------------
model.run_mcmc(n_walkers=n_walkers, n_iterations=n_iterations)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(model.sampler.acceptance_fraction)))

# ------------
# Analyze & Plot results
# ------------
#discard the first 50 steps (burn in) - keep at 50!!
#Flattens the chain to have a flat list of samples
samples = model.sampler.chain[:, 50:, :].reshape((-1, model.total_parameter_count))

# save chains + model
with gzip.open('model.pickle.gz', 'wb') as model_output:
	model_output.write(pickle.dumps(model))

# TODO: move code below to another file that reads the pickled output, 
# e.g. one that plots the chains, one that generates the triangle plot,
# some combination, etc.

# Save the samples into a text file to be read by other codes
np.savetxt("samples.text",samples)

# Plot the chains to check for convergence
#labels=["$\\rm norm$","$\\rm slope$"]
#if np.size(labels) != np.size(samples[0,:]):
#	print "size labels: {0} / size samples {1}".format(labels, np.size(samples[0,:]))
#	print("Caution: The number of label names is not correct!")
labels = model.model_parameter_names()
if show_plots:
	for i in xrange(np.size(labels)):
		pl.clf()
		pl.plot(samples[:,0],'-b')
		pl.xlabel("$\\rm Chain$")
		pl.ylabel(labels[i])
		pl.show()

## add plot distributions with histograms instead of triangle
#fig = triangle.corner(samples,labels=["$norm$","$slope$"])
fig = triangle.corner(samples, labels=labels)
fig.savefig("triangle.png")

#add analysis of the samples -- producing numbers to quote!
#global minimum -- percentiles -- et al.

# try:
# 	cf.run_mcmc_analysis(plot=False)
# except MCMCDidNotConverge:
# 	...







