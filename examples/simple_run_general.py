#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
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

#emcee parameters
n_walkers = 30
n_iterations = 5000

# Use MPI to distribute the computations
MPI = True 

#Select your component options
# PL = nuclear continuum
# HOST = host galaxy
# FE = Fe forest

#For the moment we have only implemented individual components
#Fe and host galaxy components need more work - see tickets
#To do: implement combined - Gisella - see tickets

PL = False#True
HOST = False
FE = False#True#
BC =  True#False#
BpC = True#False#

show_plots = False

# ----------------
# Read in spectrum
# ----------------

#if PL:
#	datafile = "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat"

if HOST:
	datafile = "../Data/FakeData/fake_host_spectrum.dat"

if FE:
	#datafile = "../Data/FakeData/for_gisella/fake_host_spectrum.dat"
	datafile = "../Data/FakeData/Iron_comp/fakeFe1_deg.dat"
	#datafile = "../Fe_templates/FeSimdata_BevWills_0p05.dat"

if BC:
	datafile = "../Data/FakeData/BaC_comp/FakeBac01_deg.dat"
if BC and BpC:
	datafile = "../Data/FakeData/BaC_comp/FakeBac_lines01_deg.dat"
	

# do you think there will be any way to open generic fits file and you specify hdu, npix, midpix, wavelength stuff
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
spectrum = Spectrum(maskType="Cont+Fe")#"Emission lines")#
spectrum.wavelengths = wavelengths
spectrum.flux = flux
spectrum.flux_error = flux_err	

# ------------
# Initialize model
# ------------
model = Model()
model.print_parameters = False#True#

# -----------------
# Initialize components
# -----------------
if PL:
	nuclear_comp = NuclearContinuumComponent()
	model.components.append(nuclear_comp)
	
if FE:
	fe_comp = FeComponent()
	model.components.append(fe_comp)

if HOST:
	host_galaxy_comp = HostGalaxyComponent()	
	model.components.append(host_galaxy_comp)
	
if BC or BpC:
	balmer_comp = BalmerCombined(BalmerContinuum=BC, BalmerPseudocContinuum=BpC)
	model.components.append(balmer_comp)

model.data_spectrum = spectrum # add data

# ------------
# Run MCMC
# ------------
model.run_mcmc(n_walkers=n_walkers, n_iterations=n_iterations)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(model.sampler.acceptance_fraction)))


# -------------
# save chains & model
# ------------
with gzip.open('model.pickle.gz', 'wb') as model_output:
	model_output.write(pickle.dumps(model))









