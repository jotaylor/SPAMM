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
from spamm.components.ReddeningLaw import Extinction
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
n_iterations = 1000

# Use MPI to distribute the computations
MPI = True 

#Select your component options
# PL = nuclear continuum
# HOST = host galaxy
# FE = Fe forest

#For the moment we have only implemented individual components
#Fe and host galaxy components need more work - see tickets
#To do: implement combined - Gisella - see tickets

PL = True#False#
HOST = False
FE = False#True#
BC =  True#False#
BpC = True#False#
Calzetti_ext = False#True#
SMC_ext = False#True#
MW_ext = False#True#
AGN_ext = False#True#
LMC_ext = False#True#
max_wave =[]
min_wave =[]
fluxes = []
flux_errors = []
waves = []

if PL:
	datafile = "../Data/FakeData/PLcompOnly/fakepowlaw6_werr.dat"
	wavelengths_PL, flux_PL, flux_err_PL = np.loadtxt(datafile, unpack=True)
	z = 0.0
	wavelengths_PL/=1.+z
	max_wave.append(np.max(wavelengths_PL))
	min_wave.append(np.min(wavelengths_PL))
	waves.append(wavelengths_PL)
	fluxes.append(flux_PL)
	flux_errors.append(flux_err_PL)

if HOST:
	datafile = "../Data/FakeData/Host_comp/sb3_werr.dat"
	wavelengths_host, flux_host, flux_err_host = np.loadtxt(datafile, unpack=True)
	z = 2.0
	wavelengths_host/=1.+z
	max_wave.append(np.max(wavelengths_host))
	min_wave.append(np.min(wavelengths_host))
	waves.append(wavelengths_host)
	fluxes.append(flux_host)
	flux_errors.append(flux_err_host)
	
if FE:
	#datafile = "../Data/FakeData/for_gisella/fake_host_spectrum.dat"
	datafile = "../Data/FakeData/Iron_comp/fakeFe1_deg.dat"
	wavelengths_fe, flux_fe, flux_err_fe = np.loadtxt(datafile, unpack=True)
	z = 0.5
	wavelengths_fe/=1.+z
	max_wave.append(np.max(wavelengths_fe))
	min_wave.append(np.min(wavelengths_fe))
	waves.append(wavelengths_fe)
	fluxes.append(flux_fe)
	flux_errors.append(flux_err_fe)
	
if BC and not BpC:
	datafile = "../Data/FakeData/BaC_comp/FakeBac01_deg.dat"
	wavelengths_bc, flux_bc, flux_err_bc = np.loadtxt(datafile, unpack=True)
	z = 0.0
	wavelengths_bc/=1.+z
	max_wave.append(np.max(wavelengths_bc))
	min_wave.append(np.min(wavelengths_bc))
	waves.append(wavelengths_bc)
	fluxes.append(flux_bc)
	flux_errors.append(flux_err_bc)
if BC and BpC:
	datafile = "../Data/FakeData/BaC_comp/FakeBac_lines01_deg.dat"
	wavelengths_bc, flux_bc, flux_err_bc = np.loadtxt(datafile, unpack=True)
	z = 0.0
	wavelengths_bc/=1.+z
	max_wave.append(np.max(wavelengths_bc))
	min_wave.append(np.min(wavelengths_bc))
	waves.append(wavelengths_bc)
	fluxes.append(flux_bc)
	flux_errors.append(flux_err_bc)
	
print('wavebounds',	min_wave,max_wave)
print('size',np.shape(fluxes[0]),np.shape(fluxes[1]))
a = fluxes[0]
a = np.reshape(a, np.size(a))
#print('size', np.shape(a), np.shape(squeeze(fluxes[0])))
#exit()
wavelengths = np.arange(np.max(min_wave),np.min(max_wave),0.5)
if np.size(wavelengths)%2 !=0:
	wavelengths= wavelengths[:-1]
print('wavebounds',	np.min(wavelengths),np.max(wavelengths))
fluxes_arr = np.zeros((len(fluxes),np.size(wavelengths)))
var_arr = np.zeros((len(flux_errors),np.size(wavelengths)))

for jj in range(len(fluxes)):
	fluxresize = fluxes[jj]
	fluxresize = np.reshape(fluxresize, np.size(fluxresize))
	flux_errresize = flux_errors[jj]
	flux_errresize = np.reshape(flux_errresize, np.size(flux_errresize))
	wavesresize = waves[jj]
	wavesresize = np.reshape(wavesresize, np.size(wavesresize))
	#print('waveresize',wavesresize)
	bb = np.interp(wavelengths,wavesresize,fluxresize)
	fluxes_arr[jj,:] = bb#np.interp(wavelengths,wavesresize,fluxresize)
	diffw =wavesresize[1]-wavesresize[0]
	#print('diffw',diffw)
	for xx in range(np.size(wavelengths)):
		#print('sdfs',abs(wavesresize-wavelengths[xx]))
		diffwave = abs(wavesresize-wavelengths[xx])
		sorted = np.squeeze((diffwave < diffw).nonzero())
		#print('sorted',sorted)
		if np.size(sorted) ==2:
			var_arr[jj,xx] = (flux_errresize[sorted[0]]*(wavesresize[sorted[1]] - wavelengths[xx])/diffw)**2+ (flux_errresize[sorted[1]]*(wavelengths[xx]-wavesresize[sorted[0]])/diffw)**2
		if np.size(sorted) ==1:
			#print('size',np.size(sorted))
			var_arr[jj,xx] = flux_errresize[sorted]**2
		if np.size(sorted) ==3:
			#print('size',np.shape(sorted))
			#print('sorted',sorted[1])
			var_arr[jj,xx] = flux_errresize[sorted[1]]**2
			
			

flux = np.sum(fluxes_arr,axis=0)
var = np.sum(var_arr,axis=0)
flux_err = np.sqrt(var)

print('sizes',np.shape(flux),np.shape(wavelengths))

spectrum = Spectrum()#maskType="Emission lines reduced")#"Cont+Fe")#
spectrum.wavelengths = wavelengths
spectrum.flux = flux
spectrum.flux_error = flux_err

# ------------
# Initialize model
# ------------
model = Model()
model.print_parameters = False#True#False#

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
	
if Calzetti_ext or SMC_ext or MW_ext or AGN_ext or LMC_ext:
	ext_comp = Extinction(MW=MW_ext,AGN=AGN_ext,LMC=LMC_ext,SMC=SMC_ext, Calzetti=Calzetti_ext)
	model.components.append(ext_comp)

model.data_spectrum = spectrum # add data

# ------------
# Run MCMC
# ------------
model.run_mcmc(n_walkers=n_walkers, n_iterations=n_iterations)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(model.sampler.acceptance_fraction)))


# -------------
# save chains & model
# ------------
with gzip.open('model.pickle.combination.gz', 'wb') as model_output:
	model_output.write(pickle.dumps(model))









