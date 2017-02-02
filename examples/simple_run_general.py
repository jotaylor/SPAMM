#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import gzip
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import matplotlib.pyplot as pl
from astropy import units

import triangle

sys.path.append(os.path.abspath("../"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction
from spamm.components.MaskingComponent import Mask
import warnings
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from pysynphot import observation
    from pysynphot import spectrum as pysynspec
    
def pysnblueshift(z,spectrum):
    z_blue = 1.0/(1.+z)-1.
    sp = pysynspec.ArraySourceSpectrum(wave=spectrum.wavelengths, flux=spectrum.flux)
    sp_rest = sp.redshift(z_blue)
    return sp_rest.wave,sp_rest.flux

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

#emcee parameters
n_walkers = 50
n_iterations = 500

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
HOST = False#True#
FE = True#False#
BC =  True#False#
BpC = True#False#
Calzetti_ext = False#True#
SMC_ext = False
MW_ext = False
AGN_ext = False
LMC_ext = False
maskType="Emission lines reduced"#None#"Emission lines reduced"#"Continuum"#

show_plots = False

# ----------------
# Read in spectrum
# ----------------

if PL:
    datafile = "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat"

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

#datafile= "../Data/SVA1_COADD-2925657995_42.dat"
object="2939318691"
image_num = "2"
datafile = "../Data/SVA1_COADD-"+object+"_"+image_num+".dat"

# do you think there will be any way to open generic fits file and you specify hdu, npix, midpix, wavelength stuff
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
# need to resolve
z= 0
print('redshift =',z, 'in this case')
mask = Mask(wavelengths=wavelengths,maskType=maskType)
spectrum = Spectrum(flux)#Spectrum.from_array(flux, uncertainty=flux_err, mask=mask)
spectrum.dispersion = wavelengths#*units.angstrom
spectrum.flux_error = flux_err   
spectrum.wavelengths,spectrum.flux= pysnblueshift(z,spectrum)
spectrum.mask=mask
pl.plot(spectrum.wavelengths,spectrum.flux)
pl.show()
#exit()
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
with gzip.open('model.pickle.'+object+'.'+image_num+'.gz', 'wb') as model_output:
    model_output.write(pickle.dumps(model))









