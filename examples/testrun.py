from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import gzip
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
#from spamm.components.BalmerContinuum_AK import BalmerContinuum
#from spamm.components.BalmerPseudoContinuum_AK import BalmerPseudoContinuum
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction

BpC = True#False#
BC = True#False#False#
Calzetti_ext = True#False#
SMC_ext = False#True#
MW_ext = False#True#
AGN_ext = False#True#
LMC_ext = False#True#

if BC or BpC:
	datafile = "../Data/FakeData/BaC_comp/FakeBac_lines09_deg.dat"

wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
#flux= flux+0.1*wavelengths**1
spectrum = Spectrum()
spectrum.wavelengths = wavelengths/1.85
spectrum.flux = flux
spectrum.flux_error = flux_err	

model = Model()
model.print_parameters = False

#if BC:
#	balmer_comp = BalmerContinuum()
#	model.components.append(balmer_comp)
# 	
#if BpC:
#	balmer_comp = BalmerPseudoContinuum()
#	model.components.append(balmer_comp)
# 	
if BC or BpC:
	balmer_comp = BalmerCombined(BalmerContinuum=BC, BalmerPseudocContinuum=BpC)
	model.components.append(balmer_comp)
	
if Calzetti_ext or SMC_ext or MW_ext or AGN_ext or LMC_ext:
	ext_comp = Extinction(MW=MW_ext,AGN=AGN_ext,LMC=LMC_ext,SMC=SMC_ext, Calzetti=Calzetti_ext)
	model.components.append(ext_comp)
	
model.data_spectrum = spectrum


#walker_params = balmer_comp.initial_values(model.data_spectrum)
#walker_params =  [  4.71977735e-15,   5.23714130e+03,   1.98535295e+00,  -9.55604922e+00,   9.62325639e+02,0.1]#balmer_comp.initial_values(model.data_spectrum)
walker_params =[10.e-15,   15000., 1.7, 0,6500,0.0]
#priors = balmer_comp.ln_priors(walker_params)
#print('priors_BC',priors)
#print('like', model.likelihood(model_spectrum_flux=model.data_spectrum.flux))
flux = model.model_flux(walker_params)
plt.plot(spectrum.wavelengths,spectrum.flux,'r',spectrum.wavelengths,flux,'b')
#plt.plot(spectrum.wavelengths,flux,'b')
plt.show()
