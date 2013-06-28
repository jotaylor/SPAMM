#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import emcee

from Spectrum import Spectrum

class MCMCDidNotConverge(Exception):
	pass

def ln_probability(new_params, *args):
	'''
	The function to be passed to the emcee sampler.
	
	@param params A vector in the parameter space used as input into sampler.
	@param args Additional arguments passed to this function (i.e. the Model object).
	'''

	# make sure "model" is passed in
	model = args[0]
		
	# generate model spectrum given model parameters
	print "ln_probability params = {0}".format(new_params)
	model_spectrum_flux = model.model_spectrum_flux(params=new_params)
	
	# calculate the log likelihood
	# ----------------------------
	# - compare the model spectrum to the data
	ln_likelihood = model.likelihood(model_spectrum_flux=model_spectrum_flux)
	
	# calculate the log prior
	# -----------------------	
	ln_prior = model.prior(params=new_params)
	
	return ln_likelihood + ln_prior
	

class Model(object):
	'''
	
	'''
	def __init__(self):
		self.z = None
		self._spectrum = None
		self.components = list()
		self.reddening = None
		self.model_parameters = dict()
		self.mcmc_param_vector = None
		self.mask = None
		
		
		self.model_spectrum = Spectrum()
		# precomputed wavelength range is 1000-10000Å in steps of 0.05Å
		self.model_spectrum.wavelengths = np.arange(1000, 10000, 0.05)
	
# 	@property
# 	def spectrum(self):
# 		return self._spectrum
# 	
# 	@spectrum.setter
# 	def spectrum(self, new_spectrum):
# 		self._spectrum = new_spectrum
	
	@property
	def mask(self):
		if self.spectrum is None:
			print "Attempting to read the bad pixel mask before a spectrum was defined."
			sys.exit(1)
		if self._mask is None:
			self._mask = np.ones(len(self.spectrum.wavelengths))
		else:
			return self._mask
		
	@mask.setter
	def mask(self, new_mask):
		'''
		Document me.
		
		@params mask A numpy array representing the mask.
		'''
		self._mask = new_mask

	def run_mcmc(self, n_walkers=100, n_iterations=1000):
		'''
		Method that actually calls the MCMC.
		
		@param n_walkers Number of walkers to pass to the MCMC.
		@param n_iterations Number of iterations to pass to the MCMC.
		'''
		
		# initialize walker matrix with initial parameters
		walkers_matrix = list()
		for walker in xrange(n_walkers):
			walker_params = list()
			for component in self.components:
				walker_params.append(component.initial_values(self.spectrum))
			walkers_matrix.append(walker_params)

		# create MCMC sampler
		sampler = emcee.EnsembleSampler(n_walkers,
										len(walkers_matrix[0]),
										ln_probability,
										args=[self])
		
		# run!
		sampler.run_mcmc(walkers_matrix, n_iterations)
		
		# 
	
	def parameter_vector(self):
		'''
		
		'''
		param_vector = list()
		for component in self.components:
			param_vector.append(component.parameters())
		return param_vector
	
	def model_spectrum_flux(self, params):
		'''
		Given the parameters in this model, generate a spectrum.
		
		DO NOT modify anything in this class from this method as
		it will be called by multiple MCMC walkers at the same time.
		
		@param params Vector of all paramters of all components of model.
		@returns Numpy array of flux values; use self.spectrum.wavelengths for the wavelengths.
		'''
		
		# Combine all components into a single spectrum
		# Build param vector to pass to MCMC
		
		print "params = {0}: ".format(params)
		params = list(params) # make a copy as we'll delete elements
		
		model_spectrum = Spectrum()
		model_spectrum.flux = np.zeros(len(self.model_spectrum.wavelengths))
		model_spectrum.wavelengths = self.model_spectrum.wavelengths
		
		for component in self.components:
			
			# extract parameters from full vector for each component
			p = params[0:component.parameter_count]
			del params[0:component.parameter_count]
			model_spectrum = component.add(model=self, params=p)
			
		return model_spectrum

	def likelihood(self, model_spectrum_flux=None):
		'''
		Calculate the ln likelihood of the given model spectrum.

		\f$ L \prop e^{x^2/2} \f$
		
		@params model_spectrum The model spectrum, a numpy array of flux value.
		'''
		
		assert model_spectrum_flux is not None, "'model_spectrum_flux' should not be None."
		print model_spectrum_flux
		print self.spectrum.flux
		print self.spectrum.flux_error
		ln_l = np.power(((model_spectrum_flux - self.spectrum.flux) /
					   self.spectrum.flux_error), 2)
		ln_l *= self.mask
		ln_l = np.sum(ln_l) * -0.5
		return ln_l

	def prior(self, params):
		'''
		Calculate the priors for all components in the model.
		
		@param params
		'''
		ln_p = 0
		for component in self.components:
			ln_p += sum(component.ln_priors(params=params))
		return ln_p
		
			