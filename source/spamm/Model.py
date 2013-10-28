#!/usr/bin/python
# -*- coding: utf-8 -*-

import pdb

import sys
import numpy as np
from scipy import interpolate

import emcee

from Spectrum import Spectrum

class MCMCDidNotConverge(Exception):
	pass

def ln_probability(new_params, *args):
	'''
	The likelihood of the logarithm of the model, the function to be passed to the emcee sampler.
	
	@param new_params A 1D numpy array in the parameter space used as input into sampler.
	@param args Additional arguments passed to this function (i.e. the Model object).
	'''

	# Make sure "model" is passed in - this needs access to the Model object
	# since it contains all of the information about the components.
	model = args[0] # TODO: return an error if this is not the case
		
	# generate model spectrum given model parameters
	print "ln_probability params = {0}".format(new_params)
	model_spectrum_flux = model.model_flux(params=new_params)
	
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
		self._mask = None
		
		self._data_spectrum = None
		
		self.model_spectrum = Spectrum()
		# precomputed wavelength range is 1000-10000Å in steps of 0.05Å
		self.model_spectrum.wavelengths = np.arange(1000, 10000, 0.05)
		self.model_spectrum.flux = np.zeros(len(self.model_spectrum.wavelengths))
		
		
# 	@property
# 	def spectrum(self):
# 		return self._spectrum
# 	
# 	@spectrum.setter
# 	def spectrum(self, new_spectrum):
# 		self._spectrum = new_spectrum
	
	@property
	def mask(self):
		if self.data_spectrum is None:
			print "Attempting to read the bad pixel mask before a spectrum was defined."
			sys.exit(1)
		if self._mask is None:
			self._mask = np.ones(len(self.data_spectrum.wavelengths))

		return self._mask
		
	@mask.setter
	def mask(self, new_mask):
		'''
		Document me.
		
		@params mask A numpy array representing the mask.
		'''
		self._mask = new_mask

	def append_component(self, component=None):
		'''
		Add a new component to the model.
		'''
		print "Adding component"
		self.components.append(component)

	@property
	def data_spectrum(self):
		return self._data_spectrum
	
	@data_spectrum.setter
	def data_spectrum(self, new_data_spectrum):
		'''
		
		'''
		self._data_spectrum = new_data_spectrum

		if len(self.components) == 0:
			raise Exception("Components must be added before defining the data spectrum.")

		for component in self.components:
			component.initialize(data_spectrum=new_data_spectrum)

	def run_mcmc(self, n_walkers=100, n_iterations=10):
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
				#walker_params.append(component.initial_values(self.data_spectrum))
				walker_params = walker_params + component.initial_values(self.data_spectrum)
			walkers_matrix.append(walker_params)
		#print "matrix: {0}".format(walkers_matrix)
		#sys.exit(1)
		# create MCMC sampler
		sampler = emcee.EnsembleSampler(nwalkers=n_walkers,
										dim=len(walkers_matrix[0]),
										lnpostfn=ln_probability,
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
	
	def model_flux(self, params):
		'''
		Given the parameters in this model, generate a spectrum.
		
		DO NOT modify anything in this class from this method as
		it will be called by multiple MCMC walkers at the same time.
		
		@param params 1D numpy array of all parameters of all components of model.
		@returns Numpy array of flux values; use self.data_spectrum.wavelengths for the wavelengths.
		'''
		
		# Combine all components into a single spectrum
		# Build param vector to pass to MCMC
		
		print "params = {0}: (type: {1})".format(params, type(params))
		#params2 = [x for x in params[0]] #list(params) # make a copy as we'll delete elements
		params2 = np.copy(params)
		
		self.model_spectrum.flux = np.zeros(len(self.model_spectrum.wavelengths))
		#model_spectrum.wavelengths = self.model_spectrum.wavelengths
		
		#pdb.set_trace()
		for component in self.components:

			# extract parameters from full vector for each component
			p = params2[0:component.parameter_count]

			# add the flux of the component to the model spectrum
			self.add_component(component=component, parameters=p)
			
			# remove the parameters for this component from the list
			params2 = params2[component.parameter_count:]
			
		return self.model_spectrum.flux

	def add_component(self, component=None, parameters=None):
		'''
		Add the specified component to the model's 'model_spectrum'
		on the model_spectrum's wavelength grid.
		'''
		# get the component's flux
		component_flux = component.flux(wavelengths=self.model_spectrum.wavelengths,
									    parameters=parameters)
		self.model_spectrum.flux += component_flux

	def likelihood(self, model_spectrum_flux=None):
		'''
		Calculate the ln likelihood of the given model spectrum.

		\f$ L \prop e^{x^2/2} \f$
		
		@params model_spectrum The model spectrum, a numpy array of flux value.
		'''
		
		assert model_spectrum_flux is not None, "'model_spectrum.flux' should not be None."
		print "Model spectrum flux: {0}".format(model_spectrum_flux)
		#print self.data_spectrum.flux
		#print self.data_spectrum.flux_error
		
		# This creates a function that can be used to interpolate
		# values based on the data.
		f = interpolate.interp1d(self.model_spectrum.wavelengths,
		                         self.model_spectrum.flux)
		interp_model_flux = [f(x) for x in self.data_spectrum.wavelengths]
		
		ln_l = np.power(((self.data_spectrum.flux - interp_model_flux) / self.data_spectrum.flux_error), 2)
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
		
			