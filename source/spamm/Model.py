#!/usr/bin/python
# -*- coding: utf-8 -*-

import pdb

import sys
import numpy as np
from scipy import interpolate

import emcee

from Spectrum import Spectrum

iteration_count = 0

class MCMCDidNotConverge(Exception):
	pass

def ln_posterior(new_params, *args):
	'''
	The logarithm of the posterior function -- to be passed to the emcee sampler.
	
	@param new_params A 1D numpy array in the parameter space used as input into sampler.
	@param args Additional arguments passed to this function (i.e. the Model object).
	'''
	global iteration_count
	iteration_count = iteration_count + 1
	if iteration_count % 500 == 0:
		print "iteration count: {0}".format(iteration_count)

	# Make sure "model" is passed in - this needs access to the Model object
	# since it contains all of the information about the components.
	model = args[0] # TODO: return an error if this is not the case
		
	# generate model spectrum given model parameters
	model_spectrum_flux = model.model_flux(params=new_params)
	
	# calculate the log likelihood
	# ----------------------------
	# - compare the model spectrum to the data
	ln_likelihood = model.likelihood(model_spectrum_flux=model_spectrum_flux)
	
	# calculate the log prior
	# -----------------------	
	ln_prior = model.prior(params=new_params)
	
	return ln_likelihood + ln_prior # adding two lists
	

class Model(object):
	'''
	
	'''
	def __init__(self, wavelength_start=1000, wavelength_end=10000, wavelength_delta=0.05):
		self.z = None
		self.components = list()
		#self.reddening = None
		#self.model_parameters = dict()
		#self.mcmc_param_vector = None

		# private properties
		self._mask = None
		self._data_spectrum = None
		
		# the emcee.EnsembleSampler object
		self.sampler = None

		# the output of the emcee object
		#self.sampler_output = None
		
		self.model_spectrum = Spectrum()
		# precomputed wavelength range is 1000-10000Å in steps of 0.05Å
		self.model_spectrum.wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_delta)
		self.model_spectrum.flux = np.zeros(len(self.model_spectrum.wavelengths))
		
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

	def run_mcmc(self, n_walkers=100, n_iterations=100):
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
				walker_params = walker_params + component.initial_values(self.data_spectrum)
			walkers_matrix.append(walker_params)

		global iteration_count
		iteration_count = 0

		# create MCMC sampler
		self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers,dim=len(walkers_matrix[0]),lnpostfn=ln_posterior,args=[self])
		
		# run!
		#self.sampler_output = self.sampler.run_mcmc(walkers_matrix, n_iterations)
		self.sampler.run_mcmc(walkers_matrix, n_iterations)
		
	@property
	def total_parameter_count(self):
		''' Return the total number of parameters of all components. '''
		total_no_parameters = 0
		for c in self.components:
			total_no_parameters += c.parameter_count
		return total_no_parameters
	
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
		
		print "params = {0}".format(params)

		# make a copy as we'll delete elements
		params2 = np.copy(params)
		
		self.model_spectrum.flux = np.zeros(len(self.model_spectrum.wavelengths))
		
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
		Calculate the ln likelihood of the given model spectrum 

		\f$ ln(L) = -0.5 \Sum_n {\left[ \fract{(flux_{Obs}-flux_{model})^2}{\sigma^2} + ln(2 \pi \sigma^2) \right]}\f$

		@params model_spectrum The model spectrum, a numpy array of flux value.
		The model is interpolated over the data wavelength grid.
		'''
		
		assert model_spectrum_flux is not None, "'model_spectrum.flux' should not be None."
		
		# This creates a function that can be used to interpolate
		# values based on the data.
		f = interpolate.interp1d(self.model_spectrum.wavelengths,
		                         self.model_spectrum.flux)

		#It is much more efficient to not use a for loop here.
		#interp_model_flux = [f(x) for x in self.data_spectrum.wavelengths]
		interp_model_flux = f(self.data_spectrum.wavelengths)
		
		ln_l = np.power(((self.data_spectrum.flux - interp_model_flux) / self.data_spectrum.flux_error), 2)+np.log(2*np.pi*np.power(self.data_spectrum.flux_error,2))
		ln_l *= self.mask
		ln_l = -0.5*(np.sum(ln_l))
		return ln_l

	def prior(self, params):
		'''
		Calculate the ln priors for all components in the model.
		
		@param params
		'''
		ln_p = 0
		for component in self.components:
			ln_p += sum(component.ln_priors(params=params))
		return ln_p
		
			
