#!/usr/bin/python

import emcee

from . import Spectrum

def ln_probability(new_params, *args):
	'''
	The function to be passed to the emcee sampler.
	
	@param params A vector in the parameter space used as input into sampler.
	@param args Additional arguments passed to this function (i.e. the Model object).
	'''

	# make sure "model" is passed in
	model = args[0]
		
	# generate model spectrum given model parameters
	model_spectrum = model.model_spectrum(new_params)
	
	# calculate the log likelihood
	# - compare the model spectrum to the data
	model.
	
	# calculate the log prior
	
	return ln_likelihood + ln_prior
	

class Model(object):
	'''
	
	'''
	def __init__():
		self.z = None
		self.spectrum = None
		self.components = list()
		self.reddening = None
		self.model_parameters = dict()
		self.mcmc_param_vector = None
		self.model_spectrum = None

	def run_mcmc(self, n_walkers=100, n_iterations=1000):
		''' Method that actually calls the MCMC. '''
		
		# initialize walker matrix with initial parameters
		walkers_matrix = list()
		for walker in xrange(n_walkers):
			walker_params = list()
			for component in self.components:
				walker_params.append(component.initial_values)
			walkers_matrix.append(walker_params)

		# create MCMC sampler
		sampler = emcee.EnsembleSampler(n_walkers,
										len(walkers_matrix[0]),
										ln_probability,
										self)
		
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
	
	def model_spectrum(self, params):
		''' Given the parameters in this model, generate a spectrum. '''
		
		# Combine all components into a single spectrum
		# Build param vector to pass to MCMC
				
		model_spectrum = numpy.zeros(len(self.spectrum.wavelengths))
		for component in self.components:
			model_spectrum = component.add(model_spectrum, params=params)
			
		return model_spectrum





		