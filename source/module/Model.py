#!/usr/bin/python

import emcee

from . import Spectrum

def ln_probability(params, *args):
	'''
	The function to be passed to the emcee sampler.
	
	@param params A vector in the parameter space used as input into sampler.
	@param args Additional arguments passed to this function (i.e. the Model object).
	'''

	# make sure "model" is passed in
	model = args[0]
	
	# generate model spectrum given model parameters
	model_spectrum = model.model_spectrum(params)
	
	# calculate the log likelihood
	
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

	def run_mcmc(self):
		''' Method that actually calls the MCMC. '''
		
		# define parameters
		
		# initialize walkers
		
		# initialize emcee
		sampler = emcee.EnsembleSampler(nwalkers,
										ndim,
										ln_probability,
										self)
		
		# call emcee
		sampler.run_mcmc(<walkers matrix>, <iteration>)
		
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
		model_spectrum = numpy.zeros(len(self.spectrum.wavelengths))
		for component in self.components:
			model_spectrum = component.add(model_spectrum, params=params)
			
		return model_spectrum





		