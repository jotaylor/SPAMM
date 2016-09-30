#!/usr/bin/python
# -*- coding: utf-8 -*-

import pdb

import sys
import numpy as np
from scipy import interpolate

import emcee

from .Spectrum import Spectrum

iteration_count = 0

class MCMCDidNotConverge(Exception):
	pass

def ln_posterior(new_params, *args):
	'''
	The logarithm of the posterior function -- to be passed to the emcee sampler.
	
	:param new_params: A 1D numpy array in the parameter space used as input into sampler.
	:param args: Additional arguments passed to this function (i.e. the Model object).
	'''
	global iteration_count
	iteration_count = iteration_count + 1
	if iteration_count % 20 == 0:
		print("iteration count: {0}".format(iteration_count))

	# Make sure "model" is passed in - this needs access to the Model object
	# since it contains all of the information about the components.
	model = args[0] # TODO: return an error if this is not the case
		
	# calculate the log prior
	# -----------------------	
	ln_prior = model.prior(params=new_params)
	if ln_prior < 0:
		return ln_prior 
	else:	# only calculate flux and therefore likelihood if parameters lie within bounds of priors to save computation time
		# ----------------------------
		# - compare the model spectrum to the data
		# generate model spectrum given model parameters
		model_spectrum_flux = model.model_flux(params=new_params)
		# calculate the log likelihood
		ln_likelihood = model.likelihood(model_spectrum_flux=model_spectrum_flux)
		return ln_likelihood + ln_prior # adding two lists
	

class Model(object):
	'''
	
	'''
	def __init__(self, wavelength_start=1000, wavelength_end=10000, wavelength_delta=0.05):
		'''
		
		:param wavelength_start: document me!
		:param wavelength_end: document me!
		:param wavelength_delta: document me!
		'''
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
		#self.model_spectrum.wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_delta)
		self.model_spectrum.wavelengths = None
		self.model_spectrum.flux = None # np.zeros(len(self.model_spectrum.wavelengths))
		
		# Flag to allow Model to interpolate components' wavelength grid to match data
		# if component grid is more course than data
		# TODO - document better!
		self.downsample_data_if_needed = False
		self.upsample_components_if_needed = False
		
		# debugging
		self.print_parameters = False
		
	@property
	def mask(self):
		'''
		
		'''
		if self.data_spectrum is None:
			print("Attempting to read the bad pixel mask before a spectrum was defined.")
			sys.exit(1)
		if self._mask is None:
			self._mask = np.ones(len(self.data_spectrum.wavelengths))

		return self._mask
		
	@mask.setter
	def mask(self, new_mask):
		'''
		Document me.
		
		:params mask: A numpy array representing the mask.
		'''
		self._mask = new_mask

	@property
	def data_spectrum(self):
		return self._data_spectrum
	
	@data_spectrum.setter
	def data_spectrum(self, new_data_spectrum):
		'''
		
		All components of the model must be set before setting the data (this method).

		:param new_data_spectrum: document me!
		'''
		self._data_spectrum = new_data_spectrum

		if len(self.components) == 0:
			raise Exception("Components must be added before defining the data spectrum.")

		# the data spectrum defines the model wavelength grid
		self.model_spectrum.wavelengths = np.array(new_data_spectrum.wavelengths)
		self.model_spectrum.flux = np.zeros(len(self.model_spectrum.wavelengths))

		# Check that all components are on the same wavelength grid.
		# If they are not, AND the flag to interpolate them has been set, AND they are not
		# more course than the data, interpolate. If not, fail.
		need_to_downsample_data = False
		components_to_upsample = dict()
				
		gs = 0 # grid spacing
		worst_component = None # holds component with most course wavelength grid spacing

		for component in self.components:
			component.initialize(data_spectrum=new_data_spectrum)
			
			if component.grid_spacing > gs:
				gs = component.grid_spacing
				worst_component = component
		
		if gs > new_data_spectrum.grid_spacing:
			
			if self.upsample_components_if_needed:
				# The code will interpolate to the data anyway,
				# AND the user has allowed this for coursely sampled components
				# to be upsampled to the data. This was done above.
				pass
			elif self.downsample_data_if_needed:
				# We will downsample the data, the resulting grid will be different than the
				# input data, and the user has allowed this.
				
				# downsample data to "worst" component; create new Spectrum data object
				downsampled_spectrum = new_data_spectrum.copy()
				
				# downsample
				downsampled_spectrum.wavelengths = np.arange(new_data_spectrum[0], new_data_spectrum[-1], gs)
				downsampled_spectrum.flux = scipy.interpolate.interp1d(x=downsampled_spectrum.wavelengths,
																	   y=new_data_spectrum.flux,
																	   kind='linear')
																	   
				self.model_spectrum.wavelengths = np.array(downsampled_spectrum.wavelengths)
				
				# now need to reinitialize all components with new data
				for component in self.components:
					component.initialize(data_spectrum=downsampled_spectrum)
			else:
				assert True, ("The component '{0}' has courser wavelength grid spacing \n".format(worst_component) +
							  "than the data. Either increase the spacing of the component or use one of " +
							  "the flags on the Model class ('upsample_components_if_needed', " +
							  "'downsample_data_if_needed') to override this.")


	def run_mcmc(self, n_walkers=100, n_iterations=100):
		'''
		Method that actually calls the MCMC.
		
		:param n_walkers: Number of walkers to pass to the MCMC.
		:param n_iterations: Number of iterations to pass to the MCMC.
		'''
		
		# initialize walker matrix with initial parameters
		walkers_matrix = list() # must be a list, not an np.array
		for walker in range(n_walkers):
			walker_params = list()
			for component in self.components:
				walker_params = walker_params + component.initial_values(self.data_spectrum)
			walkers_matrix.append(walker_params)

		global iteration_count
		iteration_count = 0

		# Create MCMC sampler
		# - to enable multiproccessing, set threads > 1.
		# - if using multiprocessing, the "lnpostfn" and "args" parameters must be pickleable.
		self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=len(walkers_matrix[0]),
											 lnpostfn=ln_posterior, args=[self],
											 threads=1)
		
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
		
		:rtype: document me!
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
		
		:param params: 1D numpy array of all parameters of all components of model.
		:rtype: Numpy array of flux values; use self.data_spectrum.wavelengths for the wavelengths.
		'''
		
		# Combine all components into a single spectrum
		# Build param vector to pass to MCMC
		
		if self.print_parameters:
			print("params = {0}".format(params))

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
		
		:param component: fill in doc
		:param parameters: fill in doc
		
		'''
		# get the component's flux
		component_flux = component.flux(spectrum=self.data_spectrum, parameters=parameters)
		self.model_spectrum.flux += component_flux

	def model_parameter_names(self):
		'''
		Returns a list of all component parameter names.
		
		:rtype: list of strings
		'''
		labels = list()
		for c in self.components:
			labels = labels + [x for x in c.model_parameter_names]
		return labels

	def likelihood(self, model_spectrum_flux=None):
		r'''
		Calculate the ln likelihood of the given model spectrum 

		.. math:: ln(L) = -0.5 \sum_n {\left[ \frac{(flux_{Obs}-flux_{model})^2}{\sigma^2} + ln(2 \pi \sigma^2) \right]}

		The model is interpolated over the data wavelength grid.

		:param model_spectrum_flux: The model spectrum, a numpy array of flux value.
		:rtype: float likelihood value
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
		
		:param params: describe me!
		:rtype: describe me
		'''
		
		# make a copy as we'll delete elements
		p = np.copy(params)

		ln_p = 0
		for component in self.components:
			ln_p += sum(component.ln_priors(params=p[0:component.parameter_count]))

			# remove the parameters for this component from the list
			p = p[component.parameter_count:]

		return ln_p