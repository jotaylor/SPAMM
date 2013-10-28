#!/usr/bin/python

import numpy as np
from .ComponentBase import Component

def runningMeanFast(x, N):
	'''
	x = array of points
	N = window width
	Ref: http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
	'''
	return np.convolve(x, np.ones((N,))/N)[(N-1):]

class NuclearContinuumComponent(Component):
	'''
	Description of the NuclearComponent here.
	
	This component has two parameters:
	
	slope : 
	minimization : 
	
	'''
	def __init__(self):
		super(NuclearContinuumComponent, self).__init__()
		self.model_parameter_names.append("normalization")
		self.model_parameter_names.append("slope")
		self._norm_wavelength =  None
		
	def initial_values(self, spectrum=None):
		'''
		
		Needs to sample from prior distribution.
		'''
		#print "NuclearContinuumComponent:initial_values"
		
		high = max(runningMeanFast(spectrum.flux, 5))
		normalization_init = np.random.uniform(low=0,
											   high=high)

		slope_init = np.random.uniform(low=-3.0, high=3.0)
		return [normalization_init, slope_init]

	def initialize(self, data_spectrum=None):
		'''
		Perform any initializations where the data is optional.
		'''
		if data_spectrum is None:
			raise Exception("The data spectrum must be specified to initialize {0}.".format("NuclearContinuumComponent"))
		self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths)

	def normalization_wavelength(self, data_spectrum_wavelength=None):
		'''
		
		'''
		if self._norm_wavelength is None:
			if data_spectrum_wavelength is None:
				raise Exception("The wavelength array of the data spectrum must be specified.")
			self._norm_wavelength = np.median(data_spectrum_wavelength)
		return self._norm_wavelength

	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''
		
		return [0]
		
		#ln_p = list(ln(p)
		
		# normalization prior
		
		#ln_p.append(...)
		
		# slope prior

		#ln_p.append(...)

		return ln_p

	def flux(self, wavelengths=None, parameters=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. Will use the initial parameters if none are specified.
		'''
		assert len(parameters) == len(self.model_parameter_names), "The wrong number of indices were provided: {0}".format(parameters)
		
		# calculate flux
		flux = parameters[0] * \
		       np.power((wavelengths / self.normalization_wavelength()), parameters[1])
		
		return flux

# 	def add_component(self, model=None, params=None):
# 		'''
# 		Add the nuclear continuum component to the provided model spectrum.
# 		
# 		@param model_spectrum The model spectrum to add this component to (type Spectrum)
# 		'''
# 		if params == None:
# 			params = self.initial_values(spectrum=model.data_spectrum)
# 
# 		print "params: type: {0}, {1}".format(type(params), params)
# 		assert len(params) == 2, "The wrong number of indices were provided: {0}".format(params)
# 
# 		model_spectrum_flux = params[0] * np.power((model.model_spectrum.wavelengths / model.data_spectrum.normalization_wavelength), params[1])
# 		return model_spectrum_flux
