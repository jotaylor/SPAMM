#!/usr/bin/python

import numpy as np
from .ComponentBase import Component

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

	def initial_values(self, spectrum=None):
		'''
		
		Needs to sample from prior distribution.
		'''
		low = spectrum.wavelengths / spectrum.normalization_wavelength)
		normalization_init = np.rand.uniform(low=low,
											 high=)

		slope_init = np.rand.uniform(low=-3.0, high=0.0)
		return [normalization_init, slope_init]

	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''
		ln_p = list()
		
		# normalization prior
		
		#ln_p.append(...)
		
		# slope prior

		#ln_p.append(...)

		return ln_p

	def add(self, model=None, params=None):
		'''
		Add the nuclear continuum component to the provided model spectrum.
		
		@param model_spectrum The model spectrum to add this component to (type Spectrum)
		'''
		assert len(params) == 2, "The wrong number of indices were provided."

		model_spectrum_flux = params[0] * np.power(model.model_spectrum.wavelengths / model.spectrum.normalization_wavelength), params[1])
		return model_spectrum
