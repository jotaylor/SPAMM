#!/usr/bin/python

from abc import ABCMeta, abstractmethod
import numpy as np

class Component(object):
	'''
	Description of Component class here.
	
	This class is abstract; use (or create) subclasses of this class.
	'''
	__metaclass__ = ABCMeta

	def __init__(self):
		self.z = None
		self.reddening_law = None
		self.model_parameters = list()
		self.model_parameter_names = list()

	def parameters(self):
		''' Returns the parameters of this component as a list. '''
		if self.z:
			return [self.z] + self.model_parameters
		else:
			return self.model_parameters

	@property
	def parameter_count(self):
		''' Returns the number of parameters of this component. '''
		if self.z:
			return len(self.model_parameters + 1)
		else:
			return len(self.model_parameters)

	@abstractmethod
	def initial_values(self, spectrum=None):
		pass
		
	@abstractmethod
	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''

	@abstractmethod
	def add(self, model=None, params=None):
		'''
		Add this component to the given Spectrum.
		
		@param spectrum The spectrum to add this component to (type: Spectrum object).
		@param params A list of parameters for this component.
		'''
		pass
		
class HostGalaxyComponent(Component):
	'''
	Description of the HostGalaxyComponent here.
	'''
	def __init__(self):
		super(HostGalaxyComponent, self).__init__()
		
	def initial_values(self, spectrum=None):
		'''
		Returns a list of initial values for this component.
		'''
		pass

	def add(self, model=None, params=None):
		assert 1, "Fill in here!"

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

class FeComponent(Component):
	'''
	Description of the FeComponent here.
	
	'''
	def __init__(self):
		super(FeComponent, self).__init__()
		#self.model_parameters.append(...)
		#self.model_parameter_names.append("...")

	def initial_values(self, spectrum=None):
		assert 1, "Fille in here!"
		pass

	def add(self, model=None, params=None):
		assert 1, "Fill in here!"

