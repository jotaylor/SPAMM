#!/usr/bin/python

from abc import ABCMeta, abstractmethod

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
	def initial_values(self):
		pass
		
	@abstractmethod
	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''

	def add(self, spectrum=None, params=None):
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
		pass
		
	def initial_values(self):
		'''
		Returns a list of initial values for this component.
		'''
		...


class NuclearComponent(Component):
	'''
	Description of the NuclearComponent here.
	'''
	def __init__(self):
		self.model_parameters.append(1)
		self.model_parameter_names.append("normalization")

		self.model_parameters.append(-1.3)
		self.model_parameter_names.append("slope")

	def initial_values(self):
		...

	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''
		ln_p = list()
		
		# normalization prior
		
		ln_p.append(...)
		
		# slope prior

		ln_p.append(...)

		return ln_p

class FeComponent(Component):
	'''
	Description of the FeComponent here.
	'''
	def __init__(self):
		pass
		#self.model_parameters.append(...)
		#self.model_parameter_names.append("...")

	def initial_values(self):
		...


