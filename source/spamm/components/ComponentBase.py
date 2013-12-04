#!/usr/bin/python

from abc import ABCMeta, abstractmethod
import numpy as np

class Component(object):
	'''
	Description of Component class here.
	
	This class is abstract; use (or create) subclasses of this class.
	Functionality that is common to all subclasses should be implemented here.
	'''
	__metaclass__ = ABCMeta

	def __init__(self):
		self.z = None
		self.reddening_law = None
		#self.model_parameters = list()
		self.model_parameter_names = list()

# 	def parameters(self):
# 		''' Returns the parameters of this component as a list. '''
# 		if self.z:
# 			return [self.z] + self.model_parameters
# 		else:
# 			return self.model_parameters

	def parameter_index(self, parameter_name):
		''' '''
		for idx, pname in enumerate(self.model_parameter_names):
			if parameter_name == pname:
				return idx
		
		return None
	
	@property
	def parameter_count(self):
		''' Returns the number of parameters of this component. '''
		if self.z:
			return len(self.model_parameter_names + 1)
		else:
			return len(self.model_parameter_names)

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
	def flux(self, wavelengths=None, parameters=None):
		pass

# 	@abstractmethod
# 	def add(self, model=None, params=None):
# 		'''
# 		Add this component to the given Spectrum.
# 		
# 		@param spectrum The spectrum to add this component to (type: Spectrum object).
# 		@param params A list of parameters for this component.
# 		'''
# 		pass
		
	@abstractmethod
	def initialize(self, data_spectrum=None):
		pass
