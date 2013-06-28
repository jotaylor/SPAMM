#!/usr/bin/python

from abc import ABCMeta, abstractmethod

class Component(object):

	__metaclass__ = ABCMeta

	def __init__(self):
		self.z = None
		self.reddening_law = None
		self.model_parameters = list()
		self.model_parameter_names = list()

	def parameters(self):
		if self.z:
			return [self.z] + self.model_parameters
		else:
			return self.model_parameters

	@property
	def parameter_count(self):
		if self.z:
			return len(self.model_parameters + 1)
		else:
			return len(self.model_parameters)

	@abstractmethod
	def initial_values(self):
		pass
		

	def add(self, spectrum=None, params=None):
		'''
		Add this component to the given Spectrum.
		
		@param spectrum
		'''
		pass

class HostGalaxyComponent(Component):
	
	def __init__(self):
		pass
		
	def initial_values(self):
		...

class NuclearComponent(Component):

	def __init__(self):
		self.model_parameters.append(1)
		self.model_parameter_names.append("normalization")

		self.model_parameters.append(-1.3)
		self.model_parameter_names.append("slope")

	def initial_values(self):
		...


class FeComponent(Component):
	
	def __init__(self):
		pass
		#self.model_parameters.append(...)
		#self.model_parameter_names.append("...")

	def initial_values(self):
		...


