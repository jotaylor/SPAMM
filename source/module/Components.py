#!/usr/bin/python

class Component(object):

	def __init__(self):
		self.z = None
		self.reddening = None
		self.model_parameters = list()
		self.model_parameter_names = list()

	def parameters(self):
		if self.z:
			return [self.z] + self.model_parameters
		else:
			return self.model_parameters

	def parameter_count(self):
		if self.z:
			return len(self.model_parameters + 1)
		else:
			return len(self.model_parameters)

	def add(self, spectrum=None, params=None):
		'''
		Add this component to the given Spectrum.
		
		@param spectrum
		'''
		pass

class HostGalaxyComponent(Component):
	
	def __init__(self):
		pass

class NuclearComponent(Component):

	def __init__(self):
		self.model_parameters.append(1)
		self.model_parameter_names.append("normalization")

		self.model_parameters.append(-1.3)
		self.model_parameter_names.append("slope")


class FeComponent(Component):
	
	def __init__(self):
		pass
		#self.model_parameters.append(...)
		#self.model_parameter_names.append("...")



