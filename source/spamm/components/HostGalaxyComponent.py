#!/usr/bin/python

import numpy as np

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

