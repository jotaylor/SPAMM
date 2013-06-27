#!/usr/bin/python

'''   '''

class Spectrum(object):
	'''	The Spectrum object. '''
	
	def __init__(self, z=None):
		self.wavelengths = None
		self.flux = None
		self.flux_error = None

class SDSSSpectrum(Spectrum):
	
	def __init__(self, file=None):
		datafile = open(file)
		<... read  wavelenths, spectrum >
		self.flux = ...