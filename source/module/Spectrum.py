#!/usr/bin/python

'''   '''

class Spectrum(object):
	'''
	The Spectrum object.
	'''
	def __init__(self, z=None):
		'''
		The Spectrum initialization.
		
		@param z Redshift z.
		'''
		self.wavelengths = None
		self.flux = None
		self.flux_error = None

class SDSSSpectrum(Spectrum):
	'''
	A Spectrum class that can process SDSS spectrum FITS files.
	'''	
	def __init__(self, filepath=None):
		'''
		SDSS initialization.
		
		@param filepath The full file path to the SDSS spectrum file.
		'''
		datafile = open(filepath)
		<... read  wavelenths, spectrum >
		self.flux = ...