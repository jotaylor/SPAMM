#!/usr/bin/python

'''   '''

import numpy as np

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
		self._norm_wavelength = None
	
	@property
	def normalization_wavelength(self):
		if self._norm_wavelength is None:
			self._norm_wavelength = np.median(self.wavelengths)
		return self._norm_wavelength

class SDSSSpectrum(Spectrum):
	'''
	A Spectrum class that can process SDSS spectrum FITS files.
	
	To be implemented...
	'''	
	def __init__(self, filepath=None):
		'''
		SDSS initialization.
		
		@param filepath The full file path to the SDSS spectrum file.
		'''
		datafile = open(filepath)
		#<... read  wavelenths, spectrum >
		#self.flux = ...