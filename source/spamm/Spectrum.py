#!/usr/bin/python
# -*- coding: utf-8 -*-

'''   '''

import scipy
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
		self._wavelengths = None
		self._flux = None
		self.flux_error = None
		self._norm_wavelength = None
		self._flux_at_norm_wavelength = None
	
	@property
	def wavelengths(self):
		return self._wavelengths
	
	@wavelengths.setter
	def wavelengths(self, new_w):
		self._wavelengths = new_w
		self._flux_at_norm_wavelength = None
	
	@property
	def flux(self):
		return self._flux
	
	@flux.setter
	def flux(self, new_flux):
		self._flux = new_flux
		self._flux_at_norm_wavelength = None
	
	@property
	def normalization_wavelength(self):
		if self._norm_wavelength is None:
			self._norm_wavelength = np.median(self.wavelengths)
		return self._norm_wavelength
	
	def flux_at_normalization_wavelength(self):
		''' Returns the flux at the normalization wavelength. '''
		if self._flux_at_norm_wavelength == None:
			f = scipy.interpolate.interp1d(self.wavelengths, self.flux) # returns function
			self._flux_at_norm_wavelength = f(self.normalization_wavelength)
		return self._flux_at_norm_wavelength
		
	def grid_spacing(self):
		''' Return the spacing of the wavelength grid in Ångstroms. Does not support variable grid spacing. '''
		return self.wavelengths[1] - self.wavelengths[0]
	

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
