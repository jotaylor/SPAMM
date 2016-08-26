#!/usr/bin/python
# -*- coding: utf-8 -*-

'''   '''

import scipy
import numpy as np

class Spectrum(object):
	'''
	The Spectrum object.
	'''
	def __init__(self, z=None, maskType=None,mask_FWHM_broad=5000,mask_FWHM_narrow=1000):#FWHM in km/s
		'''
		The Spectrum initialization.
		
		@param z Redshift z.
		'''
		self._wavelengths = None
		self._flux = None
		self.flux_error = None
		self._norm_wavelength = None
		self._flux_at_norm_wavelength = None
		self.maskType = maskType
		self._mask = None
		self.maskFWHM_broad = mask_FWHM_broad #estimate of FWHM in broad lines
		self.maskFWHM_narrow = mask_FWHM_narrow #estimate of FWHM in narrow lines
		
	
	@property
	def wavelengths(self):
		return self._wavelengths
	
	@wavelengths.setter
	def wavelengths(self, new_w):
		self._wavelengths = new_w
		self._flux_at_norm_wavelength = None
	
	@property
	def flux(self):
		#return self._flux
		return np.ma.masked_array(self._flux,mask=self.mask)
		
	@property
	def mask(self):
		if getattr(self,'_mask',None) is None:
			self._mask = [False]*len(self.wavelengths)
			if self.maskType != None:
				wavelength = np.array(self.wavelengths)
				wavebound = self.defaultreadmaskregions()
				for k in range(len(wavebound)):
					select = np.nonzero((wavelength >= wavebound[k][0]) & (wavelength <= wavebound[k][1]))
					for x in select[0]:
						self._mask[x] = True 
				if "Emission" not in self.maskType:
					self._mask = list(np.invert(self._mask))
		return self._mask
	
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
		
	def defaultreadmaskregions(self):
		cont = [[1275,1295],[1315,1330],[1351,1362],[1452,1520],[1680,1735],[1786,1834],[1940,2040],[2148,2243],[4770,4800],[5100,5130],[2190,2210],[3007,3027],[2190,2210],[3600,3700]]
		pureFe = [[1350,1365],[1427,1480],[1490,1505],[1705,1730],[1780,1800],[1942,2115],[2250,2300],[2333,2445],[2470,2625],[2675,2755],[2855,3010],[4500,4700]]
		#broadlines = [1215.68,1215.23,1218,1239,1243,1256,1263,1304,1307,1335,1394,1403,1402,1406,1486,1488,1531,1548,1551,1549,1640,1661,1666,1670.79,1720,1750,1786,1814,1855,1863,1883,1889,1909,2141,2321,2326,2335,2665,2796,2803,3203,3646,3835,3888.6,3934,3969,3970,4101.76,4340.5,4471.5,4686,4861,5875,5891,6562.9,7676,8446,8498,8542,8662]
		broadlines_complete = [1215.68,1215.23,1218,1239,1243,1256,1263,1304,1307,1335,1394,1403,1402,1406,1486,1488,1531,1548,1551,1640,1661,1666,1670.79,1720,1750,1786,1814,1855,1863,1883,1889,1909,2141,2321,2326,2335,2665,2796,2803,3203,4101.76,4340.5,4471.5,4686,4861,5875,5891,6562.9,7676,8446,8498,8542,8662]
		narrowlines_complete = [3426,3727,3869,4959,5007,6087,6300,6374,6548,6583,6716,6731,9069,9532]
		broadlines_reduced = [1215.68,1239,1243,1256,1263,1394,1403,1402,1406,1548,1551,1855,1863,1883,1889,1909,2796,2803,4101.76,4340.5,4861,5875,5891,6562.9]
		narrowlines_reduced = [3727,3869,4959,5007,6300,6548,6583,6716,6731]
		c = 3.e5 #km/s
		v_c_broad = self.maskFWHM_broad/c
		v_c_narrow = self.maskFWHM_narrow/c
		lineregions_complete = []
		for x in broadlines_complete:
			lineregions_complete.append([x-x*2*v_c_broad,x+x*2*v_c_broad]) # avoid lines by 3 FWHMs
		for i in narrowlines_complete:
			lineregions_complete.append([x-x*2*v_c_narrow,x+x*2*v_c_narrow])
		lineregions_reduced = []
		for x in broadlines_reduced:
			lineregions_reduced.append([x-x*2*v_c_broad,x+x*2*v_c_broad]) # avoid lines by 3 FWHMs
		for i in narrowlines_reduced:
			lineregions_reduced.append([x-x*2*v_c_narrow,x+x*2*v_c_narrow])
		if self.maskType == "Continuum":
			return cont
		if self.maskType == "FeRegions":
			return pureFe
		if self.maskType == "Cont+Fe":
			cont.extend(pureFe)
			return cont
		if self.maskType == "Emission lines complete":
			return lineregions_complete
		if self.maskType == "Emission lines reduced":
			return lineregions_reduced
		
		
		
		
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
