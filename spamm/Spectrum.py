#!/usr/bin/python
# -*- coding: utf-8 -*-

'''   '''

import scipy
import numpy as np
from specutils.core.generic import Spectrum1DRef
from astropy.units import Quantity
from utils.rebin_spec import rebin_spec

class Spectrum(Spectrum1DRef):
    '''
    Inherit from Spectrum1DRef in specutils.
    '''
    def __init__(self, data, wcs=None, *args, **kwargs):
        '''
        The Spectrum initialization.
        @param z Redshift z.
        '''
        #self.flux_error = None
        self._norm_wavelength = None
        self._norm_wavelength_flux = None
        super(Spectrum, self).__init__(data, wcs=wcs, *args, **kwargs)
        self.wavelengths = None

    @property
    def norm_wavelength(self):
        if self._norm_wavelength is None:
            self._norm_wavelength = np.median(self.wavelengths)
        return self._norm_wavelength
        
    @property
    def norm_wavelength_flux(self):
        ''' Returns the flux at the normalization wavelength. '''
        if self._norm_wavelength_flux == None:
            f = scipy.interpolate.interp1d(self.wavelengths, self.flux) # returns function
            self._norm_wavelength_flux = f(self.norm_wavelength)
        return self._norm_wavelength_flux

    def grid_spacing(self):
        ''' Return the spacing of the wavelength grid in Ångstroms. Does not support variable grid spacing. '''
        return self.wavelengths[1] - self.wavelengths[0]

    @property
    def dispersion(self):
        self._dispersion = super(Spectrum, self).dispersion

        dispersion = np.array(
            Quantity(self._dispersion, unit=self.dispersion_unit))

        return dispersion

    @dispersion.setter
    def dispersion(self, new_d):
        self._dispersion = new_d
        self._wavelengths = new_d

    @property
    def wavelengths(self):
        wavelengths = self._dispersion
        
        return wavelengths

    @wavelengths.setter
    def wavelengths(self, new_wl):
        self._wavelengths = new_wl
        self._dispersion = new_wl
        
    @property
    def log_wave(self):
        log_wave = np.log(self.wavelengths)
        
        return log_wave
    
    @property
    def log_grid(self):
        ln_wave = self.log_wave
        log_grid_spacing = np.r_[ln_wave.min():ln_wave.max():1j*ln_wave.size]
        
        return log_grid_spacing
    
    @property
    def log_grid_spacing(self):
        
        return self.log_grid[1]-self.log_grid[0]

    @property
    def flux(self):
        self._flux = super(Spectrum, self).flux
        flux = np.array(Quantity(self._flux, unit=self.unit))
        flux = np.ma.masked_array(flux,mask=self.mask)
        return flux

    @flux.setter
    def flux(self, new_flux):
#        self._flux = new_flux
        self._data = new_flux
        self._norm_wavelength_flux = None

#    def bin_spectrum(self):
    @property
    def log_spectrum(self):
        ln_wave  = self.log_wave
        ln_wavenew = self.log_grid
        
        return rebin_spec(ln_wave,self.flux, ln_wavenew)
        
    def rebin_spectrum(self, wave_new):
        rebin_flux = rebin_spec(self.wavelengths, self.flux, wave_new)
        self.flux = rebin_flux
        self.wavelengths = wave_new
        return

#! need to finalize this 
