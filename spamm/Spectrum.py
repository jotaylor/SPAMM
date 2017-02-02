#!/usr/bin/python
# -*- coding: utf-8 -*-

'''   '''

import scipy
import numpy as np
from specutils.core.generic import Spectrum1DRef
from astropy.units import Quantity

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
        self._flux_at_norm_wavelength = None
        super(Spectrum, self).__init__(data, wcs=wcs, *args, **kwargs)
        self.wavelengths = None

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
    def flux(self):
        self._flux = super(Spectrum, self).flux
        flux = np.array(Quantity(self._flux, unit=self.unit))
        flux = np.ma.masked_array(flux,mask=self.mask)
        return flux

    @flux.setter
    def flux(self, new_flux):
#        self._flux = new_flux
        self._data = new_flux
        self._flux_at_norm_wavelength = None
        
    
