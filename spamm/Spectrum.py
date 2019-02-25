#!/usr/bin/python
# -*- coding: utf-8 -*-

'''   '''

import scipy
import numpy as np
from specutils import Spectrum1D
from astropy.units import Quantity

from utils.parse_pars import parse_pars
FLUX_UNIT = parse_pars()["global"]["flux_unit"]
WL_UNIT = parse_pars()["global"]["wl_unit"]

#-----------------------------------------------------------------------------#

class Spectrum(Spectrum1D):
    '''
    Inherit from Spectrum1DRef in specutils.
    '''
    def __init__(self, spectral_axis=None, flux=None, spectral_axis_unit=WL_UNIT, 
                 flux_unit=FLUX_UNIT, *args, **kwargs):
        self.flux_unit = flux_unit
        # Not sure these are needed
        self._norm_wavelength = None
        self._norm_wavelength_flux = None
        # Set equal to None in order to inherit from Spectrum while overriding
        # ability to set/get in child class here.
        self._spectral_axis = None
        self._flux = None
        super(Spectrum, self).__init__(spectral_axis=spectral_axis*spectral_axis_unit, 
                                       flux=flux*flux_unit, *args, **kwargs)

    @property
    def norm_wavelength(self):
        if self._norm_wavelength is None:
            self._norm_wavelength = np.median(self.spectral_axis)
        return self._norm_wavelength

    @property
    def norm_wavelength_flux(self):
        ''' Returns the flux at the normalization wavelength. '''
        if self._norm_wavelength_flux == None:
            f = scipy.interpolate.interp1d(self.spectral_axis, self.flux) # returns function
            self._norm_wavelength_flux = Quantity(f(self.norm_wavelength), self.flux_unit)
        return self._norm_wavelength_flux

    def grid_spacing(self):
        ''' Return the spacing of the wavelength grid in Ångstroms. Does not support variable grid spacing. '''
        return self.spectral_axis[1] - self.spectral_axis[0]

    @property
    def spectral_axis(self):
        # This is necessary to override parent class inability to set/get
        if self._spectral_axis == None:
            self._spectral_axis = super(Spectrum, self).spectral_axis
        return self._spectral_axis

    @spectral_axis.setter
    def spectral_axis(self, new_wl):
        spectral_axis = Quantity(new_wl, unit=self.spectral_axis_unit)
        self._spectral_axis = spectral_axis
        self._norm_wavelength = None

    @property
    def flux(self):
        # This is necessary to override parent class inability to set/get
        if self._flux == None:
            self._flux = super(Spectrum, self).flux
        return self._flux

    @flux.setter
    def flux(self, new_flux):
        flux = Quantity(new_flux, unit=self.flux_unit)
        self._flux = flux
        self._norm_wavelength_flux = None

#TODO need to add log_grid_spacing, spectrum binning
