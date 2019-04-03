#!/usr/bin/python
# -*- coding: utf-8 -*-

'''   '''

import scipy
import numpy as np
from specutils.wcs.wcs_wrapper import WCSWrapper 
from specutils import Spectrum1D
from astropy.units import Quantity

from utils.parse_pars import parse_pars
FLUX_UNIT = parse_pars()["global"]["flux_unit"]
WL_UNIT = parse_pars()["global"]["wl_unit"]

#-----------------------------------------------------------------------------#

class Spectrum(Spectrum1D):
    '''
    Inherit from Spectrum1DRef in specutils. Wavelength (spectral_axis) and
    flux are unit-less in this child class. Units are instead stored as 
    attributes.
    '''
    def __init__(self, spectral_axis, flux, flux_error=None, spectral_axis_unit=WL_UNIT, 
                 flux_unit=FLUX_UNIT, *args, **kwargs):
        
        # If wavelength and flux have units, strip them off. This must be done
        # first so units don't get multipled in super().
        if type(spectral_axis) is Quantity:
            spectral_axis = spectral_axis.value
        if type(flux) is Quantity:
            flux = flux.value
        
        super(Spectrum, self).__init__(spectral_axis=spectral_axis*spectral_axis_unit, 
                                       flux=flux*flux_unit, *args, **kwargs)
        self.flux_unit = flux_unit
        self.flux_error = flux_error
        self.uncertainty = flux_error
        self._flux = flux
        self._norm_wavelength = None
        self._norm_wavelength_flux = None
        self._spectral_axis = spectral_axis
        self._spectral_axis_unit = spectral_axis_unit

    def __getstate__(self):
        odict = self.__dict__
        del odict["_wcs"]
        return odict

    def __setstate__(self, d):
        d["_wcs"] = WCSWrapper.from_array(d["_spectral_axis"] * d["_spectral_axis_unit"]) 
        self.__dict__ = d

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
            self._norm_wavelength_flux = f(self.norm_wavelength)
        return self._norm_wavelength_flux

    def grid_spacing(self):
        ''' Return the spacing of the wavelength grid in Ångstroms. Does not support variable grid spacing. '''
        return self.spectral_axis[1] - self.spectral_axis[0]

    @property
    def spectral_axis(self):
        # This is necessary to override parent class inability to set/get
        if self._spectral_axis is None:
            self._spectral_axis = super(Spectrum, self).spectral_axis
        return self._spectral_axis

    @spectral_axis.setter
    def spectral_axis(self, new_wl):
        spectral_axis = new_wl
        self._spectral_axis = spectral_axis
        self._norm_wavelength = None

    @property
    def flux(self):
        # This is necessary to override parent class inability to set/get
        if self._flux is None:
            self._flux = super(Spectrum, self).flux
        return self._flux

    @flux.setter
    def flux(self, new_flux):
        flux = new_flux
        self._flux = flux
        self._norm_wavelength_flux = None

#TODO need to add log_grid_spacing, spectrum binning
