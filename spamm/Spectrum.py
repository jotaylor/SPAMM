#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy
import numpy as np

# old version of specutils
#from specutils.wcs.wcs_wrapper import WCSWrapper

# new version requires this instead in conjunction with the change made further below:
from astropy.wcs import WCS

from specutils import Spectrum1D
from astropy.units import Quantity
from astropy.nddata import NDUncertainty, StdDevUncertainty

from utils.parse_pars import parse_pars
FLUX_UNIT = parse_pars()["global"]["flux_unit"]
WL_UNIT = parse_pars()["global"]["wl_unit"]

#-----------------------------------------------------------------------------#

class Spectrum(Spectrum1D):
    '''
    Inherit from Spectrum1DRef in specutils. Wavelength (spectral_axis) and
    flux are unit-less in this child class. Units are instead stored as 
    attributes.

    Args:
        spectral_axis (array-like or :obj:`astropy.units.Quantity`): Wavelength values.
        flux (array-like or :obj:`astropy.units.Quantity`) : Flux values.
        flux_error (array-like or :obj:`astropy.nddata.NDUncertainty`, or :obj:`astropy.units.Quantity) : 
            Error on `flux` values.
        spectral_axis_unit (:obj:`astropy.units.Unit`, optional) : Wavelength unit
        flux_unit (:obj:`astropy.units.Unit`, optional) : Flux unit. 
    '''

    def __init__(self, spectral_axis, flux, flux_error, spectral_axis_unit=WL_UNIT, 
                 flux_unit=FLUX_UNIT, *args, **kwargs):
        
        # If wavelength and flux have units, strip them off. This must be done
        # first so units don't get multiplied in super().
        if isinstance(spectral_axis, Quantity):
            spectral_axis_unit = spectral_axis.unit
            spectral_axis = spectral_axis.value
        
        if isinstance(flux, Quantity):
            flux_unit = flux.unit
            flux = flux.value
        
        if isinstance(flux_error, NDUncertainty):
            uncertainty = flux_error
            if flux_error.unit is not None:
                assert flux_error.unit == flux_unit, "Flux and flux error units must match" 
            flux_error = flux_error.array

        elif isinstance(flux_error, Quantity):
            uncertainty = StdDevUncertainty(flux_error)
            assert flux_error.unit == flux_unit, "Flux and flux error units must match" 
            flux_error = flux_error.value
            
        else:
            uncertainty = StdDevUncertainty(flux_error, unit=flux_unit)

        assert len(spectral_axis) == len(flux), "Wavelength and flux arrays must be the same length"
        assert len(flux) == len(flux_error), "Flux and flux error arrays must be the same length"

        super(Spectrum, self).__init__(spectral_axis=spectral_axis*spectral_axis_unit, 
                                       flux=flux*flux_unit, uncertainty=uncertainty,
                                       *args, **kwargs)
        self.uncertainty = uncertainty
        self.flux_unit = flux_unit
        self.flux_error = flux_error
        self._flux = flux
        self._norm_wavelength = None
        self._norm_wavelength_flux = None
        self._spectral_axis = spectral_axis
        self._spectral_axis_unit = spectral_axis_unit

    def __getstate__(self):
        odict = self.__dict__
        if "_wcs" in odict:
            del odict["_wcs"]
        return odict

    # Newest version of specutils necessitated the change of the below code

    # old version:
    # def __setstate__(self, d):
    #     d["_wcs"] = WCSWrapper.from_array(d["_spectral_axis"] * d["_spectral_axis_unit"]) 
    #     self.__dict__ = d

    # new version:
    def __setstate__(self, d):
        d["_wcs"] = WCS(naxis=1)
        d["_wcs"].wcs.crpix = [0]
        d["_wcs"].wcs.crval = [d["_spectral_axis"][0]]
        d["_wcs"].wcs.cdelt = [d["_spectral_axis"][1] - d["_spectral_axis"][0]]
        d["_wcs"].wcs.ctype = ["WAVE"]
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
        """
        Unit-less flux values of Spectrum instance.
        """
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
