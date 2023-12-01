#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy
import numpy as np
from astropy.wcs import WCS

from specutils import Spectrum1D
from astropy.units import Quantity
from astropy.nddata import NDUncertainty, StdDevUncertainty

from utils.parse_pars import parse_pars
FLUX_UNIT = parse_pars()["global"]["flux_unit"]
WL_UNIT = parse_pars()["global"]["wl_unit"]

class Spectrum(Spectrum1D):
    """
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
    """

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
        """
        This method is used for pickling a Spectrum object. It returns a dictionary that represents 
        the state of the object.

        The World Coordinate System (WCS) for the Spectrum object is not picklable, so it's removed 
        from the state dictionary before it's returned.

        Returns:
            dict: A dictionary representing the state of the Spectrum object, without the WCS.
        """
        odict = self.__dict__
        if "_wcs" in odict:
            del odict["_wcs"]
        return odict

    def __setstate__(self, d):
        """
        This method is used for unpickling a Spectrum object. It sets the state of the object 
        using the state dictionary `d` that was saved during pickling.

        The World Coordinate System (WCS) for the Spectrum object is reconstructed from the 
        spectral axis data in the state dictionary.

        Args:
            d (dict): A dictionary containing the state of the Spectrum object.

        Returns:
            None
        """
        d["_wcs"] = WCS(naxis=1)
        d["_wcs"].wcs.crpix = [0]
        d["_wcs"].wcs.crval = [d["_spectral_axis"][0]]
        d["_wcs"].wcs.cdelt = [d["_spectral_axis"][1] - d["_spectral_axis"][0]]
        d["_wcs"].wcs.ctype = ["WAVE"]
        self.__dict__ = d

    @property
    def norm_wavelength(self):
        """
        This property represents the normalized wavelength of the Spectrum object. 
        It is calculated as the median of the spectral axis. The calculated value is 
        cached to avoid recalculation.

        Returns:
            float: The normalized wavelength.
        """
        if self._norm_wavelength is None:
            self._norm_wavelength = np.median(self.spectral_axis)
        return self._norm_wavelength

    @property
    def norm_wavelength_flux(self):
        """
        This property represents the flux at the normalized wavelength of the Spectrum object. 
        It is calculated by interpolating the flux over the spectral axis. The calculated value 
        is cached to avoid recalculation.

        Returns:
            float: The flux at the normalized wavelength.
        """
        if self._norm_wavelength_flux == None:
            f = scipy.interpolate.interp1d(self.spectral_axis, self.flux)
            self._norm_wavelength_flux = f(self.norm_wavelength)
        return self._norm_wavelength_flux

    def grid_spacing(self):
        """
        This method returns the spacing of the wavelength grid in Ångstroms. 
        Note that it does not support variable grid spacing.

        Returns:
            float: The spacing of the wavelength grid in Ångstroms.
        """
        return self.spectral_axis[1] - self.spectral_axis[0]

    @property
    def spectral_axis(self):
        """
        This property represents the spectral axis of the Spectrum object. 
        If the spectral axis has not been set yet, it retrieves the spectral axis 
        from the superclass and caches it.

        Returns:
            array-like: The spectral axis of the Spectrum object.
        """
        if self._spectral_axis is None:
            self._spectral_axis = super(Spectrum, self).spectral_axis
        return self._spectral_axis

    @spectral_axis.setter
    def spectral_axis(self, new_wl):
        """
        This setter method sets the spectral axis of the Spectrum object to a new value. 
        It also invalidates the cached normalized wavelength, since it may no longer be valid 
        after the spectral axis is changed.

        Args:
            new_wl (array-like): The new spectral axis.

        Returns:
            None
        """
        spectral_axis = new_wl
        self._spectral_axis = spectral_axis
        self._norm_wavelength = None

    @property
    def flux(self):
        """
        This property represents the flux values of the Spectrum object. 
        If the flux values have not been set yet, it retrieves them from the superclass.

        Returns:
            array-like: The flux values of the Spectrum object.
        """
        if self._flux is None:
            self._flux = super(Spectrum, self).flux
        return self._flux

    @flux.setter
    def flux(self, new_flux):
        """
        This setter method sets the flux values of the Spectrum object to a new value. 
        It also invalidates the cached flux at the normalized wavelength, since it may 
        no longer be valid after the flux values are changed.

        Args:
            new_flux (array-like): The new flux values.

        Returns:
            None
        """
        self._flux = new_flux
        self._norm_wavelength_flux = None

#TODO need to add log_grid_spacing, spectrum binning
