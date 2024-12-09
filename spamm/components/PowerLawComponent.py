#!/usr/bin/python

import sys
import numpy as np
from astropy.modeling.powerlaws import PowerLaw1D,BrokenPowerLaw1D

from .ComponentBase import Component
from utils.runningmeanfast import runningMeanFast
from utils.parse_pars import parse_pars

class PowerLawComponent(Component):
    """
    AGN Continuum Component. This component is a power law of the form:

    .. math::

        F_{\\lambda,\\text{PL}} = A \\left(\\frac{\\lambda}{\\lambda_0}\\right)^{\\alpha}

    This component has two parameters:

    * :math:`A`: The normalization of the power law at the wavelength :math:`\\lambda_0`.
    * :math:`\\alpha`: The slope of the power law.

    """
    def __init__(self, pars=None, broken=None):
        """
        Initialize the PowerLawComponent.

        Args:
            pars (dict, optional): Dictionary of parameters. If None, default parameters are used.
            broken (bool, optional): If True, use a broken power law. If None, use the value from parameters.
        """
        super().__init__()
        
        self.name = "power_law"

        if pars is None:
            self.inputpars = parse_pars()["power_law"]
        else:
            self.inputpars = pars

        if broken is None:
            self.broken_pl = self.inputpars["broken_pl"]
        else:
            self.broken_pl = broken
        self.model_parameter_names = []
        
        if not self.broken_pl:
            self.model_parameter_names.append("norm_PL")
            self.model_parameter_names.append("slope1")
        else:
            self.model_parameter_names.append("wave_break")
            self.model_parameter_names.append("norm_PL")
            self.model_parameter_names.append("slope1")
            self.model_parameter_names.append("slope2")

        self.norm_min = self.inputpars["pl_norm_min"]
        self.norm_max = self.inputpars["pl_norm_max"]
        self.slope_min = self.inputpars["pl_slope_min"]
        self.slope_max = self.inputpars["pl_slope_max"]
        self.wave_break_min = self.inputpars["pl_wave_break_min"]
        self.wave_break_max = self.inputpars["pl_wave_break_max"]

#-----------------------------------------------------------------------------#

#TODO could this be moved to Component.py?
    @property
    def is_analytic(self):
        """ 
        Method that stores whether component is analytic or not
        
        Returns:
            Bool (Bool): True if componenet is analytic.
        """
        return True    

#-----------------------------------------------------------------------------#

    def initial_values(self, spectrum):
        """
        Needs to sample from prior distribution.
        Return type must be a list (not an np.array).

        Called by emcee.
        
        Args:
            spectrum (Spectrum object): ?
        
        Returns:
            norm_init (array):
            slope_init (array):
        """
        pl_init = []
        if self.norm_max == "max_flux":
            self.norm_max = max(runningMeanFast(spectrum.flux, self.inputpars["boxcar_width"]))
        elif self.norm_max == "fnw":
            fnw = spectrum.norm_wavelength_flux
            self.norm_max = fnw

        if self.broken_pl:
            size = 2
            if self.wave_break_min == "min_wl":
                self.wave_break_min = min(spectrum.spectral_axis)
            if self.wave_break_max == "max_wl": 
                self.wave_break_max = max(spectrum.spectral_axis)
            wave_break_init = np.random.uniform(low=self.wave_break_min, 
                                                     high=self.wave_break_max)
            pl_init.append(wave_break_init)
        else:
            size = 1
        
        norm_init = np.random.uniform(self.norm_min, high=self.norm_max)
        pl_init.append(norm_init)

        slope_init = np.random.uniform(low=self.slope_min, high=self.slope_max, size=size)
        # pl_init should be a list of scalars
        for slope in slope_init:
            pl_init.append(slope)

        return pl_init
#TODO need to modify emcee initial_values call

#-----------------------------------------------------------------------------#

    def ln_priors(self, params):
        """
        Return a list of the natural logarithm of all the priors.

        This function calculates the natural logarithm of the priors for the parameters of the power law component. The priors are uniform linear priors within specified ranges.

        Args:
            params (dict): A dictionary containing the parameters for the power law component. Expected keys are:
                - "wave_break" (float): The wavelength break parameter (only if `self.broken_pl` is True).
                - "norm_PL" (float): The normalization parameter.
                - "slope1" (float): The slope parameter for the first segment.
                - "slope2" (float): The slope parameter for the second segment (only if `self.broken_pl` is True).

        Returns:
            list: A list containing the natural logarithm of the priors for each parameter. If a parameter is outside its allowed range, the corresponding prior is `-np.inf`.
        """
        # Need to return parameters as a list in the correct order.
        ln_priors = []

        if self.broken_pl:
            wave_break = params["wave_break"]
            if self.wave_break_min < wave_break < self.wave_break_max:
                ln_priors.append(0.)
            else:
                ln_priors.append(-np.inf)
        
        norm = params["norm_PL"]
        if self.norm_min < norm < self.norm_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf)

        slope1 = params["slope1"]
        if self.slope_min < slope1 < self.slope_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf)

        if self.broken_pl:
            slope2 = params["slope2"]
            if self.slope_min < slope2 < self.slope_max:
                ln_priors.append(0.)
            else:
                ln_priors.append(-np.inf) # Arbitrarily small number

        return ln_priors

#-----------------------------------------------------------------------------#

    def flux(self, spectrum, params):
        spectral_axis = spectrum.spectral_axis
        norm_wavelength = spectrum.norm_wavelength
    
        if self.broken_pl:
            wave_break = params["wave_break"]
            slope1 = params["slope1"]
            slope2 = params["slope2"]
            norm = params["norm_PL"]
    
            flux = np.zeros_like(spectral_axis)
            mask = spectral_axis < wave_break
            flux[mask] = norm * (spectral_axis[mask] / norm_wavelength) ** slope1
            flux[~mask] = norm * (spectral_axis[~mask] / norm_wavelength) ** slope2
        else:
            norm = params["norm_PL"]
            slope1 = params["slope1"]
    
            flux = norm * (spectral_axis / norm_wavelength) ** slope1
    
        return flux
