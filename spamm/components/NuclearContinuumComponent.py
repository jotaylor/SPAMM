#!/usr/bin/python

import sys
import numpy as np
from astropy.modeling.powerlaws import PowerLaw1D,BrokenPowerLaw1D

from .ComponentBase import Component
from utils.runningmeanfast import runningMeanFast
from utils.parse_pars import parse_pars

#-----------------------------------------------------------------------------#

class NuclearContinuumComponent(Component):
    """
    AGN Continuum Component
        \f$ F_{\lambda,{\rm PL}}=F_{\rm PL,0} \ 
        \left(\frac{\lambda}{\lambda_0}\right)^{\alpha} \f$ 
    This component has two parameters:
        normalization : \f$ F_{\rm PL,0} \f$ 
        slope : \f$ \alpha \f$ 
    
    Attributes:
        broken_powerlaw (Bool): True if a broken power law should be used.
        model_parameter_names (list): List of model parameter names,
            e.g. slope1, wave_break
        name (str): Name of component, i.e. "Nuclear"
        norm_min (): 
        norm_max ():
        slope_min ():
        slope_max ():
        wave_break_min ():
        wave_break_max ():

    """
    def __init__(self, pars=None, broken=None):
        super().__init__()
        
        self.name = "Nuclear"

        if pars is None:
            self.inputpars = parse_pars()["nuclear_continuum"]
        else:
            self.inputpars = pars

        if broken is None:
            self.broken_pl = self.inputpars["broken_pl"]
        else:
            self.broken_pl = broken
        self.model_parameter_names = list()
        
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
        Return a list of the ln of all of the priors.
#            norm: Uniform linear prior between 0 and the maximum of the 
#                spectral flux after computing running median.
#            slope: Uniform linear prior in range [-3,3]??

        Args:
            params (): ?

        Returns:
            ln_priors (list): ln of all the priors.
        """

        # Need to return parameters as a list in the correct order.
        ln_priors = []

        if self.broken_pl:
            wave_break = params[self.parameter_index("wave_break")]
            if self.wave_break_min < wave_break < self.wave_break_max:
                ln_priors.append(0.)
            else:
                ln_priors.append(-np.inf) # Arbitrarily small number
        
        norm = params[self.parameter_index("norm_PL")]
        if self.norm_min < norm < self.norm_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf) # Arbitrarily small number

        slope1 = params[self.parameter_index("slope1")]
        if self.slope_min < slope1 < self.slope_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf) # Arbitrarily small number
            # TODO - suppress "RuntimeWarning: divide by zero encountered in log" warning.
        if self.broken_pl:
            slope2 = params[self.parameter_index("slope2")]
            if self.slope_min < slope2 < self.slope_max:
                ln_priors.append(0.)
            else:
                ln_priors.append(-np.inf) # Arbitrarily small number

        return ln_priors

#-----------------------------------------------------------------------------#

    def flux(self, spectrum, parameters=None):
        """
        Compute the flux for this component for a given wavelength grid
        and parameters. Use the initial parameters if none are specified.

        Args:
            spectrum (Spectrum object): ?
            parameters (): ?

        Return:
            flux (): Flux of the componenet.
        """
        
        assert len(parameters) == len(self.model_parameter_names), \
            "The wrong number of indices were provided: {0}".format(parameters)
        
        norm = parameters[self.parameter_index("norm_PL")]
        slope1 = parameters[self.parameter_index("slope1")]
        if not self.broken_pl:   
            PL = PowerLaw1D(norm, spectrum.norm_wavelength, slope1)  
        else:
            x_break = parameters[self.parameter_index("wave_break")]
            slope2 = parameters[self.parameter_index("slope2")]
            PL = BrokenPowerLaw1D(norm, x_break, slope1, slope2)
        flux = PL(spectrum.spectral_axis)
        return flux
