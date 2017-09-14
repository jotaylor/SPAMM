#!/usr/bin/python

import sys
import numpy as np
from astropy.modeling.powerlaws import PowerLaw1D,BrokenPowerLaw1D

from .ComponentBase import Component
#! from utils import runningMeanFast
#! get boxcar_width from yaml

class NuclearContinuumComponent(Component):
    '''
    AGN Continuum Component
    \f$ F_{\lambda,{\rm PL}}=F_{\rm PL,0} \ \left(\frac{\lambda}{\lambda_0}\right)^{\alpha} \f$ 
    This component has two parameters:

    normalization : \f$ F_{\rm PL,0} \f$ 
    slope : \f$ \alpha \f$ 

    '''
    def __init__(self, broken_pl=False):
        super(NuclearContinuumComponent, self).__init__()

        self.broken_powerlaw = broken_pl
        self.model_parameter_names = list()
        if not self.broken_powerlaw:
            self.model_parameter_names.append("norm_PL")
            self.model_parameter_names.append("slope1")
        else:
            self.model_parameter_names.append("wave_break")
            self.model_parameter_names.append("norm_PL")
            self.model_parameter_names.append("slope1")
            self.model_parameter_names.append("slope2")
        self.name = "Nuclear"

#! these should be read in from yaml
        self.norm_min = None
        self.norm_max = None
        self.slope_min = None
        self.slope_max = None
        self.wave_break_min = None
        self.wave_break_max = None

    @property
    def is_analytic(self):
        return True    

    def initial_values(self, spectrum):
        '''
        Needs to sample from prior distribution.
        Return type must be a list (not an np.array).

        Called by the emcee.
        
        :param spectrum
        
        Returns:
        --------
            norm_init : array-like

            slope_init : array-like
        '''

        pl_init = []
        if pl_norm_max == "max_flux":
            self.norm_max = max(runningMeanFast(spectrum.flux, boxcar_width))

        if self.broken_pl:
            size = 2
#! need to read in wave_break_min/max from yaml            
            if self.wave_break_min == "min_wl":
                self.wave_break_min = min(spectrum.wavelength)
            if self.wave_break_max == "max_wl": 
                self.wave_break_max = max(spectrum.wavelength)
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
#! need to modify emcee initial_values call

#-------------------------#

    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.

        normalization : uniform linear prior between 0 and the maximum of the spectral flux after computing running median
        slope : uniform linear prior in range [-3,3]
        '''

        # need to return parameters as a list in the correct order
        ln_priors = list()

        if self.broken_pl:
            wave_break = params[self.parameter_index("wave_break")]
            if self.wave_break_min < wave_break < wave_break_max:
                ln_priors.append(0.)
            else:
                #arbitrarily small number
                ln_priors.append(-np.inf)
        
        norm = params[self.parameter_index("norm_PL")]
        if self.norm_min < norm < self.norm_max:
            ln_priors.append(0.)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)

        slope1 = params[self.parameter_index("slope1")]
        if self.slope_min < slope1 < self.slope_max:
            ln_priors.append(0.)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)
            # TODO - suppress "RuntimeWarning: divide by zero encountered in log" warning.
        if self.broken_pl:
            slope2 = params[self.parameter_index("slope2")]
            if self.slope_min < slope2 < self.slope_max:
                ln_priors.append(0.)
            else:
                #arbitrarily small number
                ln_priors.append(-np.inf)

        return ln_priors

    def flux(self, spectrum, parameters=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        assert len(parameters) == len(self.model_parameter_names), "The wrong number of indices were provided: {0}".format(parameters)
        
        norm = parameters[self.parameter_index("norm_PL")]
        slope1 = parameters[self.parameter_index("slope1")]
        if not broken_pl:   
            PL = PowerLaw1D(norm, spectrum.norm_wavelength, slope1)  
        else:
            x_break = parameters[self.parameter_index("wave_break")]
            slope2 = parameters[self.parameter_index("slope2")]
            PL = BrokenPowerLaw1D(norm, x_break, slope1, slope2)
        flux = PL.evaluate(spectrum.wavelengths)

#!
        return flux
