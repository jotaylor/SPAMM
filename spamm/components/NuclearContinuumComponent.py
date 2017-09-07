#!/usr/bin/python

import sys
import numpy as np
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
            self.model_parameter_names.append("norm_PL1")
            self.model_parameter_names.append("slope1")
        else:
            self.model_parameter_names.append("wave_break")
            self.model_parameter_names.append("norm_PL1")
            self.model_parameter_names.append("norm_PL2")
            self.model_parameter_names.append("slope1")
            self.model_parameter_names.append("slope2")
        self.name = "Nuclear"

        self._norm_wavelength =  None

        self.norm_min = None
        self.norm_max = None
        self.slope_min = None
        self.slope_max = None

    @property
    def is_analytic(self):
        return True    

    def initial_values(self, spectrum=None):
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

        if self.broken_pl:
            size = 2
            self.wave_break_min = min(spectrum.wavelength)
            self.wave_break_max = max(spectrum.wavelength)
            self.wave_break_init = np.random.uniform(low=self.wave_break_min, 
                                                     high=self.wave_break_max,
                                                     size=1)
            pl_init.append(self.wave_break_init)
        else:
            size = 1
        norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max, size=size)
        pl_init.append(norm_init)

        self.norm_min = 0
        self.norm_max = max(runningMeanFast(spectrum.flux, boxcar_width))

        self.slope_min = -3.0 #[x for x in slopes]
        self.slope_max = 3.0

        slope_init = np.random.uniform(low=self.slope_min, high=self.slope_max, size=size)
        pl_init.append(slope_init)

        return pl_init
#! need to modify emcee initial_values call

    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.

        normalization : uniform linear prior between 0 and the maximum of the spectral flux after computing running median
        slope : uniform linear prior in range [-3,3]
        '''

        # need to return parameters as a list in the correct order
        ln_priors = list()

        norm = params[self.parameter_index("norm_PL")]
        slope = params[self.parameter_index("slope")]
        if self.norm_min < norm < self.norm_max:
            ln_priors.append(0.)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)
            
        if self.slope_min < slope < self.slope_max:
            ln_priors.append(0.)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)
            # TODO - suppress "RuntimeWarning: divide by zero encountered in log" warning.

        return ln_priors

    def flux(self, spectrum=None, parameters=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        assert len(parameters) == len(self.model_parameter_names), "The wrong number of indices were provided: {0}".format(parameters)
        
        if not broken_pl:   
            norm = parameters[self.parameter_index("norm_PL")]
            slope = parameters[self.parameter_index("slope")]
            normalized_wavelengths = spectrum.wavelengths / spectrum.norm_wavelength
            flux = norm * np.power(normalized_wavelengths, slope)
#!        else:
#!
        return flux
