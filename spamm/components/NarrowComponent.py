#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from .ComponentBase import Component
from utils.runningmeanfast import runningMeanFast
from utils.parse_pars import parse_pars

#-----------------------------------------------------------------------------#

from astropy.modeling.functional_models import Gaussian1D

class NarrowComponent(Component):
    def __init__(self, pars=None):
        super().__init__()

        self.name = "Narrow"

        if pars is None:
            self.inputpars = parse_pars()["narrow_lines"]
        else:
            self.inputpars = pars

        self.wavelengths = self.inputpars['wavelengths']

        # Define parameters
        self.model_parameter_names = ['width']
        for i in range(len(self.wavelengths)):
            self.model_parameter_names.extend([f'amp_{i+1}', f'center_{i+1}'])

        c = 299792.458
        # Retrieve ranges for amplitude and mean parameters from inputpars
        self.width_min = self.inputpars['width_min']
        self.width_max = self.inputpars['width_max']
        self.amp_min = self.inputpars['amp_min']
        self.amp_max = self.inputpars['amp_max']
        self.center_size = self.inputpars['center_size']
        self.center_size = [(wl - self.center_size/2, wl + self.center_size/2) for wl in self.wavelengths]

    @property   
    def is_analytic(self):
        return True

    def initial_values(self, spectrum):
        initial_values = []

        # Set initial values for width
        width = np.random.uniform(low=self.width_min, high=self.width_max)
        initial_values.append(width)

        # Set initial values for amplitude and center
        if self.amp_max == "max_flux":
            self.amp_max = max(runningMeanFast(spectrum.flux, self.inputpars["boxcar_width"]))
        elif self.amp_max == "fnw":
            fnw = spectrum.norm_wavelength_flux
            self.amp_max = fnw

        for i in range(len(self.wavelengths)):
            initial_values.extend([np.random.uniform(low=self.amp_min, high=self.amp_max),
                                   np.random.uniform(low=self.center_size[i][0], high=self.center_size[i][1])])
            
        return initial_values

    def ln_priors(self, params):
        ln_priors = []

        width = params[self.parameter_index('width')]
        if self.width_min < width < self.width_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf)
        
        for i in range(len(self.wavelengths)):
            amp = params[self.parameter_index(f'amp_{i+1}')]
            if self.amp_min < amp < self.amp_max:
                ln_priors.append(0.)
            else:
                ln_priors.append(-np.inf)
            
            center = params[self.parameter_index(f'center_{i+1}')]
            if self.center_size[i][0] < center < self.center_size[i][1]:
                ln_priors.append(0.)
            else:
                ln_priors.append(-np.inf)

        return ln_priors
    
    def flux(self, spectrum, params):
        """
        Compute the flux for this component for a given wavelength grid
        and parameters. Use the initial parameters if none are specified.

        Args:
            spectrum (Spectrum object): The input spectrum.
            parameters (list): The parameters for the Gaussian emission lines.

        Return:
            total_flux (numpy array): The input spectrum with the Gaussian emission lines added.
        """
        
        assert len(params) == len(self.model_parameter_names), \
            f"The wrong number of indices were provided: {params}"
        
        total_flux = np.zeros(len(spectrum.spectral_axis))
        width = params[self.parameter_index('width')]
        for i in range(len(self.wavelengths)):
            amp = params[self.parameter_index(f'amp_{i+1}')]
            center = params[self.parameter_index(f'center_{i+1}')]
            gaussian = Gaussian1D(amplitude=amp, mean=center, stddev=width)
            total_flux += gaussian(spectrum.spectral_axis)
        return total_flux