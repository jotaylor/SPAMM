#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from .ComponentBase import Component
from utils.runningmeanfast import runningMeanFast
from utils.parse_pars import parse_pars
import json
import os
ABSPATH = os.path.dirname(os.path.realpath(__file__))

#-----------------------------------------------------------------------------#

from astropy.modeling.functional_models import Gaussian1D



class EmissionComponent(Component):
    """
    Emission Component for modeling emission lines.

    This component initializes emission lines from a JSON file and sets up
    the necessary parameters for modeling.

    Attributes:
        name (str): Name of the component, i.e., "Emission".
        inputpars (dict): Dictionary of input parameters parsed from configuration.
        lines (list): List of EmissionLine objects initialized from the JSON file.
        model_parameter_names (list): List of model parameter names, including 'narrow_width' and parameters from each emission line.
        narrow_width_min (float): Minimum value for the narrow width parameter.
        narrow_width_max (float): Maximum value for the narrow width parameter.
        narrow_amp_min (float): Minimum value for the narrow amplitude parameter.
        narrow_amp_max (float): Maximum value for the narrow amplitude parameter.
        broad_width_min (float): Minimum value for the broad width parameter.
        broad_width_max (float): Maximum value for the broad width parameter.
        broad_amp_min (float): Minimum value for the broad amplitude parameter.
        broad_amp_max (float): Maximum value for the broad amplitude parameter.
        loc_size (float): Location size parameter.

    Args:
        pars (dict, optional): Dictionary of parameters. If None, default parameters are used.
        wl (optional): Wavelength parameter (not used in this implementation).
        mask (optional): Mask parameter (not used in this implementation).
    """
    def __init__(self, pars=None, wl=None, mask=None):
        super().__init__()

        self.name = "Emission"
        self.inputpars = parse_pars()["emission_lines"]
    
        # Load emission lines from JSON file
        with open('emission_lines.json') as f:
            emission_lines = json.load(f)

        # Initialize lines
        self.lines = [EmissionLine(line) for line in emission_lines]

        # Define parameters
        self.model_parameter_names = ['narrow_width']

        for line in self.lines:
            self.model_parameter_names.extend(line.params)

        # Retrieve ranges for amplitude and width parameters from inputpars
        self.narrow_width_min = self.inputpars['narrow_width_min']
        self.narrow_width_max = self.inputpars['narrow_width_max']
        self.narrow_amp_min = self.inputpars['narrow_amp_min']
        self.narrow_amp_max = self.inputpars['narrow_amp_max']

        self.broad_width_min = self.inputpars['broad_width_min']
        self.broad_width_max = self.inputpars['broad_width_max']
        self.broad_amp_min = self.inputpars['broad_amp_min']
        self.broad_amp_max = self.inputpars['broad_amp_max']

        self.loc_size = self.inputpars['loc_size']

    @property
    def is_analytic(self):
        return True

    def initial_values(self, spectrum, manual_initial_values=None):

        if manual_initial_values is not None:
            return list(manual_initial_values.values())
        
        initial_values = []

        # Set initial value for narrow width
        narrow_width = np.random.uniform(low=self.narrow_width_min, high=self.narrow_width_max)
        initial_values.append(narrow_width)

        for line in self.lines:
            if line.is_broad:
                initial_values.extend([np.random.uniform(low=self.narrow_amp_min, high=self.narrow_amp_max),
                                       np.random.uniform(low=line.wavelength - self.loc_size/2, high=line.wavelength + self.loc_size/2)])
                for _ in range(line.num_components):
                    initial_values.extend([np.random.uniform(low=self.broad_amp_min, high=self.broad_amp_max),
                                           np.random.uniform(low=line.wavelength - self.loc_size/2, high=line.wavelength + self.loc_size/2),
                                           np.random.uniform(low=self.broad_width_min, high=self.broad_width_max)])
            else:
                initial_values.extend([np.random.uniform(low=self.narrow_amp_min, high=self.narrow_amp_max),
                                       np.random.uniform(low=line.wavelength - self.loc_size/2, high=line.wavelength + self.loc_size/2)])
        
        return initial_values

    def ln_priors(self, params):
        ln_priors = []

        narrow_width = params['narrow_width']
        if self.narrow_width_min < narrow_width < self.narrow_width_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf)

        for line in self.lines:
            if line.is_broad:
                amp = params[f'{line.name}_amp_0']
                if self.broad_amp_min < amp < self.broad_amp_max:
                    ln_priors.append(0.)
                else:
                    ln_priors.append(-np.inf)
                
                loc = params[f'{line.name}_loc_0']
                if line.wavelength - self.loc_size/2 < loc < line.wavelength + self.loc_size/2:
                    ln_priors.append(0.)
                else:
                    ln_priors.append(-np.inf)
                
                for comp in range(line.num_components):
                    amp = params[f'{line.name}_amp_{comp+1}']
                    if self.broad_amp_min < amp < self.broad_amp_max:
                        ln_priors.append(0.)
                    else:
                        ln_priors.append(-np.inf)
                    
                    loc = params[f'{line.name}_loc_{comp+1}']
                    if line.wavelength - self.loc_size/2 < loc < line.wavelength + self.loc_size/2:
                        ln_priors.append(0.)
                    else:
                        ln_priors.append(-np.inf)
                    
                    width = params[f'{line.name}_width_{comp+1}']
                    if self.broad_width_min < width < self.broad_width_max:
                        ln_priors.append(0.)
                    else:
                        ln_priors.append(-np.inf)
            else:
                amp = params[f'{line.name}_amp_0']
                if self.narrow_amp_min < amp < self.narrow_amp_max:
                    ln_priors.append(0.)
                else:
                    ln_priors.append(-np.inf)
                
                loc = params[f'{line.name}_loc_0']
                if line.wavelength - self.loc_size/2 < loc < line.wavelength + self.loc_size/2:
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

        c_kms = 299792.458  # Speed of light in km/s

        total_flux = np.zeros(len(spectrum.spectral_axis))
        narrow_width = params['narrow_width']

        for line in self.lines:
            if line.is_broad:
                # First do the narrow part of the broad line

                amp = params[f'{line.name}_amp_0']
                mean = params[f'{line.name}_loc_0']
                width = narrow_width * mean / c_kms

                total_flux += amp * np.exp(-(spectrum.spectral_axis - mean)**2 / (2*width**2))

                # Then add the broad components
                for comp in range(line.num_components):
                    amp = params[f'{line.name}_amp_{comp+1}'] # Arbitrary flux unit
                    mean = params[f'{line.name}_loc_{comp+1}'] # Ångströms
                    broad_width = params[f'{line.name}_width_{comp+1}'] * mean / c_kms # Ångströms to km/s
                    total_flux += amp * np.exp(-(spectrum.spectral_axis - mean)**2 / (2*broad_width**2))

            else:
                amp = params[f'{line.name}_amp_0']
                mean = params[f'{line.name}_loc_0']
                width = narrow_width * mean / c_kms
                total_flux += amp * np.exp(-(spectrum.spectral_axis - mean)**2 / (2*width**2))

        return total_flux
    
class EmissionLine:
    def __init__(self, lparams):
        self.name = lparams['name']
        self.wavelength = lparams['wavelength']
        self.is_broad = lparams.get('is_broad', False)
        self.num_components = lparams.get('num_components', 1)
        if self.is_broad:
            self.params = [f"{self.name}_amp_0", f"{self.name}_loc_0"]
            self.params += [f"{self.name}_{suffix}_{i+1}" for i in range(self.num_components) for suffix in ('amp', 'loc', 'width')]
        else:
            self.params = [f"{self.name}_{suffix}_{i}" for i in range(self.num_components) for suffix in ('amp', 'loc')]
    
    