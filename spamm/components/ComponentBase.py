#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from abc import ABC, abstractmethod
import numpy as np
import sys

from utils.parse_pars import parse_pars
PARS = parse_pars()

#-----------------------------------------------------------------------------#

# Compatible with python 2 & 3.
class Component(ABC):
    '''
    Description of Component class here.

    This class is abstract; use (or create) subclasses of this class.
    Functionality that is common to all subclasses should be implemented here.
    '''

    def __init__(self):
        self.z = None # redshift
        self.reddening_law = None
        #self.model_parameters = list()
        #self.model_parameter_names = list()

        # wavelength grid defined by the data
        # interpolate to this if necessary (i.e. no analytical component)
        self.data_wavelength_grid = None
        self.interpolated_flux = None # based on data, defined in initialize()

    def parameter_index(self, parameter_name):
        '''
        Returns the index of the given parameter in the model_parameter_names list.

        This method uses the index method of the list to find the first occurrence of the given parameter name. If the parameter name is not found in the list, it returns None.

        Args:
            parameter_name (str): The name of the parameter to find.

        Returns:
            int or None: The index of the parameter in the model_parameter_names list, or None if the parameter is not found.
        '''

        try:
            return self.model_parameter_names.index(parameter_name)
        except ValueError:
            return None

    @property
    @abstractmethod
    def is_analytic(self):
        ''' Subclasses must provide an implementation of this property. '''
        pass

    @property
    def parameter_count(self):
        ''' Returns the number of parameters of this component. '''
        if self.z:
            return len(self.model_parameter_names + 1)
        else:
            return len(self.model_parameter_names)

    @abstractmethod
    def initial_values(self, spectrum=None):
        ''' Return type must be a list (not an np.array) '''
        pass

    @abstractmethod
    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.

        @param params
        '''
        pass

    def native_wavelength_grid(self):
        '''
        Returns the wavelength grid native to this component.

        This needs to be overridden by subclasses.
        '''
        if self.is_analytic:
            pass # implement in subclass
        else:
            assert True, "The method 'native_wavelength_grid' must be defined for {0}.".format(self.__class__.__name__)

    @abstractmethod
    def flux(self, wavelengths=None, params=None):
        pass

    def grid_spacing(self):
        ''' Return the spacing of the wavelength grid in Ångstroms. Does not support variable grid spacing. '''
        if self.is_analytic:
            # analytic components don't have grid spacing
            return None
        else:
            return self.native_wavelength_grid[1] - self.native_wavelength_grid[0]

    # TODO: also, shouldn't this be an abstract method? -oliver
    def initialize(self, data_spectrum=None):
        '''
        Initialize the component with a given data spectrum.

        This method is intended to be overridden by subclasses. If a subclass does not provide its own implementation,
        an AssertionError will be raised when this method is called.

        Parameters:
        data_spectrum (Spectrum): The data spectrum to initialize the component with. Default is None.

        Raises:
        AssertionError: If the component is not analytic and a subclass has not provided its own implementation.
        '''
        # Check that the component wavelength grid is not more coarse than the data wavelength grid
        assert self.is_analytic, f"The 'initialize' method of the component '{self.__class__.__name__}' must be defined."

# TODO - what if component grid is not uniform? currently require that it be.
            # self.data_wavelength_grid = np.array(data_spectrum.spectral_axis)
            # data_delta_wavelength = data_spectrum.spectral_axis[1] - data_spectrum.spectral_axis[0]
            # comp_delta_wavelength = self.native_wavelength_grid[1] - native_wavelength_grid[0]

            
            #if comp_delta_wavelength > data_delta_wavelength:


    @property
    def fast_interp(self):
        '''Determines if fast interpolation should be used instead of rebin_spec'''
        if PARS["rebin_spec"] is False:
            return True
        else:
            return False
        
