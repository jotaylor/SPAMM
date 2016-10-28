#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from six import with_metaclass
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import numpy as np
import sys

# Compatible with python 2 & 3.
class Component(with_metaclass(ABCMeta, object)):
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
        ''' '''
        for idx, pname in enumerate(self.model_parameter_names):
            if parameter_name == pname:
                return idx

        return None

    @property
    def is_analytic(self):

        # define this property in the subclass
        print("Please define the 'is_analytic' property for the class '{0}'.".format(__class__.__name__))
        sys.exit(1)

    @property
    def parameter_count(self):
        ''' Returns the number of parameters of this component. '''
        if self.z:
            return len(self.model_parameter_names + 1)
        else:
            return len(self.model_parameter_names)

    @abstractmethod
    def initial_values(self, spectrum=None):
        ''' Return type must be a list (not an np.array. '''
        pass

    @abstractmethod
    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.

        @param params
        '''

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
    def flux(self, wavelengths=None, parameters=None):
        pass

    def grid_spacing(self):
        ''' Return the spacing of the wavelength grid in Ångstroms. Does not support variable grid spacing. '''
        if self.is_analytic:
            # analytic components don't have grid spacing
            return None
        else:
            return self.native_wavelength_grid[1] - self.native_wavelength_grid[0]

    def initialize(self, data_spectrum=None):
        ''' '''
        # Check that the component wavelength grid is not more coarse than the data wavelength grid
        if self.is_analytic:
            pass
        else:

            assert True, "The 'initialize' method of the component '{0}' must be defined.".format(self.__class__.__name__)


            #self.data_wavelength_grid = np.array(data_spectrum.wavelengths)
            #data_delta_wavelength = data_spectrum.wavelengths[1] - data_spectrum.wavelengths[0]
            #comp_delta_wavelength = self.native_wavelength_grid[1] - native_wavelength_grid[0]

            # TODO - what if component grid is not uniform? currently require that it be.
            #if comp_delta_wavelength > data_delta_wavelength:


