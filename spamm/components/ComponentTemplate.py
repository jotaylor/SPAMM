#!/usr/bin/python

# To create a new component class, copy this file and fill in the values as instructed.
# Follow comments the begin with "[replace]", then delete the comment.

import sys
import numpy as np
from .ComponentBase import Component

# [replace] "TemplateComponent" with the name of the new component
class TemplateComponent(Component):
    '''
    Describe your component here.

    This component has n parameters:

    parameter1 : a short description of parameter 1
    parameter2 : a short description of parameter 2
    ...
    parametern : a short description of parameter n

    '''
    def __init__(self):
        # [replace] fill in the same name you gave above
        super(TemplateComponent, self).__init__()

        # [replace] give the parameters names (spaces are ok), one line for each parameter
        self.model_parameter_names = list() # this may need to be defined as a method
        self.model_parameter_names.append("parameter n")

        self._norm_wavelength =  None

        # [replace] define variables for min/max values for each parameter range
        self.min_parameter1 = None
        self.max_parameter1 = None
        # etc.

    @property
    def is_analytic(self):
        return True/False # choose the appropriate value.

    # This must be defined if the component is NOT analytical.
    def native_wavelength_grid(self):
        ''' Returns the wavelength grid native to this component. '''

    def initial_values(self, spectrum=None):
        '''

        Needs to sample from prior distribution.
        '''

        # call super() implementation
        super(NuclearContinuumComponent, self).initialize()

        # [replace] calculate/define minimum and maximum values for each parameter.
        self.min_parameter1 = ...
        self.max_parameter1 = ...

        # [replace] this is an example of a random flat distribution
        # See for other options: http://docs.scipy.org/doc/numpy/reference/routines.random.html
        parameter1_init = np.random.uniform(low=self.min_parameter1,
                                                                                high=self.max_parameter1)

        self.min_parameter2 = ...
        self.max_parameter2 = ...

        # [replace] this is an example of a random lognormal distribution
        # parameter2_init = np.random.lognormal(mean=, sigma=, size=)

        # [replace] return a list of all parameter_init values
        # NOTE: Order is important! Place them in the same order they were defined
        #       in __init__ above.
        return [parameter1_init, parameter2_init]

    def initialize(self, data_spectrum=None):
        '''
        Perform any initializations where the data is optional.
        '''
        if data_spectrum is None:
            raise Exception("The data spectrum must be specified to initialize" + 
                                            "{0}.".format(self.__class__.__name__))
        self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths)

    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.

        @param params
        '''

        # need to return parameters as a list in the correct order
        ln_priors = list()

        # [replace] Put code here that calculates the ln of the priors
        # given the value of the parameters.

        # [replace] Get the current value of the parameters. Use the names
        # as defined in __init__() above.
        parameter1 = params["par_1"]
        parametern = params["par_2"]

        # [replace] append each parameter, in the correct order, to the "ln_priors" list

        return ln_priors

    def flux(self, wavelengths=None, params=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        assert len(params) == len(self.model_parameter_names), ("The wrong number " +
                                                                "of indices were provided: {0}".format(params))

        # calculate flux of the component
        # [replace] fill in the flux calculation
        flux = ...

        return flux

