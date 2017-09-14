#!/usr/bin/python

import re
import sys
import numpy as np
import scipy.interpolate
import numpy as np
import scipy.integrate

from utils.runningmeanfast import runningMeanFast
from utils.gaussian_kernel import gaussian_kernel

from .ComponentBase import Component
from ..Spectrum import Spectrum

class HostGalaxyComponent(Component):
    '''
    Host Galaxy Component
    \f$ F_{\lambda,\rm Host}\ =\ \sum_{1}^{N} F_{\rm Host,i} HostTempl_{\lambda,i}(\sig_*) \f$
    This component has N templates and N+1 parameters. 

    normalization: \f$ F_{\rm Host,i} \f$ for each of the N templates.

    stellar line dispersion: \f$ \sig_* \f$

    '''

    def __init__(self):
        super(HostGalaxyComponent, self).__init__()

        self.host_gal = self.load_templates()
        self.interp_host_gal = [] # interpolated to data provided
        self.interp_norm_flux = []
        self.name = "HostGalaxy"

        self.norm_min = None # np.array([None for i in range(self.n_templates)])
        self.norm_max = None # np.array([None for i in range(self.n_templates)])

        self.stellar_disp_min = None
        self.stellar_disp_max = None
        
#! need to read these from yaml
        self.norm_min = hg_norm_min
        self.norm_max = hg_norm_max
        self.stellar_disp_min = stellar_disp_min
        self.stellar_disp_max = stellar_disp_max

    @property
    def is_analytic(self):
        return False

    def load_templates(self):
        template_list = glob.glob(os.path.join(host_galaxy_models, "*"))
        assert len(template_list) != 0, "No host galaxy templates found in specified diretory {0}".format(host_galaxy_models)
    
        # read in all of the templates
        self.host_gal = list()
    
        for template_filename in template_list:
            with open(template_filename) as template_file:
                host = Spectrum()
                host.wavelengths, host.flux = np.loadtxt(template_filename, unpack=True)
                self.host_gal.append(template)

        return self.host_gal

    @property
    def model_parameter_names(self):
        '''
        Returns a list of model parameter names.
        Since the number of parameters depends on the number of templates (only
        known at run time), this must be provided by a method.

        The parameters are normalization, one for each template, followed by stellar dispersion.
        '''
        parameter_names = list()
        for i in range(1, len(self.host_gal)+1):
            parameter_names.append("norm_{0}".format(i))
        parameter_names.append("stellar_disp")
        return parameter_names
    
    @property
    def parameter_count(self):
        ''' Returns the number of parameters of this component. '''
        no_parameters = len(self.host_gal) + 1
        
        return no_parameters

    def initial_values(self, spectrum):
        '''

        Needs to sample from prior distribution.
        Return type must be a single list (not an np.array).
        '''

#! read in boxcar_width from yaml

#! need to read in norm_min/max from yaml
        if self.norm_max == "max_flux":
            flux_max = max(runningMeanFast(spectrum.flux, boxcar_width))
            self.norm_max = flux_max

        # the size parameter will force the result to be a numpy array - not the case
        # if the inputs are single-valued (even if in the form of an array)
        norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max, size=len(self.host_gal))

        stellar_disp_init = np.random.uniform(low=self.stellar_disp_min, high=self.stellar_disp_max)

        return norm_init.tolist() + [stellar_disp_init]

    def initialize(self, data_spectrum):
        '''
        Perform any initializations where the data is optional.
        '''

        for template in self.host_gal:
            f = scipy.interpolate.interp1d(template.wavelengths, template.flux) # returns function
            self.interp_host_gal.append(f(data_spectrum.wavelengths))
            self.interp_norm_flux.append(f(data_spectrum._norm_wavelength))

    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.
        
        @param params
        '''

        # need to return parameters as a list in the correct order
        ln_priors = list()
        
        norm = list()
        for i in range(1, len(self.host_gal)+1):
            norm.append(params[self.parameter_index("norm_{0}".format(i))])

        stellar_disp = params[self.parameter_index("stellar_disp")]

        # Flat prior within the expected ranges.
        for i in range(len(self.host_gal)):
            if self.norm_min < norm[i] < self.norm_max:
                ln_priors.append(0.0)
            else:
                ln_priors.append(np.inf)

        # Stellar dispersion parameter
        if self.stellar_disp_min < stellar_disp < self.stellar_disp_max:
            ln_priors.append(0.0)
        else:
            ln_priors.append(-np.inf)
        
        return ln_priors

    def flux(self, spectrum, parameters=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        
        norm = list()
        for i in range(1, len(self.host_gal)+1):
            norm.append(parameters[self.parameter_index("norm_{0}".format(i))])

        stellar_disp = parameters[self.parameter_index("stellar_disp")]

        assert len(parameters) == self.parameter_count, \
                "The wrong number of indices were provided: {0}".format(parameters)

        #Convolve to increase the velocity dispersion. Need to
        #consider it as an excess dispersion above that which
        #is intrinsic to the template. For the moment, the
        #implicit assumption is that each template has an
        #intrinsic velocity dispersion = 0 km/s.
        
        #Create the dispersion-convolution matrix.
        #Kmat = self.stellar_disp_matrix(stellar_disp,spectrum)
        Kmat = np.identity(len(spectrum.wavelengths))

        self.host_gal_final = np.zeros(len(spectrum.wavelengths)) # calculate flux on this array
        for i in range(len(self.host_gal)):
            convolved_template = Kmat.dot(self.interp_host_gal[i])
            self.host_gal_final += norm[i]/self.interp_norm_flux[i] * convolved_template

        return self.host_gal_final
#! still need to finalize this function below
    def stellar_disp_matrix(self, stellar_disp, spectrum):

        Kmat = np.zeros((len(spectrum.wavelengths),len(spectrum.wavelengths)))
        lam = spectrum.wavelengths
        for k,lamk in enumerate(spectrum.wavelengths):
            sig = stellar_disp * lamk/3.e5 #Assume the dispersion is provided in km/s.

            #To speed things up, we'll only consider bins with central
            #wavelengths within 5 sigma of the current spectral bin.

            #Get the bin indices that are closest to +/- 5 sigma.
            lmin = np.argmin(abs((lamk-lam)/sig - 5.))
            lmax = np.argmin(abs((lamk-lam)/sig + 5.))

            #See if we are near the bounds and determine
            #the kernel normalization accordingly.
            if lmin>0 and lmax<len(lam):
                norm = sig*(2.*np.pi)**0.5
            else:
                if lmin==0:
                    a = lam[lmin]-0.5*(lam[lmin+1]-lam[lmin])
                    b = lam[lmax]+0.5*(lam[lmax+1]-lam[lmax])
                else:
                    a = lam[lmin]-0.5*(lam[lmin]-lam[lmin-1])
                    b = lam[lmax]+0.5*(lam[lmax]-lam[lmax-1])
                norm = scipy.integrate.quad(gaussian_kernel, a, b, args=(lamk,sig))[0]

            for l in range(lmin,lmax+1):
                if l==0:
                    a = lam[l]-0.5*(lam[l+1]-lam[l])
                    b = lam[l]+0.5*(lam[l+1]-lam[l])
                elif l==len(lam)-1:
                    a = lam[l]-0.5*(lam[l]-lam[l-1])
                    b = lam[l]+0.5*(lam[l]-lam[l-1])
                else:
                    a = lam[l]-0.5*(lam[l]-lam[l-1])
                    b = lam[l]+0.5*(lam[l+1]-lam[l])
                Kmat[k,l] = scipy.integrate.quad(gaussian_kernel, a, b, args=(lamk,sig))[0]/norm

        return Kmat

