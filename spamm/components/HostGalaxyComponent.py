#!/usr/bin/python

import re
import sys
import numpy as np
import scipy.interpolate
import numpy as np
import scipy.integrate
import pyfftw 
from scipy import signal
from astropy.convolution import Gaussian1DKernel, convolve
import warnings
import math
from scipy.fftpack.helper import next_fast_len
from astropy import constants

from utils.runningmeanfast import runningMeanFast
from utils.gaussian_kernel import gaussian_kernel
from utils.fftwconvolve_1d import fftwconvolve_1d
from utils.find_nearest_index import find_nearest

from .ComponentBase import Component
from ..Spectrum import Spectrum

#-----------------------------------------------------------------------------#

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
        self.interp_host_gal = [] 
        self.interp_norm_flux = []
        self.name = "HostGalaxy"

        self.norm_min = None # np.array([None for i in range(self.n_templates)])
        self.norm_max = None # np.array([None for i in range(self.n_templates)])

        self.stellar_disp_min = None
        self.stellar_disp_max = None

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

    def initialize(self, data_spectrum=None):
        '''
        Perform any initializations using data spectrum.
        '''
        if data_spectrum is None:
            raise Exception("The data spectrum must be specified to initialize" + 
                            "{0}.".format(self.__class__.__name__))

        self._flux_arrays = np.zeros(len(data_spectrum.wavelengths)) # calculate flux on this array

        # We'll eventually need to convolve these in constant 
        # velocity space, so /ebin to equal log bins
        self.rebin_log_templates = list()

#        fnw = self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths) # flux at normalization wavelength


#! need to verify if this is necessary            
        for i,template in enumerate(self.host_gal):
            # This method lets you interpolate beyond the wavelength 
            #coverage of the template if/when the data covers beyond it.  
            # Function returns 0 outside the wavelength coverage of the template.
            # To broaden in constant velocity space, you need to rebin the 
            #templates to be in equal bins in log(lambda) space.
            equal_log_bins = np.linspace(min(np.log(template.wavelengths)), max(np.log(template.wavelengths)), num = len(template.wavelengths))
#! need to verify Spectrum method name
            template_fluxes_rebin_equal_log_fluxes = Spectrum.bin_spectrum(np.log(template.wavelengths), template.flux, equal_log_bins) # do the rebinning

            
            rebinwavelengths,rebinflux = equal_log_bins, template_fluxes_rebin_equal_log_fluxes
            template_equal_log_rebin_spec = Spectrum(rebinflux)
            template_equal_log_rebin_spec.wavelengths=rebinwavelengths
            self.rebin_log_templates.append(template_equal_log_rebin_spec)


#! need to verify Spectrum method name
#! Do we rebin or interpolate here?
            self.interp_host_gal.append(Spectrum.bin_spectrum(template.wavelengths, template.flux, data_spectrum.wavelengths))
            self.interpolated_normalization_flux.append(np.interp(fnw,template.wavelengths, template.flux,left=0,right=0))

    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.
        
        @param params
        '''
        
        # need to return parameters as a list in the correct order
        ln_priors = list()
        
        normalization = list()
        for i in range(1, len(self.host_gal)+1):
            normalization.append(params[self.parameter_index("normalization_{0}".format(i))])
        
        stellar_dispersion = params[self.parameter_index("stellar dispersion")]
        # Normalization parameter

        
        # Flat prior within the expected ranges.
        for i in range(len(self.host_gal)):
            if self.norm_min[i] < normalization[i] < self.norm_max[i]:
                ln_priors.append(0.0)
            else:
                ln_priors.append(-np.inf)
        #print('norm',np.sum(normalization),self.norm_max[0],self.norm_min[0],normalization)
        #exit()
#! why is this here? another prior is added
        if np.sum(normalization) <= np.max(self.norm_max):
                ln_priors.append(0.0)
        else:
                ln_priors.append(-np.inf)
        
        # Stellar dispersion parameter
        if self.stellar_dispersion_min < stellar_dispersion < self.stellar_dispersion_max:
            ln_priors.append(0.0)
        else:
            ln_priors.append(-np.inf)
        #print('ln_priors',ln_priors)
        #print('norm',np.sum(normalization),np.max(self.norm_max),np.min(self.norm_min),normalization)
        #exit()
        # ln_prior_norms is an array, need to return a 1D array of parameters to emcee
        return ln_priors



    def flux(self, spectrum=None, parameters=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        
        normalization = list()
        for i in range(1, len(self.host_gal)+1):
            normalization.append(parameters[self.parameter_index("normalization_{0}".format(i))])
        stellar_dispersion = parameters[self.parameter_index("stellar dispersion")]
        parameters_host = normalization
        parameters_host.append(stellar_dispersion)

        assert len(parameters_host) == self.parameter_count, \
                "The wrong number of indices were provided: {0}".format(parameters)

                #Convolve to increase the velocity dispersion. Need to
                #consider it as an excess dispersion above that which
                #is intrinsic to the template. For the moment, the
                #implicit assumption is that each template has an
                #intrinsic velocity dispersion = 0 km/s.
                

        norm = list()
        interpolated_convolved_templates = list()
        # The next two parameters are lists of size len(self.host_gal)
        norm_waves = self.normalization_wavelength(data_spectrum_wavelength=spectrum.wavelengths)
        c_kms = constants.c.to("km/s")
        log_norm_waves = np.log(norm_waves)
        self._flux_arrays[:] = 0.0
        template_stellar_dispersion = 0.0 ## need to specify this value. For the moments we are assuming it is zero.
        sd_over_c = (stellar_dispersion-template_stellar_dispersion)/(c_km_per_s)
            
        for i in range(len(self.host_gal)):
            # Want to smooth and convolve in log space, since d(log(lambda)) ~ dv/c and we can broaden based on a constant velocity width
            # Compare smoothing (v/c) to bin size, and that tells you how many bins wide your Gaussian to convolve over is
            # sigma_conv is the width to broaden over, as given in Eqn 1 of Vestergaard and Wilkes 2001 (essentially the first line below this)
            sigma_conv = sd_over_c
            equal_log_bin_size = self.rebin_log_templates[i].wavelengths[2] - self.rebin_log_templates[i].wavelengths[1]
            sig_norm = sigma_conv/equal_log_bin_size
            kernel = signal.gaussian(1000,sig_norm)/(np.sqrt(2*math.pi)*sig_norm)
            if np.size(self.rebin_log_templates[i].flux)%2 > 0: #check to see if length of array is even. If it is odd need to remove last index as fftwconvolution only works on even size arrays
                self.rebin_log_templates[i].flux = self.rebin_log_templates[i].flux[:-1]
                self.rebin_log_templates[i].wavelengths = self.rebin_log_templates[i].wavelengths[:-1]
            #convolve flux (in log space) with gaussian broadening kernel
            fftwconvolved_flux = fftwconvolve_1d(self.rebin_log_templates[i].flux, kernel)

            #shift spectrum back into linear space
#! need to check Spectrum.bin_spectrum()
            interpolated_template_convolved = Spectrum.bin_spectrum(self.rebin_log_templates[i].wavelengths,fftwconvolved_flux,np.log(spectrum.wavelengths))
            interpolated_template_convolved_normalization_flux = np.interp(log_norm_waves,self.rebin_log_templates[i].wavelengths,	\
            fftwconvolved_flux,left=0,right=0) #the left and right statements just set the flux value to zero if the specified log_norm_waves is outside the bounds of self.rebin_log_templates[i].wavelengths
            
            # Find NaN errors early from dividing by zero.
            assert interpolated_template_convolved_normalization_flux != 0., "Interpolated convolution flux valued at 0 at the location of peak template flux!"
            interpolated_convolved_templates.append(interpolated_template_convolved)
            norm.append(parameters[i] / interpolated_template_convolved_normalization_flux) # Scale normalization parameter to flux in template
            self._flux_arrays += norm[i] * interpolated_convolved_templates[i]

        return self._flux_arrays


