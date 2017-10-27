#!/usr/bin/python

import re
import sys
import numpy as np
import scipy.interpolate
import numpy as np
import scipy.integrate
from scipy import signal
from astropy.convolution import Gaussian1DKernel, convolve
import warnings
import math
from astropy import constants

from utils.runningmeanfast import runningMeanFast
from utils.gaussian_kernel import gaussian_kernel
from utils.fftwconvolve_1d import fftwconvolve_1d
from utils.find_nearest_index import find_nearest
from utils.parse_input import parse_input
from utils.parse_pars import parse_pars

from .ComponentBase import Component
from ..Spectrum import Spectrum

PARS = parse_pars()

#-----------------------------------------------------------------------------#

class HostGalaxyComponent(Component):
    """
    Host Galaxy Component:
    \f$ F_{\lambda,\rm Host}\ =\ \sum_{1}^{N} F_{\rm Host,i} 
        HostTempl_{\lambda,i}(\sig_*) \f$
    This component has N templates and N+1 parameters. 
        normalization: \f$ F_{\rm Host,i} \f$ for each of the N templates.
        stellar line dispersion: \f$ \sig_* \f$

    Attributes:
        host_gal (list): List of Spectrum objects for each template file.
        interp_host (list): 
        interp_norm_flux (list):
#TODO should name be in Component?
        name (str): Name of component, i.e. "Nuclear"
        norm_min ():
        norm_max ():
        stellar_disp_min ():
        stellar_disp_max ():
    """

    def __init__(self):
        super(HostGalaxyComponent, self).__init__()

        self.host_gal = self.load_templates()
        self.interp_host = [] 
        self.interp_norm_flux = []
        self.name = "HostGalaxy"

        self.norm_min = PARS["hg_norm_min"]
        self.norm_max = PARS["hg_norm_max"]
        self.stellar_disp_min = PARS["hg_stellar_disp_min"]
        self.stellar_disp_max = PARS["hg_stellar_disp_max"]

#-----------------------------------------------------------------------------#

#TODO could this be moved to Component.py?
    @property
    def is_analytic(self):
        """ 
        Method that stores whether component is analytic or not.
        
        Returns:
            Bool (Bool): True is component is analytic.
        """
        
        return False

#-----------------------------------------------------------------------------#

    def load_templates(self):
        """
        Read in all of the host galaxy models.

        Returns:
            self.host_gal (list): List of all host galaxy model Spectrum objects.
        """

        template_list = glob.glob(os.path.join(host_galaxy_models, "*"))
        assert len(template_list) != 0, \
        "No host galaxy templates found in specified diretory {0}".format(host_galaxy_models)
    
        self.host_gal = []
    
        for template_filename in template_list:
            with open(template_filename) as template_file:
                host = Spectrum()
                host.wavelengths, host.flux = np.loadtxt(template_filename, unpack=True)
                self.host_gal.append(host)

        return self.host_gal

#-----------------------------------------------------------------------------#

    @property
    def model_parameter_names(self):
        """
        Determine the list of model parameter names. The number of 
        parameters depends on the number of templates (only known at run time), 
        The parameters are normalization, one for each template, followed by 
        stellar dispersion.
        
        Returns:
            parameter_names (list): List of all model parameter names.
        """

        parameter_names = []
        for i in range(1, len(self.host_gal)+1):
            parameter_names.append("norm_{0}".format(i))
        parameter_names.append("stellar_disp")
        
        return parameter_names

#-----------------------------------------------------------------------------#

    @property
    def parameter_count(self):
        """ 
        Returns the number of parameters of this component. 
        
        Returns: 
            no_parameters (int): Number of componenet parameters.
        """

        #TODO why is it len() +1 ?
        no_parameters = len(self.host_gal) + 1
        
        return no_parameters
    
#-----------------------------------------------------------------------------#

    def initial_values(self, spectrum):
        """
        Needs to sample from prior distribution.

        Args:
            spectrum (Spectrum object): ?

        Returns: 
            list (list): 
        """

        if self.norm_max == "max_flux":
            flux_max = max(runningMeanFast(spectrum.flux, PARS["boxcar_width"]))
            self.norm_max = flux_max

        # The size parameter will force the result to be a numpy array - not the case
        # if the inputs are single-valued (even if in the form of an array)
        norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max, size=len(self.host_gal))

        stellar_disp_init = np.random.uniform(low=self.stellar_disp_min, high=self.stellar_disp_max)

        return norm_init.tolist() + [stellar_disp_init]

#-----------------------------------------------------------------------------#

    def initialize(self, data_spectrum):
        """
        Perform any initializations using the data spectrum.

        Args:
            data_spectrum (Spectrum object): ?

        """
        
        # Calculate flux on this array
        self._flux_arrays = np.zeros(len(data_spectrum.wavelengths)) 

        # We'll eventually need to convolve these in constant 
        # velocity space, so rebin to equal log bins
        self.log_host = []

        fnw = data_spectrum.flux_at_normalization_wavelength


#TODO need to verify if this is necessary            
        # This method lets you interpolate beyond the wavelength 
        #coverage of the template if/when the data covers beyond it.  
        # Function returns 0 outside the wavelength coverage of the template.
        # To broaden in constant velocity space, you need to rebin the 
        #templates to be in equal bins in log(lambda) space.
        for i,template in enumerate(self.host_gal):
            equal_log_bins = np.linspace(min(np.log(template.wavelengths)), 
                                         max(np.log(template.wavelengths)), 
                                         num = len(template.wavelengths))
#TODO need to verify Spectrum method name
            # Bin template fluxes in equal log bins
            binned_template_flux = Spectrum.bin_spectrum(np.log(template.wavelengths), 
                                                         template.flux, 
                                                         equal_log_bins)

            
            binned_wl, binned_flux = equal_log_bins, binned_template_flux
            binned_spectrum = Spectrum.from_array(binned_flux, dispersion=binned_wl)
            self.log_host.append(binned_spectrum)
#TODO need to verify Spectrum method name
#TODO Do we rebin or interpolate here?
            self.interp_host.append(Spectrum.bin_spectrum(template.wavelengths, 
                                                              template.flux, 
                                                              data_spectrum.wavelengths))
            self.interp_norm_flux.append(np.interp(fnw, 
                                                   template.wavelengths, 
                                                   template.flux,
                                                   left=0,
                                                   right=0))

#-----------------------------------------------------------------------------#

    def ln_priors(self, params):
        """
        Return a list of the ln of all of the priors.
        
        @param params
        """
        
        # need to return parameters as a list in the correct order
        ln_priors = []
        
        norm = []
        for i in range(1, len(self.host_gal)+1):
            norm.append(params[self.parameter_index("norm_{0}".format(i))])
        
        stellar_disp = params[self.parameter_index("stellar_disp")]
        
        # Flat prior within the expected ranges.
        for i in range(len(self.host_gal)):
            if self.norm_min[i] < norm[i] < self.norm_max[i]:
                ln_priors.append(0.0)
            else:
                ln_priors.append(-np.inf)

#TODO why is this here? another prior is added
        if np.sum(norm) <= np.max(self.norm_max):
                ln_priors.append(0.0)
        else:
                ln_priors.append(-np.inf)
        
        # Stellar dispersion parameter
        if self.stellar_disp_min < stellar_disp < self.stellar_disp_max:
            ln_priors.append(0.0)
        else:
            ln_priors.append(-np.inf)
        
        return ln_priors

#-----------------------------------------------------------------------------#

    def flux(self, spectrum, parameters):
        """
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        """
        
        norm = []
        for i in range(1, len(self.host_gal)+1):
            norm.append(parameters[self.parameter_index("norm_{0}".format(i))])
        stellar_disp = parameters[self.parameter_index("stellar_disp")]
        parameters_host = norm
        parameters_host.append(stellar_disp)

        assert len(parameters_host) == self.parameter_count, \
                "The wrong number of indices were provided: {0}".format(parameters)

        #Convolve to increase the velocity dispersion. Need to
        #consider it as an excess dispersion above that which
        #is intrinsic to the template. For the moment, the
        #implicit assumption is that each template has an
        #intrinsic velocity dispersion = 0 km/s.

#TODO is norm meant ot be redefined here?
        norm = []
        conv_hosts = []
        
        # The next two parameters are lists of size len(self.host_gal)
        norm_wl = spectrum.normalization_wavelength
        c_kms = constants.c.to("km/s")
        log_norm_wl = np.log(norm_wl)
        tmpl_stellar_disp = PARS["hg_template_stellar_disp"] 
        sigma_conv = (stellar_disp - tmpl_stellar_disp) / c_kms
            
        # Want to smooth and convolve in log space, 
        # since d(log(lambda)) ~ dv/c and we can broaden 
        # based on a constant velocity width
        # Compare smoothing (v/c) to bin size, and that tells you how 
        # many bins wide your Gaussian to convolve over is
        # sigma_conv is the width to broaden over, as given in Eqn 1 of 
        # Vestergaard and Wilkes 2001 (essentially the first line below this)
        for i in range(len(self.host_gal)):
            bin_size = spectrum.log_grid_spacing
#TODO cross-check with Spectrum ^^
            sigma_norm = sigma_conv / bin_size
            sigma_size = PARS["hg_kernel_size_sigma"] * sigma_norm
#TODO can we use astropy below vv
#TODO check if astropy convultion speed has improved
            kernel = signal.gaussian(sigma_size, sigma_norm) / \
                     (np.sqrt(2 * math.pi) * sigma_norm)
            # Check to see if length of array is even. 
            # If it is odd, remove last index
            # fftwconvolution only works on even size arrays
            if len(self.log_host[i].flux) % 2 != 0: 
                self.log_host[i].flux = self.log_host[i].flux[:-1]
                self.log_host[i].wavelengths = self.log_host[i].wavelengths[:-1]
            # Convolve flux (in log space) with gaussian broadening kernel
            log_conv_host_flux = fftwconvolve_1d(self.log_host[i].flux, kernel)

#TODO need to check Spectrum.bin_spectrum()
            # Shift spectrum back into linear space.
            # the left and right statements just set the flux value 
            # to zero if the specified log_norm_wl is outside the 
            #bounds of self.log_host[i].wavelengths
            conv_host = Spectrum.bin_spectrum(self.log_host[i].wavelengths,
                                                         log_conv_host_flux,
                                                         np.log(spectrum.wavelengths))
            conv_host_norm_flux = conv_host.flux_at_normalization_wavelength
            
            # Find NaN errors early from dividing by zero.
#TODO check below syntax vv
            conv_host_norm_flux = np.nan_to_num(conv_host_norm_flux)
            conv_hosts.append(conv_host)
            
            # Scale normalization parameter to flux in template
            norm.append(parameters[i] / conv_host_norm_flux) 
            self._flux_arrays += norm[i] * conv_hosts[i]

        return self._flux_arrays

