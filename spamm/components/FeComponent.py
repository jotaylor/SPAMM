#!/usr/bin/python

import re
import sys
import numpy as np
import pyfftw 
from scipy import signal
from astropy.convolution import Gaussian1DKernel, convolve
import warnings
from astropy.constants import c
import glob
import os
import math

from utils.runningmeanfast import runningMeanFast
from utils.gaussian_kernel import gaussian_kernel
from utils.fftwconvolve_1d import fftwconvolve_1d
from utils.find_nearest_index import find_nearest
from utils.parse_pars import parse_pars
from utils.rebin_spec import rebin_spec

from .ComponentBase import Component
from ..Spectrum import Spectrum

#-----------------------------------------------------------------------------#

class FeComponent(Component):
    """
    Fe II and III pseudo-continuum of blended emission lines.

    We will use a linear combination of N broadened and scaled iron templates:
    $F_{\lambda,\,{\rm Fe}} = \sum_{i=1}^N F_{{\rm Fe},\,0,\,i} FeTemplate_{\lambda,\,i}(\sigma_i)$
    where $FeTemplate_{\lambda,\,i}$ is iron template $i$ at wavelength $\lambda$,
    $F_{{\rm Fe},\,0,\,i}$ is the template normalization, and $\simga_i$ is the width
    of the broadening kernal.

    This component has 2 kinds of parameters:
        - template normalization for each template
        - FWHM of iron lines being tested for the templates (in units of km/s)

    Parameters by number
    0-(i-1): normalizations of templates 1-i
    i-(2i-1): FWHM of the lines in templates 1 - i

    Attributes:
        ...
    """

    def __init__(self, pars=None):
        super(FeComponent, self).__init__()

        if pars is None:
            self.inputpars = parse_pars()["fe_forest"]
        else:
            self.inputpars = pars
        
        self.load_templates()
        self.model_parameter_names = [f"fe_norm_{x}" for x in range(1, len(self.fe_templ)+1)]
        self.model_parameter_names.append("fe_width") 
        self.interp_fe = []
        self.interp_fe_norm_flux = []
        self.name = "FeForest"
        
        self.norm_min = self.inputpars["fe_norm_min"]
        self.norm_max = self.inputpars["fe_norm_max"]
        self.width_min = self.inputpars["fe_width_min"]
        self.width_max = self.inputpars["fe_width_max"]
        self.templ_width= self.inputpars["fe_template_width"]
        if self.width_min < self.templ_width:
            print(f"Specified minimum Fe width too small, setting Fe width minimum to intrinsic template width = {self.templ_width}")
            self.width_min = self.templ_width

#-----------------------------------------------------------------------------#

    def load_templates(self):
        """Read in all of the Fe templates."""

        # Sort the templates alphabetically.
        template_list = sorted(glob.glob(os.path.join(self.inputpars["fe_templates"], "*")))
        assert len(template_list) != 0, \
        f"No Fe templates found in specified diretory {self.inputpars['fe_templates']}"

        self.fe_templ = []

        for template_filename in template_list:
            wavelengths, flux = np.loadtxt(template_filename, unpack=True)
            flux = np.where(flux<0, 1e-19, flux)
            fe = Spectrum(spectral_axis=wavelengths, flux=flux, flux_error=flux)

            self.fe_templ.append(fe)

#-----------------------------------------------------------------------------#

    def is_analytic(self):
        """ 
        Method that stores whether component is analytic or not.
        
        Returns:
            Bool (Bool): True is component is analytic.
        """

        return False

#-----------------------------------------------------------------------------#

    @property                                                 
    def parameter_count(self):
        """ 
        Returns the number of parameters of this component. 
        
        Returns: 
            no_parameters (int): Number of componenet parameters.
        """

        no_parameters = len(self.fe_templ) + 1

        return no_parameters

#-----------------------------------------------------------------------------#

    def initial_values(self, spectrum):
        """

        Needs to sample from prior distribution.
        Return type must be a single list (not an np.array).
        These are the first guess for the parameters to be fit for in emcee.
        In the case of the Fe Component, this would be the normalization and, for now, FWHM of iron lines.
        Note that the returns will be a list, the first 'i' elements of which are the normalizations and the 
        second 'i' elements of which are the FWHM for a loaded number of 'i' templates.
        
        Args:
            spectrum (Spectrum object): 

        Returns:
            list (list): 
        """

        if self.norm_max == "max_flux":
            flux_max = max(runningMeanFast(spectrum.flux, self.inputpars["boxcar_width"])) 
            self.norm_max = flux_max
        elif self.norm_max == "fnw":
            fnw = spectrum.norm_wavelength_flux
            self.norm_max = fnw 

        norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max, 
                                      size=len(self.fe_templ))

        width_init = np.random.uniform(low=self.width_min, 
                                       high=self.width_max) 
        
        return norm_init.tolist() + [width_init]

#-----------------------------------------------------------------------------#

    def initialize(self, data_spectrum):
        """
        Perform all necessary initializations for the iron component, 
        such as reading in the templates, rebinning them, and 
        interpolating them on the grid scale of the data spectrum.
        """

        self.flux_arrays = np.zeros(len(data_spectrum.spectral_axis)) 

        # We'll eventually need to convolve these in constant velocity space, 
        # so rebin to equal log bins
        # log_fe.wavelength is in log space but flux is not
        self.log_fe = []

        nw = data_spectrum.norm_wavelength

        for i,template in enumerate(self.fe_templ):
            log_fe_wl = np.linspace(min(np.log(template.spectral_axis)), 
                                         max(np.log(template.spectral_axis)), 
                                         num=len(template.spectral_axis))
# TODO need to verify Spectrum method name
#            binned_flux = Spectrum.bin_spectrum(np.log(template.spectral_axis),
#                                                         template.flux,
#                                                         equal_log_bins)
            if self.fast_interp:
                log_fe_flux = np.interp(log_fe_wl, 
                                        np.log(template.spectral_axis), 
                                        template.flux, 
                                        left=0, 
                                        right=0)
            else:
                log_fe_flux = rebin_spec(log_fe_wl, 
                                         np.log(template.spectral_axis), 
                                         template.flux)

            
            log_fe_spectrum = Spectrum(spectral_axis=log_fe_wl, flux=log_fe_flux, 
                                       flux_error=log_fe_flux)
            self.log_fe.append(log_fe_spectrum)
#            self.interp_fe.append(Spectrum.bin_spectrum(template.spectral_axis,
#                                                              template.flux,
#                                                              data_spectrum.spectral_axis))
            
            if self.fast_interp:
                fe_flux = np.interp(data_spectrum.spectral_axis, 
                                    template.spectral_axis,
                                    template.flux,
                                    left=0, 
                                    right=0)
            else:
                fe_flux = rebin_spec(data_spectrum.spectral_axis, 
                                     template.spectral_axis,
                                     template.flux)

            self.interp_fe.append(fe_flux)
            
            # This gives us the flux of the template at the normalization
            # wavelength associated with the data spectrum. 
            self.interp_fe_norm_flux.append(np.interp(nw,
                                                   template.spectral_axis,
                                                   template.flux,
                                                   left=0,
                                                   right=0))

#-----------------------------------------------------------------------------#

    def ln_priors(self, params):
        """
        Return a list of the ln of all of the priors.

        Args:
            params (): ?

        Returns:
            ln_priors (list): List of the ln of all priors.
        """

        # Need to return parameters as a list in the correct order
        ln_priors = []

        norm = []

        for i in range(1, len(self.fe_templ)+1):
            norm.append(params[f"fe_norm_{i}"])
        
        width = params["fe_width"]

        # Flat prior within the expected ranges.
        for i in range(len(self.fe_templ)):
            if self.norm_min < norm[i] < self.norm_max:
                ln_priors.append(0.0)
            else:
                ln_priors.append(-np.inf)
        if self.width_min < width < self.width_max:
            ln_priors.append(0.)
        else:
            ln_priors.append(-np.inf)

        return ln_priors

#-----------------------------------------------------------------------------#

    def flux(self, spectrum, params):
        """
        Returns the flux for this component for a given wavelength grid
        and parameters.  The parameters should be a list of length 
        (2 x Number of templates)
        
        Args:                                                                          
            spectrum (Spectrum object):                                                
            parameters (): ?
                                                                                       
        Returns:                                                                       
            flux_arrays (): ?                                                          
        """

        # The next two parameters are lists of size len(self.fe_templ)
        norm_wl = spectrum.norm_wavelength
        c_kms = c.to("km/s").value
        log_norm_wl = np.log(norm_wl)
        width = params["fe_width"]
        self.flux_arrays = np.zeros(len(spectrum.spectral_axis)) 
        for i in range(len(self.fe_templ)):	
            norm_i = params[f"fe_norm_{i+1}"]
        
            # Want to smooth and convolve in log space, since 
            # d(log(lambda)) ~ dv/c and we can broaden based on a constant 
            # velocity width. Compare smoothing (v/c) to bin size, and that 
            # tells you how many bins wide your Gaussian to convolve over is
            # sigma_conv is the width to broaden over, as given in Eqn 1 
            # of Vestergaard and Wilkes 2001 
            # (essentially the first line below this)
            # NOTE: log_fe.spectral_axis is in log space, but flux is not!
            sigma_conv = np.sqrt(width**2 - self.templ_width**2) / \
                         (c_kms * 2.*np.sqrt(2.*np.log(2.)))

            bin_size = self.log_fe[i].spectral_axis[2] - self.log_fe[i].spectral_axis[1]
#TODO cross-check with Spectrum ^^
            sigma_norm = np.ceil(sigma_conv / bin_size)
            sigma_size = self.inputpars["fe_kernel_size_sigma"] * sigma_norm
            kernel = signal.gaussian(sigma_size, sigma_norm) / \
                     (np.sqrt(2 * math.pi) * sigma_norm)
            
#            # Convolve flux (in log space) with gaussian broadening kernel
#            # Check to see if length of array is even. 
#            # If it is odd, remove last index
#            # fftwconvolution only works on even size arrays
#            if len(self.log_fe[i].flux) % 2 != 0:
#                self.log_fe[i].flux = self.log_fe[i].flux[:-1]
#                self.log_fe[i].spectral_axis = self.log_fe[i].spectral_axis[:-1]
#            log_conv_fe_flux = fftwconvolve_1d(self.log_fe[i].flux, kernel)
        
            log_conv_fe_flux = np.convolve(self.log_fe[i].flux, kernel,mode="same")
#TODO need to check Spectrum.bin_spectrum()
            # Shift spectrum back into linear space.
            # the left and right statements just set the flux value 
            # to zero if the specified log_norm_wl is outside the 
            #bounds of self.log_fe[i].spectral_axis
#            conv_fe = Spectrum.bin_spectrum(self.log_fe[i].spectral_axis,
#                                                         log_conv_fe_flux,
#                                                         np.log(spectrum.spectral_axis))
            # log(convolute template flux) rebinned onto log space of data spectrum WL.
            if self.fast_interp:
                conv_fe_flux = np.interp(np.log(spectrum.spectral_axis),
                                         self.log_fe[i].spectral_axis,
                                         log_conv_fe_flux,
                                         left=0,
                                         right=0)
            else:
                conv_fe_flux = rebin_spec(np.log(spectrum.spectral_axis),
                                          self.log_fe[i].spectral_axis,
                                          log_conv_fe_flux)
            
            conv_fe_nw = np.median(self.fe_templ[i].spectral_axis)
            conv_fe_norm_flux = np.interp(conv_fe_nw, spectrum.spectral_axis, conv_fe_flux) 
            spectrum_norm_flux = np.interp(conv_fe_nw, spectrum.spectral_axis, spectrum.flux) 

            # Find NaN errors early from dividing by zero.
#TODO check below syntax vv
            conv_fe_norm_flux = np.nan_to_num(conv_fe_norm_flux)

            # Scale normalization parameter to flux in template
            self.flux_arrays += (norm_i / conv_fe_norm_flux) * conv_fe_flux
        return self.flux_arrays
