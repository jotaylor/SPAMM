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
from scipy.signal.signaltools import _next_regular

from .ComponentBase import Component
from ..Spectrum import Spectrum

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from pysynphot import observation
    from pysynphot import spectrum as pysynspec

#NORMALIZATON = 0
#STELLAR_DISPERSION = 1

c_km_per_s = 299792.458 # speed of light in km/s

def fftwconvolve_1d(in1, in2):
    outlen = in1.shape[-1] + in2.shape[-1] - 1 
    origlen = in1.shape[-1]
    n = _next_regular(outlen) 
    tr1 = pyfftw.interfaces.numpy_fft.rfft(in1, n) 
    tr2 = pyfftw.interfaces.numpy_fft.rfft(in2, n) 
    sh = np.broadcast(tr1, tr2).shape 
    dt = np.common_type(tr1, tr2) 
    pr = pyfftw.n_byte_align_empty(sh, 16, dt) 
    np.multiply(tr1, tr2, out=pr) 
    out = pyfftw.interfaces.numpy_fft.irfft(pr, n) 
    index_low = int(outlen/2.)-int(np.floor(origlen/2))
    index_high = int(outlen/2.)+int(np.ceil(origlen/2))
    return out[..., index_low:index_high].copy() 

def find_nearest(input_list,value):
    '''
    Find nearest entry in an array to a specified value.
    list = list of floats
    value = desired value to find closest match to in the array
    return = value closest to input value from input_list
    Ref: http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''
    idx = (np.abs(np.asarray(input_list, dtype = float)-value)).argmin()
    return input_list[idx]

def rebin_spec(wave, specin, wavnew):
    '''
    Rebin spectra to bins used in wavnew.
    Ref: http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/
    '''
    spec = pysynspec.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = pysynspec.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

    return obs.binflux

def runningMeanFast(x, N):
    '''
    x = array of points
    N = window width
    Ref: http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    '''
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


########
## TO DO: 
##
## - Change template loading procedure to a more flexible, class level loading.
#
## - Implement some kind of global variable such that code can find
##   the location of the templates when ran from an arbitrary folder.
#
## - Implement stellar dispersion parameter in templates. For now, the
##   parameter is fit for but does not affect the likelihood function,
##   so output distribution corresponds exactly to prior.
#
########

class HostGalaxyComponent(Component):
    '''
    Host Galaxy Component
    \f$ F_{\lambda,\rm Host}\ =\ \sum_{1}^{N} F_{\rm Host,i} HostTempl_{\lambda,i}(\sig_*) \f$
    This component has N templates and N+1 parameters. 

    normalization: \f$ F_{\rm Host,i} \f$ for each of the N templates.

    stellar line dispersion: \f$ \sig_* \f$

    '''

    #This dictionary will eventually hold the templates. Implement lazy loading. 
   # _templates = None

    def __init__(self):
        super(HostGalaxyComponent, self).__init__()

        self._templates = None
        self.interpolated_templates = None # interpolated to data provided
        self.name = "HostGalaxy"


        #self.template_wave, self.template_flux, self.n_templates = self._load_host_templates()
#       self.template_flux_model_grid = None
        self.interpolated_normalization_flux = None
#        for i in range(self.n_templates):
#            self.model_parameter_names.append("normalization_host_template_{0:04d}".format(i))

#        self.model_parameter_names.append("normalization") # type np.array
#        self.model_parameter_names.append("stellar_dispersion")

        self._flux_arrays = None # defined in initialize()
        self._norm_wavelength = None

        self.norm_min = None # np.array([None for i in range(self.n_templates)])
        self.norm_max = None # np.array([None for i in range(self.n_templates)])

        self.stellar_dispersion_min = None
        self.stellar_dispersion_max = None

    @property
    def templates(self):
        if self._templates == None:
            self._load_host_templates()
        return self._templates

    @property
    def model_parameter_names(self):
        '''
        Returns a list of model parameter names.
        Since the number of parameters depends on the number of templates (only
        known at run time), this must be provided by a method.

        The parameters are normalization, one for each template, followed by stellar dispersion.
        '''
        parameter_names = list()
        for i in range(1, len(self.templates)+1):
            parameter_names.append("normalization_{0}".format(i))
        parameter_names.append("stellar dispersion")
        return parameter_names

    @property
    def is_analytic(self):
        return False


    def _load_host_templates(self, template_set=None):

        # determine the file name
        if template_set is None:
            template_set_file_name = "../Data/HostModels/HostGalaxy_Kevin/trueascii.lst"
            #template_set_file_name = "../Templates/Host_templates/default_list_of_templates.txt"
        else:
            raise Exception("Host galaxy template set '{0}' not found.".format(template_set))
            #print template_set,"is not available"
            #sys.exit()
        
        # get the list of filenames of the templates
        template_filenames = list()
        with open(template_set_file_name) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                else:
                    template_filenames.append(line.rstrip("\n"))
        
        
        # read in all of the templates
        self._templates = list()

        for template_filename in template_filenames:
            with open(template_filename) as template_file:
                wavelengths, flux = np.loadtxt(template_filename, unpack=True)
                template = Spectrum(flux)
                template.wavelengths = wavelengths
                self._templates.append(template)

    def initial_values(self, spectrum=None):
        '''

        Needs to sample from prior distribution.
        Return type must be a single list (not an np.array).
        '''

        boxcar_width = 5 # width of smoothing function

        self.flux_max = max(runningMeanFast(spectrum.flux, boxcar_width))

        self.norm_min = np.zeros(len(self.templates))
        self.norm_max = np.zeros(len(self.templates)) + self.flux_max

        # the size parameter will force the result to be a numpy array - not the case
        # if the inputs are single-valued (even if in the form of an array)
        norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max*0.1, size=self.norm_min.size)

        self.stellar_dispersion_min = 30.0
        self.stellar_dispersion_max = 1000.0
        stellar_dispersion_init = np.random.uniform(low=self.stellar_dispersion_min, high=self.stellar_dispersion_max)

        return norm_init.tolist() + [stellar_dispersion_init]

    def normalization_wavelength(self, data_spectrum_wavelength=None):
        '''
        Returns a single value.
        '''
        if self._norm_wavelength is None:
            if data_spectrum_wavelength is None:
                raise Exception("The wavelength array of the data spectrum must be specified.")
            self._norm_wavelength = np.median(data_spectrum_wavelength)
        return self._norm_wavelength

    def initialize(self, data_spectrum=None):
        '''
        Perform any initializations using data spectrum.
        '''
        if data_spectrum is None:
            raise Exception("The data spectrum must be specified to initialize" + 
                            "{0}.".format(self.__class__.__name__))

        self._flux_arrays = np.zeros(len(data_spectrum.wavelengths)) # calculate flux on this array

        self.interpolated_templates = list()
        self.interpolated_normalization_flux = list()
        
        # We'll eventually need to convolve these in constant velocity space, so rebin to equal log bins
        self.rebin_log_templates = list()
        self.interpolated_templates_logspace_rebin = list()

        fnw = self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths) # flux at normalization wavelength

        #for template in self.templates:
        #    f = scipy.interpolate.interp1d(template.wavelengths, template.flux) # returns function
        #    #self.interpolated_templates.append(f(data_spectrum.wavelengths))
        #    template_rebin_fluxes = rebin_spec(template.wavelengths, template.flux, data_spectrum.wavelengths) # do the rebinning
        #    self.interpolated_templates.append(template_rebin_fluxes)
        #    
        #    ## fnw = flux at normalized wavelength##
        #    #if self._norm_wavelength is None:
        #    #    self._norm_wavelength = np.median(data_spectrum.wavelengths)
        #    #fnw = self._norm_wavelength#
        #    fnw=self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths)
        #    self.interpolated_normalization_flux.append(f(fnw))
            
        for i,template in enumerate(self.templates):
            # This method lets you interpolate beyond the wavelength coverage of the template if/when the data covers beyond it.  
            # Function returns 0 outside the wavelength coverage of the template.
            # To broaden in constant velocity space, you need to rebin the templates to be in equal bins in log(lambda) space.
            equal_log_bins = np.linspace(min(np.log(template.wavelengths)), max(np.log(template.wavelengths)), num = len(template.wavelengths))
            template_fluxes_rebin_equal_log_fluxes = rebin_spec(np.log(template.wavelengths), template.flux, equal_log_bins) # do the rebinning

            
            rebinwavelengths,rebinflux = equal_log_bins, template_fluxes_rebin_equal_log_fluxes
            template_equal_log_rebin_spec = Spectrum(rebinflux)
            template_equal_log_rebin_spec.wavelengths=rebinwavelengths
            self.rebin_log_templates.append(template_equal_log_rebin_spec)


            self.interpolated_templates.append(rebin_spec(template.wavelengths, template.flux, data_spectrum.wavelengths))
            self.interpolated_templates_logspace_rebin.append(rebin_spec(equal_log_bins, template_fluxes_rebin_equal_log_fluxes,np.log(data_spectrum.wavelengths)))
            self.interpolated_normalization_flux.append(np.interp(fnw,template.wavelengths, template.flux,left=0,right=0))


    @property
    def native_wavelength_grid(self):### do we need this (I assume templates may have different spacing)
        for template in self.templates:
            template1grid = template.wavelengths
        return template1grid
        #assert False, "finish this code"
    
    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.
        
        @param params
        '''
        
        # need to return parameters as a list in the correct order
        ln_priors = list()
        
        normalization = list()
        for i in range(1, len(self.templates)+1):
            normalization.append(params[self.parameter_index("normalization_{0}".format(i))])
        
        stellar_dispersion = params[self.parameter_index("stellar dispersion")]
        # Normalization parameter

        # Flat prior within the expected ranges.
#        ln_prior_norms = np.zeros(len(self.templates))
#        for i in range(len(self.templates)):
#            if self.norm_min[i] < normalization[i] < self.norm_max[i]:
#                ln_prior_norms[i] = 0.0
#            else:
#                ln_prior_norms[i] = -np.inf
#
#        # Stellar dispersion parameter
#        if self.stellar_dispersion_min < stellar_dispersion < self.stellar_dispersion_max:
#            ln_prior_stellar_dispersion = 0.0
#        else:
#            ln_prior_stellar_dispersion = -np.inf
#        
#        # ln_prior_norms is an array, need to return a 1D array of parameters to emcee
#        return ln_prior_norms.tolist() + [ln_prior_stellar_dispersion]
        
        # Flat prior within the expected ranges.
        for i in range(len(self.templates)):
            if self.norm_min[i] < normalization[i] < self.norm_max[i]:
                ln_priors.append(0.0)
            else:
                ln_priors.append(-np.inf)
        #print('norm',np.sum(normalization),self.norm_max[0],self.norm_min[0],normalization)
        #exit()
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

    @property
    def parameter_count(self):
        ''' Returns the number of parameters of this component. '''
        no_parameters = len(self.templates) + 1
        if self.z:
            return no_parameters + 1
        else:
            return no_parameters


    def flux(self, spectrum=None, parameters=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        
        normalization = list()
        for i in range(1, len(self.templates)+1):
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
                
#        #Create the dispersion-convolution matrix.
#        Kmat = self.stellar_dispersion_matrix(stellar_dispersion,spectrum)
#        #Kmat = np.identity(len(spectrum.wavelengths))
#        #flux = np.zeros(wavelengths.shape)
#        #print "******* {0}".format(parameters)
#        norm = list() # parameter normalization
#        for i in range(len(self.templates)):
#            norm.append(parameters_host[i] / self.interpolated_normalization_flux[i]) # * spectrum.flux_at_normalization_wavelength())
##        norm = parameters[0:-1] / self.interpolated_normalization_flux
#        self._flux_arrays = 0.0
#        for i in range(len(self.templates)):
#            convolved_template = Kmat.dot(self.interpolated_templates[i])
#            self._flux_arrays += norm[i] * convolved_template
#            #self._flux_arrays += norm[i] * self.interpolated_templates[i]

        norm = list()
        interpolated_convolved_templates = list()
        # The next two parameters are lists of size len(self.templates)
        norm_waves = self.normalization_wavelength(data_spectrum_wavelength=spectrum.wavelengths)
        log_norm_waves = np.log(norm_waves)
        self._flux_arrays[:] = 0.0
        sd_over_c = stellar_dispersion/(c_km_per_s)
            
        for i in range(len(self.templates)):
            # Want to smooth and convolve in log space, since d(log(lambda)) ~ dv/c and we can broaden based on a constant velocity width
            # Compare smoothing (v/c) to bin size, and that tells you how many bins wide your Gaussian to convolve over is
            # sigma_conv is the width to broaden over, as given in Eqn 1 of Vestergaard and Wilkes 2001 (essentially the first line below this)
            sigma_conv = sd_over_c
            equal_log_bin_size = self.rebin_log_templates[i].wavelengths[2] - self.rebin_log_templates[i].wavelengths[1]
            sig_norm = sigma_conv/equal_log_bin_size
            kernel = signal.gaussian(1000,sig_norm)/(np.sqrt(2*math.pi)*sig_norm)
            if np.size(self.rebin_log_templates[i].flux)%2 > 0:
                self.rebin_log_templates[i].flux = self.rebin_log_templates[i].flux[:-1]
                self.rebin_log_templates[i].wavelengths = self.rebin_log_templates[i].wavelengths[:-1]
            fftwconvolved = fftwconvolve_1d(self.rebin_log_templates[i].flux, kernel)
            interpolated_template_convolved = np.interp(np.log(spectrum.wavelengths),self.rebin_log_templates[i].wavelengths,	\
            fftwconvolved,left=0,right=0)
            interpolated_template_convolved_normalization_flux = np.interp(log_norm_waves,self.rebin_log_templates[i].wavelengths,	\
            fftwconvolved,left=0,right=0) # since in log space, need log_norm_waves here!
            # Find NaN errors early from dividing by zero.
            assert interpolated_template_convolved_normalization_flux != 0., "Interpolated convolution flux valued at 0 at the location of peak template flux!"
            interpolated_convolved_templates.append(interpolated_template_convolved)
            norm.append(parameters[i] / interpolated_template_convolved_normalization_flux) # Scale normalization parameter to flux in template
            self._flux_arrays += norm[i] * interpolated_convolved_templates[i]

        return self._flux_arrays

#    def stellar_dispersion_matrix(self, stellar_dispersion, spectrum=None):
#
#        Kmat = np.zeros((len(spectrum.wavelengths),len(spectrum.wavelengths)))
#        lam = spectrum.wavelengths
#        for k,lamk in enumerate(spectrum.wavelengths):
#            sig = stellar_dispersion * lamk/3.e5 #Assume the dispersion is provided in km/s.
#
#            #To speed things up, we'll only consider bins with central
#            #wavelengths within 5 sigma of the current spectral bin.
#
#            #Get the bin indices that are closest to +/- 20 sigma.
#            lmin = np.argmin(abs((lamk-lam)/sig - 3.))
#            lmax = np.argmin(abs((lamk-lam)/sig + 3.))
#
#            #See if we are near the bounds and determine
#            #the kernel normalization accordingly.
#            if lmin>0 and lmax<len(lam):
#                norm = sig*(2.*np.pi)**0.5
#            else:
#                if lmin==0:
#                    a = lam[lmin]-0.5*(lam[lmin+1]-lam[lmin])
#                    b = lam[lmax]+0.5*(lam[lmax+1]-lam[lmax])
#                else:
#                    a = lam[lmin]-0.5*(lam[lmin]-lam[lmin-1])
#                    b = lam[lmax]+0.5*(lam[lmax]-lam[lmax-1])
#                norm = scipy.integrate.quad(self.gaussian_kernel,a,b,args=(lamk,sig))[0]
#
#            for l in range(lmin,lmax+1):
#                if l==0:
#                    a = lam[l]-0.5*(lam[l+1]-lam[l])
#                    b = lam[l]+0.5*(lam[l+1]-lam[l])
#                elif l==len(lam)-1:
#                    a = lam[l]-0.5*(lam[l]-lam[l-1])
#                    b = lam[l]+0.5*(lam[l]-lam[l-1])
#                else:
#                    a = lam[l]-0.5*(lam[l]-lam[l-1])
#                    b = lam[l]+0.5*(lam[l+1]-lam[l])
#                Kmat[k,l] = scipy.integrate.quad(self.gaussian_kernel,a,b,args=(lamk,sig))[0]/norm
#
#        return Kmat

    def gaussian_kernel(self,x,mu,sig):
        return np.exp(-0.5*((x-mu)/sig)**2)

