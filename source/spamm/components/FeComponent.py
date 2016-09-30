#!/usr/bin/python

import re
import sys
import numpy as np
import pyfftw 
from scipy import signal
from astropy.convolution import Gaussian1DKernel, convolve
import warnings

# Suppress warnings about not having templates for pysynphot
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=UserWarning)
	from pysynphot import observation
	from pysynphot import spectrum

from .ComponentBase import Component
from ..Spectrum import Spectrum

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

def runningMeanFast(x, N):
	'''
	x = array of points
	N = window width
	Ref: http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
	'''
	return np.convolve(x, np.ones((N,))/N)[(N-1):]
	
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
	spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
	f = np.ones(len(wave))
	filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
	obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

	return obs.binflux	
	
########
## TO DO: 
##
## - Change template loading procedure to a more flexible, class level loading.
#
## - Implement some kind of global variable such that code can find
##   the location of the templates when ran from an arbitrary folder.
##	-As of now, assumes the file that gives the templates also gives their widths in km/s (read in 
##		via a basic numpy.loadtxt call, so separated by space)
#
## - Should there be a check that there is no overlap in the wavelength coverage of the templates?
## - Ask which parameters should be returned.  As of now, get FWHM of the iron lines and their flux at the norm wavelength
## - Able to do Lorentzian broadening instead?
## - Different method of rebinning?
########


class FeComponent(Component):
	'''
	Fe II and III pseudo-continuum of blended emission lines.
	
	We will use a linear combination of N broadened and scaled iron templates:
	$F_{\lambda,\,{\rm Fe}} = \sum_{i=1}^N F_{{\rm Fe},\,0,\,i} FeTemplate_{\lambda,\,i}(\sigma_i)$
	where $FeTemplate_{\lambda,\,i}$ is iron template $i$ at wavelength $\lambda$,
	$F_{{\rm Fe},\,0,\,i}$ is the template normalization, and $\simga_i$ is the width
	of the broadening kernal.
	
	This component has 2 kinds of parameters:
	
	parameter1 : template normalization for each template
	parameter2 : FWHM of iron lines being tested for the templates (in units of km/s)
	
	
	Parameters by number
	0-(i-1): normalizations of templates 1-i
	i-(2i-1): FWHM of the lines in templates 1 - i
	
	'''

	def __init__(self):
		super(FeComponent, self).__init__()

		self._templates = None
		self.rebin_log_templates = None # storage for templates where wavelengths are rebinned to be equally distributed in ln() space
		self.interpolated_templates = None # interpolated to data provided (normal spacing)
		self.interpolated_templates_logspace_rebin = None # broaden in d(log(lambda)) space to do constant velocity broadening
		self.interpolated_normalization_flux = None # flux at normalization wavelength

		# Add the Fe-template specific parameters
		self._kernel_type = "Gaussian"			# to be set to either Gaussian or Lorentzian once things kick up; not used for now
		
		self._flux_arrays = None # defined in initialize()
		self._norm_wavelength = None # this is now going to a be a list, giving the location of max flux (in wavelength) for each template.

		# Set boundaries for priors on all params to fit for
		self.norm_min = None # units will be the same as the flux for the data spectrum
		self.norm_max = None 

		# Widths for Gaussian broadening kernel, units will be in km/s
		self.fwhm_min = None # units will be in km/s
		self.fwhm_max = None
		
		self._template_inherent_widths = None # will prevent us from trying to fit widths smaller than the template's
	
	@property
	def templates(self):
		if self._templates == None:	# if no templates are present, load them
			self._load_fe_templates()
		return self._templates

	@property
	def model_parameter_names(self):
		'''
		Returns a list of model parameter names.
		Since the number of parameters depends on the number of templates (only
		known at run time), this must be provided by a method.
		
		The parameters are normalization, one for each template, followed by FWHM of each template.
		'''
		parameter_names = list()
		for i in range(1, len(self.templates)+1):
			parameter_names.append("iron_normalization_template_{0}".format(i))
		for i in range(1, len(self.templates)+1):
			parameter_names.append("iron_FWHM_template_{0}".format(i))
		return parameter_names

	@property
	def is_analytic(self):
		return False

	def _load_fe_templates(self, template_set=None):

		# Determine the file name
		# File should be formatted such that each line is /path/to/template.file template_width(in km/s)
		if template_set is None:
			template_set_file_name = "../Fe_templates/default_list_of_templates.txt"
		else:
			raise Exception("Fe galaxy template set '{0}' not found.".format(template_set))
			
		# Get the list of filenames of the templates
		template_filenames = list()
		with open(template_set_file_name) as file:
			for line in file:
				if line.startswith("#"):
					continue
				else:
					template_filenames.append(line.rstrip("\n"))
		

		# Read in all of the templates
		self._templates = list()
		self._template_inherent_widths = list()
		
		for line in template_filenames:
			template = Spectrum()
			try:
				template_filename, template_inherent_width = line.split()[0], line.split()[1] # split on space to get inherent width as well
			except:
				print("Template list should be formatted in two columns, the first being the path to the template and the second being the width of the template in km/s!  Exiting now.\n")
				sys.exit()
			template.wavelengths, template.flux = np.loadtxt(template_filename, unpack=True)
			self._templates.append(template)
			self._template_inherent_widths.append(float(template_inherent_width)) 
			print("\n\nAssuming width of " + str(template_inherent_width) + " km/s for template " + str(template_filename) + ".\n") # sanity check
	
	def initial_values(self, spectrum=None):
		'''
	
		Needs to sample from prior distribution.
		Return type must be a single list (not an np.array).
		These are the first guess for the parameters to be fit for in emcee.
		In the case of the Fe Component, this would be the normalization and, for now, FWHM of iron lines.
		Note that the returns will be a list, the first 'i' elements of which are the normalizations and the 
		second 'i' elements of which are the FWHM for a loaded number of 'i' templates.
		'''

		boxcar_width = 10 # width of smoothing function
		flux_max = 10.*max(runningMeanFast(spectrum.flux, boxcar_width)) # does this need to be increased?
		
		# Set normalization to go from zero (the template isn't present) to the max flux found in the running mean of boxcar_width
		self.norm_min = np.zeros(len(self.templates))
		self.norm_max = np.zeros(len(self.templates)) + flux_max
		norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max, size=self.norm_min.size)
		
		# Based on priors stated in compon_descr_template.pdf, log-normal prior in range 500-20000 km/s
		# UPDATE: Changed to min at the width of the template rather than 500 km/s
		self.fwhm_min = np.ones(len(self.templates)) * min(self._template_inherent_widths)	# units of km/s
		self.fwhm_max = np.ones(len(self.templates)) * 20000.
		fwhm_init_log10 = np.random.uniform(low=np.log10(self.fwhm_min), high=np.log10(self.fwhm_max), size=len(self.fwhm_min)) # uniformly sample log space between limits
		fwhm_init = np.power(10., fwhm_init_log10) # put back in linear space
		
		return norm_init.tolist() + fwhm_init.tolist()	# should be list of length (number of parameters * len(self.templates))
		
	def initialize(self, data_spectrum=None):
		'''
		Perform all necessary initializations for the iron component, such as reading in teh templates, 
		rebinning them, and interpolating them on the grid scale of the data spectrum.
		'''
		if data_spectrum is None:
			raise Exception("The data spectrum must be specified to initialize" + 
					"{0}.".format(self.__class__.__name__))

		self._flux_arrays = np.zeros(len(data_spectrum.wavelengths)) # calculate flux on this array, initialized to be zero (filled in later with .flux method)

		# Initialize as blank lists
		self.interpolated_templates = list()
		self.interpolated_normalization_flux = list()
		
		# We'll eventually need to convolve these in constant velocity space, so rebin to equal log bins
		self.rebin_log_templates = list()
		self.interpolated_templates_logspace_rebin = list()
		
		fnw = self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths) # flux at normalization wavelength
		
		
		for i,template in enumerate(self.templates):
			# This method lets you interpolate beyond the wavelength coverage of the template if/when the data covers beyond it.  
			# Function returns 0 outside the wavelength coverage of the template.
			# To broaden in constant velocity space, you need to rebin the templates to be in equal bins in log(lambda) space.
			equal_log_bins = np.linspace(min(np.log(template.wavelengths)), max(np.log(template.wavelengths)), num = len(template.wavelengths))
			template_fluxes_rebin_equal_log_fluxes = rebin_spec(np.log(template.wavelengths), template.flux, equal_log_bins) # do the rebinning
			
			template_equal_log_rebin_spec = Spectrum()
			template_equal_log_rebin_spec.wavelengths, template_equal_log_rebin_spec.flux = equal_log_bins, template_fluxes_rebin_equal_log_fluxes
			self.rebin_log_templates.append(template_equal_log_rebin_spec)
			
		
			self.interpolated_templates.append(np.interp(data_spectrum.wavelengths,template.wavelengths, template.flux,left=0,right=0))
			self.interpolated_templates_logspace_rebin.append(np.interp(np.log(data_spectrum.wavelengths),equal_log_bins, template_fluxes_rebin_equal_log_fluxes,left=0,right=0))
			self.interpolated_normalization_flux.append(np.interp(fnw[i],template.wavelengths, template.flux,left=0,right=0))
		
		
	def native_wavelength_grid(self):
		'''
		For now, this stitches the wavelengths of all templates together.  Should be fixed later on to handle cases where template wavelengths overlap?
		'''
		wavelengths = list()
		for template in self.templates:
			wavelengths += list(template.wavelengths)
		return wavelengths
		
			
	def normalization_wavelength(self, data_spectrum_wavelength=None):
		'''
		Returns a list of wavelengths, the wavelength in the data spectrum closest to the maximal flux wavelength in the template spectra.
		Should this be changed back to median wavelength of the templates?
		'''
		if self._norm_wavelength is None:
			if data_spectrum_wavelength is None:
				raise Exception("The wavelength array of the data spectrum must be specified.")
			norm_wavelengths_set = list()
			for template in self.templates:
				template_max_flux_wavelength = template.wavelengths[np.argmax(np.asarray(template.flux))] # find wavelength corresponding to max flux of template
				nearest_data_wavelength = find_nearest(data_spectrum_wavelength, template_max_flux_wavelength) # find closest data wavelength
				assert abs(template_max_flux_wavelength - nearest_data_wavelength) < 100., "Check overlap of spectrum and template wavelengths!"
				norm_wavelengths_set.append(nearest_data_wavelength)
			self._norm_wavelength = norm_wavelengths_set
		return self._norm_wavelength
		
	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		Parameters by number 
		0 - (i - 1):  normalizations of the "i" templates
		i - (2i - 1): FWHM of the lines in the "i" templates	
				
		'''

		# Need to return parameters as a list in the correct order
		ln_priors = list()
		
		# Get each parameter
		normalizations = params[0:len(self.templates)]	# first (number of templates) parameters are normalizations
		fe_fwhm = params[len(self.templates):] # second (number of templates) parameters are dispersions
		
		# Sanity checks
		assert len(normalizations) == len(fe_fwhm), "Different number of normalizations and dispersions!"
		assert len(normalizations) == len(self.templates), "Different number of normalizations and templates!"
		
		# Normalization parameter
		# Flat prior within the expected ranges.  Should it be less than infinity?
		# Width parameter
		ln_prior_norms = np.zeros(len(self.templates))
		ln_prior_fwhm =  np.zeros(len(self.templates))
		
		for i in range(len(self.templates)):
			if self.norm_min[i] < normalizations[i] < self.norm_max[i]:
				ln_prior_norms[i] = 0
			else:
				ln_prior_norms[i] = -np.inf
			if self.fwhm_min[i] < fe_fwhm[i] < self.fwhm_max[i]:
				ln_prior_fwhm[i] = 0
			else:
				ln_prior_fwhm[i] = -np.inf
		return ln_prior_norms.tolist() + ln_prior_fwhm.tolist()

	@property
	def parameter_count(self):
		''' Returns the number of parameters of this component. '''
		no_parameters = len(self.templates)*2	# the template normalization and broadening (for each template)
		if self.z:
			return no_parameters + 1
		else:
			return no_parameters


	def flux(self, spectrum=None, parameters=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters.  The parameters should be a list of length (2 x Number of templates)
		'''

		assert len(parameters) == self.parameter_count, \
				"The wrong number of indices were provided: {0}".format(parameters)
		norm = list() # parameter normalization
		interpolated_convolved_templates = list()
		# The next two parameters are lists of size len(self.templates)
		norm_waves = self.normalization_wavelength(data_spectrum_wavelength=spectrum.wavelengths)
		log_norm_waves = np.log(norm_waves)
		self._flux_arrays[:] = 0.0
		
		for i in range(len(self.templates)):	
			# Parameter len(self.templates) + i gives Gaussian width of that template in this run
			fwhm = parameters[i + len(self.templates)]
			fwhm_over_c = fwhm/(c_km_per_s)
			if fwhm < (self._template_inherent_widths[i]):
				# Arbitrarily large flux, model will not be a good fit if narrower than template width.  Preferably infinity, but that doesn't play nice with the Model.likelihood method
				interpolated_convolved_templates.append((np.ones(len(spectrum.wavelengths), dtype = float) * 1.e50))
				norm.append(1.)
			else:
				# Want to smooth and convolve in log space, since d(log(lambda)) ~ dv/c and we can broaden based on a constant velocity width
				# Compare smoothing (v/c) to bin size, and that tells you how many bins wide your Gaussian to convolve over is
				# sigma_conv is the width to broaden over, as given in Eqn 1 of Vestergaard and Wilkes 2001 (essentially the first line below this)
				sigma_conv = np.sqrt(np.square(fwhm_over_c) - np.square(self._template_inherent_widths[i]/c_km_per_s))/(2.*np.sqrt(2.*np.log(2.)))
				equal_log_bin_size = self.rebin_log_templates[i].wavelengths[2] - self.rebin_log_templates[i].wavelengths[1]
				sig_norm = sigma_conv/equal_log_bin_size
				kernel = signal.gaussian(1000,sig_norm)/(np.sqrt(2*math.pi)*sig_norm)
				fftwconvolved = fftwconvolve_1d(self.rebin_log_templates[i].flux, kernel)
				interpolated_template_convolved = np.interp(np.log(spectrum_real.wavelengths),self.rebin_log_templates[i].wavelengths,	\
				fftwconvolved,left=0,right=0)
				interpolated_template_convolved_normalization_flux = np.interp(log_norm_waves[i],rebin_log_template.wavelengths,	\
				fftwconvolved,left=0,right=0) # since in log space, need log_norm_waves here!
				# Find NaN errors early from dividing by zero.
				assert interpolated_template_convolved_normalization_flux != 0., "Interpolated convolution flux valued at 0 at the location of peak template flux!"
				interpolated_convolved_templates.append(interpolated_template_convolved)
				norm.append(parameters[i] / interpolated_template_convolved_normalization_flux) # Scale normalization parameter to flux in template
			self._flux_arrays += norm[i] * interpolated_convolved_templates[i]
		return self._flux_arrays
