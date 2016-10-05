#!/usr/bin/python

import re
import sys
import numpy as np
import scipy.interpolate
import numpy as np
import scipy.integrate

from .ComponentBase import Component
from ..Spectrum import Spectrum

#NORMALIZATON = 0
#STELLAR_DISPERSION = 1

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
	_templates = None

	def __init__(self):
		super(HostGalaxyComponent, self).__init__()

		self._templates = None
		self.interpolated_templates = None # interpolated to data provided
		self.name = "HostGalaxy"
		
		#self.template_wave, self.template_flux, self.n_templates = self._load_host_templates()
#		self.template_flux_model_grid = None
		self.interpolated_normalization_flux = None
#		for i in range(self.n_templates):
#			self.model_parameter_names.append("normalization_host_template_{0:04d}".format(i))

#		self.model_parameter_names.append("normalization") # type np.array
#		self.model_parameter_names.append("stellar_dispersion")
		
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
			template_set_file_name = "../Host_templates/default_list_of_templates.txt"
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
				template = Spectrum()
				template.wavelengths, template.flux = np.loadtxt(template_filename, unpack=True)
				self._templates.append(template)

# 		try:
# 			template_filenames_file = open(template_set_file_name,"r")
# 		except IOError:
# 			print "Cannot open file '{0}'.\n".format(template_filenames_file)
# 			sys.exit()
# 		
# 		self._templates = list()
# 		for template_filename in template_filenames_file:
# 			if template_filename.startswith("#"):
# 				continue
# 			else:
# 				template_filename = template_filename.rstrip("\n")
# 			with open(template_filename) as template_file:
# 				for line in template_file:
# 					# skip comments
# 					if line.startswith("\#"):
# 						continue
# 					else:
# 						filename = line
# 		
# 					template = Spectrum()
# 					template.wavelengths, template.flux = np.loadtxt(filename, unpack=True)
# 		
# 					self._templates.append(template)

	def initial_values(self, spectrum=None):
		'''
	
		Needs to sample from prior distribution.
		Return type must be a single list (not an np.array).
		'''

		boxcar_width = 5 # width of smoothing function

		flux_max = max(runningMeanFast(spectrum.flux, boxcar_width))

		self.norm_min = np.zeros(len(self.templates))
		self.norm_max = np.zeros(len(self.templates)) + flux_max
		
		# the size parameter will force the result to be a numpy array - not the case
		# if the inputs are single-valued (even if in the form of an array)
		norm_init = np.random.uniform(low=self.norm_min, high=self.norm_max, size=self.norm_min.size)

		self.stellar_dispersion_min = 30.0
		self.stellar_dispersion_max = 600.0
		stellar_dispersion_init = np.random.uniform(low=self.stellar_dispersion_min, high=self.stellar_dispersion_max)

		return norm_init.tolist() + [stellar_dispersion_init]
       
#		host_params_init = list()
#        if isinstance(norm_init,float):
#           host_params_init.append(norm_init)
#       else:
#            host_params_init.extend(norm_init)
#		host_params_init.append(stellar_dispersion_init)
#		
#		return host_params_init


	def initialize(self, data_spectrum=None):
		'''
		Perform any initializations where the data is optional.
		'''
		if data_spectrum is None:
			raise Exception("The data spectrum must be specified to initialize" + 
					"{0}.".format(self.__class__.__name__))

		self._flux_arrays = np.zeros(len(data_spectrum.wavelengths)) # calculate flux on this array

		self.interpolated_templates = list()
		self.interpolated_normalization_flux = list()
		
		for template in self.templates:
			f = scipy.interpolate.interp1d(template.wavelengths, template.flux) # returns function
			self.interpolated_templates.append(f(data_spectrum.wavelengths))
			# fnw = flux at normalized wavelength
			fnw = self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths)
			self.interpolated_normalization_flux.append(f(fnw))
		
#         #Check if we have created the interpolated versions of
#         #the templates in the model grid. If not, do it. 
#         if self.template_flux_model_grid is None:
# 	        self.template_flux_model_grid = list()
# 	        self.template_flux_at_norm_wavelength = np.zeros(self.n_templates)
# 	        
# 	        for i in range(self.n_templates):
# 	            f = scipy.interpolate.interp1d(self.template_wave[i], self.template_flux[i]) # returns function
# 	            self.template_flux_model_grid.append(f(wavelengths))
# 	            self.template_flux_at_norm_wavelength[i] = f(self.normalization_wavelength())
# 	        
# 	        self.template_flux_model_grid = np.array(self.template_flux_model_grid)


	def native_wavelength_grid(self):
		assert False, "finish this code"
	
	def normalization_wavelength(self, data_spectrum_wavelength=None):
		'''
		Returns a single value.
		'''
		if self._norm_wavelength is None:
			if data_spectrum_wavelength is None:
				raise Exception("The wavelength array of the data spectrum must be specified.")
			self._norm_wavelength = np.median(data_spectrum_wavelength)
		return self._norm_wavelength

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
		params[self.parameter_index("stellar dispersion")]

		stellar_dispersion = params[self.parameter_index("stellar dispersion")]
		parameters = normalisation
		parameters.append(stellar_dispersion)
		# Normalization parameter

		# Flat prior within the expected ranges.
		ln_prior_norms = np.zeros(len(self.templates))
		for i in range(len(self.templates)):
			if self.norm_min[i] < normalization[i] < self.norm_max[i]:
				ln_prior_norms[i] = 0.0
			else:
				ln_prior_norms[i] = -np.inf

		# Stellar dispersion parameter
		if self.stellar_dispersion_min < stellar_dispersion < self.stellar_dispersion_max:
			ln_prior_stellar_dispersion = 0.0
		else:
			ln_prior_stellar_dispersion = -np.inf
		
		# ln_prior_norms is an array, need to return a 1D array of parameters to emcee
		return ln_prior_norms.tolist() + [ln_prior_stellar_dispersion]

	@property
	def parameter_count(self):
		''' Returns the number of parameters of this component. '''
		no_parameters = len(self.templates) + 1
		if self.z:
			return no_parameters + 1
		else:
			return no_parameters


	def flux(self, spectrum=None, params=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. Will use the initial parameters if none are specified.
		'''
		
		normalization = list()
		for i in range(1, len(self.templates)+1):
			normalization.append(params[self.parameter_index("normalization_{0}".format(i))])
		params[self.parameter_index("stellar dispersion")]

		stellar_dispersion = params[self.parameter_index("stellar dispersion")]
		parameters = normalisation
		parameters.append(stellar_dispersion)

		assert len(parameters) == self.parameter_count, \
				"The wrong number of indices were provided: {0}".format(parameters)

                #Convolve to increase the velocity dispersion. Need to
                #consider it as an excess dispersion above that which
                #is intrinsic to the template. For the moment, the
                #implicit assumption is that each template has an
                #intrinsic velocity dispersion = 0 km/s.
		stellar_dispersion = parameters[-1]
		#Create the dispersion-convolution matrix.
		#Kmat = self.stellar_dispersion_matrix(stellar_dispersion,spectrum)
		Kmat = np.identity(len(spectrum.wavelengths))

		#flux = np.zeros(wavelengths.shape)
		#print "******* {0}".format(parameters)
		norm = list() # parameter normalization
		for i in range(len(self.templates)):
			norm.append(parameters[i] / self.interpolated_normalization_flux[i]) # * spectrum.flux_at_normalization_wavelength())
#		norm = parameters[0:-1] / self.interpolated_normalization_flux
		self._flux_arrays[:] = 0.0
		for i in range(len(self.templates)):
			convolved_template = Kmat.dot(self.interpolated_templates[i])
			self._flux_arrays += norm[i] * convolved_template
			#self._flux_arrays += norm[i] * self.interpolated_templates[i]

		return self._flux_arrays

        def stellar_dispersion_matrix(self, stellar_dispersion, spectrum=None):

		Kmat = np.zeros((len(spectrum.wavelengths),len(spectrum.wavelengths)))
		lam = spectrum.wavelengths
		for k,lamk in enumerate(spectrum.wavelengths):
			sig = stellar_dispersion * lamk/3.e5 #Assume the dispersion is provided in km/s.

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
				norm = scipy.integrate.quad(self.gaussian_kernel,a,b,args=(lamk,sig))[0]

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
				Kmat[k,l] = scipy.integrate.quad(self.gaussian_kernel,a,b,args=(lamk,sig))[0]/norm
                
		return Kmat

	def gaussian_kernel(self,x,mu,sig):
		return np.exp(-0.5*((x-mu)/sig)**2)

