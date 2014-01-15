#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
import re
import scipy.interpolate


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
		
                self.template_wave, self.template_flux, self.n_templates = self.load_host_templates()
                self.template_flux_model_grid = None
                self.template_flux_at_norm_wavelength = None

		for i in range(self.n_templates):
			self.model_parameter_names.append(
				"normalization_host_template_{0:04d}".format(i))
		self.model_parameter_names.append("stellar_dispersion")
			
		self._norm_wavelength = None

		self.norm_min = np.array([None for i in range(self.n_templates)])
		self.norm_max = np.array([None for i in range(self.n_templates)])
        
		self.stellar_dispersion_min = None
		self.stellar_dispersion_max = None
        

        def load_host_templates(self,template_set="default"):

                template_set_file_name = ""
                if template_set == "default":
                        template_set_file_name = "../Host_templates/default_list_of_templates.txt"
                else:
                        #Crash. Not sure what is the best pythonic way
                        #to do it, so this may need to be
                        #changed.
                        print "Incorrect set of host galaxy templates."
                        print template_set,"is not available"
                        sys.exit()

                twave = list()
                tflux = list()
                ntemp = 0
                try:
                        list_of_templates = open(template_set_file_name,"r")
                except IOError:
                        print "Cannot open file",template_set_file_name
                        sys.exit()
                
                        
                for line in list_of_templates:
                        if re.match("\#",line):
                                continue
                        auxwave = list()
                        auxflux = list()
                        cat = open(line.split()[0],"r")
                        for lline in cat:
                                if re.match("\#",lline):
                                        continue
                                x = [float(ix) for ix in lline.split()]
                                auxwave.append(x[0])
                                auxflux.append(x[1])
                        cat.close()
                        twave.append(auxwave)
                        tflux.append(auxflux)
                        ntemp += 1
                list_of_templates.close()
                
                twave = np.array(twave)
                tflux = np.array(tflux)
                
                return twave, tflux, ntemp



	def initial_values(self, spectrum=None):
		'''
	
		Needs to sample from prior distribution.
		'''

   		boxcar_width = 5 # width of smoothing function
             
		flux_max = max(runningMeanFast(spectrum.flux, boxcar_width))
        
		self.norm_min = np.zeros(self.norm_min.shape)
		self.norm_max = np.zeros(self.norm_max.shape) + flux_max
		norm_init = np.random.uniform(low=self.norm_min,high=self.norm_max)

		self.stellar_dispersion_min = 30.
		self.stellar_dispersion_max = 600.
		stellar_dispersion_init = np.random.uniform(low=self.stellar_dispersion_min,high=self.stellar_dispersion_max)
        
		host_params_init = list()
                if isinstance(norm_init,float):
                        host_params_init.append(norm_init)
                else:
                        host_params_init.extend(norm_init)
		host_params_init.append(stellar_dispersion_init)
		
		return host_params_init



	def initialize(self, data_spectrum=None):
		'''
		Perform any initializations where the data is optional.
		'''
		if data_spectrum is None:
			raise Exception("The data spectrum must be specified to initialize" + 
					"{0}.".format(self.__class__.__name__))
		self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths)
		

	def normalization_wavelength(self, data_spectrum_wavelength=None):
		'''
		
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
		
		host_norm = list()
                for i in range(self.n_templates):
			host_norm.append(params[self.parameter_index("normalization_host_template_{0:04d}".format(i))])
		stellar_dispersion = params[self.parameter_index("stellar_dispersion")]

		# Flat prior within the expected ranges.
                for i in range(self.n_templates):
			if self.norm_min[i] < host_norm[i] < self.norm_max[i]:
				ln_priors.append(np.log(1))
			else:
				ln_priors.append(-1.e17)
                if self.stellar_dispersion_min < stellar_dispersion < self.stellar_dispersion_max:
			ln_priors.append(np.log(1))
                else:
			ln_priors.append(-1.e17)

		return ln_priors


	def flux(self, wavelengths=None, parameters=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. Will use the initial parameters if none are specified.
		'''

		assert len(parameters) == len(self.model_parameter_names), "The wrong number of indices were provided: {0}".format(parameters)

                #Check if we have created the interpolated versions of
                #the templates in the model grid. If not, do it. 
                if self.template_flux_model_grid is None:
                        self.template_flux_model_grid = list()
                        self.template_flux_at_norm_wavelength = np.zeros(self.n_templates)
                        for i in range(self.n_templates):
                                f = scipy.interpolate.interp1d(self.template_wave[i],self.template_flux[i])
                                self.template_flux_model_grid.append(f(wavelengths))
                                self.template_flux_at_norm_wavelength[i] = f(self.normalization_wavelength())
                        self.template_flux_model_grid = np.array(self.template_flux_model_grid)

		# calculate flux of the component
                stellar_dispersion = parameters[-1] #Not implemented yet.
                flux = np.zeros(wavelengths.shape)
                norm = parameters[:self.n_templates]/self.template_flux_at_norm_wavelength
                for i in range(self.n_templates):
                        flux += norm[i]*self.template_flux_model_grid[i]

		return flux
