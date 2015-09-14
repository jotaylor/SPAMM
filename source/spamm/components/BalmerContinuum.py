#!/usr/bin/python

# To create a new component class, copy this file and fill in the values as instructed.
# Follow comments the begin with "[replace]", then delete the comment.

import sys
import numpy as np
from .ComponentBase import Component

# [replace] "TemplateComponent" with the name of the new component
class BalmerContinuum(Component):
	'''
	Analytic model of the BalmerContinuum (BC) based on Grandi et
	al. (1982) and Kovacevic & Popovic (2013).

	F(lambda) = F(3646 A) * B(T_e) * (1 - e^-tau(lambda))
	tau(lambda) = tau(3646) * (lambda/3646)^3
	
	This component has 3 parameters:
	
	parameter1 : The flux normalization at the Balmer Edge lambda = 3646 A, F(3646)
	parameter2 : The electron temperture T_e for the Planck function B(T_e)
	parameter3 : The optical depth at the Balmer edge, tau(3646)

	note that all constants, and the units, are absorbed in the
	parameter F(3646 A).  To impliment this appropriately, the
	planck function is normalized at 3646 A, so that F(3646)
	actually represents the flux at this wavelength.
	
	priors:
	p1 :  Flat between 0 and the measured flux at 3466 A.
	p2 :  Flat between 5 000 and 20 000 Kelvin
	p3 :  Flat between 0.1 and 2.0
			
	'''
	def __init__(self):
		# [replace] fill in the same name you gave above
		super(BalmerContinuum, self).__init__()

		# [replace] give the parameters names (spaces are ok), one line for each parameter
		self.model_parameter_names = list() # this may need to be defined as a method
		self.model_parameter_names.append("normalization")
		self.model_parameter_names.append("Te")
		self.model_parameter_names.append("tauBE")
		
		self._norm_wavelength =  None
		
		# [replace] define variables for min/max values for each parameter range
		self.normalization_min = None
		self.normalization_max = None

		self.Te_min = None
		self.Te_max = None

		self.tauBE_min = None
		self.tauBE_max = None
		# etc.
		
	@property
	def is_analytic(self):
		return True
	

	def get_norm(self):
		#use redshift? (are we always fitting in the observed frame?)
		if self.BCmax is None:
			self._norm_wavelength = np.median(data_spectrum_wavelength)
		return self._norm_wavelength

	def initial_values(self, spectrum=None):
		'''
		Needs to sample from prior distribution.
		'''

		# [replace] calculate/define minimum and maximum values for each parameter.
		if spectrum is None:
			raise Exception("Need a data spectrum from which to estimate maximum flux at 3646 A")
		m = abs(spectrum.wavelengths - 3646.) == np.min(abs(spectrum.wavelengths - 3646.))
		BCmax = spectrum.flux[m]

		self.normalization_min = 0
		self.normalization_max = BCmax
		normalization_init = np.random.uniform(low=self.normalization_min,
						       high=self.normalization_max)


		self.Te_min = 5.e3
		self.Te_max = 20.e3
		Te_init = np.random.uniform(low=self.Te_min,
					    high=self.Te_max)
		self.tauBE_min = 0.0
		self.tauBE_max = 2.0
		tauBE_init = np.random.uniform(low=self.tauBE_min,
					       high=self.tauBE_max)

		return [normalization_init, Te_init, tauBE_init]

#	def initialize(self, data_spectrum=None):
#		'''
#		Perform any initializations where the data is optional.
#		'''
#		if data_spectrum is None:
#			raise Exception("The data spectrum must be specified to initialize" + 
#							"{0}.".format(self.__class__.__name__))
#		self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.wavelengths)

	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''
		
		# need to return parameters as a list in the correct order
		ln_priors = list()
		
		#get the parameters
		normalization = params[self.parameter_index("normalization")]
		Te            = params[self.parameter_index("Te")]
		tauBE         = params[self.parameter_index("tauBE")]


		#Flat priors, appended in order
		if self.normalization_min < normalization < self.normalization_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		if self.Te_min < Te < self.Te_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		if self.tauBE_min < tauBE < self.tauBE_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		#reuturn
		return ln_priors
		
	def flux(self, spectrum=None, parameters=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. Will use the initial parameters if none are specified.
		'''

		assert len(parameters) == len(self.model_parameter_names), ("The wrong number " +
									"of indices were provided: {0}".format(parameters))
		
		p = self.planckfunc(spectrum.wavelengths,parameters[1])
		a = self.absorbterm(spectrum.wavelengths,parameters[2])

		pnorm = self.planckfunc(3646.,parameters[1])
		anorm = self.absorbterm(3646.,parameters[2])

		flux = parameters[0]*p*a/(pnorm*anorm)
		

		N = sp.r_[3:250]
		L =  Balmer(N)
		lines = self.genlines(spectrum.wavelengths,L)
		I     = self.Iratio(L,L[4],T_e)
		lines *= sp.repeat(I.reshape(I.size,1),ploty.shape[1],axis=1)
		lines  = sp.sum(lines,axis = 0)

		m = abs(spectrum.wavelengths - 3646.) == np.min(abs(spectrum.wavelegnths - 3646.))
		norm = parameters[0]/lines[m]
		lines *= norm
		
		flux += lines

		return flux

#	def flux(self, wavelengths=None, parameters=None):
#		'''
#		Returns the flux for this component for a given wavelength grid
#		and parameters. Will use the initial parameters if none are specified.
#		'''
#		assert len(parameters) == len(self.model_parameter_names), ("The wrong number " +
#									"of indices were provided: {0}".format(parameters))
#		
#		p = planckfunc(wavelengths,parameters[1])
#		a = asorbterm(wavelengths,parameters[2])
#
#		flux = parameters[0]*p*a
#		
#		return flux

	@staticmethod
	def planckfunc(wv,T):
		#assumes angstroms
		c = 2.998e10
		h = 6.626e-27
		k = 1.381e-16

		power = h*c/(k*T*(wv*1.e-8))
		denom = np.exp(power) - 1

		return 1./(wv)**5/denom

	@staticmethod
	def absorbterm(wv,tau0):
		#assumes angstroms
		tau = tau0*(wv/3646.)**3
		return 1 - np.exp(-tau)

	@staticmethod
	def Balmerseries(n):
		#assumes angstroms
		ilambda = 1./912*(0.25 - 1./n**2)
		return 1./ilambda

	@staticmethod
	def genlines(lgrid,lcent):
		#assumes angstroms
		LL = lgrid - lcent.reshape(lcent.size,1)
		return sp.exp(- (LL)**2/50.)/sp.sqrt(2*sp.pi*50)
	
	@staticmethod
	def Iratio(l1,l2,T):
		#assumes angstroms
		dE = h*c*1.e8*(1./l1 - 1./l2)
		return sp.exp(- dE/k/T)
