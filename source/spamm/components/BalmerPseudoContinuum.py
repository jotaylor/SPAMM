#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
import pickle

#in cgs.  Code genereally assumes that wavelengths are in angstroms,
#these are taken care of in helper functions
c = 2.998e10

def balmerseries(n):
	#central wavelenths of the balmer series
	#assumes angstroms
	ilambda = 1./911.3*(0.25 - 1./n**2)
	return 1./ilambda

def genlines(lgrid,lcent,shift,width):
	#cacluates the (gaussian) emission line fluxes
	#lgrid is the array of wavelengths at which to evaluate the line fluxes
	#lcent is an array with wavelength centers
	#lshift, lwidth are the offset and the width of the lines

	#to make it go fast, the idea is to give each line it's own
	#array, so this returns a 2d grid, whith one axis being the
	#line and the other axis being the fluxes of that line.
	lcent -= shift*lcent
	LL = lgrid - lcent.reshape(lcent.size,1)
	
	lwidth =  width*lcent.reshape(lcent.size,1)
	return np.exp(- LL**2 /lwidth**2)
	
coeff = pickle.load(open('../SH95recombcoeff/coeff.interpers.pickle','rb'))
def iratio(n,T):
	coef_use = [ coef_interp(n,T) for coef_interp in coeff ] 
	#returns Htheta (E = 10 to E = 2) first
	return np.array(coef_use[-8::-1])


def makelines(wv,T,n,shift,width):
	#Take the helper functions above, and sum the high order balmer lines
	#H-zeta is at 8, maybe start higher?
	N = np.r_[10:51]
#	N = np.r_[3:200]
	L =  balmerseries(N)

	lines = genlines(wv,L,shift,width)
	I     = iratio(n,T)

	scale = np.repeat(I.reshape(I.size,1),lines.shape[1],axis=1)
	lines *= scale

#	for i in range(lines.shape[0]):
#		plt.plot(wv,lines[i],'k')
	lines  = np.sum(lines,axis = 0)

#	plt.plot(wv,lines,'k')
#	plt.show()

	return lines


class BalmerPseudoContinuum(Component):
	'''
	Analytic model of the high-order Balmer lines, making up the Pseudo continuum near 3656 A.

	Line profiles are Gaussians (for now).  The line ratios are fixed to Storey &^ Hummer 1995, case B, n_e = 10^10 cm^-3.

	This component has 3 parameters:
	
	parameter1 : The flux normalization near the Balmer Edge lambda = 3656 A, F(3656)
	parameter2 : A shift of the line centroids
	parameter3 : The width of the Gaussians

	note that all constants, and the units, are absorbed in the
	parameter F(3656 A).  
	
	priors:
	p1 :  Flat, between 0 and the observed flux F(3656).
	p2 :  Determined from Hbeta, if applicable.
	p3 :  Determined from Hbeta, if applicable.
			
	'''
	def __init__(self):
		# [replace] fill in the same name you gave above
		super(BalmerPseudoContinuum, self).__init__()

		self.model_parameter_names = list() 

		# paramters for the lines
		self.model_parameter_names.append("normalization")
		self.model_parameter_names.append("temperature")
		self.model_parameter_names.append("density")
		self.model_parameter_names.append("loffset")
		self.model_parameter_names.append("lwidth")

		
		self._norm_wavelength =  None
		
		self.normalization_min = None
		self.normalization_max = None

		self.temperature_min = None
		self.temperature_max = None

		self.density_min = None
		self.density_max = None

		self.loffset_min = None
		self.loffset_max = None

		self.lwidth_min = None
		self.lwidth_max = None# etc.


		#store look up tables of high-order balmer line emissivities

		
	@property
	def is_analytic(self):
		return True
	

	def initial_values(self, spectrum=None):
		'''
		
		Needs to sample from prior distribution.
		'''
		if spectrum is None:
			raise Exception("Need a data spectrum from which to estimate maximum flux at 3656 A")
		#shouldn't exceed 
		#need to put in redshift!?
		m = abs(spectrum.wavelengths - 3656.) == np.min(abs(spectrum.wavelengths - 3656.))
		BCmax = spectrum.flux[m]

		self.normalization_min = 0
		self.normalization_max = BCmax
		normalization_init = np.random.uniform(low=self.normalization_min,
						       high=self.normalization_max)

		self.temperature_min = 500.
		self.temperature_max = 30000.
		temperature_init = np.random.uniform(low=self.temperature_min,
						       high=self.temperature_max)

		self.density_min = 1.e8
		self.density_max = 1.e14
		density_init = np.random.uniform(low=self.density_min,
						       high=self.density_max)

		self.loffset_min = -10.0
		self.loffset_max =  10.0
		loffset_init = np.random.uniform( low=self.loffset_min,
						 high=self.loffset_max)

		self.lwidth_min = 1.0
		self.lwidth_max = 1000.
		lwidth_init = np.random.uniform( low=self.lwidth_min,
						 high=self.lwidth_max)


		return [normalization_init, loffset_init,lwidth_init]


	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''
		
				# need to return parameters as a list in the correct order
		ln_priors = list()
		
		#get the parameters
		normalization = params[self.parameter_index("normalization")]
		temperature   = params[self.parameter_index("temperature")]
		density       = params[self.parameter_index("density")]
		loffset       = params[self.parameter_index("loffset")]
		lwidth        = params[self.parameter_index("lwidth")]

		#Flat priors, appended in order
		if self.normalization_min < normalization < self.normalization_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		if self.temperature_min < temperature < self.temperature_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		if self.density_min < density < self.density_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		if self.loffset_min < loffset < self.loffset_max:
			ln_priors.append(np.log(1))
		else:
			ln_priors.append(-1.e17)

		if self.lwidth_min < lwidth < self.lwidth_max:
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
		newedge = 3656*(1 - parameters[3]/c)

		flux = makelines(spectrum.wavelengths,
				 parameters[1],parameters[2],
				 parameters[3]/c,parameters[4]/c)

		m = abs(spectrum.wavelengths - newedge) == np.min(abs(spectrum.wavelengths - newedge))
		fnorm = flux[m]

		flux = parameters[0]*flux/fnorm

		return flux

