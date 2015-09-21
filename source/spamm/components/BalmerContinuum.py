#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

#in cgs.  Code genereally assumes that wavelengths are in angstroms,
#these are taken care of in my helper functions
c = 2.998e10
h = 6.626e-27
k = 1.381e-16

def planckfunc(wv,T):
	#returns dimensionless
	#assumes angstroms
	power = h*c/(k*T*(wv*1.e-8))
	denom = np.exp(power) - 1

	return 1./(wv)**5/denom

def absorbterm(wv,tau0):
	#calculates [1 - e^(-tau)] (optically-thin emitting slab)
	#assumes angstroms
	tau = tau0*(wv/3646.)**3
	return 1 - np.exp(-tau)

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
	

def iratio(l1,l2,T):
	#assuming LTE, this cacluates the ratio of the line fluxes.
	#assumes angstroms

	coef = np.genfromtxt('/home/rrlyrae/fausnaugh/repos/mcmc_deconvol/Data/SH95recomb/temp/BLcoeff.dat',usecols=1)
#	return coef[::-1]
	dE = h*c*1.e8*(1./l1 - 1./l2)
	boltzman = np.exp(-dE/k/T)

#	iout = np.r_[coef[::-1],boltzman[48::]*coef[0]]
	iout = boltzman
	return iout


def makelines(wv,T,shift,width):
	#Take the helper functions above, and sum the high order balmer lines
	#H detla is at 6, maybe start higher?
	N = np.r_[3:51]
	N = np.r_[3:200]
	L =  balmerseries(N)

	lines = genlines(wv,L,shift,width)
	I     = iratio(L,L[47],T)

	scale = np.repeat(I.reshape(I.size,1),lines.shape[1],axis=1)
	lines *= scale

#	for i in range(lines.shape[0]):
#		plt.plot(wv,lines[i],'k')
	lines  = np.sum(lines,axis = 0)

#	plt.plot(wv,lines,'k')
#	plt.show()

	return lines

def log_conv(x,y,w):
	lnx  = np.log(x)
	dlnx = (lnx[-1] - lnx[0])/lnx.size
	lnxnew = np.r_[lnx[0] : lnx[-1] + dlnx*0.99 : dlnx]
	lnxnew[-1] = lnx[-1]

	#linear interpolation for now.....
	interp = interp1d(lnx,y)
	yrebin = interp(lnxnew)

	if lnx.size %2 == 0:
		kx = np.r_[-lnx.size//2 : lnx.size//2 + 1]*dlnx
	else:
		kx = np.r_[-lnx.size//2 : lnx.size//2 + 2]*dlnx		
	k  = np.exp(- (kx)**2/(w)**2)
	k /= np.sum(k)

	ysmooth = np.convolve(yrebin,k,'same')

	interp = interp1d(lnxnew,ysmooth)
	assert interp(lnx).size == x.size

	return interp(lnx)

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
		super(BalmerContinuum, self).__init__()

		self.model_parameter_names = list()

		# parameters for the continuum
		self.model_parameter_names.append("normalization")
		self.model_parameter_names.append("Te")
		self.model_parameter_names.append("tauBE")

		# paramters for the lines
		self.model_parameter_names.append("loffset")
		self.model_parameter_names.append("lwidth")
		
		self._norm_wavelength =  None
		
		# [replace] define variables for min/max values for each parameter range
		self.normalization_min = None
		self.normalization_max = None

		self.Te_min = None
		self.Te_max = None

		self.tauBE_min = None
		self.tauBE_max = None

		self.loffset_min = None
		self.loffset_max = None

		self.lwidth_min = None
		self.lwidth_max = None
		# etc.
		
	@property
	def is_analytic(self):
		return True
	

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

		self.loffset_min = -10.0
		self.loffset_max =  10.0
		loffset_init = np.random.uniform( low=self.loffset_min,
						 high=self.loffset_max)

		self.lwidth_min = 1.0
		self.lwidth_max = 1000.
		tauBE_init = np.random.uniform( low=self.lwidth_min,
					       high=self.lwidth_max)


		return [normalization_init, Te_init, tauBE_init,loffset_init,lwidth_init]


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
		
		newedge = 3646*(1 - parameters[3]/c)

		p = planckfunc(spectrum.wavelengths,parameters[1])
		a = absorbterm(spectrum.wavelengths,parameters[2])
		flux = a*p
		flux[spectrum.wavelengths > newedge] = 0
		flux = log_conv(spectrum.wavelengths,flux,parameters[4]/c)

		m = abs(spectrum.wavelengths - newedge) == np.min(abs(spectrum.wavelengths - newedge))
		fnorm = flux[m]

		flux = parameters[0]*flux/fnorm
 
		lflux = makelines(spectrum.wavelengths,parameters[1],parameters[3]/c,parameters[4]/c)
		lfnorm = lflux[m]

		norm = parameters[0]/lfnorm

		lflux *= norm
#		lflux[spectrum.wavelengths <= newedge] = 0

#		plt.plot(spectrum.wavelengths,flux,'b')
#		plt.plot(spectrum.wavelengths,lflux,'r')
#		plt.show()
		
		flux += lflux

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
