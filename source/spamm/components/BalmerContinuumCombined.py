#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal.signaltools import _next_regular 
import pyfftw 
import pickle
from pysynphot import observation
from pysynphot import spectrum as pysynphot_spec

#in cgs.  Code genereally assumes that wavelengths are in angstroms,
#these are taken care of in helper functions
c = 2.998e10
h = 6.626e-27
k = 1.381e-16
E0 = 2.179e-11

def rebin_spec(wave, specin, wavnew):
	'''
	Rebin spectra to bins used in wavnew.
	Ref: http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/
	'''
	spec = pysynphot_spec.ArraySourceSpectrum(wave=wave, flux=specin)
	f = np.ones(len(wave))
	filt = pysynphot_spec.ArraySpectralElement(wave, f, waveunits='angstrom')
	obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

	return obs.binflux	

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
	lcent -= shift*lcent # what units is everything in 
	LL = lgrid - lcent.reshape(lcent.size,1) #(is this simply x-x0)
	
	lwidth =  width*lcent.reshape(lcent.size,1)
	return np.exp(- LL**2 /lwidth**2)
	
coeff = pickle.load(open('../SH95recombcoeff/coeff.interpers.pickle','rb'))
def iratio_SH(n,T): # 
	"""Grabs intensity values from Storey and Hammer 1995 results""" 
	coef_use = [ coef_interp(n,T) for coef_interp in coeff ] 
	#returns Htheta (E = 10 to E = 2) first
	return np.array(coef_use[-1::-1])
	
def iratio_50_400(N,init_iratio,T): # 
	"""Estimates relative intensity values for lines 50-400 using from Kovacevic et al 2014""" 
	I50 = init_iratio
	maxline = N.max()
	iratios = np.zeros(N.max()-49)
	iratios[0] = I50
	n = np.arange(50,maxline)
	#print('n',n)
	#print('iratios[0]',iratios[0])
	for i in n:
		iratios[i-49] = iratios[i-50]*np.exp(E0*(1./i**2-1./(i-1)**2)/(k*T))
	return iratios[1:]

def makelines(wv,T,n,shift,width):
	#Take the helper functions above, and sum the high order balmer lines
	#H-zeta is at 8, maybe start higher?
	N = np.r_[3:51]
#	N = np.r_[3:120]
	L =  balmerseries(N)

	lines = genlines(wv,L,shift,width)
	
	if N.max() == 51:
		I = iratio_SH(n,T)
	else:
		sh_iratio =iratio_SH(n,T)
		I51 = sh_iratio.reshape(sh_iratio.size)
		I51beyond = iratio_50_400(N,I51[-1],T)
		I = np.append(I51,I51beyond)

	scale = np.repeat(I.reshape(I.size,1),lines.shape[1],axis=1)
	lines *= scale

	lines  = np.sum(lines,axis = 0)


	return lines

pyfftw.interfaces.cache.enable() # Cache for the "planning" 
pyfftw.interfaces.cache.set_keepalive_time(1.0) 
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

def planckfunc(wv,T):
	#returns dimensionless
	#assumes angstroms
	power = h*c/(k*T*(wv*1.e-8))
	denom = np.exp(power) - 1
	# where is 2hc**2 term?
	return 1./(wv)**5/denom

def absorbterm(wv,tau0):
	#calculates [1 - e^(-tau)] (optically-thin emitting slab)
	#assumes angstroms
	tau = tau0*(wv/3646.)**3
	return 1 - np.exp(-tau)


def log_conv(x,y,w): 
	# calculates convolution with lwidth in log wavelength space, which is equivalent to uniform in velocity space
	lnx  = np.log(x)
	lnxnew = np.r_[lnx.min():lnx.max():1j*lnx.size]
	lnxnew[0]  = lnx[0]
	lnxnew[-1] = lnx[-1]

	#rebin spectrum in equally spaced log wavelengths
	yrebin = rebin_spec(lnx,y, lnxnew)

	dpix = w/(lnxnew[1] - lnxnew[0])
	kw = round(5*dpix)
	kx = np.r_[-kw:kw+1]
	k  = np.exp(- (kx)**2/(dpix)**2)
	k /= abs(np.sum(k))

	ysmooth = fftwconvolve_1d(yrebin, k)
	ysmooth -=np.min(ysmooth)
	assert ysmooth.size == x.size
	#rebin spectrum to original wavelength values
	return rebin_spec(np.exp(lnxnew),ysmooth,x)


def BC_flux(spectrum=None, parameters=None):
		'''
		Analytic model of the BalmerContinuum (BC) based on Grandi et
		al. (1982) and Kovacevic & Popovic (2013).
	
		F(lambda) = F(3646 A) * B(T_e) * (1 - e^-tau(lambda))
		tau(lambda) = tau(3646) * (lambda/3646)^3
		
		This component has 5 parameters:
		
		parameter1 : The flux normalization at the Balmer Edge lambda = 3646 A, F(3646)
		parameter2 : The electron temperture T_e for the Planck function B(T_e)
		parameter3 : The optical depth at the Balmer edge, tau(3646)
		parameter4 : A shift of the line centroids
		parameter5 : The width of the Gaussians
		
	
		note that all constants, and the units, are absorbed in the
		parameter F(3646 A).  To impliment this appropriately, the
		planck function is normalized at 3646 A, so that F(3646)
		actually represents the flux at this wavelength.
		
		priors:
		p1 :  Flat between 0 and the measured flux at 3466 A.
		p2 :  Flat between 5 000 and 20 000 Kelvin
		p3 :  Flat between 0.1 and 2.0
		p4 :  Determined from Hbeta, if applicable.
		p5 :  Determined from Hbeta, if applicable.			
		'''
		
		newedge = 3646*(1 - parameters[3]/c)

		p = planckfunc(spectrum.wavelengths,parameters[1])
		a = absorbterm(spectrum.wavelengths,parameters[2])
		flux = a*p

		m = abs(spectrum.wavelengths - newedge) == np.min(abs(spectrum.wavelengths - newedge))
		fnorm = flux[m]
		
		flux[spectrum.wavelengths > newedge] = 0

		flux = parameters[0]*flux/fnorm
 
		flux = log_conv(spectrum.wavelengths,flux,parameters[4]/c)
		
		return flux
		
def BpC_flux(spectrum=None, parameters=None):
		'''
		Analytic model of the high-order Balmer lines, making up the Pseudo continuum near 3666 A.

		Line profiles are Gaussians (for now).  The line ratios are fixed to Storey &^ Hummer 1995, case B, n_e = 10^10 cm^-3.

		This component has 3 parameters:
	
		parameter1 : The flux normalization near the Balmer Edge lambda = 3666 A, F(3666)
		parameter4 : A shift of the line centroids
		parameter5 : The width of the Gaussians

		note that all constants, and the units, are absorbed in the
		parameter F(3656 A).  
	
		priors:
		p1 :  Flat, between 0 and the observed flux F(3656).
		p4 :  Determined from Hbeta, if applicable.
		p5 :  Determined from Hbeta, if applicable.
			
		'''
		ckms = 2.998e5
		newedge = 3666*(1 - parameters[3]/ckms)

		flux = makelines(spectrum.wavelengths,
				 parameters[1],parameters[2],
				 parameters[3]/ckms,parameters[4]/ckms)

		m = abs(spectrum.wavelengths - newedge) == np.min(abs(spectrum.wavelengths - newedge))
		fnorm = flux[m]
		
		newedge = 3646*(1 - parameters[3]/ckms)
		flux[spectrum.wavelengths < newedge] = 0

		flux = parameters[0]*flux/fnorm 

		return flux


class BalmerCombined(Component):
	'''
	Model of the combined BalmerContinuum (BC) based on Grandi et
	al. (1982) and Kovacevic & Popovic (2014).It contains two components: an analytical 
	function to describe optically thick clouds with a uniform temperature for wavelength <3646A
	and the sum of higher order Balmer lines which merge into a pseudo-continuum for wavelength >=3646A.
	The resulting flux is therefore given by the combination of these two components, here BC_flux + BpC_flux. 
	When initialising set which components to use'''


	def __init__(self, BalmerContinuum=False, BalmerPseudocContinuum=False):
		super(BalmerCombined, self).__init__()

		self.model_parameter_names = list()

		# parameters for the continuum
		self.model_parameter_names.append("normalization")
		self.model_parameter_names.append("Te")
		self.model_parameter_names.append("tauBE")

		# paramters for the lines
		self.model_parameter_names.append("loffset")
		self.model_parameter_names.append("lwidth")
		
		self._norm_wavelength =  None
		
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
		
		self.BC = BalmerContinuum
		self.BpC = BalmerPseudocContinuum
		
		# etc.
		
	@property
	def is_analytic(self):
		return True
	

	def initial_values(self, spectrum=None):
		'''
		Needs to sample from prior distribution.
		These are the first guess for the parameters to be fit for in emcee, unless specified elsewhere.
		'''

		#  calculate/define minimum and maximum values for each parameter.
		if spectrum is None:
			raise Exception("Need a data spectrum from which to estimate maximum flux at 3646 A")
		
		if self.normalization_min == None or self.normalization_max == None:
			m = np.nonzero(abs(spectrum.wavelengths - 3646.) == np.min(abs(spectrum.wavelengths - 3646.)))
			print('m',m)
			BCmax = np.max(spectrum.flux[m[0]-10:m[0]+10])
			print('BCmax',BCmax)
			self.normalization_min = 0
			self.normalization_max = BCmax
		normalization_init = np.random.uniform(low=self.normalization_min,
						       high=self.normalization_max)
						       
		if self.Te_min == None or self.Te_max == None:
			self.Te_min = 5.e3
			self.Te_max = 20.e3
		Te_init = np.random.uniform(low=self.Te_min,
					    high=self.Te_max)
					    
		if self.tauBE_min == None or self.tauBE_max == None:
			self.tauBE_min = 0.0
			self.tauBE_max = 2.0
		tauBE_init = np.random.uniform(low=self.tauBE_min,
					       high=self.tauBE_max)

		if self.loffset_min == None or self.loffset_max == None:
			self.loffset_min = -10.0
			self.loffset_max =  10.0
		loffset_init = np.random.uniform( low=self.loffset_min,
						 high=self.loffset_max)

		if self.lwidth_min == None or self.lwidth_max == None:
			self.lwidth_min = 1.0
			self.lwidth_max = 10000.
		lwidth_init = np.random.uniform( low=self.lwidth_min,
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
		normalization = params[0]#[self.parameter_index("normalization")]
		Te            = params[1]#[self.parameter_index("Te")]
		tauBE         = params[2]#[self.parameter_index("tauBE")]
		loffset       = params[3]#[self.parameter_index("loffset")] # determined in blamer pseudo continuum, right?
		lwidth        = params[4]#[self.parameter_index("lwidth")]


		#Flat priors, appended in order
		if self.normalization_min < normalization < self.normalization_max:
			ln_priors.append(0)
		else:
			ln_priors.append(-1.e100)

		if self.Te_min < Te < self.Te_max:
			ln_priors.append(0)
		else:
			ln_priors.append(-1.e100)

		if self.tauBE_min < tauBE < self.tauBE_max:
			ln_priors.append(0)
		else:
			ln_priors.append(-1.e100)

		if self.loffset_min < loffset < self.loffset_max:
			ln_priors.append(0)
		else:
			ln_priors.append(-1.e100)

		if self.lwidth_min < lwidth < self.lwidth_max:
			ln_priors.append(0)
		else:
			ln_priors.append(-1.e100)


		return ln_priors
		
		
		
	def flux(self, spectrum=None, parameters=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. 
		'''
		if self.BC and self.BpC:
			flux_BC = BC_flux(spectrum=spectrum, parameters=parameters)
			flux_BpC = BpC_flux(spectrum=spectrum, parameters=parameters)
			flux_est=[flux_BC[i]+flux_BpC[i] for i in xrange(len(flux_BpC))]

		else:
			if self.BC:
				flux_est = BC_flux(spectrum=spectrum, parameters=parameters)
			if self.BpC:
				flux_est = BpC_flux(spectrum=spectrum, parameters=parameters)
		return flux_est




