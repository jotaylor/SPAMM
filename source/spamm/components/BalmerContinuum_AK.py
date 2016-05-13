#!/usr/bin/python

import sys
import re
import numpy as np
from scipy.signal.signaltools import _next_regular 
from .ComponentBase import Component
from .BalmerContinuumSuper import BalmerSuper
from scipy.interpolate import interp1d
from scipy.integrate import simps
import pyfftw 

import matplotlib.pyplot as plt

#in cgs.  Code genereally assumes that wavelengths are in angstroms,
#these are taken care of in helper functions
c = 2.998e5
h = 6.626e-27
k = 1.381e-16

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
    index_low = int(outlen/2.)-int(origlen/2)
    index_high = int(outlen/2.)+int(origlen/2) # might need to check if consistently gets right length out
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

	#linear interpolation for now.....
	#interp = interp1d(np.exp(lnx),y)#this is stupid, but round-off error is affecting things
	yrebin = np.interp(np.exp(lnxnew),np.exp(lnx),y,left=0,right=0)#interp(np.exp(lnxnew))

	dpix = w/(lnxnew[1] - lnxnew[0])
	kw = round(5*dpix)
	kx = np.r_[-kw:kw+1]
	k  = np.exp(- (kx)**2/(dpix)**2)
	k /= abs(np.sum(k))

	ysmooth = fftwconvolve_1d(yrebin, k)

	assert ysmooth.size == x.size

	return np.interp(np.exp(lnx),np.exp(lnxnew),ysmooth,left=0,right=0) # does this need to be converted back to list ?
	

class BalmerContinuum(BalmerSuper):
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

		
		
	def flux(self, spectrum=None, parameters=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. Will use the initial parameters if none are specified.
		'''
		# should you have a line to state that if parameters=None use initial values?
		assert len(parameters) == len(self.model_parameter_names), ("The wrong number " +
									"of indices were provided: {0}".format(parameters))
		
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




