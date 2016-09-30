#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
from scipy.interpolate import interp1d
from scipy.integrate import simps
import pyfftw 

import matplotlib.pyplot as plt

#in cgs.  Code genereally assumes that wavelengths are in angstroms,
#these are taken care of in helper functions
c = 2.998e10
h = 6.626e-27
k = 1.381e-16

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
def iratio(n,T): # what does this function do? guessing finds normalisation given tau and optical thickness from Blamer continuum calculation
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


def BC_flux(spectrum=None, parameters=None):
    '''
    Returns the flux for this component for a given wavelength grid
    and parameters. Will use the initial parameters if none are specified.
    '''
    # should you have a line to state that if parameters=None use initial values?


    newedge = 3646*(1 - parameters[3]/c)

    p = planckfunc(spectrum.wavelengths,parameters[1])
    a = absorbterm(spectrum.wavelengths,parameters[2])
    flux = a*p
    flux[spectrum.wavelengths > newedge] = 0

    m = abs(spectrum.wavelengths - newedge) == np.min(abs(spectrum.wavelengths - newedge))
    fnorm = flux[m]

    flux = parameters[0]*flux/fnorm

    flux = log_conv(spectrum.wavelengths,flux,parameters[4]/c)

    return flux

def BpC_flux(self, spectrum=None, parameters=None):
    '''
    Returns the flux for this component for a given wavelength grid
    and parameters. Will use the initial parameters if none are specified.
    '''
    newedge = 3656*(1 - parameters[3]/c)

    flux = makelines(spectrum.wavelengths,
                     parameters[1],parameters[2],
                     parameters[3]/c,parameters[4]/c)

    m = abs(spectrum.wavelengths - newedge) == np.min(abs(spectrum.wavelengths - newedge))
    fnorm = flux[m]

    flux = parameters[0]*flux/fnorm # shouldn't we check if this hasn't been calculated in Balmer

    return flux


class BalmerSuper(Component, BalmerContinuum=False, BalmerPseudocContinuum=False):
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
        super(BalmerSuper, self).__init__()

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
        '''

        #  calculate/define minimum and maximum values for each parameter.
        if spectrum is None:
            raise Exception("Need a data spectrum from which to estimate maximum flux at 3646 A")

        if self.normalization_min == None or self.normalization_max == None:
            m = abs(spectrum.wavelengths - 3646.) == np.min(abs(spectrum.wavelengths - 3646.))

            BCmax = spectrum.flux[m]

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
            self.lwidth_max = 1000.
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
            ln_priors.append(-1.e17)

        if self.Te_min < Te < self.Te_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-1.e17)

        if self.tauBE_min < tauBE < self.tauBE_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-1.e17)

        if self.loffset_min < loffset < self.loffset_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-1.e17)

        if self.lwidth_min < lwidth < self.lwidth_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-1.e17)

#		if self.normalization_min < normalization < self.normalization_max and self.Te_min < Te < self.Te_max and  self.tauBE_min < tauBE < self.tauBE_max and self.loffset_min < loffset < self.loffset_max and self.lwidth_min < lwidth < self.lwidth_max:
#			ln_priors.append(0.)
#		else:
#			ln_priors.append(-1.e17)

        return ln_priors

    @property
    def flux(self):
        assert len(parameters) == len(self.model_parameter_names), ("The wrong number " +
                                                                "of indices were provided: {0}".format(parameters))
        if BC:
            flux_est 
        return flux_est




