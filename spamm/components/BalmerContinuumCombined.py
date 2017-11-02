#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.fftpack.helper import next_fast_len
import matplotlib.pyplot as plt
from astropy.constants import c, h, k_B, Ryd
from astropy.analytic_functions import blackbody_lambda

#TODO this needs to be integrated into Spectrum eventually
from utils.rebin_spec import rebin_spec
from utils.fftwconvolve_1d import fftwconvolve_1d

# Constants are in cgs.  
c = c.to("cg/s")
h = h.to("cg/s")
k = k_B.to("cg/s")
R = Ryd.to("1/Angstrom")
E0 = 2.179e-11
balmer_edge = 3646 # Angstroms

#-----------------------------------------------------------------------------#

def balmerseries(n):
    """
    Calculate a Balmer series line wavelength [Angstroms]
    
    Args:
        n (int or array of ints): Quantum number of the electron.

    Returns:
        float (float): Wavelength of the transition from n=n to n=2.
    """
    
    ilambda = R * (0.25 - 1./n**2)
    return 1. / ilambda

#-----------------------------------------------------------------------------#

def genlines(lgrid, lcent, shift, width, ltype):
    """
    Calculate Gaussian emission line fluxes. 

    Args:
        lgrid (array): Wavelengths at which to evaluate the line fluxes.
        lcent (array): Wavelength centers of each line.
        shift (int or float): Offset of lines.
        width (int or float): Width of lines.
        ltype (str): Type of line; gaussian or lorentzian.
    
    Returns:
        array (array): A 2d grid, whith one axis being the line and the other 
        being the fluxes of that line.
    """
    
    #To speed up code, give each line its own array. 
    lcent -= shift*lcent # what units is everything in 
    LL = lgrid - lcent.reshape(lcent.size,1) #(is this simply x-x0)

    lwidth =  width*lcent.reshape(lcent.size,1)
    if ltype == "gaussian":
        return np.exp(- LL**2 /lwidth**2)
    elif ltype == "lorentzian":
        return lwidth / (LL**2 + lwidth**2)
    else:
        raise ValueError("Variable 'ltype' ({0}) must be 'gaussian' or 'lorentzian'".
                         format(ltype))

#-----------------------------------------------------------------------------#

def balmer_ratio_SH(n_e, maxN, T, nlines):
    """
    Calculate the ratio between Balmer lines.
    (Intensity values from Storey and Hammer 1`995 results)
    
    Args:
        n_e (float): Electron density.
        maxN (int): ?
        T (): Temperature?
        nlines (int): Number of lines.

    Returns:
        array (array): Ratio of Balmer lines, with Htheta (N=10 -> N=2) first.
    """

    import pickle
    if nlines < 50:
        coeff = pickle.load(open('../SH95recombcoeff/coeff.interpers.pickle','rb'),
                            encoding="latin1")
        coef_use = [coef_interp(n_e, T) for coef_interp in coeff] 
        
        return np.array(coef_use[-1:-maxN+1:-1])

#-----------------------------------------------------------------------------#
#TODO merge all iratio's
#TODO iratio_50_400_noT was created since original was not fitting data- what
# is long term solution?

def iratio_50_400(N,init_iratio,T): # 
    """
    Estimate relative intensity values for lines 50-400 using from 
    Kovacevic et al 2014.
    """ 
    #T=15000.
    I50 = init_iratio[1]
    maxline = N.max()
    iratios = np.zeros(N.max()-49)
    iratios[0] = I50
    n = np.arange(50,maxline)
    #print('n',n)
    #print('iratios[0]',iratios[0])
    for i in n:
        iratios[i-49] = iratios[i-50]*np.exp(E0*(1./i**2-1./(i-1)**2)/(k*T))
    return iratios[1:]

#-----------------------------------------------------------------------------#

def iratio_50_400_noT(N,init_i): # 
    """
    Estimate relative intensity values for lines 50-400 using from 
    Kovacevic et al 2014, where we estimate excitation temperature from 
    last 2 Storey and Hummer intensities.
    """ 
    init_ratio = init_i[1]/init_i[0]
    T_ext = E0*(1./50.**2-1./49.**2)/(k*np.log(init_ratio))
    #print('T_ext',T_ext)
    #T_ext=15000.
    maxline = N.max()
    iratios = np.zeros(N.max()-49)
    iratios[0] = init_i[1]
    nn = np.arange(50,maxline)
    for i in nn:
        iratios[i-49] = iratios[i-50]*np.exp(E0*(1./i**2-1./(i-1)**2)/(k*T_ext))
    return iratios[1:]

#-----------------------------------------------------------------------------#

def makelines(sp_wavel, T, n_e, shift, width, ltype):
    """
    Args:
        sp_wavel (array): Wavelengths at which to evaluate the line fluxes.
        T (int or float): Electron temperature.
        n_e (float): Electron density.
        shift (int or float): Offset from expected Balmer wavelengths.
        width (int or float): Width of emission line. 
    """
    #Take the helper functions above, and sum the high order balmer lines
    #H-zeta is at 8, maybe start higher?
    nlines = np.arange(3,400)
    lcenter =  balmerseries(nlines)

#TODO define ltype somewhere (yaml)
    lcent -= shift*lcenter
    LL = sp_wavel - lcent.reshape(lcent.size, 1) #(is this simply x-x0)
    lwidth =  width*lcent.reshape(lcent.size,1)
    if ltype == "gaussian":
        lines = np.exp(- LL**2 /lwidth**2)
    elif ltype == "lorentzian":
        lines = lwidth / (LL**2 + lwidth**2)
    else:
        raise ValueError("Variable 'ltype' ({0}) must be 'gaussian' or 'lorentzian'".
                         format(ltype))

#TODO put this logic in functions themselves
#I -> lflux 
    if nlines.max() <= 51:
        lflux = balmer_ratio_SH(n_e,nlines.max(),T)
    else:
        sh_iratio =balmer_ratio_SH(n_e,51,T)
        I51 = sh_iratio.reshape(sh_iratio.size)
        I51beyond = iratio_50_400_noT(nlines,I51[-2:])
        #I51beyond = iratio_50_400(nlines,I51[-2:],T)
        lflux = np.append(I51,I51beyond)

    scale = np.repeat(lflux.reshape(lflux.size,1),lines.shape[1],axis=1)
    lines *= scale

    balmer_lines = np.sum(lines,axis = 0)

    return balmer_lines

#-----------------------------------------------------------------------------#

#TODO add convolution function to utils
def log_conv(x, y, w): 
    """
    Perform convolution in log space. 

    Args:
        x (): Function 1.
        y (): Function 2.
        w (): 

    Returns:
        array (array):
    """
    
    # Calculates convolution with lwidth in log wavelength space, 
    # which is equivalent to uniform in velocity space.
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

#-----------------------------------------------------------------------------#

#TODO add these functions to the class
# Use parameter names from class
def BC_flux(spectrum=None, parameters=None):
    """
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
    """

    # Wavelength at which Balmer components merge.
    edge_wl = balmer_edge*(1 - parameters[3]/c)
   
#TODO need WL as quantity objects for better astropy functionality
    blackbody = blackbody_lambda(spectrum.wavelengths, parameters[1])
    #calculates [1 - e^(-tau)] (optically-thin emitting slab)
    #assumes angstroms
    tau = parameters[2]*(sp_wavel/balmer_edge.)**3
    absorption = 1 - np.exp(-tau)
    bc_flux = absorption * blackbody

#TODO import from utils
    norm_index = find_nearest(spectrum.wavelengths, edge_wl) 
    fnorm = bc_flux[norm_index]
    bc_flux[spectrum.wavelengths > edge_wl-0.5] = 0.
    
    bc_flux *= parameters[0]/fnorm
    
    bc_flux = log_conv(spectrum.wavelengths,bc_flux,parameters[4]/c)

    return bc_flux

#-----------------------------------------------------------------------------#

#TODO pseudo continuum should be first when moved to class
def BpC_flux(spectrum=None, parameters=None):
    """
    Analytic model of the high-order Balmer lines, making up the Pseudo continuum near 3666 A.

    Line profiles are Gaussians (for now).  The line ratios are fixed to Storey &^ Hummer 1995, case B, n_e = 10^10 cm^-3.
    This component has 3 parameters:
    
    parameter1 : The flux normalization near the Balmer Edge lambda = 3666 A, F(3666)
    parameter4 : A shift of the line centroids
    parameter5 : The width of the Gaussians
    parameter6 : The log of the electron density
    
    note that all constants, and the units, are absorbed in the
    parameter F(3656 A).  
    
    priors:
    p1 :  Flat, between 0 and the observed flux F(3656).
    p4 :  Determined from Hbeta, if applicable.
    p5 :  Determined from Hbeta, if applicable.
    p6 :  Flat between 2 and 14.
            
    note that all constants, and the units, are absorbed in the
    parameter F(3656 A).  
    """

    c_kms = c.to("km/s")
    edge_wl = balmer_edge*(1 - parameters[3]/ckms)
    
    n_e =10.**parameters[5]
    T_e= parameters[1]
    bpc_flux = makelines(spectrum.wavelengths,
             T_e,n_e,
             parameters[3]/ckms,parameters[4]/ckms)
    
    norm_index = find_nearest(spectrum.wavelengths, edge_wl) 
    fnorm = bpc_flux[norm_index]
    
    bpc_flux[spectrum.wavelengths < edge_wl] = 0
    bpc_flux *= parameters[0]/fnorm

    return bpc_flux

#-----------------------------------------------------------------------------#

class BalmerCombined(Component):
    """
    Model of the combined BalmerContinuum (BC) based on Grandi et
    al. (1982) and Kovacevic & Popovic (2014).It contains two components: an analytical 
    function to describe optically thick clouds with a uniform temperature for wavelength <3646A
    and the sum of higher order Balmer lines which merge into a pseudo-continuum for wavelength >=3646A.
    The resulting flux is therefore given by the combination of these two components, here BC_flux + BpC_flux. 
    When initialising set which components to use"""


    def __init__(self, BalmerContinuum=False, BalmerPseudocContinuum=False):
        super().__init__()

        self.model_parameter_names = []
        self.name = "Balmer"

        # parameters for the continuum
        self.model_parameter_names.append("normalization_BC")
        self.model_parameter_names.append("Te")
        self.model_parameter_names.append("tauBE")

        # paramters for the lines
        self.model_parameter_names.append("loffset")
        self.model_parameter_names.append("lwidth")
        
        self.model_parameter_names.append("logNe")
        
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
        
        self.logNe_min = None
        self.logNe_max = None
        
        self.BC = BalmerContinuum
        self.BpC = BalmerPseudocContinuum
        
        # etc.
        
#-----------------------------------------------------------------------------#

    @property
    def is_analytic(self):
        return True
    
#-----------------------------------------------------------------------------#

    def initial_values(self, spectrum=None):
        """
        Needs to sample from prior distribution.
        These are the first guess for the parameters to be fit for in emcee, unless specified elsewhere.
        """

        #  calculate/define minimum and maximum values for each parameter.
        if spectrum is None:
            raise Exception("Need a data spectrum from which to estimate maximum flux at 3646 A")
        
        if self.normalization_min == None or self.normalization_max == None:
            m = np.nonzero(abs(spectrum.wavelengths - balmer_edge.) == np.min(abs(spectrum.wavelengths - balmer_edge.)))
            print('m',m)
            BCmax = np.max(spectrum.flux[m[0][0]-10:m[0][0]+10])
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
            self.lwidth_min = 100.
            self.lwidth_max = 10000.
        lwidth_init = np.random.uniform( low=self.lwidth_min,
                           high=self.lwidth_max)
                           
        if self.logNe_min == None or self.logNe_max == None:
            self.logNe_min = 2
            self.logNe_max = 14
        logNe_init = np.random.uniform(low=self.logNe_min,
                        high=self.logNe_max)


        return [normalization_init, Te_init, tauBE_init,loffset_init,lwidth_init,logNe_init]

#-----------------------------------------------------------------------------#

    def ln_priors(self, params):
        """
        Return a list of the ln of all of the priors.
        
        @param params
        """
        
        # need to return parameters as a list in the correct order
        ln_priors = list()
        
        
        
        #get the parameters
        normalization = params[self.parameter_index("normalization_BC")]
        Te            = params[self.parameter_index("Te")]
        tauBE         = params[self.parameter_index("tauBE")]
        loffset       = params[self.parameter_index("loffset")]
        lwidth        = params[self.parameter_index("lwidth")]
        logNe         = params[self.parameter_index("logNe")]
        

        
        #Flat priors, appended in order
        if self.normalization_min < normalization < self.normalization_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-np.inf)

        if self.Te_min < Te < self.Te_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-np.inf)

        if self.tauBE_min < tauBE < self.tauBE_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-np.inf)

        if self.loffset_min < loffset < self.loffset_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-np.inf)

        if self.lwidth_min < lwidth < self.lwidth_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-np.inf)
            
        if self.logNe_min < logNe < self.logNe_max:
            ln_priors.append(0)
        else:
            ln_priors.append(-np.inf)


        return ln_priors
        
#-----------------------------------------------------------------------------#

    def flux(self, spectrum=None, parameters=None):
        """
        Returns the flux for this component for a given wavelength grid
        and parameters. 
        """
        #get the parameters
        normalization = parameters[self.parameter_index("normalization_BC")]
        Te            = parameters[self.parameter_index("Te")]
        tauBE         = parameters[self.parameter_index("tauBE")]
        loffset       = parameters[self.parameter_index("loffset")]
        lwidth        = parameters[self.parameter_index("lwidth")]
        logNe         = parameters[self.parameter_index("logNe")]
        
        Bparameters = [normalization,Te,tauBE,loffset,lwidth,logNe]
        #print('balmerpameters',balmerparameters)
        #print('pameters',parameters)
        
        if self.BC and self.BpC:
            flux_BC = BC_flux(spectrum=spectrum, parameters=Bparameters)
            flux_BpC = BpC_flux(spectrum=spectrum, parameters=Bparameters)
            flux_est=[flux_BC[i]+flux_BpC[i] for i in xrange(len(flux_BpC))]

        else:
            if self.BC:
                flux_est = BC_flux(spectrum=spectrum, parameters=Bparameters)
            if self.BpC:
                flux_est = BpC_flux(spectrum=spectrum, parameters=Bparameters)
        return flux_est
