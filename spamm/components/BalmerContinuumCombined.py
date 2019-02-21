#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.fftpack.helper import next_fast_len
import matplotlib.pyplot as plt
from astropy.constants import c, h, k_B, Ryd
from astropy.modeling.blackbody import blackbody_lambda

#TODO this needs to be integrated into Spectrum eventually
from utils.rebin_spec import rebin_spec
#from utils.fftwconvolve_1d import fftwconvolve_1d
from utils.find_nearest_index import find_nearest_index
from utils.parse_pars import parse_pars
#import line_profiler

PARS = parse_pars()["balmer_continuum"]

# Constants are in cgs.  
c = c.cgs
h = h.cgs
k = k_B.cgs
R = Ryd.to("1/Angstrom")
E0 = 2.179e-11
balmer_edge = 3646 # Angstroms


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
        self.model_parameter_names.append("bc_norm")
        self.model_parameter_names.append("bc_Te")
        self.model_parameter_names.append("bc_tauBE")
        # paramters for the lines
        self.model_parameter_names.append("bc_loffset")
        self.model_parameter_names.append("bc_lwidth")
        self.model_parameter_names.append("bc_logNe")
#        self.model_parameter_names.append("lscale")
        
        self._norm_wavelength =  None
        
        self.normalization_min = PARS["bc_norm_min"]
        self.normalization_max = PARS["bc_norm_max"]

        self.Te_min = PARS["bc_Te_min"]
        self.Te_max = PARS["bc_Te_max"]

        self.tauBE_min = PARS["bc_tauBE_min"]
        self.tauBE_max = PARS["bc_tauBE_max"]

        self.loffset_min = PARS["bc_loffset_min"]
        self.loffset_max = PARS["bc_loffset_max"]

        self.lwidth_min = PARS["bc_lwidth_min"]
        self.lwidth_max = PARS["bc_lwidth_max"]
        
        self.logNe_min = PARS["bc_logNe_min"]
        self.logNe_max = PARS["bc_logNe_max"]
        
#        self.lscale_min = None
#        self.lscale_max = None
        
        self.BC = BalmerContinuum
        self.BpC = BalmerPseudocContinuum
        
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
        
        if self.normalization_max == "bcmax_flux":
            be_index = find_nearest_index(spectrum.wavelengths, balmer_edge) 
            bcmax = np.max(spectrum.flux[be_index-10:be_index+10])
            self.normalization_max = bcmax
        normalization_init = np.random.uniform(low=self.normalization_min,
                               high=self.normalization_max)
        
        Te_init = np.random.uniform(low=self.Te_min,
                        high=self.Te_max)
                        
        tauBE_init = np.random.uniform(low=self.tauBE_min,
                           high=self.tauBE_max)

        loffset_init = np.random.uniform( low=self.loffset_min,
                         high=self.loffset_max)

        lwidth_init = np.random.uniform( low=self.lwidth_min,
                           high=self.lwidth_max)
                           
        logNe_init = np.random.uniform(low=self.logNe_min,
                        high=self.logNe_max)
                        
#        if self.lscale_min == None or self.lscale_max == None:
#            self.lscale_min = 0
#            self.lscale_max = 2
#        lscale_init = np.random.uniform(low=self.lscale_min,
#                        high=self.lscale_max)


        return [normalization_init, Te_init, tauBE_init,loffset_init,lwidth_init,logNe_init]#,lscale_init]

#-----------------------------------------------------------------------------#

    def ln_priors(self, params):
        """
        Return a list of the ln of all of the priors.
        
        @param params
        """
        
        # need to return parameters as a list in the correct order
        ln_priors = list()
        
        
        
        #get the parameters
        normalization = params[self.parameter_index("bc_norm")]
        Te            = params[self.parameter_index("bc_Te")]
        tauBE         = params[self.parameter_index("bc_tauBE")]
        loffset       = params[self.parameter_index("bc_loffset")]
        lwidth        = params[self.parameter_index("bc_lwidth")]
        logNe         = params[self.parameter_index("bc_logNe")]
#        lscale        = params[self.parameter_index("lscale")]
        

        
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
            
#        if self.lscale_min < lscale < self.lscale_max:
#            ln_priors.append(0)
#        else:
#            ln_priors.append(-np.inf)

        return ln_priors
        
#-----------------------------------------------------------------------------#
    
    #TODO add convolution function to utils
    
    def log_conv(self,wavelength, orig_flux, width_lines): 
        
        """
        Perform convolution in log space. 
        
        Args:
            wavelength (): wavelength of original spectrum
            orig_flux (): Flux of spectrum before convolution
            width_lines (): width of the lines wanted in v/c 
        
        Returns:
            array (array):
        """
        
        # Calculates convolution with lwidth in log wavelength space, 
        # which is equivalent to uniform in velocity space.
        ln_wave  = np.log(wavelength)
        ln_wavenew = np.r_[ln_wave.min():ln_wave.max():1j*ln_wave.size]
        ln_wavenew[0]  = ln_wave[0]
        ln_wavenew[-1] = ln_wave[-1]
        
        #rebin spectrum in equally spaced log wavelengths
        if self.fast_interp:
            flux_rebin = np.interp(ln_wavenew, ln_wave, orig_flux)
        else:
            flux_rebin = rebin_spec(ln_wave, orig_flux, ln_wavenew)
        
        dpix = width_lines/(ln_wavenew[1] - ln_wavenew[0])
        kernel_width = round(5*dpix)
        kernel_x = np.r_[-kernel_width:kernel_width+1]
        kernel  = np.exp(- (kernel_x)**2/(dpix)**2)
        kernel /= abs(np.sum(kernel))
        
        flux_conv = np.convolve(flux_rebin, kernel,mode='same')
        assert flux_conv.size == wavelength.size
        #rebin spectrum to original wavelength values
        if self.fast_interp:
            orig_wl = np.interp(wavelength, np.exp(ln_wavenew), flux_conv)
        else:
            orig_wl = rebin_spec(np.exp(ln_wavenew), flux_conv, wavelength)
        
        return orig_wl
        
#-----------------------------------------------------------------------------#
    
    def balmerseries(self, line_orders):
        """
        Calculate a Balmer series line wavelength [Angstroms]
        
        Args:
            line_orders (int or array of ints): Quantum number of the electron.
    
        Returns:
            float (float): Wavelength of the transition from n=n to n=2.
        """
        
        ilambda = R.value * (0.25 - 1./line_orders**2)
        return 1. / ilambda
    
    
#-----------------------------------------------------------------------------#
    
    def balmer_ratio(self, n_e, line_orders ,T): # 
        """
        Calculate the ratio between Balmer lines.
        (Intensity values from Storey and Hammer 1`995 results and Kovacevic et al 2014)
        
        Args:
            n_e (float): Electron density.
            T (): Electron Temperature
            lines (int array): array of lines to include.
    
        Returns:
            array (array): Ratio of Balmer lines, with Htheta (N=10 -> N=2) first.
        """
        """
        Estimate relative intensity values for lines 3-400 using from 
        Kovacevic et al 2014.
        """ 
        import pickle
        
        maxline = int(line_orders.max())
        minline = int(line_orders.min())
        flux_ratios = np.zeros(maxline-minline+1)
        n = np.arange(minline,maxline+1)
    
        coeff = pickle.load(open('../Data/SH95recombcoeff/coeff.interpers.pickle','rb'),
                            encoding="latin1")
        coef_use = [coef_interp(n_e, T) for coef_interp in coeff] 
    
        for i in n:
            if i <=50:
                flux_ratios[i-minline] = coef_use[-i+2]
            else:
                flux_ratios[i-minline] = flux_ratios[i-minline-1]*np.exp(E0*(1./i**2-1./(i-1)**2)/(k.value*T))
        return flux_ratios
    
#-----------------------------------------------------------------------------#
    #@profile
    def makelines(self, sp_wavel, T, n_e, shift, width):
        """
        Args:
            sp_wavel (array): Wavelengths at which to evaluate the line fluxes.
            T (int or float): Electron temperature.
            n_e (float): Electron density.
            shift (int or float): Offset from expected Balmer wavelengths.
            width (int or float): Width of emission line. 
        """

        line_orders = np.arange(PARS["bc_lines_min"],PARS["bc_lines_max"]) 
        lcenter =  self.balmerseries(line_orders)
    
        lcenter -= shift*lcenter
        LL = sp_wavel - lcenter.reshape(lcenter.size, 1) #(is this simply x-x0)
        lwidth =  width*lcenter.reshape(lcenter.size,1)
        ltype = PARS["bc_line_type"]
        if ltype == "gaussian":
            lines = np.exp(- LL**2 /lwidth**2)
        elif ltype == "lorentzian":
            lines = lwidth / (LL**2 + lwidth**2)
        else:
            raise ValueError("Variable 'ltype' ({0}) must be 'gaussian' or 'lorentzian'".
                             format(ltype))
    
        lflux = self.balmer_ratio(n_e,line_orders,T)

        scale = np.repeat(lflux.reshape(lflux.size,1),lines.shape[1],axis=1)


        lines *= scale
    
        balmer_lines = np.sum(lines,axis = 0)
    
        return balmer_lines
    

       
#-----------------------------------------------------------------------------#
    
    def BpC_flux(self, spectrum=None, parameters=None):
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
        normalization = parameters[self.parameter_index("bc_norm")]
        Te            = parameters[self.parameter_index("bc_Te")]
        tauBE         = parameters[self.parameter_index("bc_tauBE")]
        loffset       = parameters[self.parameter_index("bc_loffset")]
        lwidth        = parameters[self.parameter_index("bc_lwidth")]
        logNe         = parameters[self.parameter_index("bc_logNe")]
#        lscale         = parameters[self.parameter_index("lscale")]
    
        c_kms = c.to("km/s")
        edge_wl = balmer_edge*(1 - loffset/c_kms.value)
        
        n_e =10.**logNe
        bpc_flux = self.makelines(spectrum.wavelengths,
                 Te,n_e,
                 loffset/c_kms.value,lwidth/c_kms.value)
                 
        
        norm_index = find_nearest_index(spectrum.wavelengths, edge_wl) 
        
        
        fnorm = bpc_flux[norm_index]
        bpc_flux[spectrum.wavelengths <= spectrum.wavelengths[norm_index]] = 0
        bpc_flux *= normalization/fnorm
#        bpc_flux *= lscale
    
        return bpc_flux
    
#-----------------------------------------------------------------------------#
    
    def BC_flux(self, spectrum=None, parameters=None):
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
        #get the parameters
        normalization = parameters[self.parameter_index("bc_norm")]
        Te            = parameters[self.parameter_index("bc_Te")]
        tauBE         = parameters[self.parameter_index("bc_tauBE")]
        loffset       = parameters[self.parameter_index("bc_loffset")]
        lwidth        = parameters[self.parameter_index("bc_lwidth")]
        logNe         = parameters[self.parameter_index("bc_logNe")]
        
        # Wavelength at which Balmer components merge.
        edge_wl = balmer_edge*(1 - loffset/c.value)

    #TODO need WL as quantity objects for better astropy functionality
        blackbody = blackbody_lambda(spectrum.wavelengths, Te)
        #calculates [1 - e^(-tau)] (optically-thin emitting slab)
        #assumes angstroms
        tau = tauBE*(spectrum.wavelengths/balmer_edge)**3
        absorption = 1 - np.exp(-tau)
        bc_flux = absorption * blackbody
        bc_flux = self.log_conv(spectrum.wavelengths,bc_flux,lwidth/c.value)
    
        norm_index = find_nearest_index(spectrum.wavelengths, edge_wl) 
        fnorm = bc_flux[norm_index]
        bc_flux[spectrum.wavelengths > spectrum.wavelengths[norm_index]] = 0.
        bc_flux *= normalization/fnorm
        
        return bc_flux
    

    
#-----------------------------------------------------------------------------#

    
    def flux(self, spectrum=None, parameters=None):
        """
        Returns the flux for this component for a given wavelength grid
        and parameters. 
        """
        
        
        if self.BC and self.BpC:
            flux_BC = self.BC_flux(spectrum=spectrum, parameters=parameters)
            flux_BpC = self.BpC_flux(spectrum=spectrum, parameters=parameters)
            flux_est=[flux_BC[i]+flux_BpC[i] for i in range(len(flux_BpC))]

        else:
            if self.BC:
                flux_est = self.BC_flux(spectrum=spectrum, parameters=parameters)
            if self.BpC:
                flux_est = self.BpC_flux(spectrum=spectrum, parameters=parameters)
                
        return np.array(flux_est)
