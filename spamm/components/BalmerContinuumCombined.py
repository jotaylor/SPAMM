#!/usr/bin/python

import numpy as np
from .ComponentBase import Component
from astropy.constants import c, h, k_B, Ryd
from astropy import units as u
import pickle

# replacing this with astropy's blackbody function
#from astropy.modeling.blackbody import blackbody_lambda
from astropy.modeling.physical_models import BlackBody
#from utils.blackbody import blackbody

#TODO this needs to be integrated into Spectrum eventually
from utils.rebin_spec import rebin_spec
#from utils.fftwconvolve_1d import fftwconvolve_1d
from utils.find_nearest_index import find_nearest_index
from utils.parse_pars import parse_pars
#import line_profiler

# Constants are in cgs.  
c = c.cgs
c_cms = c.cgs.value
c_kms = c.to("km/s").value
h = h.cgs.value
k = k_B.cgs.value
R = Ryd.to("1/Angstrom").value
E0 = 2.179e-11
balmer_edge = 3646 # Angstroms

class BalmerCombined(Component):

    """
    Model of the combined BalmerContinuum (BC) based on Grandi et
    al. (1982) and Kovacevic & Popovic (2014). It contains two components: an analytical 
    function to describe optically thick clouds with a uniform temperature for wavelength <=3646A
    and the sum of higher order Balmer lines which merge into a pseudo-continuum for wavelength >3646A.
    The resulting flux is therefore given by the combination of these two components, here BC_flux + BpC_flux. 
    When initialising set which components to use
    """

    def __init__(self, pars=None, BalmerContinuum=False, BalmerPseudocContinuum=False):
        super().__init__()
        
        if pars is None:
            self.inputpars = parse_pars()["balmer_continuum"]
        else:
            self.inputpars = pars

        self.model_parameter_names = []
        self.name = "Balmer"
        self.coeff = pickle.load(open('../data/SH95recombcoeff/coeff.interpers.pickle','rb'), encoding="latin1")

        # parameters for the continuum
        self.model_parameter_names.append("bc_norm")
        self.model_parameter_names.append("bc_Te")
        self.model_parameter_names.append("bc_tauBE")

        # parameters for the lines
        self.model_parameter_names.append("bc_loffset")
        self.model_parameter_names.append("bc_lwidth")
        self.model_parameter_names.append("bc_logNe")
#        self.model_parameter_names.append("lscale")
        
        self._norm_wavelength =  None
        
        self.normalization_min = self.inputpars["bc_norm_min"]
        self.normalization_max = self.inputpars["bc_norm_max"]

        self.Te_min = self.inputpars["bc_Te_min"]
        self.Te_max = self.inputpars["bc_Te_max"]

        self.tauBE_min = self.inputpars["bc_tauBE_min"]
        self.tauBE_max = self.inputpars["bc_tauBE_max"]

        self.loffset_min = self.inputpars["bc_loffset_min"]
        self.loffset_max = self.inputpars["bc_loffset_max"]

        self.lwidth_min = self.inputpars["bc_lwidth_min"]
        self.lwidth_max = self.inputpars["bc_lwidth_max"]
        
        self.logNe_min = self.inputpars["bc_logNe_min"]
        self.logNe_max = self.inputpars["bc_logNe_max"]
        
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
            be_index = find_nearest_index(spectrum.spectral_axis, balmer_edge) 
            bcmax = np.max(spectrum.flux[be_index-10:be_index+10])
            self.normalization_max = bcmax
        normalization_init = np.random.uniform(low=self.normalization_min,
                               high=self.normalization_max)
        
        Te_init = np.random.uniform(low=self.Te_min,
                        high=self.Te_max)
                        
        tauBE_init = np.random.uniform(low=self.tauBE_min,
                           high=self.tauBE_max)

        loffset_init = np.random.uniform(low=self.loffset_min,
                         high=self.loffset_max)

        lwidth_init = np.random.uniform(low=self.lwidth_min,
                           high=self.lwidth_max)
                           
        logNe_init = np.random.uniform(low=self.logNe_min,
                        high=self.logNe_max)
                        
#        if self.lscale_min == None or self.lscale_max == None:
#            self.lscale_min = 0
#            self.lscale_max = 2
#        lscale_init = np.random.uniform(low=self.lscale_min,
#                        high=self.lscale_max)


        return [normalization_init, Te_init, tauBE_init, loffset_init, lwidth_init, logNe_init] #,lscale_init]

#-----------------------------------------------------------------------------#

    def ln_priors(self, params):
        """
        Calculate the natural logarithm of priors for each parameter.

        This function checks if each parameter is within its allowed range. 
        If it is, the function appends 0 (since ln(1) = 0) to the list of priors. 
        If it's not, it appends negative infinity (since ln(0) = -inf).

        Parameters:
        params (list): List of parameters to check. 
        The order is: normalization, Te, tauBE, loffset, lwidth, logNe.

        Returns:
        list: The natural logarithm of the prior for each parameter.
        """
        # Initialize an empty list to store the natural logarithm of priors
        ln_priors = list()

        # Extract the parameters from the input list using their respective indices
        normalization = params[self.parameter_index("bc_norm")]
        Te            = params[self.parameter_index("bc_Te")]
        tauBE         = params[self.parameter_index("bc_tauBE")]
        loffset       = params[self.parameter_index("bc_loffset")]
        lwidth        = params[self.parameter_index("bc_lwidth")]
        logNe         = params[self.parameter_index("bc_logNe")]
#        lscale        = params[self.parameter_index("lscale")]
        
        # Check each parameter against its allowed range
        # If a parameter is within its range, append 0 to ln_priors (since ln(1) = 0)
        # If a parameter is outside its range, append -inf to ln_priors (since ln(0) = -inf)
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

        # Return the list of natural logarithm of priors
        return ln_priors
        
#-----------------------------------------------------------------------------#
    
    #TODO add convolution function to utils
    
    def log_conv(self, wave, flux, line_width):
        """
        Perform convolution in log space. 

        This method convolves the input spectrum with a Gaussian kernel in logarithmic wavelength space. 
        This is equivalent to performing the convolution in velocity space with a kernel that has a 
        constant width in terms of velocity.
        
        Args:
            wave (numpy.ndarray): Wavelength array of the original spectrum.
            flux (numpy.ndarray): Flux array of the spectrum before convolution.
            line_width (float): Desired width of the lines in units of v/c.
        
        Returns:
            rebinned_flux (numpy.ndarray): Convolved spectrum rebinned back to the original wavelength array.
        """
        # Convert the wavelength to logarithmic scale
        ln_wave = np.log(wave)
        
        # Create a new logarithmic wavelength array with the same range but uniform grid
        ln_wave_uniform = np.linspace(ln_wave.min(), ln_wave.max(), ln_wave.size)
        ln_wave_uniform[0] = ln_wave[0]
        ln_wave_uniform[-1] = ln_wave[-1]
        
        # Rebin the original spectrum onto the new logarithmic wavelength grid
        if self.fast_interp:
            rebinned_flux = np.interp(ln_wave_uniform, ln_wave, flux)
        else:
            rebinned_flux = rebin_spec(ln_wave_uniform, ln_wave, flux)
        
        # Calculate the width of the Gaussian kernel in terms of the new grid spacing
        dpix = line_width/(ln_wave_uniform[1] - ln_wave_uniform[0])
        
        # Create the Gaussian kernel over a range of -5*dpix to 5*dpix
        kernel_width = round(5*dpix)
        kernel_x = np.arange(-kernel_width, kernel_width+1)
        kernel_y = np.exp(-(kernel_x)**2 / (dpix)**2)
        
        # Normalize the kernel so that its total area is 1
        kernel_y /= abs(np.sum(kernel_y))
        
        # Convolve the rebinned spectrum with the kernel
        flux_conv = np.convolve(rebinned_flux, kernel_y, mode='same')
        assert flux_conv.size == wave.size

        # Rebin the convolved spectrum back to the original wavelength grid
        if self.fast_interp:
            rebinned_flux_conv = np.interp(wave, np.exp(ln_wave_uniform), flux_conv)
        else:
            rebinned_flux_conv = rebin_spec(wave, np.exp(ln_wave_uniform), flux_conv)
        
        # Return the convolved spectrum
        return rebinned_flux_conv
        
#-----------------------------------------------------------------------------#
    
    def balmerseries(self, line_orders):
        """
        Calculate a Balmer series line wavelength [Angstroms]
        
        Args:
            line_orders (int or array of ints): Quantum number of the electron.
    
        Returns:
            float (float): Wavelength of the transition from n=n to n=2.
        """
        
        ilambda = R * (0.25 - 1./line_orders**2)
        return 1. / ilambda
    
    
#-----------------------------------------------------------------------------#
    
    def balmer_ratio(self, n_e, line_orders, T): # 
        """
        Calculate the ratio between Balmer lines based on electron density and temperature.

        This function estimates the relative intensity values for lines 3-400 using results
        from Storey and Hummer 1995 and Kovacevic et al 2014. The ratios are calculated for
        a given electron density and temperature, and for a specified range of Balmer lines.

        Args:
            n_e (float): Electron density in cm^-3.
            T (float): Electron temperature in Kelvin.
            line_orders (numpy array): Array of integers representing the Balmer lines to include. For example, an array [3, 4, 5] would represent H-gamma, H-delta, and H-epsilon.

        Returns:
            flux_ratios (numpy array): Array of calculated Balmer line ratios. The ratios are ordered from Htheta (N=10 -> N=2) first to the maximum line order specified in the input.
        """
        maxline = int(line_orders.max())
        minline = int(line_orders.min())
        flux_ratios = np.zeros(maxline - minline + 1)
        n = np.arange(minline, maxline+1)
    
        coef_use = [coef_interp(n_e, T) for coef_interp in self.coeff] 

        for i in n:
            if i <=50:
                flux_ratios[i - minline] = coef_use[-i+2]
            else:
                flux_ratios[i - minline] = flux_ratios[i-minline-1]*np.exp(E0*(1./i**2-1./(i-1)**2)/(k*T))
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

        line_orders = np.arange(self.inputpars["bc_lines_min"], self.inputpars["bc_lines_max"]) 
        lcenter = self.balmerseries(line_orders)
    
        lcenter -= shift*lcenter
        lcenter_reshaped = lcenter.reshape(lcenter.size, 1)
        LL = sp_wavel - lcenter_reshaped #(is this simply x-x0)
        lwidth = width * lcenter_reshaped
        ltype = self.inputpars["bc_line_type"]

        if ltype == "gaussian":
            lines = np.exp(-LL**2 / lwidth**2)
        elif ltype == "lorentzian":
            lines = lwidth / (LL**2 + lwidth**2)
        else:
            raise ValueError(f"Variable 'ltype' ({ltype}) must be 'gaussian' or 'lorentzian'")
    
        lflux = self.balmer_ratio(n_e, line_orders, T)

        #scale = np.repeat(lflux.reshape(lflux.size,1), lines.shape[1], axis=1)
        scale = lflux[:, np.newaxis]

        lines *= scale
    
        balmer_lines = np.sum(lines,axis = 0)
    
        return balmer_lines
    

       
#-----------------------------------------------------------------------------#
    
    def BpC_flux(self, spectrum=None, params=None):
        """
        Analytic model of the high-order Balmer lines, making up the Pseudo continuum near 3666 A.
    
        Line profiles are Gaussians (for now).  The line ratios are fixed to Storey &^ Hummer 1995,
        case B, n_e = 10^10 cm^-3.
        
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
        normalization = params[self.parameter_index("bc_norm")]
        Te            = params[self.parameter_index("bc_Te")]
        tauBE         = params[self.parameter_index("bc_tauBE")]
        loffset       = params[self.parameter_index("bc_loffset")]
        lwidth        = params[self.parameter_index("bc_lwidth")]
        logNe         = params[self.parameter_index("bc_logNe")]
#        lscale         = params[self.parameter_index("lscale")]
    
        #c_kms = c.to("km/s")
        edge_wl = balmer_edge*(1 - loffset/c_kms)
        
        n_e =10.**logNe
        bpc_flux = self.makelines(spectrum.spectral_axis, 
                                  Te,
                                  n_e, 
                                  loffset/c_kms, 
                                  lwidth/c_kms)
                 
        
        norm_index = find_nearest_index(spectrum.spectral_axis, edge_wl) 
        
        fnorm = bpc_flux[norm_index]
        bpc_flux[spectrum.spectral_axis <= spectrum.spectral_axis[norm_index]] = 0
        bpc_flux *= normalization/fnorm
#        bpc_flux *= lscale
    
        return bpc_flux
    
#-----------------------------------------------------------------------------#
    
    def BC_flux(self, spectrum=None, params=None):
        """
        Analytic model of the BalmerContinuum (BC) based on Grandi et
        al. (1982) and Kovacevic & Popovic (2013).
    
        F(lambda) = F(3646 A) * B(T_e) * (1 - e^-tau(lambda))
        tau(lambda) = tau(3646) * (lambda/3646)^3
    
        This component has 5 parameters:
    
        parameter1 : The flux normalization at the Balmer Edge lambda = 3646 A, F(3646)
        parameter2 : The electron temperture T_e for the Planck function B(T_e)
        parameter3 : The optical depth at the Balmer edge, tau(3646)
        parameter4 : A shift of the line centroids (km/s)
        parameter5 : The width of the Gaussians (km/s)
    
    
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
        normalization = params[self.parameter_index("bc_norm")]
        Te            = params[self.parameter_index("bc_Te")]
        tauBE         = params[self.parameter_index("bc_tauBE")]
        loffset       = params[self.parameter_index("bc_loffset")]
        lwidth        = params[self.parameter_index("bc_lwidth")]
        #logNe         = params[self.parameter_index("bc_logNe")]
        
        #c_kms = c.to("km/s")

        # Wavelength at which Balmer components merge.
        edge_wl = balmer_edge*(1 - loffset/c_kms)

    #TODO need WL as quantity objects for better astropy functionality
        # Create a blackbody model with the temperature Te
        blackbody_model = BlackBody(temperature=Te*u.K)

        # Evaluate the model at the wavelengths in spectrum.spectral_axis
        blackbody = blackbody_model(spectrum.spectral_axis)

        # Calculates [1 - e^(-tau)] (optically-thin emitting slab)
        # Assumes angstroms
        tau = tauBE*(spectrum.spectral_axis / balmer_edge)**3
        absorption = 1 - np.exp(-tau)
        
        bc_flux = absorption * blackbody
        bc_flux = self.log_conv(spectrum.spectral_axis, bc_flux, lwidth/c_kms)
    
        norm_index = find_nearest_index(spectrum.spectral_axis, edge_wl) 
        fnorm = bc_flux[norm_index]
        bc_flux[spectrum.spectral_axis > spectrum.spectral_axis[norm_index]] = 0.
        bc_flux *= normalization/fnorm
        
        return bc_flux
    

    
#-----------------------------------------------------------------------------#

    
    def flux(self, spectrum=None, params=None):
        """
        Returns the flux for this component for a given wavelength grid
        and parameters. 
        """
        if self.BC and self.BpC:
            #start_time = timeit.default_timer()
            flux_BC = self.BC_flux(spectrum=spectrum, params=params)
            #end_time = timeit.default_timer()
            #print(f"Execution time (BC): {end_time - start_time} seconds")

            #start_time = timeit.default_timer()
            flux_BpC = self.BpC_flux(spectrum=spectrum, params=params)
            #end_time = timeit.default_timer()
            #print(f"Execution tim (BpC): {end_time - start_time} seconds")

            flux_est = flux_BC + flux_BpC

        else:
            if self.BC:
                flux_est = self.BC_flux(spectrum=spectrum, params=params)
            if self.BpC:
                flux_est = self.BpC_flux(spectrum=spectrum, params=params)
                
        return np.array(flux_est)
