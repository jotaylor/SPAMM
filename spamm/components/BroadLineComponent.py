#!/usr/bin/python

import numpy as np

from .ComponentBase import Component
from utils.runningmeanfast import runningMeanFast

def Gaussian(x, cenwave, sigma, ampl):
    '''
    x = array of wavelenengths 
    cenwave = Gaussian central wavelength
    sigma = Gaussian width
    ampl = Gaussian amplitude  

    \f$ Gauss = \frac{amplitude}{sigma \sqrt{2\pi}}e^{-\frac{1}{2} \left(\frac{x-cenwave}{sigma}\right)^2}\f$
    '''
    gauss = ampl/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-cenwave)/sigma,2))

    return gauss

def gauss_hermite(x,cenwave,sigma,amplitude,h3,h4,h5,h6)

    w=(x-cenwave)/sigma
    alpha=1./np.sqrt(2.*np.pi)*np.exp(-0.5*np.power(w,2))
    H3=(w*(2*np.power(w,2)-3))/np.sqrt(3)
    H4=(np.power(w,2)*(4.*np.power(w,2)-12.)+3.)/(2.*np.sqrt(6.))
    H5=(w*(np.power(w,2)*(4*np.power(w,2)-20)+15))/(2.*np.sqrt(15.))
    H6=(np.power(w,2)*(np.power(w,2)*(8*np.power(w,2)-60.)+90.)-15.)/(12.*np.sqrt(5))

    ghermite=amplitude*alpha/sigma*(1+h3*H3+h4*H4+h5*H5+h6*H6)

    return ghermite

class BroadLineComponent(Component):
    '''
    Broad Emission Line component
    Standard functional form for broad emission lines:  Gaussian function plus 6th order 
    Gauss-Hermite polynomial

    Gaussian function:
    \f$ F_{\lambda} = \frac{f_{\rm peak, G}}{\sigma_{\rm G} \sqrt{2\pi}}e^{-\frac{1}{2}
    \left(\frac{\lambda - \mu_{\rm G}}{\sigma_{\rm G}}\right)^2}\f$

    6th order Gauss-Hermite polynomial (van der Marel \& Franx 1993, ApJ, 407, 525):
    \f$ F_{\lambda} = [f_{\rm peak, GH} \alpha(w)/\sigma_{\rm GH}]\left(1 + \sum_{j=3}^{6}h_jH_j(w) \right) \f$
    where:
    \f$ w\equiv (\lambda - \mu_{\rm GH})/\sigma_{\rm GH} \f$
    and
    \f$ \alpha(w) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}w^2} \f$

The \f$ H_j \f$ coefficients can be found in Cappellari et al.\ (2002, ApJ, 578, 787):
\f$ H_3(w) = \frac{w(2w^2-3)}{\sqrt{3}} \f$
\f$ H_4(w) = \frac{w^2(4w^2-12)+3}{2\sqrt{6}} \f$
\f$ H_5(w) = \frac{w[w^2(4w^2-20)+15]}{2\sqrt{15}} \f$
\f$ H_6(w) = \frac{w^2[w^2(8w^2-60)+90]-15}{12\sqrt{5}} \f$

    This component has 10 parameters:

    Gaussian central wavelength: \f$ \mu_{\rm G} \f$
    Gaussian width: \f$ \sigma_{\rm G} \f$
    Gaussian amplitude : \f$ \f_{\rm peak, G} \f$
    Gauss-Hermite central wavelength: \f$ \mu_{\rm G} \f$
    Gaussian width: \f$ \sigma_{\rm G} \f$			
    Gauss-Hermite amplitude : \f$ f_{\rm peak, GH} \f$
    Gauss-Hermite moment: \f$ h_3 \f$
    Gauss-Hermite moment: \f$ h_4 \f$
    Gauss-Hermite moment: \f$ h_5 \f$
    Gauss-Hermite moment: \f$ h_6 \f$	
    '''

    def __init__(self, pars=None):
        super().__init__()

        self.model_parameter_names = list() # this may need to be defined as a method
        self.model_parameter_names.append("Gauss_cenwave")
        self.model_parameter_names.append("Gauss_width")
        self.model_parameter_names.append("Gauss_amplitude")
        self.model_parameter_names.append("GH_cenwave")
        self.model_parameter_names.append("GH_width")
        self.model_parameter_names.append("GH_amplitude")
        self.model_parameter_names.append("GH_h3")
        self.model_parameter_names.append("GH_h4")
        self.model_parameter_names.append("GH_h5")
        self.model_parameter_names.append("GH_h6")

        self.norm_wavelength = None		
        self.min_cenwave = None
        self.max_cenwave = None
        self.min_width = None
        self.max_width = None
        self.min_amplitude = None
        self.max_amplitude = None
        self.min_h = None
        self.max_h = None

    @property
    def is_analytic(self):
        """ 
        Method that stores whether component is analytic or not
        
        Returns:
            Bool (Bool): True if componenet is analytic.
        """
        return True


    def initial_values(self, spectrum):
        '''

        Needs to sample from prior distribution.
        '''

        # call super() implementation
        #super(BroadLineComponent, self).initialize()

        # [replace] calculate/define minimum and maximum values for each parameter.
        c = 299792.458 # km/s
        boxcar_width=5

        self.min_cenwave = (-6000./c+1.)*llab
        self.max_cenwave = (6000./c+1.)*llab

        Gauss_cenwave_init = np.random.uniform(low=self.min_cenwave, high=self.max_cenwave)

        GH_cenwave_init = np.random.uniform(low=self.min_cenwave, high=self.max_cenwave)

        self.min_width = 0.
        self.max_width =(12000./c+1.)*llab

        Gauss_width_init = np.random.uniform(low=self.min_width, high=self.max_width)

        GH_width_init = np.random.uniform(low=self.min_width, high=self.max_width)


        self.min_amplitude = 0.
        self.max_amplitude = max(runningMeanFast(spectrum.flux, boxcar_width))

        Gauss_amplitude_init = np.random.uniform(low=self.min_amplitude,high=self.max_amplitude)

        GH_amplitude_init = np.random.uniform(low=self.min_amplitude,high=self.max_amplitude)

        self.min_h = -0.3
        self.max_h = 0.3

        GH_h3_init=np.random.uniform(low=self.min_h,high=self.max_h)
        GH_h4_init=np.random.uniform(low=self.min_h,high=self.max_h)
        GH_h5_init=np.random.uniform(low=self.min_h,high=self.max_h)
        GH_h6_init=np.random.uniform(low=self.min_h,high=self.max_h)

        return [Gauss_cenwave_init, Gauss_width_init, Gauss_amplitude_init, \ 
        GH_cenwave_init, GH_width_init, GH_amplitude_init, GH_h3_init, GH_h4_init, \
        GH_h5_init, GH_h6_init]

    def initialize(self, data_spectrum=None):
        '''
        Perform any initializations where the data is optional.
        '''
        if data_spectrum is None:
            raise Exception("The data spectrum must be specified to initialize" + 
                                            "{0}.".format(self.__class__.__name__))
        self.normalization_wavelength(data_spectrum_wavelength=data_spectrum.spectral_axis)

    def ln_priors(self, params):
        '''
        Return a list of the ln of all of the priors.

        For each component:
        cenwave: uniform linear prior in range [-6000,6000] km/s
        width: uniform linear prior in range [0,12000] km/s
        amplitude: uniform linear prior between 0 and  and the maximum of the spectral 
                flux after computing running median

        For the Gauss-Hermite polynomial:
        h_j moments: uniform linear prior in range [-0.3,0.3]

        '''

        # need to return parameters as a list in the correct order
        ln_priors = list()

        Gauss_cenwave = params["Gauss_cenwave"]
        Gauss_width = params["Gauss_width"]
        Gauss_amplitude = params["Gauss_amplitude"]
        GH_cenwave = params["GH_cenwave"]
        GH_width = params["GH_width"]
        GH_amplitude = params["GH_amplitude"]
        GH_h3 = params["GH_h3"]
        GH_h4 = params["GH_h4"]
        GH_h5 = params["GH_h5"]
        GH_h6 = params["GH_h6"]

        if self.min_cenwave < Gauss_cenwave < self.max_cenwave:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_width < Gauss_width < self.max_width:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_amplitude < Gauss_amplitude < self.max_amplitude:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_cenwave < GH_cenwave < self.max_cenwave:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_width < GH_width < self.max_width:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_amplitude < GH_amplitude < self.max_amplitude:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_h < GH_h3 < self.max_h:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_h < GH_h4 < self.max_h:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_h < GH_h5 < self.max_h:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        if self.min_h < GH_h6 < self.max_h:
            ln_priors.append(np.log(1))
        else:
            #arbitrarily small number
            ln_priors.append(-1.e17)

        return ln_priors

    def flux(self, wavelengths=None, params=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        assert len(params) == len(self.model_parameter_names), ("The wrong number " +
                                                                "of indices were provided: {0}".format(params))

        flux = gaussian(spectrum.spectral_axis,Gauss_cenwave,Gauss_width,Gauss_amplitude)+\
        gauss_hermite(spectrum.spectral_axis,GH_cenwave,GH_width,GH_amplitude,GH_h3,GH_h4,\
        GH_h5,GH_h6)

        return flux

