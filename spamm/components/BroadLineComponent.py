#!/usr/bin/python

# To create a new component class, copy this file and fill in the values as instructed.
# Follow comments the begin with "[replace]", then delete the comment.

import sys
import numpy as np
from scipy.optimize import curve_fit

from .ComponentBase import Component
from utils.runningmeanfast import runningMeanFast
from utils.parse_pars import parse_pars

PARS = parse_pars()["emission_lines"]

def gauss_hermite(x,cenwave,sigma,amplitude,h3,h4,h5,h6, type="6thOrder"):

    w=(x-cenwave)/sigma
    alpha=np.exp(-0.5*np.power(w,2))
    if type == "Gauss":
        ghermite = amplitude*alpha/sigma
    H3=(w*(2*np.power(w,2)-3))/np.sqrt(3)
    H4=(np.power(w,2)*(4.*np.power(w,2)-12.)+3.)/(2.*np.sqrt(6.))
    if type=="4thOrder":
        ghermite = amplitude*alpha/sigma*(1+h3*H3+h4*H4)
    H5=(w*(np.power(w,2)*(4*np.power(w,2)-20)+15))/(2.*np.sqrt(15.))
    H6=(np.power(w,2)*(np.power(w,2)*(8*np.power(w,2)-60.)+90.)-15.)/(12.*np.sqrt(5))
    if type=="6thOrder":
        ghermite=amplitude*alpha*(1+h3*H3+h4*H4+h5*H5+h6*H6)

    return ghermite

def line_fit_prelim(x,a,b,G_cenwave,G_sigma,G_amplitude,GH_cenwave,GH_sigma,GH_amplitude,h3,h4,h5,h6):
    linear = a+b*x#
    nl = gauss_hermite(x,G_cenwave,G_sigma,G_amplitude,0,0,0,0, type="Gauss")
    bl = gauss_hermite(x,GH_cenwave,GH_sigma,GH_amplitude,h3,h4,h5,h6, type="6th Order")
    return linear+nl+bl

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

    def __init__(self,type = "6thOrder", llab=None):
        super(BroadLineComponent, self).__init__()

        self.type = type
        self.llab = llab
        self.model_parameter_names = list() # this may need to be defined as a method
        self.model_parameter_names.append("Gauss_cenwave")
        self.model_parameter_names.append("Gauss_width")
        self.model_parameter_names.append("Gauss_amplitude")
        self.model_parameter_names.append("GH_cenwave")
        self.model_parameter_names.append("GH_width")
        self.model_parameter_names.append("GH_amplitude")
        if type == "6thOrder" or type=="4thOrder":
            self.model_parameter_names.append("GH_h3")
            self.model_parameter_names.append("GH_h4")
            if type == "6thOrder":
                self.model_parameter_names.append("GH_h5")
                self.model_parameter_names.append("GH_h6")

        self.norm_wavelength=None
        self.min_cenwave = None
        self.max_cenwave = None
        self.min_width = None
        self.max_width = None
        self.min_amplitude = None
        self.max_amplitude = None
        self.min_h = None
        self.max_h = None
        self.name = "BroadLine"

    @property
    def is_analytic(self):
        return True


    def initial_values(self, spectrum=None):
        '''

        Needs to sample from prior distribution.
        '''

        # call super() implementation
        super(BroadLineComponent, self).initialize()

        # [replace] calculate/define minimum and maximum values for each parameter.
        light_speed=299792.458 #km/s
        boxcar_width=5

        self.min_cenwave =(-PARS["el_shift"]/light_speed+1.)*self.llab
        self.max_cenwave =(PARS["el_shift"]/light_speed+1.)*self.llab

        Gauss_cenwave_init = np.random.uniform(low=self.min_cenwave, high=self.max_cenwave)

        GH_cenwave_init = np.random.uniform(low=self.min_cenwave, high=self.max_cenwave)

        self.nl_min_width = (PARS["el_narrow_min"]/light_speed)*self.llab
        self.nl_max_width =(PARS["el_narrow_max"]/light_speed)*self.llab

        Gauss_width_init = np.random.uniform(low=self.nl_min_width, high=self.nl_max_width)

        self.bl_min_width = (PARS["el_broad_min"]/light_speed)*self.llab
        self.bl_max_width =(PARS["el_broad_max"]/light_speed)*self.llab

        GH_width_init = np.random.uniform(low=self.bl_min_width, high=self.bl_max_width)


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

        if PARS["el_prefit"] == True:
            window_min = (-PARS["el_prefit_window"]/light_speed+1.)*self.llab
            window_max = (PARS["el_prefit_window"]/light_speed+1.)*self.llab
            indexselect_line = np.where((spectrum.wavelengths>window_min) & (spectrum.wavelengths<window_max) & np.isfinite(spectrum.flux) )
            p0 = [np.nanmedian(spectrum.flux),0,self.llab,1,1,self.llab,1,1,0,0,0,0]
            print('p0',p0)
            bounds_min = [-np.inf,-np.inf, self.min_cenwave,self.nl_min_width,self.min_amplitude,self.min_cenwave,self.bl_min_width,self.min_amplitude,self.min_h,self.min_h,self.min_h,self.min_h]
            bounds_max = [np.inf,np.inf,self.max_cenwave,self.nl_max_width,self.max_amplitude,self.max_cenwave,self.bl_max_width,self.max_amplitude,self.max_h,self.max_h,self.max_h,self.max_h]
            print('bounds_min',bounds_min)
            print('bounds_max',bounds_max)
            popt_amp_A, pcov_amp_A = curve_fit(line_fit_prelim,spectrum.wavelengths[indexselect_line],spectrum.flux[indexselect_line],p0=p0,bounds=[bounds_min,bounds_max])
            a,b,Gauss_cenwave_init,Gauss_width_init,Gauss_amplitude_init,GH_cenwave_init,GH_width_init,GH_amplitude_init,GH_h3_init,GH_h4_init,GH_h5_init,GH_h6_init  =  np.random.multivariate_normal(popt_amp_A,pcov_amp_A,1).T


        if self.type == "Gauss":
            return [Gauss_cenwave_init, Gauss_width_init, Gauss_amplitude_init, \
            GH_cenwave_init, GH_width_init, GH_amplitude_init]

        if self.type == "4thOrder":
            return [Gauss_cenwave_init, Gauss_width_init, Gauss_amplitude_init, \
            GH_cenwave_init, GH_width_init, GH_amplitude_init, GH_h3_init, GH_h4_init]

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

        Gauss_cenwave = params[self.parameter_index("Gauss_cenwave")]
        Gauss_width = params[self.parameter_index("Gauss_width")]
        Gauss_amplitude = params[self.parameter_index("Gauss_amplitude")]
        GH_cenwave = params[self.parameter_index("GH_cenwave")]
        GH_width = params[self.parameter_index("GH_width")]
        GH_amplitude = params[self.parameter_index("GH_amplitude")]
        if self.type == "6thOrder" or self.type=="4thOrder":
            GH_h3 = params[self.parameter_index("GH_h3")]
            GH_h4 = params[self.parameter_index("GH_h4")]
            if self.type == "6thOrder":
                GH_h5 = params[self.parameter_index("GH_h5")]
                GH_h6 = params[self.parameter_index("GH_h6")]


        if self.min_cenwave < Gauss_cenwave < self.max_cenwave:
            ln_priors.append(0)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)

        if self.nl_min_width < Gauss_width < self.nl_max_width:
            ln_priors.append(0)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)

        if self.min_amplitude < Gauss_amplitude < self.max_amplitude:
            ln_priors.append(0)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)

        if self.min_cenwave < GH_cenwave < self.max_cenwave:
            ln_priors.append(0)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)

        if self.bl_min_width < GH_width < self.bl_max_width:
            ln_priors.append(0)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)

        if self.min_amplitude < GH_amplitude < self.max_amplitude:
            ln_priors.append(0)
        else:
            #arbitrarily small number
            ln_priors.append(-np.inf)
        if self.type == "6thOrder" or self.type=="4thOrder":
            if self.min_h < GH_h3 < self.max_h:
                ln_priors.append(0)
            else:
                #arbitrarily small number
                ln_priors.append(-np.inf)

            if self.min_h < GH_h4 < self.max_h:
                ln_priors.append(0)
            else:
                    #arbitrarily small number
                ln_priors.append(-np.inf)
        if self.type == "6thOrder":
            if self.min_h < GH_h5 < self.max_h:
                ln_priors.append(0)
            else:
                #arbitrarily small number
                ln_priors.append(-np.inf)

            if self.min_h < GH_h6 < self.max_h:
                ln_priors.append(0)
            else:
                #arbitrarily small number
                ln_priors.append(-np.inf)

        return ln_priors

    def flux(self, spectrum=None, parameters=None):
        '''
        Returns the flux for this component for a given wavelength grid
        and parameters. Will use the initial parameters if none are specified.
        '''
        assert len(parameters) == len(self.model_parameter_names), ("The wrong number " +
                                                                "of indices were provided: {0}".format(parameters))
        Gauss_cenwave = parameters[self.parameter_index("Gauss_cenwave")]
        Gauss_width = parameters[self.parameter_index("Gauss_width")]
        Gauss_amplitude = parameters[self.parameter_index("Gauss_amplitude")]
        GH_cenwave = parameters[self.parameter_index("GH_cenwave")]
        GH_width = parameters[self.parameter_index("GH_width")]
        GH_amplitude = parameters[self.parameter_index("GH_amplitude")]
        if self.type == "6thOrder" or type=="4thOrder":
            GH_h3 = parameters[self.parameter_index("GH_h3")]
            GH_h4 = parameters[self.parameter_index("GH_h4")]
            if self.type == "6thOrder":
                GH_h5 = parameters[self.parameter_index("GH_h5")]
                GH_h6 = parameters[self.parameter_index("GH_h6")]
            else:
                GH_h5 = 0.
                GH_h6 = 0.
        else:
            GH_h3 = 0.0
            GH_h4 = 0.0
            GH_h5 = 0.
            GH_h6 = 0.

        flux = gauss_hermite(spectrum.wavelengths,Gauss_cenwave,Gauss_width,Gauss_amplitude,0,0,0,0,type="Gauss")+\
        gauss_hermite(spectrum.wavelengths,GH_cenwave,GH_width,GH_amplitude,GH_h3,GH_h4,\
        GH_h5,GH_h6,type=self.type)

        return flux
