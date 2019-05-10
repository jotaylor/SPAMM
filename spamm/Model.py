#!/usr/bin/python

import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy import units as u

import emcee

from .Spectrum import Spectrum

iteration_count = 0

#-----------------------------------------------------------------------------#

class MCMCDidNotConverge(Exception):
    pass

#-----------------------------------------------------------------------------#

# TODO arg pos is not even used vv
def sort_on_runtime(pos):
    """
    Description.

    Args:
        pos (): ?? 

    Returns:
        ndarray (ndarray): ?
        ndarray (ndarray): ? 
    """

    p = np.atleast_2d(p)
    idx = np.argsort(p[:, 0])[::-1]
    
    return p[idx], idx

#-----------------------------------------------------------------------------#

def ln_posterior(new_params, *args):
    """
    Return the logarithm of the posterior function, to be passed to the emcee 
    sampler.

    Args:
        new_params (ndarray): Array in the parameter space used as input 
            into sampler.
        args: Additional arguments passed to this function 
            (i.e. the Model object).

    Returns:
        list (list): List of model likelihoods and priors.
    """

    global iteration_count
    iteration_count = iteration_count + 1
    if iteration_count % 20 == 0:
        print("iteration count: {0}".format(iteration_count))

    # Make sure "model" is passed in - this needs access to the Model object
    # since it contains all of the information about the components.
    model = args[0] # TODO: return an error if this is not the case

    # Calculate the log prior.
    ln_prior = model.prior(params=new_params)
    if not np.isfinite(ln_prior):
        return -np.inf
    
    # Only calculate flux and therefore likelihood if parameters lie within 
    # bounds of priors to save computation time.
    # Compare the model spectrum to the data, generate model spectrum given 
    # model parameters, and calculate the log likelihood.
    else:    
        model_spectrum_flux = model.model_flux(params=new_params)
        ln_likelihood = model.likelihood(model_spectrum_flux=model_spectrum_flux)
    
        return ln_likelihood + ln_prior 

#-----------------------------------------------------------------------------#

#TODO do we need all these commented attributes??
class Model(object):
    """
    A class to describe model objects..

    Attributes:
        _mask ():
        _data_spectrum():
        z ():
        components ():
        mpi (): 
        sampler ():
        model_spectrum (Spectrum object): 
        downsample_data_if_needed (Bool):
        upsample_components_if_needed (Bool):
        print_parameters (Bool): Used for debugging.
    """
    
    def __init__(self, wavelength_start=1000, wavelength_end=10000, 
                 wavelength_delta=0.05, mpi=False):
        """
        Args:
            wavelength_start (): 
            wavelength_end (): 
            wavelength_delta (float):
            mpi (Bool):
        """

        self._mask = None
        self._data_spectrum = None
        
        self.z = None
        self.components = []
        self.mpi = mpi

        self.sampler = None
        #self.sampler_output = None

        wl_init = np.arange(wavelength_start, wavelength_end, wavelength_delta)
        self.model_spectrum = Spectrum(spectral_axis = wl_init,
                                       flux = np.zeros(len(wl_init)),
                                       flux_error=np.zeros(len(wl_init)))

        # Flag to allow Model to interpolate components' wavelength grid to 
        # match data if component grid is more course than data.
        # TODO - document better!
        self.downsample_data_if_needed = False
        self.upsample_components_if_needed = False

        self.print_parameters = False

# TODO is this needed? vvvv

#        self.reddening = None
#        self.model_parameters = {}
#        self.mcmc_param_vector = None
#    @property
#    def mask(self):
#        '''
#
#        '''
#        if self.data_spectrum is None:
#            print("Attempting to read the bad pixel mask before a spectrum was defined.")
#            sys.exit(1)
#        if self._mask is None:
#            self._mask = np.ones(len(self.data_spectrum.spectral_axis))
#
#        return self._mask
#
#    @mask.setter
#    def mask(self, new_mask):
#        '''
#        Document me.
#
#        :params mask: A numpy array representing the mask.
#        '''
#        self._mask = new_mask

#-----------------------------------------------------------------------------#

    @property
    def data_spectrum(self):
        """
        All components of the model must be set before setting the data.

        Returns:
            _data_spectrum (Spectrum object): ?
        """

        return self._data_spectrum

#-----------------------------------------------------------------------------#

    @data_spectrum.setter
    def data_spectrum(self, new_data_spectrum):
        """
        Args:
            new_data_spectrum (Spectrum object): ?
        """

        self._data_spectrum = new_data_spectrum

        if len(self.components) == 0:
            raise Exception("Components must be added before defining the data spectrum.")

        # The data spectrum defines the model wavelength grid.
        self.model_spectrum.spectral_axis = np.array(new_data_spectrum.spectral_axis)
        self.model_spectrum.flux = np.zeros(len(self.model_spectrum.spectral_axis))

        # Check that all components are on the same wavelength grid.
        # If they are not, *and* the flag to interpolate them has been set, 
        # *and* they are not more coarse than the data, interpolate. 
        # If not, fail.
        need_to_downsample_data = False
        components_to_upsample = {}

        gs = 0 # grid spacing
        worst_component = None # holds component with most course wavelength grid spacing

        for component in self.components:
            component.initialize(data_spectrum=new_data_spectrum)
             
            if component.grid_spacing() and component.grid_spacing() > gs:
                gs = component.grid_spacing()
                worst_component = component

        if gs > new_data_spectrum.grid_spacing():

            # The code will interpolate to the data anyway,
            # AND the user has allowed this for coursely sampled components
            # to be upsampled to the data. This was done above.
            if self.upsample_components_if_needed:
                pass
            
            # We will downsample the data to the "worst" component. The 
            # resulting grid will be different than the input data.
            elif self.downsample_data_if_needed:
                downsampled_spectrum = new_data_spectrum.copy()
                downsampled_spectrum.spectral_axis = np.arange(new_data_spectrum[0], 
                                                            new_data_spectrum[-1], 
                                                            gs)
                downsampled_spectrum.flux = interp1d(x=downsampled_spectrum.spectral_axis,
                                                     y=new_data_spectrum.flux,
                                                     kind="linear")
                self.model_spectrum.spectral_axis = np.array(downsampled_spectrum.spectral_axis)

                # Reinitialize all components with new data.
                for component in self.components:
                    component.initialize(data_spectrum=downsampled_spectrum)

            else:
                # TODO WHY IS THIS AN ASSERT
                assert True, (
                "The component '{0}' has courser wavelength grid spacing ".
                format(worst_component) + "than the data. Either increase the "
                "spacing of the component or use one of the flags on the "
                "Model class ('upsample_components_if_needed', "
                "'downsample_data_if_needed') to override this.")

#-----------------------------------------------------------------------------#

    def run_mcmc(self, n_walkers=100, n_iterations=100):
        """
        Run emcee MCMC.
    
        Args:
            n_walkers (int): Number of walkers to pass to the MCMC.
            n_iteratins (int): Number of iterations to pass to the MCMC. 
        """

        # Initialize walker matrix with initial parameters
        walkers_matrix = [] # must be a list, not an np.array
        for walker in range(n_walkers):
            walker_params = []
            for component in self.components:
                walker_params = walker_params + component.initial_values(self.data_spectrum)
            walkers_matrix.append(walker_params)

        global iteration_count
        iteration_count = 0

        # Create MCMC sampler. To enable multiproccessing, set threads > 1.
        # If using multiprocessing, the "lnpostfn" and "args" parameters 
        # must be pickleable.
        if self.mpi:
            # Initialize the multiprocessing pool object.
            from emcee.utils import MPIPool
            pool = MPIPool(loadbalance=True)
            if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
            self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, 
                                                 dim=len(walkers_matrix[0]),
                                                 lnpostfn=ln_posterior, 
                                                 args=[self], pool=pool,
                                                 runtime_sortingfn=sort_on_runtime)
            self.sampler.run_mcmc(walkers_matrix, n_iterations)
            pool.close()

        else:
            self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, 
                                                 dim=len(walkers_matrix[0]),
                                                 lnpostfn=ln_posterior, args=[self],
                                                 threads=1)
        
        #self.sampler_output = self.sampler.run_mcmc(walkers_matrix, n_iterations)
        self.sampler.run_mcmc(walkers_matrix, n_iterations)

#-----------------------------------------------------------------------------#

# TODO should there be a getter without a setter? vv
    @property
    def total_parameter_count(self):
        """
        Return the total number of parameters of all components.
        
        Returns:
            total_no_parameters (int): Total number of parameters for 
                all components.
        """
        
        total_no_parameters = 0
        for c in self.components:
            total_no_parameters += c.parameter_count
        
        return total_no_parameters

#-----------------------------------------------------------------------------#

    def parameter_vector(self):
        """
        Return the names? of all component parameters..

        Returns:
            param_vector (list): List of all component parameter names?
        """

        param_vector = []
        for component in self.components:
            param_vector.append(component.parameters())

        return param_vector

#-----------------------------------------------------------------------------#

    def model_flux(self, params):
        """
        Given the parameters in the model, generate a spectrum. This method is
        called by multiple MCMC walkers at the same time, so edit with caution.

        Args:
            params (ndarray): 1D numpy array of all parameters of all 
                components of the model.

        Returns:
            self.model_spectrum.flux (ndarray): Array of flux values?
        """

        # Combine all components into a single spectrum. Build an array of
        # parameters to pass to emcee. 
        if self.print_parameters:
            print("params = {0}".format(params))

        # Make a copy since we'll delete elements.
        # Note: np.copy does a deepcopy
        params2 = np.copy(params)

        self.model_spectrum.flux = np.zeros(len(self.model_spectrum.spectral_axis))

        # Extract parameters from full array for each component.
        for component in self.components:
            p = params2[0:component.parameter_count]

            # Add the flux of each component to the model spectrum, 
            # except for extinction
            if component.name != "Extinction":
                self.add_component(component=component, parameters=p)
            else:
                self.reddening(component=component, parameters=p)

            # Remove the parameters for this component from the list
            params2 = params2[component.parameter_count:]

        return self.model_spectrum.flux

#-----------------------------------------------------------------------------#

    def add_component(self, component, parameters):
        """
        Add the specified component's flux to the model's flux.
            
        Args:
            component (Component Object): Component to add to model.
            parameters (): ?
        """

        component_flux = component.flux(spectrum=self.data_spectrum, parameters=parameters)
        self.model_spectrum.flux += component_flux

#-----------------------------------------------------------------------------#

    def reddening(self, component, parameters):
        """
        Apply reddening to the model spectrum flux.
    
        Args:
            component (Component Object): Component to add to model.
            parameters (): ?
        """

        extinction = component.extinction(spectrum=self.data_spectrum, params=parameters)
        extinct_spectra= np.array(self.model_spectrum.flux)*extinction
        self.model_spectrum.flux = extinct_spectra

        #for j in range(len(self.data_spectrum.spectral_axis)):
        #    self.model_spectrum.flux[j] *= extinction[j]

#-----------------------------------------------------------------------------#

    def model_parameter_names(self):
        """
        Return a list of all component parameter names.

        Returns:
            labels (list): List of component names.
        """

        labels = []
        for c in self.components:
            labels = labels + [x for x in c.model_parameter_names]
        return labels

#-----------------------------------------------------------------------------#

# TODO model_spectrum_flux is not used vvv
    def likelihood(self, model_spectrum_flux):
        """
        Calculate the ln(likelihood) of the given model spectrum.
        The model is interpolated over the data wavelength grid.
        ln(L) = -0.5 \sum_n {\left[ \frac{(flux_{Obs}-flux_{model})^2}{\sigma^2} 
            + ln(2 \pi \sigma^2) \right]}
    
        Args:
            model_spectrum_flux (): The model spectrum, a numpy array of flux value.
    
        Returns:
            ln_l (float): Sum of ln(likelihood) values?
        """

        # Create an interpolation function.
        f = interp1d(self.model_spectrum.spectral_axis,
                                 self.model_spectrum.flux)

        # It is much more efficient to not use a for loop here.
        # interp_model_flux = [f(x) for x in self.data_spectrum.wavelength]
        interp_model_flux = f(self.data_spectrum.spectral_axis)

        ln_l = np.power(( (self.data_spectrum.flux - interp_model_flux) / self.data_spectrum.flux_error), 2) + np.log(2 * np.pi * np.power(self.data_spectrum.flux_error, 2))
        #ln_l *= self.mask
        ln_l = -0.5 * np.sum(ln_l)

        return ln_l

#-----------------------------------------------------------------------------#

    def prior(self, params):
        """
        Calculate the ln(priors) for all components in the model.
    
        Args:
            params (): ? 

        Returns:
            ln_p (float): Sum of ln(prior) values?
        """

        # Make a copy since we'll delete elements.
        p = np.copy(params)

        ln_p = 0
        for component in self.components:
            ln_p += sum(component.ln_priors(params=p[0:component.parameter_count]))

            # Remove the parameters for this component from the list.
            p = p[component.parameter_count:]

        return ln_p
