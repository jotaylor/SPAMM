#!/usr/bin/env python

'''
This script is a copy of "simply_run.py", but moves the sampler out of the Model
in an attempt to make the multiprocessing capability of emcee work.

This works, but the result is considerably slower. This stands as a proof of
concept at the moment, but should not be used.
'''

import os
import sys
import gzip
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as pl

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent

# TODO: astropy units for spectrum

# -----------------------------------------------------------------
# This block of code tells Python to drop into the debugger
# if there is an uncaught exception when run from the command line.
def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.pm()
sys.excepthook = info
# -----------------------------------------------------------------

n_walkers = 30
n_iterations = 800
show_plots = False

# ----------------
# Read in spectrum
# ----------------

# ------------
# Initialize model
# ------------
global model
model = Model()
model.print_parameters = False

# -----------------
# Initialize components
# -----------------
if False:
    nuclear_comp = NuclearContinuumComponent()

    datafile = "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat"
    #datafile = "../Data/FakeData/PLcompOnly/fakepowlaw2_werr.dat"
    wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
    spectrum = Spectrum()
    spectrum.wavelengths = wavelengths
    spectrum.flux = flux
    spectrum.flux_error = flux_err

    model.components.append(nuclear_comp)

if True:
    host_galaxy_comp = HostGalaxyComponent()

    datafile = "../Data/FakeData/for_gisella/fake_host_spectrum.dat"
    wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
    spectrum = Spectrum()
    spectrum.wavelengths = wavelengths
    spectrum.flux = flux
    spectrum.flux_error = flux_err

    model.components.append(host_galaxy_comp)

model.data_spectrum = spectrum # add data

# ------------
# Run MCMC
# ------------
if False:
    model.run_mcmc(n_walkers=n_walkers, n_iterations=n_iterations)
    sampler = Model.sampler

if True:

    import emcee

    global iteration_count
    iteration_count = 0

    def ln_posterior(new_params, *args):
        '''
        The logarithm of the posterior function -- to be passed to the emcee sampler.

        :param new_params: A 1D numpy array in the parameter space used as input into sampler.
        :param args: Additional arguments passed to this function (i.e. the Model object).
        '''
        global iteration_count
        iteration_count = iteration_count + 1
        if iteration_count % 2000 == 0:
            print("iteration count: {0} model id: {1}".format(iteration_count, hex(id(args[0]))))

        # Make sure "model" is passed in - this needs access to the Model object
        # since it contains all of the information about the components.
        global model
        #model = args[0] # TODO: return an error if this is not the case

        # generate model spectrum given model parameters
        model_spectrum_flux = model.model_flux(params=new_params)

        # calculate the log likelihood
        # ----------------------------
        # - compare the model spectrum to the data
        ln_likelihood = model.likelihood(model_spectrum_flux=model_spectrum_flux)

        # calculate the log prior
        # -----------------------	
        ln_prior = model.prior(params=new_params)

        return ln_likelihood + ln_prior # adding two lists

    # initialize walker matrix with initial parameters
    walkers_matrix = list() # must be a list, not an np.array
    for walker in range(n_walkers):
        walker_params = list()
        for component in model.components:
            walker_params = walker_params + component.initial_values(model.data_spectrum)
        walkers_matrix.append(walker_params)

    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=len(walkers_matrix[0]),
                                                                     lnpostfn=ln_posterior, args=[model],
                                                                     threads=4)

    # run!
    #self.sampler_output = self.sampler.run_mcmc(walkers_matrix, n_iterations)
    sampler.run_mcmc(walkers_matrix, n_iterations)



print(("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))))


# ------------
# Analyze & Plot results
# ------------
#discard the first 50 steps (burn in) - keep at 50!!
#Flattens the chain to have a flat list of samples
samples = sampler.chain[:, 50:, :].reshape((-1, model.total_parameter_count))

# save chains + model
with gzip.open('model.pickle.gz', 'wb') as model_output:
    model_output.write(pickle.dumps(model))

# TODO: move code below to another file that reads the pickled output, 
# e.g. one that plots the chains, one that generates the triangle plot,
# some combination, etc.

# Save the samples into a text file to be read by other codes
np.savetxt("samples.text",samples)

# Plot the chains to check for convergence
#labels=["$\\rm norm$","$\\rm slope$"]
#if np.size(labels) != np.size(samples[0,:]):
#	print "size labels: {0} / size samples {1}".format(labels, np.size(samples[0,:]))
#	print("Caution: The number of label names is not correct!")
labels = model.model_parameter_names()
if show_plots:
    for i in range(np.size(labels)):
        pl.clf()
        pl.plot(samples[:,0],'-b')
        pl.xlabel("$\\rm Chain$")
        pl.ylabel(labels[i])
        pl.show()

## add plot distributions with histograms instead of triangle
#fig = triangle.corner(samples,labels=["$norm$","$slope$"])
fig = triangle.corner(samples, labels=labels)
fig.savefig("triangle.png")

#add analysis of the samples -- producing numbers to quote!
#global minimum -- percentiles -- et al.

# try:
# 	cf.run_mcmc_analysis(plot=False)
# except MCMCDidNotConverge:
# 	...







