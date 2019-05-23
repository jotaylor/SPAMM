#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import matplotlib.pyplot as plt

'''
This code is for analyzing the final posterior samples.
'''

def median_values(samples, frac=0.68):
    num_params = np.size(samples[0,:])
    result = np.zeros((num_params, 3))
    for i in range(0, num_params):
        chain = np.sort(samples[:,i])
        # calculate the median values:
        if np.size(chain)%2 == 1:   # for odd sizes of chain
            index = int(np.size(chain)/2)
            median = chain[index]
        if np.size(chain)%2 == 0:   # for even sizes of chain
            lower = int(np.size(chain)/2)-1
            higher = int(np.size(chain)/2)
            median = (chain[lower] + chain[higher])/2.
        result[i,0] = median
        # calculate the confidence intervals
        num_spread = np.size(chain)/2.0*frac
        max_index = int(np.size(chain)/2.0 + num_spread)
        min_index = int(np.size(chain)/2.0 - num_spread)
        max_value = chain[max_index]
        min_value = chain[min_index]
        max_error = abs(median - max_value)
        min_error = abs(median - min_value)
        result[i,1] = min_error
        result[i,2] = max_error

    print("Calculating median and "+str(frac*100)+"% confidence intervals (min, max).")
    return result


def mean_values(samples):
    num_params = np.size(samples[0,:])
    result = np.zeros((num_params, 2))
    for i in range(0, num_params):
        chain = samples[:,i]
        result[i,0] = np.mean(chain)
        result[i,1] = np.std(chain)

    print("Calculating mean and standard deviation.")
    return result


def plot_chains(samples, labels):
    num_params = np.size(samples[0,:])
    fig = plt.figure()
    #####
    # The following is from triangle.py:
    K = num_params
    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.15         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig = plt.figure(figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                    wspace=whspace, hspace=whspace)
    #####
    for i in range(0, num_params):
        chain = samples[:,i]
        ax = fig.add_subplot(num_params, 1, i+1)
        ax.plot(chain, '-b')
        ax.set_ylabel(labels[i])
    ax.set_xlabel("MCMC Chain Iteration")
    print("Plotting the MCMC chains.")
    return fig


def plot_posteriors(samples, labels, boxes=20, params=None):
    num_params = np.size(samples[0,:])
    fig = plt.figure()
    #####
    # The following is from triangle.py:
    K = num_params
    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.3         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig = plt.figure(figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                    wspace=whspace, hspace=whspace)
    #####
    for i in range(0, num_params):
        chain = samples[:,i]
        ax = fig.add_subplot(num_params, 1, i+1)
        ax.hist(chain, boxes)
        if params is not None:
            try:
                ax.axvline(params[labels[i]], color="r", linestyle="dashed", linewidth=2)
                ax.set_title("Actual {0}={1}".format(labels[i], params[labels[i]]))
            except KeyError:
                pass
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Posterior PDF")
    print("Plotting the model posterior PDFs.")
    return fig


def plot_spectra(model, samples):
    data_spectrum = model.data_spectrum   # lambda, flux, flux_error
    num_samples = 110  #np.size(samples[:,0])

    plt.clf()
    plt.ion()
    for i in range(0, num_samples):
        # plot the data
        plt.errorbar(data_spectrum.spectral_axis, data_spectrum.flux, 
                yerr=data_spectrum.flux_error, mfc='blue', mec='blue', ecolor='b',
                fmt='.')

        # plot each of the model components individually
        k = 0
    for component in model.components:
        component_flux = component.flux(spectrum=model.data_spectrum,
                parameters=samples[i,k:len(component.model_parameter_names)])  # flux
        k = len(component.model_parameter_names)
        plt.plot(data_spectrum.spectral_axis, component_flux, '-r')

    # plot the sum of the model components
        model_spectrum = model.model_flux(params=samples[i,:])  # flux
        plt.plot(data_spectrum.spectral_axis, model_spectrum)

