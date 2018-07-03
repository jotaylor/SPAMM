#!/usr/bin/env python

'''
This script reads and a pickled Model object and creates analysis plots. 
The Model object contains chains, a sample spectrum, and templates used.

Usage:
    > python analyze_model_run.py model.pickle.gz
'''

#import os
import sys
import gzip
import argparse
#import inspect
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

import triangle
#sys.path.append(os.path.abspath("../source"))
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.Analysis import *

#-----------------------------------------------------------------------------#

def make_chain_plots(model_filename):
    """
    Create analysis plots:
        - chain values vs. iteration number
        - histogram of chain values 
        - triangle plot of chains

    Args:
        model_filename (str): Name of input zipped model pickle file.
    """
    try:
        p_data = pickle.loads(gzip.open(model_filename).read())
    except UnicodeDecodeError:
        p_data = pickle.loads(gzip.open(model_filename).read(), encoding="latin1")
    model = p_data["model"]
    params = p_data["comp_params"] 
        
#   We can do this intelligently based on number of iterations.
    samples = model.sampler.chain[:, 50:, :].reshape((-1, model.total_parameter_count))
    
    if np.size(samples) == 0:
        print("WARNING, size of samples is 0! Exiting analysis code now...")
        exit()
    
    fig = triangle.corner(samples, labels=model.model_parameter_names())
    figname = "plots/{0}_triangle.png".format(model_filename)
    fig.savefig(figname)
    print("\tWrote {0}".format(figname))
    
    # First, calculate the median and confidence intervals
    #	where frac is the fraction of samples within the
    #	quoted uncertainties.  frac = 0.68 is the default.
    #	Columns are median, -error1, +error2.
    print(median_values(samples, frac=0.68))
    # Second, calculate the mean and standard deviation.
    #	Columns are mean, standard deviation.
    print(mean_values(samples))
    
    # Third, plot the MCMC chains as a function of iteration.
    #	You can easily tell if the chains are converged because you can
    #	no longer tell where the individual particle chains are sliced together.
    #	For testing, I saved 2000 iterations and ignored the first 1000.
    fig = plot_chains(samples, labels=model.model_parameter_names())
    figname = "plots/{0}_chain.png".format(model_filename)
    fig.savefig(figname)
    print("\tWrote {0}".format(figname))
    
    # Fourth, plot the posterior PDFs for each parameter.  
    #	These are histograms of the MCMC chains.  We should add
    #	parameter names to this and the previous plots at some point.
    #	boxes = 20 is the default.
    fig = plot_posteriors(samples, labels=model.model_parameter_names(), boxes=20, params=params)
    figname = "plots/{0}_posterior.png".format(model_filename)
    fig.savefig(figname)
    print("\tWrote {0}".format(figname))
    
#    # Fifth, plot the spectrum fits from the posterior PDF.
#    fig = plot_spectra(model, samples)
#    figname = "plots/{}_spectrum.png".format(model_filename)
#    fig.savefig(figname)
#    print("\tWrote {0}".format(figname))
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    # read the name of the SPAMM pickle from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filename", help="SPAMM model pickle file", type=str)
    args = parser.parse_args()
    
    model_filename = args.model_filename
    make_chain_plots(model_filename) 
