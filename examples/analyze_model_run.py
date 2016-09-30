#!/usr/bin/env python

'''
This script reads in a Model object that has been run. The "data format"
is to pickle the Model and write it to a gzipped file. It contains all of the
chains, the sample spectrum, and the templates.

This example reads this file, uncompresses and un-pickles the object,
then creates a new triangle plot.
The model object can be used as it existed in the file that wrote it.

The code then calls functions defined in Analysis.py to analyze the 
Model object that has been run.

Usage:

% analyze_model_run.py model.pickle.gz
'''

import os
import sys
import gzip
import optparse
import inspect
import cPickle as pickle
import numpy as np

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent

from spamm.Analysis import *

# read the name of the SPAMM pickle from the command line.
parser = optparse.OptionParser()
parser.add_option("-m", help="SPAMM model pickle file", dest="model_filename", action="store")

(opts, args) = parser.parse_args()

if opts.model_filename is None:
    print("\nPlease specify the file to read, e.g. \n\n% {0} -m model.pickle.gz\n\n".format(sys.argv[0]))
    sys.exit(1)

model = pickle.loads(gzip.open(opts.model_filename).read())

samples = model.sampler.chain[:, 1000:, :].reshape((-1, model.total_parameter_count))

if np.size(samples) == 0:
    print("WARNING, size of samples is 0! Exiting analysis code now...")
    exit()

fig = triangle.corner(samples, labels=model.model_parameter_names())
fig.savefig("plots/triangle.png")


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
fig.savefig("plots/chain.png")
# Fourth, plot the posterior PDFs for each parameter.  
#	These are histograms of the MCMC chains.  We should add
#	parameter names to this and the previous plots at some point.
#	boxes = 20 is the default.
fig = plot_posteriors(samples, labels=model.model_parameter_names(), boxes=20)
fig.savefig("plots/posterior.png")

# Fifth, plot the spectrum fits from the posterior PDF.
plot_spectra(model, samples)


sys.exit(0)
