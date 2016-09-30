#!/usr/bin/env python

'''
This script reads in a Model object that has been run. The "data format"
is to pickle the Model and write it to a gzipped file. It contains all of the
chains, the sample spectrum, and the templates.

This example reads this file, uncompresses and un-pickles the object,
then creates a new triangle plot.
The model object can be used as it existed in the file the wrote it.

Usage:

% read_model_run.py model.pickle.gz
'''

import os
import sys
import gzip
import optparse
import inspect
import cPickle as pickle

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent

# read the name of the SPAMM pickle from the command line.
parser = optparse.OptionParser()
parser.add_option("-m", help="SPAMM model pickle file", dest="model_filename", action="store")

(opts, args) = parser.parse_args()

if opts.model_filename is None:
    print("\nPlease specify the file to read, e.g. \n\n% {0} -m model.pickle.gz\n\n".format(sys.argv[0]))
    sys.exit(1)

model = pickle.loads(gzip.open(opts.model_filename).read())

samples = model.sampler.chain[:, 50:, :].reshape((-1, model.total_parameter_count))

fig = triangle.corner(samples, labels=model.model_parameter_names())
fig.savefig("triangle_from_pickle.png")

sys.exit(0)
