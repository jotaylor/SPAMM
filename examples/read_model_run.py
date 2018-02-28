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
try:
    import cPickle as pickle
except ImportError:
    import pickle
import triangle
import matplotlib.pyplot as pl
from matplotlib import cm,mlab,colors,ticker,rc
from matplotlib.ticker import NullFormatter

sys.path.append(os.path.abspath("../"))

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

try:
    model = pickle.loads(gzip.open(opts.model_filename).read())
except UnicodeDecodeError:
    model = pickle.loads(gzip.open(opts.model_filename).read(), encoding="latin1")

samples = model.sampler.chain[:, 20:, :].reshape((-1, model.total_parameter_count))

#rc('font', **{'family':'serif','serif':['Palatino'],'size'   : 24})
#rc('text', usetex=True)           
#nullfmt = NullFormatter() 

pl.plot(model.data_spectrum.dispersion,model.data_spectrum.flux)
pl.plot(model.model_spectrum.dispersion,model.model_spectrum.flux)
pl.savefig("fit_from_pickle.eps", format='eps', dpi=1000)
pl.close()

fig2 = triangle.corner(samples, labels=model.model_parameter_names())
fig2.savefig("triangle_from_pickle.eps")

sys.exit(0)
