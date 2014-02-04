#!/usr/bin/env python

'''
This script reads in a Model object that has been run. The "data format"
is to pickle the Model and write it to a gzipped file. It contains all of the
chains, the sample spectrum, and the templates.

This example reads this file, uncompresses and un-pickles the object.
It can then be used as it existed in the file the wrote it.
'''

import sys
import gzip
import cPickle as pickle

import triangle
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent

# example to read pickled Model

model = pickle.loads(gzip.open("model.pickle.gz").read())

#print model

samples = model.sampler.chain[:, 50:, :].reshape((-1, model.total_parameter_count))

fig = triangle.corner(samples, labels=model.model_parameter_names())
fig.savefig("triangle_from_pickle.png")
