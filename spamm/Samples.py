#! /usr/bin/env python

import os
import dill
import gzip
import numpy as np

#-----------------------------------------------------------------------------#

class Samples(object):
    """
    A class to handle and analyze samples from a model.

    Attributes:
        pname (str): Path to the pickle files.
        gif (bool): Whether to create a gif.
        last (bool): Whether to use the last sample.
        step (int): Step size for sampling.
        burn (int): Burn-in for MCMC.
        histbins (int): Number of bins for histograms.
        models (list): List of models.
        samples (array): Array of samples.
        params (list): List of parameters.
        model_name (str): Name of the model.
        model (object): The model object.
        total_parameter_count (int): Total number of parameters in the model.
        model_parameter_names (list): Names of the model parameters.
        outdir (str): Output directory.
    """

    def __init__(self, pickle_files, outdir=None, gif=False, last=False, 
                 step=100, burn=50, histbins=100):
        """
        Initialize the Samples object and load the samples from the pickle files.
        """
        self.pname = pickle_files
        self.gif = gif
        self.last = last
        self.step = step
        self.burn = burn
        self.histbins = histbins
         
        self.models, self.samples, self.params, self.model_name = get_samples(pickle_files, burn)

        try:
            self.model = self.models[0]
        except TypeError:
            self.model = self.models

        self.total_parameter_count = self.model.total_parameter_count
        self.model_parameter_names = self.model.model_parameter_names()
        self._get_stats()
        
        if outdir is None:
            outdir = os.path.dirname(self.model_name)
            if outdir == "":
                outdir = "."
        self.outdir = outdir

    def _get_stats(self):
        """
        Calculate and store the mean, median, mode, and maximum of each parameter.
        """
        self.means = []
        self.medians = []
        self.modes = []
        self.maxs = []

        for i in range(self.model.total_parameter_count):
            chain = self.samples[:,i]
            hist, bins = np.histogram(chain, self.histbins)
            binsize = bins[1] - bins[0]

            maxind = np.argmax(hist)
            max_bin = bins[maxind]
            self.maxs.append(max_bin + binsize/2.)
            self.medians.append(np.median(chain) + binsize/2.)
            self.means.append(np.average(chain) + binsize/2.)
            mode_bin = bins[np.argmax(hist)]
            self.modes.append(mode_bin + binsize/2.)

#-----------------------------------------------------------------------------#

def get_samples(pname, burn=50):
    """ 
    Some pickled model files I made were incorrectly made and a try/except
    needs to be inserted when creating sample:
        try:
            sample = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
        except TypeError:
            sample = model.sampler.chain[:, burn:, :].reshape((-1, 10))
            #samples = model.sampler.chain[:, burn:, :].reshape((-1, 16))
    """ 

    if isinstance(pname, str):
        model_name = os.path.basename(pname).split(".")[0]
        model, params = read_pickle(pname)
        assert burn < np.shape(model.sampler.chain)[1], \
            f"Chain burn value, {burn}, must be smaller than number of iterations, {np.shape(model.sampler.chain)[1]}.\nRerun with lower burn value or more iterations"
        samples = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
        allmodels = model
    else:
        allmodels = []
        allsamples = []
        for pfile in pname:
            model, params = read_pickle(pfile)
            sample = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
            allsamples.append(sample)
            allmodels.append(model)
        samples = np.concatenate(tuple(allsamples))
        model_name = f"concat_{len(samples)}"
    
    return allmodels, samples, params, model_name

#-----------------------------------------------------------------------------#

def read_pickle(pname):
    try:
        p_data = dill.loads(gzip.open(pname).read())
    except UnicodeDecodeError:
        p_data = dill.loads(gzip.open(pname).read(), encoding="latin1")
    
    model = p_data["model"]
    params = p_data["comp_params"]

    if pname == "model_20180627_1534.pickle.gz":
        params = {'fe_norm_2': 3.5356725072091589e-15, 
                  'fe_norm_3': 8.9351374726858118e-15, 
                  'no_templates': 3, 
                  'fe_width': 4208.055598607859, 
                  'fe_norm_1': 9.4233576501633248e-15}
    elif pname == "model_20180627_4259.pickle.gz":
        params = {'fe_norm_2': 8.68930476e-15, 
                  'fe_width': 5450,
                  'no_templates': 3, 
                  'fe_norm_1': 1.07988504e-14, 
                  'fe_norm_3': 6.91877436e-15,
                  'norm_PL': 5e-15,
                  'slope1': 2.5}
    elif pname == "model_20180711_4746.pickle.gz":
        params = {'bc_Te': 50250.0,
                  'bc_lines': 201.5,
                  'bc_loffset': 0.0,
                  'bc_logNe': 5.5,
                  'bc_lwidth': 5050.0,
                  'bc_norm': 3e-14,
                  'bc_tauBE': 1.0,
                  'broken_pl': False,
                  'fe_norm_1': 1.07988504e-14,
                  'fe_norm_2': 8.68930476e-15,
                  'fe_norm_3': 6.91877436e-15,
                  'fe_width': 5450,
                  'norm_PL': 5e-15,
                  'slope1': 2.5}
    elif pname == "model_20180711_3027.pickle.gz":
        params['fe_norm_1'] = 1.07988504e-14
        params['fe_norm_2'] = 8.68930476e-15
        params['fe_norm_3'] = 6.91877436e-15
        params['fe_width'] = 5450 

    
    return model, params

#-----------------------------------------------------------------------------#
