#!/usr/bin/env python

"""
Test the Iron Component code. This code can be run from teh command line:
> python test_fe.py --datafile /user/jotaylor/git/spamm//Data/FakeData/Iron_comp/fakeFe1_deg.dat
--redshift 0.5

"""

import os
import dill as pickle
import datetime
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as pl
import time
import argparse
import glob

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.components.FeComponent import FeComponent
from spamm.Spectrum import Spectrum

PARS = parse_pars()["fe_forest"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def from_file(datafile, redshift=None, 
              scale=None, subset=False, pname=None):
    print(PARS, "\n")
    templates = glob.glob(os.path.join(PARS["fe_templates"], "*"))
    print("Using datafile: {}\n".format(datafile))
    print("Templates = {}\n".format(templates))
    print("Are the parameters in utils good? If not, ctrl+c, modify them, and rerun")
    time.sleep(5)
    
    try:
        wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
    except ValueError:
        wavelengths, flux = np.loadtxt(datafile, unpack=True)
        flux_err = flux*0.05
    
    if redshift is not None:
        print("Correcting for redshift {}\n".format(redshift))
        wavelengths /= (1+float(redshift))
    if scale is not None:
        print("Scaling flux by {\n}".format(scale))
        flux *= scale
        flux_err *= scale
    if subset:
        minwl = 1e10
        maxwl = 0.
        for template in templates:
            with open(template) as f:
                t_wl, t_flux = np.loadtxt(template, unpack=True)
                maxwl = max(maxwl, max(t_wl))
                minwl = min(minwl, min(t_wl))
        print("Wavelength range of templates {} =\n{}:{}".format(templates,
              minwl, maxwl))
        print("Only using this range on datafile")
        inds = np.where((wavelengths >= minwl) & (wavelengths <= maxwl))
        wavelengths = wavelengths[inds]
        flux = flux[inds]
        flux_err = flux_err[inds]

    params = {"wl": wavelengths,
              "flux": flux,
              "err": flux_err,
              "pname": pname,
              "datafile": datafile}

    return wavelengths, flux, flux_err, params

#-----------------------------------------------------------------------------#

def create_fe(fe_params=None):
    """
    Args:
        fe_params (dictionary): Iron component parameters. Required keys are:
            - no_templates (number of templates)
            - wl (wavelength range of Fe model, redshift must be accounted for already)
            - fe_width (in km/s)
            - fe_norm_{} (1,2,3 depending on number of templates)
    """

    if fe_params is None:
        fe_params = {"no_templates": 3, "wl": WL}
        max_template_flux = 1.8119e-14
        samples = draw_from_sample.gaussian(PARS["fe_norm_min"], max_template_flux, 3)
        fe_params["fe_norm_1"] = samples[0]
        fe_params["fe_norm_2"] = samples[1]
        fe_params["fe_norm_3"] = samples[2]
        sample = draw_from_sample.gaussian(PARS["fe_width_min"], PARS["fe_width_max"])
        fe_params["fe_width"] = sample

    print("Fe params: {}".format(fe_params))
    fe = FeComponent()
    # Make a Spectrum object with dummy flux
    spectrum = Spectrum(fe_params["wl"])
    spectrum.dispersion = fe_params["wl"]
    fe.initialize(spectrum)
    comp_params = [fe_params["fe_norm_{}".format(x)] for x in range(1, fe_params["no_templates"]+1)] + [fe_params["fe_width"]]
    fe_flux = FeComponent.flux(fe, spectrum, comp_params)
    fe_err = fe_flux * 0.05

#    pl.errorbar(fe_params["wl"], fe_flux, fe_err)
#    pl.savefig("fe_data.png")

    return fe_params["wl"], fe_flux, fe_err, fe_params

#-----------------------------------------------------------------------------#
