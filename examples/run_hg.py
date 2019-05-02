#!/usr/bin/env python

"""
Test the Iron Component code. This code can be run from teh command line:
> python test_fe.py --datafile /user/jotaylor/git/spamm//Data/FakeData/Iron_comp/fakeFe1_deg.dat
--redshift 0.5

"""

import os
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
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.Spectrum import Spectrum

PARS = parse_pars()["host_galaxy"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def from_file(datafile, redshift=None, 
              scale=None, subset=False, pname=None):
    print(PARS, "\n")
    templates = glob.glob(os.path.join(PARS["hg_models"], "*"))
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

def create_hg(hg_params=None):
    """
    Args:
        hg_params (dictionary): Host galaxy component parameters. Required keys are:
            - no_templates (number of templates)
            - wl (wavelength range of HG model, redshift must be accounted for already)
            - hg_norm_{} (1,2,3 depending on number of templates)
    """

    if hg_params is None:
        hg_params = {"no_templates": 3, "wl": WL}
        max_template_flux = 6e-12
        samples = draw_from_sample.gaussian(PARS["hg_norm_min"], max_template_flux, 3)
        hg_params["hg_norm_1"] = samples[0]
        hg_params["hg_norm_2"] = samples[1]
        hg_params["hg_norm_3"] = samples[2]
        hg_params["hg_stellar_disp"] = draw_from_sample.gaussian(PARS["hg_stellar_disp_min"], PARS["hg_stellar_disp_max"])

    print("HG params: {}".format(hg_params))
    hg = HostGalaxyComponent()
    # Make a Spectrum object with dummy flux
    spectrum = Spectrum(hg_params["wl"], hg_params["wl"])
    hg.initialize(spectrum)
    comp_params = [hg_params["hg_norm_{}".format(x)] for x in range(1, hg_params["no_templates"]+1)] + [hg_params["hg_stellar_disp"]]
    hg_flux = HostGalaxyComponent.flux(hg, spectrum, comp_params)
    hg_err = hg_flux * 0.05

#    pl.errorbar(hg_params["wl"], hg_flux, hg_err)
#    pl.savefig("hg_data.png")

    return hg_params["wl"], hg_flux, hg_err, hg_params

#-----------------------------------------------------------------------------#
