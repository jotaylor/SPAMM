#!/usr/bin/env python

"""
Test the Balmer Component code. This code can be run from the command line:
> python test_bc.py --datafile /user/jotaylor/git/spamm/Data/FakeData/BaC_comp/FakeBac_lines04_deg.dat
--redshift 0.2

"""

import os
import datetime
import numpy as np
import time
import argparse
import glob

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.components.BalmerContinuumCombined import BalmerCombined as BCComponent
from spamm.Spectrum import Spectrum

PARS = parse_pars()["balmer_continuum"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def from_file(datafile, redshift=None, 
              scale=None, subset=False, pname=None):
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

def create_bc(bc_params=None):
    """
    Args:
        bc_params (dictionary): Iron component parameters. Required keys are:
            - no_templates (number of templates)
            - wl (wavelength range of BC model, redshift must be accounted for already)
            - bc_lines
            - bc_norm
            - bc_Te
            - bc_tauBE
            - bc_loffset
            - bc_lwidth
            - bc_logNe
    """
    wl = bc_params["wl"]
    del bc_params["wl"]

    if bc_params is None:
        bc_params = {"wl": WL}
        max_bc_flux = 6e-14
        bc_params["bc_lines"] = draw_from_sample.gaussian(PARS["bc_lines_min"], PARS["bc_lines_max"])
        bc_params["bc_norm"] = draw_from_sample.gaussian(PARS["bc_norm_min"], max_bc_flux)
        bc_params["bc_Te"] = draw_from_sample.gaussian(PARS["bc_Te_min"], PARS["bc_Te_max"])
        bc_params["bc_tauBE"] = draw_from_sample.gaussian(PARS["bc_tauBE_min"], PARS["bc_tauBE_max"])
        bc_params["bc_loffset"] = draw_from_sample.gaussian(PARS["bc_loffset_min"], PARS["bc_loffset_max"])
        bc_params["bc_lwidth"] = draw_from_sample.gaussian(PARS["bc_lwidth_min"], PARS["bc_lwidth_max"])
        bc_params["bc_logNe"] = draw_from_sample.gaussian(PARS["bc_logNe_min"], PARS["bc_logNe_max"])

    bc = BCComponent(BalmerContinuum=True, BalmerPseudocContinuum=True)
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(wl, wl, wl)
    bc.initialize(spectrum)
    #comp_params = {x: bc_params[x] for x in ["bc_norm", "bc_Te", "bc_tauBE", "bc_loffset", "bc_lwidth", "bc_logNe"]}
    bc_flux = BCComponent.flux(bc, spectrum, bc_params)
    bc_err = bc_flux * 0.05

#    pl.plot(bc_params["wl"], bc_flux)
#    pl.savefig("bc_data.png")
#    this = input("press enter")

    return wl, bc_flux, bc_err, bc_params

#-----------------------------------------------------------------------------#
