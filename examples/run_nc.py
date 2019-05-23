#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.Spectrum import Spectrum

PARS = parse_pars()["nuclear_continuum"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def create_nc(nc_params=None):
    if nc_params is None:
        nc_params = {"wl": WL}
        nc_params["broken_pl"] = False
        nc_params["slope1"] = draw_from_sample.gaussian(PARS["pl_slope_min"], PARS["pl_slope_max"])
        max_template_flux = 1e-13 
        nc_params["norm_PL"] = draw_from_sample.gaussian(PARS["pl_norm_min"], max_template_flux)
    print("NC params: {}".format(nc_params))
    nc = NuclearContinuumComponent(broken=nc_params["broken_pl"])
    
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(nc_params["wl"], nc_params["wl"], nc_params["wl"])
    nc.initialize(spectrum)
    if nc_params["broken_pl"] == True:
        comp_params = [nc_params["wave_break"], nc_params["norm_PL"],
                       nc_params["slope1"], nc_params["slope2"]]
    else:
        comp_params = [nc_params["norm_PL"], nc_params["slope1"]]

    nc_flux = NuclearContinuumComponent.flux(nc, spectrum, comp_params)
    nc_err = nc_flux * 0.05

    return nc_params["wl"], nc_flux, nc_err, nc_params

#-----------------------------------------------------------------------------#
