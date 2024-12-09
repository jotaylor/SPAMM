#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.components.PowerLawComponent import PowerLawComponent
from spamm.Spectrum import Spectrum

PARS = parse_pars()["power_law"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def create_pl(pl_params=None):

    wl = pl_params["wl"]
    del pl_params["wl"]
    
    if pl_params is None:
        pl_params = {"wl": WL}
        pl_params["broken_pl"] = False
        pl_params["slope1"] = draw_from_sample.gaussian(PARS["pl_slope_min"], PARS["pl_slope_max"])
        max_template_flux = 1e-13 
        pl_params["norm_PL"] = draw_from_sample.gaussian(PARS["pl_norm_min"], max_template_flux)
    print(f"PL params: {pl_params}")
    pl = PowerLaw(broken=pl_params["broken_pl"])
    
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(wl, wl, wl)
    pl.initialize(spectrum)

    pl_flux = PowerLawComponent.flux(pl, spectrum, pl_params)
    pl_err = pl_flux * 0.05
    
    
    
    return wl, pl_flux, pl_err, pl_params

#-----------------------------------------------------------------------------#
