#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.components.NarrowComponent import NarrowComponent
from spamm.Spectrum import Spectrum

PARS = parse_pars()["nuclear_continuum"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def create_ne(ne_params=None):

    wl = ne_params["wl"]
    del ne_params["wl"]
    
    print(f"ne params: {ne_params}")
    ne = NarrowComponent()
    
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(wl, wl, wl)
    ne.initialize(spectrum)

    ne_flux = NarrowComponent.flux(ne, spectrum, ne_params)
    ne_err = ne_flux * 0.05

    return wl, ne_flux, ne_err, ne_params

#-----------------------------------------------------------------------------#
