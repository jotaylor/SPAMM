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
    print(f"ne params: {ne_params}")
    ne = NarrowComponent()
    
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(ne_params["wl"], ne_params["wl"], ne_params["wl"])
    ne.initialize(spectrum)

    comp_params = [ne_params["width"],
               ne_params["amp_1"],
               ne_params["center_1"],
               ne_params["amp_2"],
               ne_params["center_2"],
               ne_params["amp_3"],
               ne_params["center_3"]
              ]

    ne_flux = NarrowComponent.flux(ne, spectrum, comp_params)
    ne_err = ne_flux * 0.05

    return ne_params["wl"], ne_flux, ne_err, ne_params

#-----------------------------------------------------------------------------#
