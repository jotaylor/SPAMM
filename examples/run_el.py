#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.compoelnts.EmissionComponent import EmissionComponent
from spamm.Spectrum import Spectrum
from astropy.modeling.functional_models import Gaussian1D

PARS = parse_pars()["emission_lines"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])


def create_el(el_params=None):

    wl = el_params["wl"]
    del el_params["wl"]
    
    print(f"el params: {el_params}")
    
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(wl, wl, wl)
    el_flux = np.zeros(len(spectrum.spectral_axis))
    narrow_width = params['narrow_width']
        for line in self.lines:
            if line.is_broad:
                # First do the narrow part of the broad line
                gaussian = Gaussian1D(amplitude=params[f'{line.name}_amp_0'], mean=params[f'{line.name}_loc_0'], stddev=narrow_width)
                el_flux += gaussian(spectrum.spectral_axis)
                # Then add the broad components
                for comp in range(line.num_components):
                    amp = params[f'{line.name}_amp_{comp+1}']
                    loc = params[f'{line.name}_loc_{comp+1}']
                    broad_width = params[f'{line.name}_width_{comp+1}']
                    gaussian = Gaussian1D(amplitude=amp, mean=loc, stddev=broad_width)
                    el_flux += gaussian(spectrum.spectral_axis)
            else:
                amp = params[f'{line.name}_amp_0']
                loc = params[f'{line.name}_loc_0']
                gaussian = Gaussian1D(amplitude=amp, mean=loc, stddev=narrow_width)
                el_flux += gaussian(spectrum.spectral_axis)
        
    el_err = el_flux * 0.05

    return wl, el_flux, el_err, el_params

#-----------------------------------------------------------------------------#
