#! /usr/bin/env python

from scipy.stats import powerlaw
from astropy.modeling.powerlaws import PowerLaw1D, BrokenPowerLaw1D
import matplotlib.pyplot as plt

def generate_pl(wavelengths, norm=None, norm_wl=None, slope1=None, slope2=None, wl_break=None, broken=False):
    if not broken:
        pl = PowerLaw1D(norm, norm_wl, slope1)
    else:
        pl = BrokenPowerLaw1D(norm, wl_break, slope1, slope2)

    flux = pl(wavelengths)

    return flux
