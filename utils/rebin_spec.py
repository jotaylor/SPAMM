#! /usr/bin/env python

from numba import jit
import numpy as np
#
@jit(nopython=True)
def  rebin_spec(old_wave,old_flux,new_wave):
    I = np.searchsorted(old_wave, new_wave)
    out_of_range = ((I == 0) | (I == old_wave.size))
    I[out_of_range] = 1  # Prevent index-errors


    x1 = old_wave[I-1]
    x2 = old_wave[I]

    y1 = old_flux[I-1]
    y2 = old_flux[I]

    xI = new_wave

    yI = y1 + (y2-y1) * (xI-x1) / (x2-x1)
    yI[out_of_range] = 0.

    return yI