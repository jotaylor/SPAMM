#! /usr/bin/env python

from pysynphot import observation
from pysynphot import spectrum as pysynphot_spec
import numpy as np

def rebin_spectrum(new_wave, old_wave, old_spec):
    """
    Rebin a spectrum to new wavelengths.

    Parameters:
    new_wave (array-like): New wavelengths for the output spectrum.
    old_wave (array-like): Original wavelengths of the input spectrum.
    old_spec (array-like): Input spectrum to be rebinned.

    Returns:
    array-like: The rebinned spectrum.

    Reference: 
    http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/
    """

    # Create a source spectrum from the old wavelengths and spectrum
    source_spec = pysynphot_spec.ArraySourceSpectrum(wave=old_wave, 
                                                     flux=old_spec)

    # Create a spectral element from the old wavelengths
    unit_filter = pysynphot_spec.ArraySpectralElement(old_wave, 
                                                      np.ones(len(old_wave)), 
                                                      waveunits='angstrom')

    # Create an observation from the source spectrum and unit filter, rebinned to the new wavelengths
    obs_spec = observation.Observation(source_spec, 
                                       unit_filter, 
                                       binset=new_wave, 
                                       force='taper')

    return obs_spec.binflux

