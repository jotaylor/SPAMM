#! /usr/bin/env python

def rebin_spec(wave, specin, wavnew):
    """
    Rebin spectra to bins used in wavnew.
    Ref: http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/
    """
    from pysynphot import observation
    from pysynphot import spectrum as pysynphot_spec
    import numpy as np

    spec = pysynphot_spec.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = pysynphot_spec.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

    return obs.binflux

