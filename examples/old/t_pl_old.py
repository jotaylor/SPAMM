#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import gzip
import dill as pickle
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as pl
from astropy import units

import triangle
#sys.path.append(os.path.abspath("../"))

from analyze_model_run import make_chain_plots
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction
from spamm.components.MaskingComponent import Mask
# TODO: astropy units for spectrum


#-----------------------------------------------------------------------------#

def perform_test(components, datafile=None, params=None):
    
    if datafile:
        try:
            wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
        except ValueError:
            wavelengths, flux= np.loadtxt(datafile, unpack=True) 
            flux_err = flux*0.05
        finally:
            pname = None
    else:
        wavelengths = params["wl"]
        flux = params["flux"]
        flux_err = params["err"]
        pname = params["pname"]

#    mask = Mask(wavelengths=wavelengths,maskType=maskType)
    spectrum = Spectrum.from_array(flux, uncertainty=flux_err)
#    spectrum.mask=mask
    spectrum.dispersion = wavelengths#*units.angstrom
    spectrum.flux_error = flux_err    
    
    # ------------
    # Initialize model
    # ------------
    model = Model()
    model.print_parameters = False#True#False#
    
    # -----------------
    # Initialize components
    # -----------------
    if PL:
        nuclear_comp = NuclearContinuumComponent(params["broken_pl"])
        model.components.append(nuclear_comp)
    
    if FE:
        fe_comp = FeComponent()
        model.components.append(fe_comp)
    
    if HOST:
        host_galaxy_comp = HostGalaxyComponent()    
        model.components.append(host_galaxy_comp)
    
    if BC or BpC:
        balmer_comp = BalmerCombined(BalmerContinuum=BC, BalmerPseudocContinuum=BpC)
        model.components.append(balmer_comp)
        
    if Calzetti_ext or SMC_ext or MW_ext or AGN_ext or LMC_ext:
        ext_comp = Extinction(MW=MW_ext,AGN=AGN_ext,LMC=LMC_ext,SMC=SMC_ext, Calzetti=Calzetti_ext)
        model.components.append(ext_comp)
    
    model.data_spectrum = spectrum # add data
    # ------------
    # Run MCMC
    # ------------
    model.run_mcmc(n_walkers=n_walkers, n_iterations=n_iterations)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(model.sampler.acceptance_fraction)))
    
    # -------------
    # save chains & model
    # ------------
    p_data = {"model": model,
              "params": params}
    
    if pname is None:
        now = datetime.datetime.now()
        pname = "model_{0}.pickle.gz".format(now.strftime("%Y%m%d_%M%S"))
    with gzip.open(pname, "wb") as model_output:
        model_output.write(pickle.dumps(p_data))
        print("Saved pickle file {0}".format(pname))
    make_chain_plots(pname)

#-----------------------------------------------------------------------------#

def test_fe():
    this = 1.

#-----------------------------------------------------------------------------#

def err_gauss(mean, std, num, divisor):
    gauss = np.random.normal(mean, std, num)
    
    return gauss/divisor

#-----------------------------------------------------------------------------#

def err_percent(y, percent):
    if percent > 1.:
        percent /= 100.

    return y*percent

#-----------------------------------------------------------------------------#

def make_PL(factor = 0.1,
            x = np.arange(1000,6000),
            norms = np.array([0.1, 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            slopes1_p = np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 2.9]),
            slopes1_n = np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3])*-1, 
            slopes2_p = np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3]),
            slopes2_n = np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3])*-1, 
            wl_ref = None,
            wl_break = None,
            err=err_gauss,
            broken_pl=None,
            pname=None):
    from random import choice as r
    import make_powerlaw

    # Default settings.
    norms *= factor
    if slopes1_p is None or slopes1_p[0] is None:
        slopes1 = [slopes1_n]
    elif slopes1_n is None or slopes1_n[0] is None:
        slopes1 = [slopes1_p]
    else:
        slopes1 = [slopes1_p, slopes1_n]

    if slopes2_p is None or slopes2_p[0] is None:
        slopes2 = [slopes2_n]
    elif slopes2_n is None or slopes2_n[0] is None:
        slopes2 = [slopes2_p]
    else:
        slopes2 = [slopes2_p, slopes2_n]

    params = {}
    slope1_i = r(slopes1)
    slope2_i = r(slopes2)
    
    params["norm_PL"] = r(norms)
    params["slope1"] = r(slope1_i)
    if broken_pl is None:
        params["broken_pl"] = r([True, False])
    else:
        params["broken_pl"] = broken_pl
    
    if not params["broken_pl"]:
        if not wl_ref:
            params["WL Ref"] = r(x)
        else:
            params["WL Ref"] = wl_ref
        params["wave_break"] = "N/A"
        params["slope2"] = "N/A"
        y = make_powerlaw.generate_pl(x, params["norm_PL"], params["WL Ref"], params["slope1"])
    
    else:
        params["WL Ref"] = "N/A"
        if not wl_break:
            params["wave_break"] = r(x)
        else:
            params["wave_break"] = wl_break
        params["slope2"] = r(slope2_i)
        y = make_powerlaw.generate_pl(x, norm=params["norm_PL"], slope1=params["slope1"],
                                      slope2=params["slope2"], wl_break=params["wave_break"],
                                      broken=True)
    if isinstance(err, (int, float)):
#        y_err = err_percent(y, err)
        y_err = np.full(len(y), 5e-16)
    else: #it's a function
        y_err = err(0., factor, len(y), 2.)
#    pl.errorbar(x, y, yerr=y_err)


#    pl.show()
#    this = input("press enter to clear plot")
#    pl.clf()
    
    params["wl"] = x
    params["flux"] = y
    params["err"] = y_err
    params["pname"] = pname
    print(params)
    perform_test(params)
    print(params)
     
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkers", dest="n_walkers", default=30, 
                        help="Number of emcee walkers")
    parser.add_argument("--iter", dest="n_iterations", default=500,
                        help="Number of emcee iterations")
    parser.add_argument("--pl", dest="PL", default=True, action="store_true",
                        help="Use Nuclear Continuum component")
    parser.add_argument("--host", dest="HOST", default=False, action="store_true",
                        help="Use Host Galaxy component")
    parser.add_argument("--fe", dest="FE", default=False, action="store_true",
                        help="Use Fe Forest component")
    parser.add_argument("--bc", dest="BC", default=False, action="store_true",
                        help="Use Balmer Continuum component")
    parser.add_argument("--bpc", dest="BpC", default=False, action="store_true",
                        help="?")
    parser.add_argument("--calzetti", dest="Calzetti_ext", default=False, action="store_true",
                        help="Extinction model")
    parser.add_argument("--smc", dest="SMC_ext", default=False, action="store_true",
                        help="Extinction model")
    parser.add_argument("--mw", dest="MW_ext", default=False, action="store_true",
                        help="Extinction model")
    parser.add_argument("--agn", dest="AGN_ext", default=False, action="store_true",
                        help="Extinction model")
    parser.add_argument("--lmc", dest="LMC_ext", default=False, action="store_true",
                        help="Extinction model")
    parser.add_argument("--mask", dest="maskType", default=None,
                        help="Mask type- can be 'Continuum', 'Emission lines reduced' or 'None'")

    percerr = [.1, .05, .01, .005]

    # 10 Broken power law tests
    factors = [1. for x in range(10)]
    xs = [[1000,10000], [2000, 5000], [3000, 8000], [4000, 10000], [4000, 7000],
          [1000, 4000], [7000, 10000], [2500, 4500], [5500, 8500], [6500, 8000]]
    wl_breaks = [7000, 3000, 6000, 5000, 6000, 2000, 9000, 4000, 6000, 7500]
    slopes1_n = [-2.2, -1.4, -0.2, -0.9, -1.8, None, None, None, None, None]
    slopes1_p = [None, None, None, None, None, 2.1,  1.3,  0.3,  .9,   0.4]
    slopes2_n = [-0.1, -0.5, 0.,   -0.5, -1.0, None, None, None, None, None]
    slopes2_p = [None, None, None, None, None, 1.5,  1.0,  0.,   0.5,  0.1]
    norms = [1e-15, 1.4e-14, 1.8e-17, 3.5e-16, 5.7e-15, 6.6e-15, 3.3e-13, 4.9e-15, 9.1e-16, 2.1e-17]

# POWER LAW 1 from excel file
#    pnames = ["fakepowerlaw1_{0}err.pickle.gz".format(x) for x in percerr]
#    for i in range(len(percerr)):
#        make_PL(factor=1., x=np.arange(1159, 2003), norms=np.array([1.8e-17]), 
#                     slopes1_n=np.array([-2.2]), slopes1_p=None,
#                     wl_ref=1581, err=percerr[i], broken_pl=False, pname=pnames[i])
    
# POWER LAW 2 from excel file
#    pnames = ["fakepowerlaw2_{0}err.pickle.gz".format(x) for x in percerr]
#    for i in range(len(percerr)):
#        make_PL(factor=1., x=np.arange(4300, 5800), norms=np.array([3.5e-16]), 
#                     slopes1_p=np.array([0.1]), slopes1_n=None,
#                     wl_ref=5050, err=percerr[i], broken_pl=False, pname=pnames[i])

# Totally random
#    wl_ranges = [(1000, 3000), (4000, 8000), (1000, 10000), (1500, 5500)]
#    for wla,wlb in wl_ranges:
#        for i in range(len(percerr)):
#            pname = "powerlaw_{0}-{1}_{2}err.pickle.gz".format(wla, wlb, percerr[i])
#            make_PL(factor=1e-15, x=np.arange(wla, wlb), err=percerr[i],
#                         pname=pname)

# Controlled broken PL
#    make_PL(factor=1., x=np.arange(1000,10000), wl_break=7000, 
#                 slopes1_n=np.array([1.3]), slopes1_p=None, 
#                 slopes2_n=np.array([0]), slopes2_p=None,
#                 broken_pl=True, err=.1, norms=np.array([1e-15]))
#

#   10 Controlled broken PL tests
    for i in range(10):
        for j in range(len(percerr)):
            make_PL(factor = factors[i], 
                         x = np.arange(xs[i][0], xs[i][1]), 
                         wl_break = wl_breaks[i], 
                         slopes1_n = np.array([slopes1_n[i]]), 
                         slopes1_p = np.array([slopes1_p[i]]), 
                         slopes2_n = np.array([slopes2_n[i]]), 
                         slopes2_p = np.array([slopes2_p[i]]),
                         broken_pl = True, 
                         err = percerr[j], 
                         norms = np.array([norms[i]]) )
    
 
