#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pdb
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


component_data = {"PL": "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat",
                  "HOST": "../Data/FakeData/fake_host_spectrum.dat",
                  "FE": #"/user/jotaylor/git/spamm/Data/FeModels/Fe_UVtemplt_A.asc",
                        "../Data/FakeData/Iron_comp/fakeFe1_deg.dat",
                        #"../Data/FakeData/for_gisella/fake_host_spectrum.dat",
                        #"../Fe_templates/FeSimdata_BevWills_0p05.dat",
                  "BC": "../Data/FakeData/BaC_comp/FakeBac01_deg.dat",
                  "BpC": "../Data/FakeData/BaC_comp/FakeBac_lines01_deg.dat"}

#-----------------------------------------------------------------------------#

def perform_test(components, datafile=None, params=None):
    # eventually need to update params to work with multiple components i.e. params["PL"]...
    
    for c in ["PL", "FE", "HOST", "BC", "BpC", "Calzetti_ext", "SMC_ext", "MW_ext", "AGN_ext", "LMC_ext"]:
        if c not in components:
            components[c] = False
    if "n_walkers" not in components:
        components["n_walkers"] = 30
    if "n_iterations" not in components:
        components["n_iterations"] = 500 

    if params is None:
        keys = list(components.keys())
        i = 0
        while datafile is None:
            if components[keys[i]] is True:
                datafile = component_data[keys[i]]
                print("Using data file {0}".format(datafile))
            i += 1
        try:
            wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
        except ValueError:
            wavelengths, flux= np.loadtxt(datafile, unpack=True) 
            flux_err = flux*0.05
        finally:
            pname = None
            flux = np.where(flux<0, 1e-19, flux)
    else:
        wavelengths = params["wl"]
        flux = params["flux"]
        flux_err = params["err"]
        pname = params["pname"]

#    mask = Mask(wavelengths=wavelengths,maskType=maskType)
#    spectrum.mask=mask
    spectrum = Spectrum.from_array(flux, uncertainty=flux_err)
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
    if components["PL"]:
        if params:
            nuclear_comp = NuclearContinuumComponent(params["broken_pl"])
        else:
            nuclear_comp = NuclearContinuumComponent()
        model.components.append(nuclear_comp)
    if components["FE"]:
        fe_comp = FeComponent()
        model.components.append(fe_comp)
    if components["HOST"]:
        host_galaxy_comp = HostGalaxyComponent()    
        model.components.append(host_galaxy_comp)
    if components["BC"] or components["BpC"]:
        balmer_comp = BalmerCombined(BalmerContinuum=BC, BalmerPseudocContinuum=BpC)
        model.components.append(balmer_comp)
    if components["Calzetti_ext"] or components["SMC_ext"] or components["MW_ext"] or components["AGN_ext"] or components["LMC_ext"]:
        ext_comp = Extinction(MW=MW_ext,AGN=AGN_ext,LMC=LMC_ext,SMC=SMC_ext, Calzetti=Calzetti_ext)
        model.components.append(ext_comp)
    
    model.data_spectrum = spectrum # add data
    
    # ------------
    # Run MCMC
    # ------------
    model.run_mcmc(n_walkers=components["n_walkers"], n_iterations=components["n_iterations"])
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
    pl.errorbar(x, y, yerr=y_err)


#    pl.show()
#    this = input("press enter to clear plot")
    pl.clf()
    
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
    parser.add_argument("--pl", dest="PL", default=False, action="store_true",
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
    args = parser.parse_args()
    
    components = vars(args)
    perform_test(components, datafile=None, params=None)
