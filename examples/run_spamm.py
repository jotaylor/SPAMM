#!/usr/bin/env python

import gzip
import dill as pickle
import datetime
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl

from utils.parse_pars import parse_pars
from analyze_model_run import make_chain_plots
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction
from spamm.components.MaskingComponent import Mask

#-----------------------------------------------------------------------------#

def spamm_wlflux(components, wl, flux, flux_err=None, 
                 n_walkers=30, n_iterations=500, brokenPL=False,
                 pname=None, comp_params=None):
    """
    Args:
        components : dictionary
            A dictionary with at least one component to model, e.g. {"FE": True}
        datafile : str
            Pathname of the datafile to be used as input.
        comp_params : dictionary
            If None, the component parameters (at minimum- wavelength, flux, flux error)
            will be determined from the datafile. If defined, comp_params *MUST* contain:
                - wl
                - flux
                - err
                - pname (can be None)
                - broken_pl (if component=PL)
    """

    t1 = datetime.datetime.now()
    for c in ["PL", "FE", "HOST", "BC", "BpC", "Calzetti_ext", "SMC_ext", "MW_ext", "AGN_ext", "LMC_ext"]:
        if c not in components:
            components[c] = False

    if flux_err is None:
        flux_err = flux*0.05
    
    spectrum = Spectrum.from_array(flux, uncertainty=flux_err)
    spectrum.dispersion = wl#*units.angstrom
    spectrum.flux_error = flux_err    
    
    # ------------
    # Initialize model
    # ------------
    model = Model()
    model.print_parameters = False
    
    # -----------------
    # Initialize components
    # -----------------
    if components["PL"]:
        if brokenPL:
            nuclear_comp = NuclearContinuumComponent(broken=True)
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
        balmer_comp = BalmerCombined(BalmerContinuum=components["BC"], 
                                     BalmerPseudocContinuum=components["BpC"])
        model.components.append(balmer_comp)
    if components["Calzetti_ext"] or components["SMC_ext"] or components["MW_ext"] or components["AGN_ext"] or components["LMC_ext"]:
        ext_comp = Extinction(MW=MW_ext, AGN=AGN_ext, LMC=LMC_ext, SMC=SMC_ext, Calzetti=Calzetti_ext)
        model.components.append(ext_comp)

    if comp_params is None:
        comp_params = {}
    for k,v in zip(("wl", "flux", "err", "components"), (wl, flux, flux_err, components)):
        if k not in comp_params:
            comp_params[k] = v

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
              "comp_params": comp_params}
    
    if pname is None:
        now = datetime.datetime.now()
        pname = "model_{0}.pickle.gz".format(now.strftime("%Y%m%d_%M%S"))
    with gzip.open(pname, "wb") as model_output:
        model_output.write(pickle.dumps(p_data))
        print("Saved pickle file {0}".format(pname))
    make_chain_plots(pname)

    t2 = datetime.datetime.now()
    print("executed in {}".format(t2-t1))
