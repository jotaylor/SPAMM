#!/usr/bin/env python

import gzip
import dill as pickle
import datetime
import numpy as np
from astropy import units as u
from specutils import Spectrum1D

from utils.parse_pars import parse_pars
from plot_spamm_results import make_plots_from_pickle
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction

ACCEPTED_COMPS = ["PL", "FE", "HOST", "BC", "BPC", "CALZETTI_EXT", "SMC_EXT", "MW_EXT", "AGN_EXT", "LMC_EXT"]

#-----------------------------------------------------------------------------#

def spamm(complist, inspectrum, par_file=None, n_walkers=30, 
          n_iterations=500, pname=None, comp_params=None):
    """
    Args:
        complist (list): A list with at least one component to model. 
            Accepted component names are listed below. They are case insensitive:
                - PL
                - FE
                - HOST
                - BC
                - BPC
                - CALZETTI_EXT
                - SMC_EXT
                - MW_EXT
                - AGN_EXT
                - LMC_EXT
        inspectrum (:obj:`spamm.Spectrum`, :obj:`specutils.Spectrum1D`, or tuple): 
            A SPAMM Spectrum object, specutils Spectrum object, or a tuple. 
            If tuple, it must be at least length 2: ((wavelength,), (flux,)). 
            It may also contain a 3rd element, the error on the flux.
        par_file (str): Location of parameters file.
        n_walkers (int): Number of walkers, or chains, to use in emcee.
        n_iterations (int): Number of iterations for each walker/chain.
        pname (str): Name of output pickle file. If None, name will be
            determined based on current run time.
        comp_params : dictionary
            Contains the known values of component parameters, with keys
            defined in each of the individual run scripts (run_XX.py).
            If None, the actual values of parameters will not be plotted.
    """

    t1 = datetime.datetime.now()
    if par_file is None:
        pars = parse_pars()
    else:
        pars = parse_pars(par_file)

    complist = [x.upper() for x in complist]
    components = {k:(True if k in complist else False) for k in ACCEPTED_COMPS}
    
    if isinstance(inspectrum, Spectrum):
        spectrum = inspectrum
        wl = inspectrum.spectral_axis
        flux = inspectrum.flux
        flux_error = None
    elif isinstance(inspectrum, Spectrum1D):
        spectrum = Spectrum(spectral_axis=inspectrum.spectral_axis, 
                            flux=inspectrum.flux, flux_error=inspectrum.uncertainty)
        wl = inspectrum.spectral_axis.value
        flux = inspectrum.flux.value
        flux_error = None
    else:
        wl, flux, flux_error = inspectrum
# This is just for testing    
#       flux_error = flux*0.05
        spectrum = Spectrum(spectral_axis=wl, flux=flux, flux_error=flux_error)

    if comp_params is None:
        comp_params = {}
    for k,v in zip(("wl", "flux", "err", "components"), (wl, flux, flux_error, components)):
        if k not in comp_params:
            comp_params[k] = v

    # ------------
    # Initialize model
    # ------------
    model = Model()
    model.print_parameters = False

    # -----------------
    # Initialize components
    # -----------------
    if components["PL"]:
        try:
            if comp_params["broken_pl"] is True:
                brokenPL = True
            else:
                brokenPL = False
        except:
            brokenPL = False
        finally:
            nuclear_comp = NuclearContinuumComponent(broken=brokenPL,
                                                     pars=pars["nuclear_continuum"])
            model.components.append(nuclear_comp)
    if components["FE"]:
        fe_comp = FeComponent(pars=pars["fe_forest"])
        model.components.append(fe_comp)
    if components["HOST"]:
        host_galaxy_comp = HostGalaxyComponent(pars=pars["host_galaxy"])
        model.components.append(host_galaxy_comp)
    if components["BC"] or components["BPC"]:
        balmer_comp = BalmerCombined(pars=pars["balmer_continuum"],
                                     BalmerContinuum=components["BC"],
                                     BalmerPseudocContinuum=components["BPC"])
        model.components.append(balmer_comp)
    if components["CALZETTI_EXT"] or components["SMC_EXT"] or components["MW_EXT"] or components["AGN_EXT"] or components["LMC_EXT"]:
        ext_comp = Extinction(MW=MW_ext, AGN=AGN_ext, LMC=LMC_ext, SMC=SMC_ext, Calzetti=Calzetti_ext)
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
              "comp_params": comp_params}

    if pname is None:
        now = datetime.datetime.now()
        pname = "model_{0}.pickle.gz".format(now.strftime("%Y%m%d_%M%S"))
    with gzip.open(pname, "wb") as model_output:
        model_output.write(pickle.dumps(p_data))
        print("Saved pickle file {0}".format(pname))
    make_plots_from_pickle(pname)

    t2 = datetime.datetime.now()
    print("executed in {}".format(t2-t1))

    return p_data
