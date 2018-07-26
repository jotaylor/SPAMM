#!/usr/bin/env python

import gzip
import dill as pickle
import datetime

from analyze_model_run import make_chain_plots
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction

#-----------------------------------------------------------------------------#

def spamm_wlflux(components, wl, flux, flux_err=None, 
                 n_walkers=30, n_iterations=500, 
                 pname=None, comp_params=None):
    """
    Args:
        components (dictionary): A dictionary with at least one component to 
            model, e.g. {"FE": True}. Accepted key values are:
                - PL
                - FE
                - HOST
                - BC
                - BpC
                - Calzetti_ext
                - SMC_ext
                - MW_ext
                - AGN_ext
                - LMC_ext
        wl (array-like): Wavelength of input data spectrum.
        flux (array-like): Flux of input data spectrum.
        flux_err (array-like): Error on flux measurement.
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
    for c in ["PL", "FE", "HOST", "BC", "BpC", "Calzetti_ext", "SMC_ext", "MW_ext", "AGN_ext", "LMC_ext"]:
        if c not in components:
            components[c] = False

    if flux_err is None:
        flux_err = flux*0.05
    
    spectrum = Spectrum.from_array(flux, uncertainty=flux_err)
    spectrum.dispersion = wl#*units.angstrom
    spectrum.flux_error = flux_err    
    
    if comp_params is None:
        comp_params = {}
    for k,v in zip(("wl", "flux", "err", "components"), (wl, flux, flux_err, components)):
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
            nuclear_comp = NuclearContinuumComponent(broken=brokenPL)
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
