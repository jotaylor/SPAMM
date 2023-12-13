#!/usr/bin/env python

import os
# Set OMP_NUM_THREADS=1 to prevent NumPy's automatic parallelization
# which can interfere with emcee's parallelization. Recommended in emcee's documentation.
os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import dill as pickle
import datetime
import numpy as np
from specutils import Spectrum1D
import timeit

from utils.parse_pars import parse_pars
from utils.mask_utils import bool_mask
from spamm.analysis import make_plots_from_pickle
from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.PowerLawComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.NarrowComponent import NarrowComponent
#from spamm.components.ReddeningLaw import Extinction

# List of accepted component names for the model
ACCEPTED_COMPS = ["PL", "FE", "HG", "BC", "BPC", "NEL", "CALZETTI_EXT", "SMC_EXT", "MW_EXT", "AGN_EXT", "LMC_EXT"]

###############################################################################

def spamm(complist, inspectrum, redshift=0.0, mask=None, par_file=None, n_walkers=32, n_iterations=500, 
          outdir=None, picklefile=None, comp_params=None, parallel=False):
    """
    Runs the SPAMM analysis on a given spectrum with specified components.

    Args:
        complist (list):
            A list of components to model. Accepted component names are: 
            "PL", "FE", "HG", "BC", "BPC", "NEL", "CALZETTI_EXT", "SMC_EXT", 
            "MW_EXT", "AGN_EXT", "LMC_EXT".
        inspectrum (spamm.Spectrum, specutils.Spectrum1D, or tuple): 
            The input spectrum to model. If a tuple, it should be in the format 
            ((wavelength,), (flux,)) or ((wavelength,), (flux,), (flux_error,)).
        mask (numpy.ndarray, list of bools, or list of tuples/lists, optional): 
            A mask for the input spectrum. If a boolean array or list, `True` 
            indicates a valid data point and `False` a point to be ignored. 
            If a list of tuples/lists, each tuple/list should contain two elements 
            representing a range in the spectrum to be included. If None, no mask is applied.
        par_file (str, optional): 
            Path to the parameters file. If None, default parameters are used.
        n_walkers (int, optional): 
            Number of walkers to use in the MCMC analysis.
        n_iterations (int, optional): 
            Number of iterations for each MCMC walker.
        outdir (str, optional): 
            Directory to save output files. If None, a directory is created based on the current run time.
        picklefile (str, optional): 
            Name of the output pickle file. If None, a name is generated based on the current run time.
        comp_params (dict, optional): 
            Known values of component parameters. Contains the known values of component parameters, 
            with keys defined in each of the individual run scripts (run_XX.py).
            If None, the actual values of parameters will not be plotted.

    Returns:
        dict: 
            A dictionary containing the model and component parameters.

    Raises:
        ValueError: 
            If an invalid component is specified in `complist`.
    """
    start_time = timeit.default_timer()

    # Parse parameter file 
    if par_file is None:
        pars = parse_pars()
    else:
        pars = parse_pars(par_file)

    # Convert component list to uppercase and create a dictionary of components
    complist = [x.upper() for x in complist]
    components = {k:(True if k in complist else False) for k in ACCEPTED_COMPS}
    
    # Process input spectrum and extract relevant data
    # Handles Spectrum, Spectrum1D objects, and tuple formats
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
        spectrum = Spectrum(spectral_axis=wl, flux=flux, flux_error=flux_error)

    # De-redshift wavelength array
    wl /= (1.0 + redshift)

    # If comp_params is not provided, initialize it as an empty dictionary
    if comp_params is None:
        comp_params = {}

    # Populate comp_params with values from the input spectrum, 
    # but only for those parameters that are not already present
    for param_name, param_value in zip(("wl", "flux", "err", "components"), (wl, flux, flux_error, components)):
        if param_name not in comp_params:
            comp_params[param_name] = param_value

    # Initialize the model
    model = Model()

    # Add each component to the model's components if it should be included in the model.
    if components["PL"]:
        is_broken = comp_params.get("broken_pl", False)
        nuclear_comp = NuclearContinuumComponent(pars=pars["nuclear_continuum"], broken=is_broken)
        model.components.append(nuclear_comp)

    if components["FE"]:
        fe_comp = FeComponent(pars=pars["fe_forest"])
        model.components.append(fe_comp)

    if components["HG"]:
        host_galaxy_comp = HostGalaxyComponent(pars=pars["host_galaxy"])
        model.components.append(host_galaxy_comp)

    if components["BC"] or components["BPC"]:
        balmer_comp = BalmerCombined(pars=pars["balmer_continuum"],
                                     BalmerContinuum=components["BC"],
                                     BalmerPseudocContinuum=components["BPC"])
        model.components.append(balmer_comp)

    if components["NEL"]:
        narrow_comp = NarrowComponent(pars=pars["narrow_lines"])
        model.components.append(narrow_comp)

    # TODO: MW_ext, AGN_ext etc. are all undefined?
    #if components["CALZETTI_EXT"] or components["SMC_EXT"] or components["MW_EXT"] or components["AGN_EXT"] or components["LMC_EXT"]:
    #    ext_comp = Extinction(MW=MW_ext, AGN=AGN_ext, LMC=LMC_ext, SMC=SMC_ext, Calzetti=Calzetti_ext)
    #    model.components.append(ext_comp)

    if mask is not None:
        # Check if mask is a boolean array/list
        if (isinstance(mask, np.ndarray) or isinstance(mask, list)) and all(isinstance(x, bool) for x in mask):
            boolmask = np.array(mask)
        # Check if mask is a list of tuples/lists
        elif isinstance(mask, (list, np.ndarray)) and all(isinstance(x, (list, tuple)) for x in mask):
            boolmask = bool_mask(wl, mask)
        else:
            raise TypeError("Mask must be a boolean array/list or a list of tuples/lists.")
        
        # Ensure wavelength and mask arrays have same length
        if len(wl) != len(boolmask):
            raise ValueError(f"Wavelength array length ({len(wl)}) and mask array length ({len(boolmask)}) must be the same.")
        # Ensure flux and mask arrays have same length
        if len(flux) != len(boolmask):
            raise ValueError(f"Flux array length ({len(flux)}) and mask array length ({len(boolmask)}) must be the same.")
    else:
        boolmask = None

    model.data_spectrum = spectrum
    model.mask = boolmask

    data_spectrum = spectrum
    components = model.components

    # Run mcmc
    model.run_mcmc(data_spectrum=data_spectrum, 
                   components=components,
                   mask=boolmask,
                   n_walkers=n_walkers,
                   n_iterations=n_iterations,
                   parallel=parallel)
    
    print(f"[SPAMM]: Mean acceptance fraction: {np.mean(model.sampler.acceptance_fraction):.3f}")

    # Save the model and component parameters to a pickle file
    p_data = {"model": model, "comp_params": comp_params}
    
    # Get the current date and time and format as a string
    nowdt = datetime.datetime.now()
    now = nowdt.strftime("%Y%m%d_%M%S")

    # If no picklefile name is provided, create one using the current date and time
    if not picklefile:
        picklefile = f"model_{now}.pickle.gz"
    else:
        # If a picklefile name is provided, ensure it has the correct extension
        picklefile = os.path.basename(picklefile)
        if not picklefile.endswith((".pickle", ".p", ".gz")):
            picklefile += ".pickle.gz"
        elif picklefile.endswith((".pickle", ".p")):
            picklefile += ".gz"

    # If no output directory is provided, create one using the current date and time
    if outdir is None:
        outdir = now

    # If the output directory does not exist, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Create the full path for the pickle file
    pname = os.path.join(outdir, picklefile)
    
    # Save the model and component parameters to a gzipped pickle file
    with gzip.open(pname, "wb") as model_output:
        model_output.write(pickle.dumps(p_data))
        print(f"[SPAMM]: Saved pickle file: {pname}")

    # Generate plots from the saved pickle file
    make_plots_from_pickle(pname, outdir)

    # Calculate and print the total execution time
    end_time = timeit.default_timer()
    print(f"[SPAMM]: Execution time: {(end_time - start_time):.3f} seconds")

    return p_data

###############################################################################

def parse_comps(argcomp):
    if len(argcomp) == 1:
        if "," in argcomp[0]:
            comps = [x for x in argcomp[0].split(",")]
        else:
            comps = argcomp
    else:
        comps = argcomp

    return comps

###############################################################################

# TODO: What is this?
# NOT SUPPORTED YET #

# if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("inspectrum", help="Input spectrum file", 
#                        type=str) 
#    parser.add_argument("--comp", nargs="*",
#                        help="List of components to use: can be  PL, FE, BC, HG")
#    parser.add_argument("--n_walkers", dest="n_walkers", default=30,
#                        help="Number of walkers")
#    parser.add_argument("--n_iterations", dest="n_iterations", default=500,
#                        help="Number of iterations per walker")
#    args = parser.parse_args()


#    comps = parse_comps(args.comp)
#    spamm(complist=comps, n_walkers=int(args.n_walkers), n_iterations=int(args.n_iterations))
