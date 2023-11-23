#! /usr/bin/env python

import os
import yaml
from astropy import units as u
ABSPATH = os.path.dirname(os.path.realpath(__file__))

def parse_pars(par_file=os.path.join(os.getcwd(), "parameters.yaml")):
    '''
    Read in SPAMM input parameters from input parameters file.

    Args:
        par_file (str): Location of parameters file.
    Returns:
        pars (dict): SPAMM input parameters.
    '''

    assert os.path.exists(par_file), f"Input parameters {par_file} is not in {os.getcwd()}"

    with open(par_file, "r") as f:
        pars = yaml.safe_load(f)
    
    pars["global"]["wl_unit"] = u.Unit(pars["global"]["wl_unit"])
    pars["global"]["flux_unit"] = u.Unit(pars["global"]["flux_unit"])
    
    return pars
