#! /usr/bin/env python

import os
import yaml
from astropy import units as u
ABSPATH = os.path.dirname(os.path.realpath(__file__))

def parse_pars(par_file=os.path.join(ABSPATH, "parameters.yaml")):
    '''
    Read in SPAMM input parameters from input parameters file.

    Args:
        par_file (str): Location of parameters file.
    Returns:
        pars (dict): SPAMM input parameters.
    '''

    assert os.path.exists(par_file), \
    "Input parameters {0} is not in {1}".format(par_file, os.getcwd())

    with open(par_file, "r") as f:
        pars = yaml.load(f)
    
    pars["global"]["wl_unit"] = u.Unit(pars["global"]["wl_unit"])
    pars["global"]["flux_unit"] = u.Unit(pars["global"]["flux_unit"])
    
    return pars
