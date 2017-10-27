#! /usr/bin/env python

import os
import yaml

def parse_pars(par_file="parameters.yaml"):
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

    return pars
