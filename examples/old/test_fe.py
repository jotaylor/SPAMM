#!/usr/bin/env python

"""
Test the Iron Component code. This code can be run from teh command line:
> python test_fe.py --datafile /user/jotaylor/git/spamm//Data/FakeData/Iron_comp/fakeFe1_deg.dat
--redshift 0.5

"""

import os
import dill as pickle
import datetime
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as pl
import time
import argparse
import glob

import test_components
from utils.parse_pars import parse_pars

PARS = parse_pars()["fe_forest"]

#-----------------------------------------------------------------------------#

def run_test(datafile, n_walkers=30, n_iterations=500, redshift=None, 
             scale=None, subset=False, pname=None):
    t1 = datetime.datetime.now()
    print(PARS)
    templates = glob.glob(os.path.join(PARS["fe_templates"], "*"))
    print("Using datafile: {}".format(datafile))
    print("Templates = {}".format(templates))
    print("Are the parameters in utils good? If not, ctrl+c, modify them, and rerun")
    time.sleep(5)
    
    try:
        wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
    except ValueError:
        wavelengths, flux = np.loadtxt(datafile, unpack=True)
        flux_err = flux*0.05
    
    if redshift is not None:
        print("Correcting for redshift {}".format(redshift))
        wavelengths /= (1+float(redshift))
    if scale is not None:
        print("Scaling flux by {}".format(scale))
        flux *= scale
        flux_err *= scale
    if subset:
        minwl = 1e10
        maxwl = 0.
        for template in templates:
            with open(template) as f:
                t_wl, t_flux = np.loadtxt(template, unpack=True)
                maxwl = max(maxwl, max(t_wl))
                minwl = min(minwl, min(t_wl))
        print("Wavelength range of templates {} =\n{}:{}".format(templates,
              minwl, maxwl))
        print("Only using this range on datafile")
        inds = np.where((wavelengths >= minwl) & (wavelengths <= maxwl))
        wavelengths = wavelengths[inds]
        flux = flux[inds]
        flux_err = flux_err[inds]

    params = {"wl": wavelengths,
              "flux": flux,
              "err": flux_err,
              "pname": pname,
              "datafile": datafile}

    return wavelengths, flux, flux_err, params
#    test_components.perform_test(components={"FE": True}, comp_params=params)
    t2 = datetime.datetime.now()    
    print("executed in {}".format(t2-t1))
#-----------------------------------------------------------------------------#


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkers", dest="n_walkers", default=30,
                        help="Number of emcee walkers")
    parser.add_argument("--iter", dest="n_iterations", default=500,
                        help="Number of emcee iterations")
    parser.add_argument("--datafile", dest="datafile",
                        default="/user/jotaylor/git/spamm/Data/FakeData/Iron_comp/fakeFe1_deg.dat",
                        help="Path to datafile")
    parser.add_argument("--redshift", dest="redshift", default=None,
                        help="If not None, correct for datafile redshift by defined number")
    parser.add_argument("--scale", dest="scale_data", default=None,
                        help="If not None, scale input data by defined number")
    parser.add_argument("--subset", default=False, action="store_true",
                        help="Switch to match datafile WL range to template WL range")
    parser.add_argument("--pname", dest="pname", default=None,
                        help="Name of output pickle file")
    args = parser.parse_args()
    
    run_test(args.datafile, args.n_walkers, args.n_iterations, 
             args.redshift, args.scale_data, args.subset, args.pname)

    
