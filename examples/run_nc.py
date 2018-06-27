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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
from astropy import units
from collections import OrderedDict

import test_components

randoms = {"wl_lo": [1000, 1500, 2000, 2500, 3000],
           "wl_hi": [6000, 6500, 7000, 7500, 8000],
           "norm_PL": np.array([0.1, 1., 2., 3., 4., 5., 6., 7., 8., 9.])*1e-15,
           "slope_factor": [1., -1], 
           "slope1": np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 2.9]),
           "slope2": np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 2.9]),
           "err_factor": [.005, .01, .05, .1],
           "broken_pl": [True, False]}

params = OrderedDict([("wl", None),
                      ("norm_PL", None),
                      ("slope_factor", None),
                      ("slope1", None),
                      ("slope2", None),
                      ("wave_break", None),
                      ("err_factor", None),
                      ("broken_pl", None),
                      ("pname", None)])

#-----------------------------------------------------------------------------#

def err_gauss(mean, std, num, divisor):
    #y_err = err_guass(0., factor, len(y), 2.)
    gauss = np.random.normal(mean, std, num)
    
    return gauss/divisor

#-----------------------------------------------------------------------------#

def err_percent(y, percent):
    if percent > 1.:
        percent /= 100.

    return y*percent

#-----------------------------------------------------------------------------#

def randomize(param, params):
    """ This requires params to be an ORDERED DICT!!!!!"""
    from random import choice as r

    if param == "wl":
        wl_lo = r(randoms["wl_lo"])
        wl_hi = r(randoms["wl_hi"])
        return np.arange(wl_lo, wl_hi)
    
    elif param == "slope1" or param == "slope2":
        if params["slope_factor"] is None:
            params["slope_factor"] = r(randoms["slope_factor"])
        return r(randoms[param]) * params["slope_factor"]
    
    elif param == "wave_break":
        return r(params["wl"])
    
    elif param == "pname":
        return params["pname"]
         
    elif param not in randoms:
        print("ERROR: {0} is not in pre-defined random dictionary, exiting".format(param))
        sys.exit()
    else:        
        return r(randoms[param])    

#-----------------------------------------------------------------------------#

def run_test(params):
    import make_powerlaw
    
    for param in params:
        if param is None:
            params[param] = randomize(param, params)
   
    if params["broken_pl"] is False:
        y = make_powerlaw.generate_pl(params["wl"], params["norm_PL"], 
                                      params["wave_break"], params["slope1"])
    else:
        y = make_powerlaw.generate_pl(params["wl"], norm=params["norm_PL"], 
                                      slope1=params["slope1"],
                                      slope2=params["slope2"], wl_break=params["wave_break"],
                                      broken=True)
    
    y_err = err_percent(y, params["err_factor"])

#    pl.errorbar(x, y, yerr=y_err)
#    pl.show()
#    this = input("press enter to clear plot")
#    pl.clf()
    
    params["wl"] = params["wl"]
    params["flux"] = y
    params["err"] = y_err

    print(params)
   
    return params["wl"], y, y_err, params 
    #test_components.perform_test(components={"PL": True}, comp_params=params)
    #print(params)
     
#-----------------------------------------------------------------------------#

def ten_tests():
    """10 Controlled broken PL tests"""

    percerr = [.1, .05, .01, .005]
    
    # 10 Broken power law tests
    xs = [[1000,10000], [2000, 5000], [3000, 8000], [4000, 10000], [4000, 7000],
          [1000, 4000], [7000, 10000], [2500, 4500], [5500, 8500], [6500, 8000]]
    wl_breaks = [7000, 3000, 6000, 5000, 6000, 2000, 9000, 4000, 6000, 7500]
    slopes1 = [-2.2, -1.4, -0.2, -0.9, -1.8, 2.1,  1.3,  0.3,  .9,   0.4]
    slopes2 = [-0.1, -0.5, 0.,   -0.5, -1.0, 1.5,  1.0,  0.,   0.5,  0.1]
    norms = [1e-15, 1.4e-14, 1.8e-17, 3.5e-16, 5.7e-15, 6.6e-15, 3.3e-13, 4.9e-15, 9.1e-16, 2.1e-17]

    for i in range(10):
        for j in range(len(percerr)):
            params["wl"] = np.arange(xs[i][0], xs[i][1])
            params["slope1"] = slopes1[i]
            params["slope2"] = slopes2[i]
            params["slope_factor"] = 1.
            params["norm_PL"] = norms[i]
            params["err_factor"] = percerr[j]
            params["broken_pl"] = True
            params["wave_break"] = wl_breaks[i]
        
            w, f, f_err, p = run_test(params)
            
    return w, f, f_err, p

#-----------------------------------------------------------------------------#

def pl1():
    # POWER LAW 1 from excel file
    params["wl"] = np.arange(1159, 2003)
    params["slope1"] = -2.2
    params["slope_factor"] = 1.
    params["norm_PL"] = 1.8e-17
    params["err_factor"] = 0.05
    params["broken_pl"] = False
    params["wave_break"] = 1581 
    params["pname"] = "fakepowerlaw1_05err.pickle.gz"
    
    w, f, f_err, p = run_test(params)
    
    return w, f, f_err, p 

#-----------------------------------------------------------------------------#

def combine_pl_shallow():
    # To combine with other componenets for complete testing.
    # Wavelength range of fakeFe1_deg.dat
    params["wl"] = np.arange(1650, 12000, .75)
    params["slope1"] = 1.7
    params["slope_factor"] = 1.
    params["norm_PL"] = 1.5e-14
    params["err_factor"] = 0.05
    params["broken_pl"] = False
    params["wave_break"] = 2000
    params["pname"] = "powerlaw_combineshallow.pickle.gz"

    w, f, f_err, p = run_test(params)
    
    return w, f, f_err, p 

#-----------------------------------------------------------------------------#

def combine_pl(wl):
    # To combine with other componenets for complete testing.
    params["wl"] = wl
    params["slope1"] = 2.5
    params["slope_factor"] = 1.
    params["norm_PL"] = 5e-15
    params["err_factor"] = 0.05
    params["broken_pl"] = False
    params["wave_break"] = 6000
    params["pname"] = "powerlaw_combine.pickle.gz"

    w, f, f_err, p = run_test(params)
    
    return w, f, f_err, p 

#-----------------------------------------------------------------------------#

# POWER LAW 2 from excel file
#    pnames = ["fakepowerlaw2_{0}err.pickle.gz".format(x) for x in percerr]
#    for i in range(len(percerr)):
#        run_test(factor=1., x=np.arange(4300, 5800), norms=np.array([3.5e-16]), 
#                     slopes1_p=np.array([0.1]), slopes1_n=None,
#                     wave_break=5050, err=percerr[i], broken_pl=False, pname=pnames[i])

# Totally random
#    wl_ranges = [(1000, 3000), (4000, 8000), (1000, 10000), (1500, 5500)]
#    for wla,wlb in wl_ranges:
#        for i in range(len(percerr)):
#            pname = "powerlaw_{0}-{1}_{2}err.pickle.gz".format(wla, wlb, percerr[i])
#            run_test(factor=1e-15, x=np.arange(wla, wlb), err=percerr[i],
#                         pname=pname)

# Controlled broken PL
#    run_test(factor=1., x=np.arange(1000,10000), wl_break=7000, 
#                 slopes1_n=np.array([1.3]), slopes1_p=None, 
#                 slopes2_n=np.array([0]), slopes2_p=None,
#                 broken_pl=True, err=.1, norms=np.array([1e-15]))
#

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    print("Running with completely random parameters...")
    ten_tests()

#    for param in params:
#        r_p = randomize(param, params)
#        params[param] = r_p
#    run_test(params)

