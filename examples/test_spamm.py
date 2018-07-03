#! /usr/bin/env python

import pdb
import numpy as np

import run_spamm, run_fe, run_nc
from utils.add_in_quadrature import add_in_quadrature

# The normalizations are drawn from a gaussian sample with mu=9.06e-15,
# sigma=3.08946e-15 (from 0->template max flux). fe_width is halfway 
# between range in parameters. WL is very close to template span (1075-7535)

FE_PARAMS = {"fe_norm_1": 1.07988504e-14,
             "fe_norm_2": 8.68930476e-15,
             "fe_norm_3": 6.91877436e-15,
             "fe_width": 5450,
             "no_templates": 3,
             "wl": np.arange(2000, 7000, .5)}
LINEOUT = "#"*75

#-----------------------------------------------------------------------------#

def test_nc_fe(fe_params=FE_PARAMS):
    print("{0}\nTESTING NUCLEAR CONTINUUM + IRON\n{0}".format(LINEOUT))
    #fe_wl, fe_flux, fe_err, fe_p = run_fe.run_test("/user/jotaylor/git/spamm/Data/FakeData/Iron_comp/fakeFe1_deg.dat", redshift=0.5)
    fe_wl, fe_flux, fe_err, fe_p = run_fe.create_fe(fe_params)
    nc_wl, nc_flux, nc_err, nc_p = run_nc.combine_pl(fe_wl)
    assert set(nc_wl-fe_wl) == {0}, "Wavelength scales do not match" 
    comb_wl = nc_wl
    comb_flux = nc_flux + fe_flux
    comb_err = add_in_quadrature(nc_err, fe_err)

    comb_p = {**nc_p, **fe_p}

    run_spamm.spamm_wlflux({"PL": True, "FE": True}, comb_wl, comb_flux, comb_err,
                           comp_params=comb_p, n_walkers=100, n_iterations=500)
                           #, pname="nc_fe.pickle.gz")


#-----------------------------------------------------------------------------#

def test_fe_fromfile(datafile="/user/jotaylor/git/spamm/Data/FakeData/Iron_comp/fakeFe1_deg.da.dat", 
                     redshift=0.5):
    print("{0}\nTESTING IRON\n{0}".format(LINEOUT))
    fe_wl, fe_flux, fe_err, fe_p = run_fe.run_test(datafile, redshift)
    run_spamm.spamm_wlflux({"FE": True}, fe_wl, fe_flux, fe_err)#, pname="fe.pickle.gz")

#-----------------------------------------------------------------------------#

def test_fe(fe_params=None):
    print("{0}\nTESTING IRON\n{0}".format(LINEOUT))
    fe_wl, fe_flux, fe_err, fe_p = run_fe.create_fe(fe_params)
    
    run_spamm.spamm_wlflux({"FE": True}, fe_wl, fe_flux, fe_err, 
                           comp_params=fe_p)

#-----------------------------------------------------------------------------#

def test_nc():
    print("{0}\nTESTING NUCLEAR CONTINUUM\n{0}".format(LINEOUT))
    nc_wl, nc_flux, nc_err, nc_p = run_nc.combine_pl()
    run_spamm.spamm_wlflux({"PL": True}, nc_wl, nc_flux, nc_err,
                           comp_params=nc_p)
                           #pname="nc.pickle.gz")

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    test_nc_fe()
#    test_nc()
#    test_fe()
