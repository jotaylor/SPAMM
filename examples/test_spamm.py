#! /usr/bin/env python

import pdb
import numpy as np

import run_spamm, run_fe, run_nc, run_bc
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

# These values are just the midpoints of the parameter space in parameters.yaml
BC_PARAMS = {"bc_norm": 3e-14,
             "bc_tauBE": 1.,
             "bc_logNe": 5.5,
             "bc_loffset": 0.,
             "bc_lwidth": 5050.,
             "bc_Te": 50250.,
             "bc_lines": 201.5,
             "wl": np.arange(2000, 7000, 0.5)}
WL = np.arange(2000, 7000, 0.5)

LINEOUT = "#"*75

#-----------------------------------------------------------------------------#

def test_nc_fe(fe_params=FE_PARAMS):
    print("{0}\nTESTING NUCLEAR CONTINUUM + IRON\n{0}".format(LINEOUT))
    #fe_wl, fe_flux, fe_err, fe_p = run_fe.run_test("/user/jotaylor/git/spamm/Data/FakeData/Iron_comp/fakeFe1_deg.dat", redshift=0.5)
    fe_wl, fe_flux, fe_err, fe_p = run_fe.create_fe(fe_params)
    nc_wl, nc_flux, nc_err, nc_p = run_nc.combine_pl(WL)
    assert set(nc_wl-fe_wl) == {0}, "Wavelength scales do not match" 
    comb_wl = WL
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
    run_spamm.spamm_wlflux({"FE": True}, fe_wl, fe_flux, fe_err,
                           comp_params=fe_p)#, pname="fe.pickle.gz")

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

def test_bc_fromfile(datafile="/user/jotaylor/git/spamm/Data/FakeData/BaC_comp/FakeBac_lines04_deg.dat", 
                     redshift=0.2, bpc=True):
    print("{0}\nTESTING BALMER CONTINUUM\n{0}".format(LINEOUT))
    bc_wl, bc_flux, bc_err, bc_p = run_bc.run_test(datafile, redshift)
    run_spamm.spamm_wlflux({"BC": True, "BpC": True}, bc_wl, bc_flux, bc_err, 
                           comp_params=bc_p, pname="bc.pickle.gz")

#-----------------------------------------------------------------------------#

def test_bc(bc_params=None):
    print("{0}\nTESTING BALMER CONTINUUM\n{0}".format(LINEOUT))
    bc_wl, bc_flux, bc_err, bc_p = run_bc.create_bc(bc_params)

    run_spamm.spamm_wlflux({"BC": True, "BpC": True}, bc_wl, bc_flux, bc_err, 
                           n_walkers=100, n_iterations=500, comp_params=bc_p)

#-----------------------------------------------------------------------------#

def test_nc_bc(bc_params=BC_PARAMS):
    print("{0}\nTESTING NUCLEAR + BALMER CONTINUUM\n{0}".format(LINEOUT))
    bc_wl, bc_flux, bc_err, bc_p = run_bc.create_bc(bc_params)
    nc_wl, nc_flux, nc_err, nc_p = run_nc.combine_pl(WL)
    assert set(nc_wl-bc_wl) == {0}, "Wavelength scales do not match" 
    comb_wl = WL
    comb_flux = nc_flux + bc_flux
    comb_err = add_in_quadrature(nc_err, bc_err)

    comb_p = {**nc_p, **bc_p}
    
    print(comb_p) 
    run_spamm.spamm_wlflux({"PL": True, "BC": True, "BpC": True}, comb_wl, comb_flux, comb_err,
                           comp_params=comb_p)#, n_walkers=100, n_iterations=500)


#-----------------------------------------------------------------------------#

def test_nc_bc_fe(bc_params=BC_PARAMS, fe_params=FE_PARAMS):
    print("{0}\nTESTING NUCLEAR + BALMER CONTINUUM + IRON\n{0}".format(LINEOUT))
    fe_wl, fe_flux, fe_err, fe_p = run_fe.create_fe(fe_params)
    bc_wl, bc_flux, bc_err, bc_p = run_bc.create_bc(bc_params)
    nc_wl, nc_flux, nc_err, nc_p = run_nc.combine_pl(WL)
    assert set(nc_wl-bc_wl) == {0}, "NC and BC wavelength scales do not match" 
    assert set(nc_wl-fe_wl) == {0}, "NC and Fe wavelength scales do not match" 
    comb_wl = WL
    comb_flux = nc_flux + bc_flux + fe_flux
    comb_err = add_in_quadrature(nc_err, bc_err, fe_err)

    comb_p = {**nc_p, **bc_p, **fe_p}

    print(comb_p) 
    run_spamm.spamm_wlflux({"PL": True, "BC": True, "BpC": True, "FE": True}, comb_wl, comb_flux, comb_err,
                           comp_params=comb_p, n_walkers=100, n_iterations=500)

#-----------------------------------------------------------------------------#


if __name__ == "__main__":
#    test_nc_fe()
#    test_nc()
#    test_fe()
#    test_bc()
#    test_nc_bc()
    test_nc_bc_fe()



