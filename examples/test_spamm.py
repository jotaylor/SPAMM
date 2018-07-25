#! /usr/bin/env python

import argparse
import numpy as np

import run_spamm, run_fe, run_nc, run_bc, run_hg
from utils.add_in_quadrature import add_in_quadrature

# The normalizations are drawn from a gaussian sample with mu=9.06e-15,
# sigma=3.08946e-15 (from 0->template max flux). fe_width is halfway 
# between range in parameters. WL is very close to template span (1075-7535)
WL = np.arange(1000, 10000, 0.5)

NC_PARAMS = {"wl": WL,
             "slope1": 2.3,
             "norm_PL": 5e-15,
             "broken_pl": False}

FE_PARAMS = {"fe_norm_1": 1.07988504e-14,
             "fe_norm_2": 6.91877436e-15,
             "fe_norm_3": 5e-15,# 8.68930476e-15, 
             "fe_width": 5450,
             "no_templates": 3,
             "wl": WL}

# These values are just the midpoints of the parameter space in parameters.yaml
BC_PARAMS = {"bc_norm": 3e-14,
             "bc_tauBE": 1.,
             "bc_logNe": 5.5,
             "bc_loffset": 0.,
             "bc_lwidth": 5050.,
             "bc_Te": 50250.,
             "bc_lines": 201.5,
             "wl": WL}

# These values are just the midpoints of the parameter space in parameters.yaml
HG_PARAMS = {"hg_norm_1": 5.e-17,
             "hg_norm_2": 5.5e-17,
             "hg_norm_3": 5.e-16,
             "hg_stellar_disp": 515,
             "no_templates": 3,
             "wl": WL}

LINEOUT = "#"*75

#-----------------------------------------------------------------------------#

def test_nc_fe(fe_params=FE_PARAMS):
    print("{0}\nTESTING NUCLEAR CONTINUUM + IRON\n{0}".format(LINEOUT))
    #fe_wl, fe_flux, fe_err, fe_p = run_fe.from_file("/user/jotaylor/git/spamm/Data/FakeData/Iron_comp/fakeFe1_deg.dat", redshift=0.5)
    fe_wl, fe_flux, fe_err, fe_p = run_fe.create_fe(fe_params)
    nc_wl, nc_flux, nc_err, nc_p = run_nc.combine_pl(WL)
    assert set(nc_wl-fe_wl) == {0}, "Wavelength scales do not match" 
    comb_wl = WL
    comb_flux = nc_flux + fe_flux
    comb_err = add_in_quadrature([nc_err, fe_err])

    comb_p = {**nc_p, **fe_p}

    run_spamm.spamm_wlflux({"PL": True, "FE": True}, comb_wl, comb_flux, comb_err,
                           comp_params=comb_p, n_walkers=100, n_iterations=500)
                           #, pname="nc_fe.pickle.gz")


#-----------------------------------------------------------------------------#

def test_fe_fromfile(datafile="/user/jotaylor/git/spamm/Data/FakeData/Iron_comp/fakeFe1_deg.dat", 
                     redshift=0.5):
    print("{0}\nTESTING IRON\n{0}".format(LINEOUT))
    fe_wl, fe_flux, fe_err, fe_p = run_fe.from_file(datafile, redshift)
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
    bc_wl, bc_flux, bc_err, bc_p = run_bc.from_file(datafile, redshift)
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
    comb_err = add_in_quadrature([nc_err, bc_err])

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
    comb_err = add_in_quadrature([nc_err, bc_err, fe_err])

    comb_p = {**nc_p, **bc_p, **fe_p}

    print(comb_p) 
    run_spamm.spamm_wlflux({"PL": True, "BC": True, "BpC": True, "FE": True}, comb_wl, comb_flux, comb_err,
                           comp_params=comb_p, n_walkers=400, n_iterations=400)

#-----------------------------------------------------------------------------#

def test_hg(hg_params=HG_PARAMS):
    print("{0}\nTESTING HOST GALAXY\n{0}".format(LINEOUT))
    hg_wl, hg_flux, hg_err, hg_p = run_hg.create_hg(hg_params)
    
    run_spamm.spamm_wlflux({"HOST": True}, hg_wl, hg_flux, hg_err, 
                           comp_params=hg_p) 

#-----------------------------------------------------------------------------#

def test_spamm(components=None, comp_params=None, n_walkers=30, n_iterations=500):
    """
    Args:
        components (list): Components to be added to the data spectrum. 
        Options are:
            - PL or NC
            - FE
            - BC or BpC
            - HOST or HG
    """

    if components is None:
        components = ["PL", "FE", "BC", "HG"]
        comp_params = {"PL": NC_PARAMS, "FE": FE_PARAMS, "BC": BC_PARAMS,
                       "HOST": HG_PARAMS}
    elif comp_params is None:
        comp_params = {"PL": NC_PARAMS, "FE": FE_PARAMS, "BC": BC_PARAMS,
                       "HOST": HG_PARAMS}
    
    all_wls = []
    all_fluxes = []
    all_errs = []
    comb_p = {}
    comp_names = {}
    for component in components:
        component = component.upper()
        if component == "PL" or component == "NC":
            comp_wl, comp_flux, comp_err, comp_p = run_nc.create_nc(comp_params["PL"])
            comp_names["PL"] = True
        elif component == "FE":
            comp_wl, comp_flux, comp_err, comp_p = run_fe.create_fe(comp_params["FE"])
            comp_names["FE"] = True
        elif component == "BC" or component == "BPC":
            comp_wl, comp_flux, comp_err, comp_p = run_bc.create_bc(comp_params["BC"])
            comp_names["BC"] = True
            comp_names["BpC"] = True
        elif component == "HG" or component == "HOST":
            comp_wl, comp_flux, comp_err, comp_p = run_hg.create_hg(comp_params["HOST"])
            comp_names["HOST"] = True
        all_fluxes.append(comp_flux)
        all_wls.append(comp_wl)
        all_errs.append(comp_err)
        comb_p = {**comb_p, **comp_p}

    comb_wl = WL #all_wls[0]
    comb_flux = np.sum(all_fluxes, axis=0)
    comb_err = add_in_quadrature(all_errs)
    
    print("{0}\nUsing components: {1}\nWith {2} walkers, {3} iterations\n{0}".format(LINEOUT, components, n_walkers, n_iterations))

#    return comb_wl, comb_flux, comb_err, all_fluxes

    run_spamm.spamm_wlflux(comp_names, comb_wl, comb_flux, comb_err, 
                           comp_params=comb_p, n_walkers=n_walkers,
                           n_iterations=n_iterations)

#-----------------------------------------------------------------------------#

def parse_comps(argcomp):
    if len(argcomp) == 1:
        if "," in argcomp[0]:
            comps = [x for x in argcomp[0].split(",")]
        else:
            comps = argcomp
    else:
        comps = argcomp

    return comps

#-----------------------------------------------------------------------------#


if __name__ == "__main__":
#    test_nc_fe()
#    test_nc()
#    test_fe()
#    test_bc()
#    test_nc_bc()
#    test_nc_bc_fe()
#    test_hg()

    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", nargs="*", 
                        help="List of components to use: can be  PL, FE, BC, HG")
    parser.add_argument("--n_walkers", dest="n_walkers", default=30,
                        help="Number of walkers")
    parser.add_argument("--n_iterations", dest="n_iterations", default=500,
                        help="Number of iterations per walker")
    args = parser.parse_args()

    comps = parse_comps(args.comp)
    test_spamm(components=comps, n_walkers=int(args.n_walkers), n_iterations=int(args.n_iterations))

