About SPAMM
=======================
Dear SPAMMer, welcome!

**SPAMM** (**S**\ pectral **P**\ roperties of **A**\ GN **M**\ odeled through **M**\ CMC) is an AGN spectral decomposition package that utilizes Bayesian methods, specifically the MCMC algorithm `emcee`_.


How SPAMM got started
---------------------
AGN spectra suffer from blending of multiple emission and absorption components that arise 
from physically distinct sources at a variety of distance scales from the central BH 
(e.g. accretion disk, host galaxy, narrow and broad emission lines). Blended features in 
AGN spectra can lead to biases that can affect BH mass measurements by more than an order 
of magnitude. Using spectral decomposition, a model is defined that attempts to 
simultaneously reproduce all AGN components. Most spectral decomposition techniques use 
best-fit optimization algorithms that are extremely inefficient in sampling the parameter 
space and do not provide robust estimates of the uncertainties and correlations between 
different spectral components. These approaches also rely on assumptions that some spectral 
regions are affected by one component only, and can be used to "fix" the parameters for 
those components. Our innovative code, SPAMM, addresses all of these issues. SPAMM is an 
open-source python package that uses a Bayesian approach and MCMC techniques to perform 
spectral decomposition on AGN spectra while providing a full description of uncertainties in the models.

