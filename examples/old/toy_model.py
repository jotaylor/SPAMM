
from module import ComponentFit

s = Spectrum(z=z)

s.apply_mask(...)
s.min_wavelength()
s.wavelengths(scale=[log,linear])

host_galaxy_component = HostGalaxyComponent()
fe_emission_component = FeComponent()

r = ReddeningLaw()
r2 = AnotherReddeningLaw()

host_galaxy_component.reddening_law = r
fe_emission_component.reddening_law = r2
#fe_emission_component.prior("random")

model = Model()
model.spectrum = s
model.components.append(host_galaxy_component)
model.components.append(fe_emission_component)

model.run_mcmc(n_walkers=200)

try:
    cf.run_mcmc_analysis(plot=False)
except MCMCDidNotConverge:
    ...

cf.plot_results(directory="/a/b/c")

cf.mc_chain(parameter=x)	


