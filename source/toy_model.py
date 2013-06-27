
from module import ComponentFit

s = Spectrum(z=z)

s.apply_mask(...)
s.min_wavelength()
s.wavelengths(scale=[log,linear])

host_galaxy_component = HostGalaxyComponent()
fe_emission_component = FeComponent()

r = Reddening()
r2 = AnotherReddening()

host_galaxy_component.reddening = r
fe_emission_component.reddening = r2

model = Model()
model.spectrum = s
model.components.append(host_galaxy_component)
model.components.append(fe_emission_component)

model.model_parameters["b"] = ...

model.build_model()
model.run_mcmc()

try:
	cf.run_mcmc_analysis(plot=False)
except MCMCDidNotConverge:
	...

cf.plot_results(directory="/a/b/c")

cf.mc_chain(parameter=x)	

	
