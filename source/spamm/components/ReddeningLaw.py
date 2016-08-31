#!/usr/bin/python

import sys
import numpy as np
from .ComponentBase import Component

def Calzetti_ext(spectrum=None, parameters=None):
	if spectrum is None:
		raise Exception("Need a data spectrum")
	ext= [1.]*len(spectrum.wavelengths)
	Rv = 4.05
	for j in range(len(spectrum.wavelengths)):
		wavelengths_um = spectrum.wavelengths[j]/10000.
		if (wavelengths_um >= 0.12) & (wavelengths_um < 0.63):
			k = 2.659*(-1.857+1.040/wavelengths_um)+4.05
			ext[j] = pow(10,-0.4*parameters[0]*k)
		if (wavelengths_um >= 0.63) & (wavelengths_um <= 2.2):
			k = 2.659*(-2.156+1.509/wavelengths_um-0.198/pow(wavelengths_um,2)+0.11/pow(wavelengths_um,3))+4.05
			ext[j] = pow(10,-0.4*parameters[0]*k)
	return ext
	
def LMC_Fitzpatrick_ext(spectrum=None, parameters=None): #Fitzpatrick 1986
	'''Large Magellanic Cloud extinction curve defined in Fitzpatrick 1986'''
	if spectrum is None:
		raise Exception("Need a data spectrum")
	ext= [1.]*len(spectrum.wavelengths)
	C1 = -0.69
	C2 = 0.89 #micrometres
	C3 = 2.55 #micrometres^-2
	x_0 = 4.608 #micrometres^-1
	gamma = 0.994 #micrometres^-1
	Rv = 3.1
	
	for j in range(len(spectrum.wavelengths)):
		wavelengths_um = spectrum.wavelengths[j]/10000.
		x = pow(wavelengths_um,-1)
		x2 = pow(x,2)
		D = x2/(pow(x2-pow(x_0,2),2)+x2*pow(gamma,2))
		F = 0.5392*pow((x-5.9),2)+0.05644*pow((x-5.9),3)
		if (x >= 5.9):
			C4 = 0.50 #micrometres^-1 
		else:
			C4 =0.0
		k = C1+Rv+C2*x+C3*x*D+C4*F
		ext[j] = pow(10,-0.4*parameters[0]*k)
	return ext
	
def MW_Seaton_ext(spectrum=None, parameters=None): #Seaton 1979
	'''Milky Way extinction curve defined in Seaton 1979'''
	if spectrum is None:
		raise Exception("Need a data spectrum")
	ext= [1.]*len(spectrum.wavelengths)
	C1 = -0.38
	C2 = 0.74 #micrometres
	C3 = 3.96 #micrometres^-2
	C4 = 0.26 #micrometres^-1
	x_0 = 4.595 #micrometres^-1
	gamma = 1.051 #micrometres^-1
	Rv = 3.1
	for j in range(len(spectrum.wavelengths)):
		wavelengths_um = spectrum.wavelengths[j]/10000.
		x = pow(wavelengths_um,-1)
		x2 = pow(x,2)
		D = x2/(pow(x2-pow(x_0,2),2)+x2*pow(gamma,2))
		F = 0.5392*pow((x-5.9),2)+0.05644*pow((x-5.9),3)
		if (x >= 5.9):
			C4 = 0.26 #micrometres^-1 
		else:
			C4 =0.0
		k = C1+Rv+C2*x+C3*x*D+C4*F
		ext[j] = pow(10,-0.4*parameters[0]*k)
	return ext
	
def SMC_Gordon_ext(spectrum=None, parameters=None): #Gordon 2003
	'''Small Magellanic Cloud extinction curve defined in Gordon 2003'''
	if spectrum is None:
		raise Exception("Need a data spectrum")
	ext= [1.]*len(spectrum.wavelengths)
	C1 = -4.96
	C2 = 2.26 #micrometres
	C3 = 0.39 #micrometres^-2
	C4 = 0.46 #micrometres^-1
	x_0 = 4.6 #micrometres^-1
	gamma = 1.0 #micrometres^-1
	Rv = 2.74
	for j in range(len(spectrum.wavelengths)):
		wavelengths_um = spectrum.wavelengths[j]/10000.
		x = pow(wavelengths_um,-1)
		x2 = pow(x,2)
		D = x2/(pow(x2-pow(x_0,2),2)+x2*pow(gamma,2))
		F = 0.5392*pow((x-5.9),2)+0.05644*pow((x-5.9),3)
		if (x >= 5.9):
			C4 = 0.26 #micrometres^-1 
		else:
			C4 =0.0
		k = C1+Rv+C2*x+C3*x*D+C4*F
		ext[j] = pow(10,-0.4*parameters[0]*k)
	return ext
	
def AGN_Gaskell_ext(spectrum=None, parameters=None): #Gaskell and Benker 2007
	'''Active galactic nuclei extinction curve defined in Gaskell and Benker 2007.
			Much flatter than galactic extinction curves.'''
	if spectrum is None:
		raise Exception("Need a data spectrum")
	ext = [1.]*len(spectrum.wavelengths)
	A = 0.000843
	B = -0.02496
	C = 0.2919
	D = -1.815
	E = 6.83
	F = -7.92
	Rv = 5.0
	for j in range(len(spectrum.wavelengths)):
		wavelengths_um = spectrum.wavelengths[j]/10000.
		x = pow(wavelengths_um,-1)
		if (x>=1.5) & (x<8):
			k = A*pow(x,5)+B*pow(x,4)+C*pow(x,3)+D*pow(x,2)+E*x+F+Rv
			ext[j] = pow(10,-0.4*parameters[0]*k)
	return ext
		

class Extinction(Component):
	'''
	Description of Extinction class goes here.
	'''
	def __init__(self,MW=False,AGN=False,LMC=False,SMC=False, Calzetti=False):
		super(Extinction, self).__init__()
		self.model_parameter_names = list()
		self.model_parameter_names.append("E(B-V)")
		
		self.EBV_min=None
		self.EBV_max=None
		
		self.name = "Extinction"
		
		self.MW = MW
		self.AGN = AGN
		self.LMC = LMC
		self.SMC = SMC
		self.Calzetti = Calzetti

		
		self._k = None
		
	@property
	def is_analytic(self):
		return True
		
	def initial_values(self, spectrum=None):
		'''
		Needs to sample from prior distribution.
		These are the first guess for the parameters to be fit for in emcee, unless specified elsewhere.
		'''

		#  calculate/define minimum and maximum values for each parameter.
		if self.EBV_min == None or self.EBV_max == None:
			self.EBV_min = 0
			self.EBV_max = 2.
		EBV_init = np.random.uniform(low=self.EBV_min,
					    high=self.EBV_max)
		return [EBV_init]
		
	def ln_priors(self, params):
		'''
		Return a list of the ln of all of the priors.
		
		@param params
		'''
		
		# need to return parameters as a list in the correct order
		ln_priors = list()
		
		#get the parameters
		EBV = params[0]
		
		#Flat priors, appended in order
		if self.EBV_min < EBV < self.EBV_max:
			ln_priors.append(0)
		else:
			ln_priors.append(-np.inf)
			
		return ln_priors
	
	def flux(self, spectrum=None):
		'''
		Returns the flux for this component for a given wavelength grid
		and parameters. Will use the initial parameters if none are specified.
		'''
		flux = [0.]*len(spectrum.wavelengths)
		return flux
		
	
	def extinction(self, spectrum=None, parameters=None):
		if spectrum is None:
			raise Exception("Need a data spectrum")
			sys.exit()
		if self.MW:
			ext = MW_Seaton_ext(spectrum=spectrum, parameters=parameters)
		if self.AGN:
			ext = AGN_Gaskell_ext(spectrum=spectrum, parameters=parameters)
		if self.LMC:
			ext = LMC_Fitzpatrick_ext(spectrum=spectrum, parameters=parameters)
		if self.SMC:
			ext = SMC_Gordon_ext(spectrum=spectrum, parameters=parameters)
		if self.Calzetti:
			ext = Calzetti_ext(spectrum=spectrum, parameters=parameters)
		if (not self.MW) & (not self.AGN) & (not self.LMC) & (not self.SMC) & (not self.Calzetti):
			ext = [1.]*len(spectrum.wavelengths)
		return ext

	
