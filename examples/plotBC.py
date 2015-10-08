
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import matplotlib.pyplot as plt
import scipy as sp



import os
import sys
import gzip
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as pl

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuum import BalmerContinuum


datafile = 'testLBT.dat'
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
spectrum = Spectrum()
spectrum.wavelengths = sp.r_[1000:5200:10000j]
spectrum.flux = sp.ones(10000)
spectrum.flux_error = sp.ones(10000)	



model = Model()
bc_comp = BalmerContinuum()	
model.components.append(bc_comp)

model.data_spectrum = spectrum # add data

norm = 1.e-15
pvector = [ norm,0.5e4,0.1,0.0,0.5e8 ]

BCflux = model.model_flux(pvector)

nwv = 4861
m = abs(spectrum.wavelengths - nwv) == min(abs(spectrum.wavelengths - nwv))
BCflux /= BCflux[m]

plt.plot(spectrum.wavelengths,BCflux,'k|')
plt.plot([1000,5000],[norm,norm],'k--')
plt.plot([3646,3646],[0,norm*1.3],'r--')


x2,y2 = sp.genfromtxt('/home/rrlyrae/fausnaugh/repos/mcmc_deconvol/Data/FakeData/BaC_comp/FakeBac_lines01.dat',unpack=1)
x3,y3 = sp.genfromtxt('/home/rrlyrae/fausnaugh/AGNstorm/Denney2009templates/Matthias_temps/balc1001001.dat',unpack=1,usecols=(0,1))
x2,y2 = sp.genfromtxt('/home/rrlyrae/fausnaugh/repos/mcmc_deconvol/Data/FakeData/BaC_comp2/FakeBac-full_nrm1.dat',unpack=1)

m = abs(x2 - nwv) == min(abs(x2 - nwv))
y2 /= y2[m]

m = abs(x3 - 3646) == min(abs(x3 - 3646))
y3 /= y3[m]

plt.plot(x2,y2,'b')
#plt.plot(x3,y3,'r')

plt.plot([3646,3646],[0,3],'r--')


plt.show()

sp.savetxt('BC.txt',sp.c_[spectrum.wavelengths,BCflux])

flist = ['BC50.txt','BC200.txt','BC250.txt','BC400.txt','BC1000.txt']
for f in flist:
    x,y = sp.genfromtxt(f,unpack=1)
    plt.plot(x,y,label=f)



#plt.legend(loc='upper right')
#plt.show()

#plt.plot(spectrum.wavelengths,BCflux,'k')

#plt.plot([1000,5000],[norm,norm],'k--')
#plt.plot([3646,3646],[0,norm*1.3],'r--')

#plt.show()

