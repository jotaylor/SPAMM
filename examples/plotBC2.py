
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
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl

import triangle

sys.path.append(os.path.abspath("../source"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuum import BalmerContinuum
from spamm.components.BalmerPseudoContinuum import BalmerPseudoContinuum


datafile = 'testLBT.dat'
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
spectrum = Spectrum()
spectrum.wavelengths = sp.r_[1000:7200:10000j]
spectrum.flux = sp.ones(10000)
spectrum.flux_error = sp.ones(10000)	



model = Model()
bc_comp = BalmerContinuum()	
bpc_comp = BalmerPseudoContinuum()
model.components.append(bc_comp)
model.components.append(bpc_comp)

model.data_spectrum = spectrum # add data

norm = 1.e-15
lwidth = 3.e8
pvector = [ norm, 1.0e4, 0.1, 0.0, lwidth, #balmer continuum
                    norm/2.5, 1.5e4, 1.e10, 0.0, lwidth] #pseudo continuum

BCflux = model.model_flux(pvector)

nwv = 3500
m = abs(spectrum.wavelengths - nwv) == min(abs(spectrum.wavelengths - nwv))
BCflux /= BCflux[m]

#F,(ax1,ax2) = plt.subplots(2,1)
F,(ax1) = plt.subplots(1,1)

ax1.plot(spectrum.wavelengths,BCflux,'k',label='spamm')
ax1.plot([1000,5000],[norm,norm],'k--')
ax1.plot([3646,3646],[0,norm*1.3],'r--')


#x2,y2 = sp.genfromtxt('FakeBac_lines01.dat',unpack=1)
x3,y3 = sp.genfromtxt('mattias_balc1001001.dat',unpack=1,usecols=(0,1))
x2,y2 = sp.genfromtxt('FakeBac_lines01_deg.dat',usecols=(0,1),unpack=1)

m = abs(x2 - nwv) == min(abs(x2 - nwv))
y2 /= y2[m]

m = abs(x3 - nwv) == min(abs(x3 - nwv))
y3 /= y3[m]

ax1.plot(x3,y3,'b',label='Matthias template')
ax1.plot(x2,y2,'r',label='Marrianne template')
#plt.plot(x3,y3,'r')

ax1.plot([3646,3646],[0,3],'r--')

z = interp1d(x3,y3,bounds_error = 0, fill_value = 0 )
norm = z(spectrum.wavelengths)
norm[norm <=1.e-1] = 1.e6
res = (z(spectrum.wavelengths) - BCflux)/norm

#ax2.plot(spectrum.wavelengths,res,'k')
ax1.set_xlim([2500,4500])
#ax2.set_xlim([2500,4500])
ax1.legend(loc='upper left')
#plt.show()
plt.savefig('Mattiascomp.pdf',fmt='pdf')
sp.savetxt('BC.txt',sp.c_[spectrum.wavelengths,BCflux])
plt.show()


F2,(ax21) = plt.subplots(1,1,sharex='col')

norm = 1.e-15
lwidth = 3.e8

T = sp.r_[1.e3:3.e4:4j]
n = sp.r_[1.e8:9.e13:4j]
#for t in T:
#    for en in n:
for i in range(4):
    t = T[i]
    en = n[i]
    pvector = [ norm, t, 0.1, 0.0, lwidth, #balmer continuum
                    norm/2.5, t, en, 0.0, lwidth] #pseudo continuum

    BCflux = model.model_flux(pvector)
    ax21.plot(spectrum.wavelengths,BCflux,label=r'T$_{\rm e}$ = %1.1f n$_{\rm e}$ = %1.1f'%(sp.log10(t),sp.log10(en)))
ax21.set_xlim([min(spectrum.wavelengths), 5500])
ax21.legend(loc='upper right')
plt.savefig('BC2.png',fmt='png')
plt.show()
