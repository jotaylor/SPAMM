import numpy as np    
def defaultreadmaskregions(maskType=None,mask_FWHM_broad=5000,mask_FWHM_narrow=1000):
        cont = [[1275,1295],[1315,1330],[1351,1362],[1452,1520],[1680,1735],[1786,1834],[1940,2040],[2148,2243],[4770,4800],[5100,5130],[2190,2210],[3007,3027],[2190,2210],[3600,3700]]
        pureFe = [[1350,1365],[1427,1480],[1490,1505],[1705,1730],[1780,1800],[1942,2115],[2250,2300],[2333,2445],[2470,2625],[2675,2755],[2855,3010],[4500,4700]]
        #broadlines = [1215.68,1215.23,1218,1239,1243,1256,1263,1304,1307,1335,1394,1403,1402,1406,1486,1488,1531,1548,1551,1549,1640,1661,1666,1670.79,1720,1750,1786,1814,1855,1863,1883,1889,1909,2141,2321,2326,2335,2665,2796,2803,3203,3646,3835,3888.6,3934,3969,3970,4101.76,4340.5,4471.5,4686,4861,5875,5891,6562.9,7676,8446,8498,8542,8662]
        broadlines_complete = [1215.68,1215.23,1218,1239,1243,1256,1263,1304,1307,1335,1394,1403,1402,1406,1486,1488,1531,1548,1551,1640,1661,1666,1670.79,1720,1750,1786,1814,1855,1863,1883,1889,1909,2141,2321,2326,2335,2665,2796,2803,3203,4101.76,4340.5,4471.5,4686,4861,5875,5891,6562.9,7676,8446,8498,8542,8662]
        narrowlines_complete = [3426,3727,3869,4959,5007,6087,6300,6374,6548,6583,6716,6731,9069,9532]
        broadlines_reduced = [1215.68,1239,1243,1256,1263,1394,1403,1402,1406,1548,1551,1855,1863,1883,1889,1909,2796,2803,4101.76,4340.5,4861,5875,5891,6562.9]
        narrowlines_reduced = [3727,3869,4959,5007,6300,6548,6583,6716,6731]
        c = 3.e5 #km/s
        v_c_broad = mask_FWHM_broad/c
        v_c_narrow = mask_FWHM_narrow/c
        lineregions_complete = []
        for x in broadlines_complete:
            lineregions_complete.append([x-x*2*v_c_broad,x+x*2*v_c_broad]) # avoid lines by 3 FWHMs
        for i in narrowlines_complete:
            lineregions_complete.append([x-x*2*v_c_narrow,x+x*2*v_c_narrow])
        lineregions_reduced = []
        for x in broadlines_reduced:
            lineregions_reduced.append([x-x*2*v_c_broad,x+x*2*v_c_broad]) # avoid lines by 3 FWHMs
        for i in narrowlines_reduced:
            lineregions_reduced.append([x-x*2*v_c_narrow,x+x*2*v_c_narrow])
        if maskType == "Continuum":
            return cont
        if maskType == "FeRegions":
            return pureFe
        if maskType == "Cont+Fe":
            cont.extend(pureFe)
            return cont
        if maskType == "Emission lines complete":
            return lineregions_complete
        if maskType == "Emission lines reduced":
            return lineregions_reduced
            
def Mask(wavelengths=None, maskType=None,mask_FWHM_broad=5000,mask_FWHM_narrow=1000):
            _mask = [False]*len(wavelengths)
            if maskType != None:
                wavelength = np.array(wavelengths)
                wavebound = defaultreadmaskregions(maskType=maskType,mask_FWHM_broad=mask_FWHM_broad,mask_FWHM_narrow=mask_FWHM_narrow)
                for k in range(len(wavebound)):
                    select = np.nonzero((wavelength >= wavebound[k][0]) & (wavelength <= wavebound[k][1]))
                    for x in select[0]:
                        _mask[x] = True 
                if "Emission" not in maskType:
                    _mask = list(np.invert(_mask))
            return _mask