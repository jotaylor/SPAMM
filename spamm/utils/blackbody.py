import numpy as np
from astropy import units as u
from scipy.constants import c, h, k

# c = c.si.value
# h = h.si.value
# k = k_B.si.value

def blackbody(wave, Te):
    """
    Calculate the blackbody flux at a given wavelength and temperature.
    
    Args:
        wave (float): Wavelength at which to calculate the blackbody flux (in meters).
        Te (float): Temperature of the blackbody (in Kelvin).
    
    Returns:
        float: Blackbody flux at the given wavelength and temperature (in erg/s/cm^2/sr/cm).
    """
    

    # Calculate the blackbody flux using Planck's law
    B = (2 * h * c**2 / wave**5) * (1 / (np.exp(h * c / (wave * k * Te)) - 1))
    

    # Return the blackbody flux
    return B