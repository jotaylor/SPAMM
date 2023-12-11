import numpy as np

def bool_mask(wave, masks):
    """
    Generates a boolean mask for a given wavelength array and a list of masks.

    Args:
        wave (numpy.ndarray): 
            1D array of wavelengths.
        masks (list of tuples): 
            List of spectral masks. Each mask is a tuple of two values 
            indicating the start and end of the spectral region to be masked.

    Returns:
        numpy.ndarray: 
            Boolean mask of the same size as `wave`. The mask is `True` for wavelengths
            not in any of the spectral masks and `False` for wavelengths within the spectral masks.
    """
    flag = np.zeros_like(wave, dtype=bool)
    for mask in masks:
        left, right = mask[0], mask[1]
        flag |= (wave > left) & (wave < right)
    
    return ~flag