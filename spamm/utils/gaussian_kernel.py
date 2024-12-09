#! /usr/bin/env python

import numpy as np

def gaussian_kernel(x, mu, sig):
    '''
    Construct a gaussian function.

    Args:
        x (array-like): Data array
        mu (float): Mean of the distribution
        sig (float): Standard deviation

    Returns:
        Gaussian function.
    '''

    return np.exp(-0.5*((x-mu)/sig)**2)
