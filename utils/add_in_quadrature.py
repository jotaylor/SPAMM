#! /usr/bin/env python

import numpy as np

def add_in_quadrature(data_in):
    """ 
    Add arrays in quadrature.
    Args: 
        data_in (list, array, or tuple): Holds the arrays of individual 
            values to be added in quadrature.
    Returns:
        sum_quad (array): The sum of the input arrays added in quadrature.
    """

    sqsum = 0.
    for data in data_in:
        data_arr = np.array(data)
        sqsum += data_arr**2
    sum_quad = np.sqrt(sqsum)

    return sum_quad
